"""Reusable, provenance-checked checkpoint reevaluation infrastructure.

This module intentionally treats reevaluation as a separate experiment object.
It reconstructs an agent from the immutable source run's resolved config,
streams one compact row per episode, and never writes to the source run.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import pickle
import re
import socket
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from .management import config_fingerprint, jsonable
from ..utils.checkpointing import normalize_checkpoint_selector, resolve_checkpoint
from ..utils.reproducibility import derive_seed


COMMON_EPISODE_SEED_SCHEME = 'common_task_episode_v1'


REEVALUATION_STATUSES = {'running', 'completed', 'failed', 'aborted', 'invalid'}
EPISODE_FIELDS = (
    'study_id',
    'config_id',
    'config_slug',
    'environment',
    'training_seed',
    'checkpoint_step',
    'task_id',
    'task_name',
    'episode_index',
    'evaluation_seed',
    'task_seed',
    'episode_seed',
    'actor_seed',
    'noise_seed',
    'success',
    'episode_return',
    'episode_length',
    'terminated',
    'truncated',
    'paired_episode_id',
    'final_info_json',
)
TASK_SUMMARY_FIELDS = (
    'task_id',
    'task_name',
    'episode_count',
    'success_count',
    'success_rate',
    'success_standard_error',
    'success_wilson_low_95',
    'success_wilson_high_95',
    'return_mean',
    'return_std',
    'episode_length_mean',
    'episode_length_std',
)


class ReevaluationError(ValueError):
    """Raised when a reevaluation source, protocol, or artifact is invalid."""


def _utc_now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as file:
        json.dump(jsonable(value), file, indent=2, sort_keys=True)
        file.write('\n')


def _read_json(path):
    with Path(path).open() as file:
        value = json.load(file)
    if not isinstance(value, Mapping):
        raise ReevaluationError(f'Expected JSON mapping: {path}')
    return dict(value)


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open('rb') as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _git_metadata(repo_root=None):
    repo_root = Path(repo_root or Path(__file__).resolve().parents[2])
    try:
        commit = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ['git', 'status', '--porcelain', '--untracked-files=all'],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        return {'git_commit': commit, 'git_dirty': bool(status)}
    except (OSError, subprocess.CalledProcessError):
        return {'git_commit': None, 'git_dirty': None}


def load_reevaluation_spec(path):
    path = Path(path).resolve()
    with path.open() as file:
        spec = yaml.safe_load(file) or {}
    if not isinstance(spec, dict):
        raise ReevaluationError(f'Reevaluation spec must be a mapping: {path}')
    required = ('reevaluation_id', 'source_study_id', 'source_run_root', 'environments', 'protocol')
    missing = [key for key in required if key not in spec]
    if missing:
        raise ReevaluationError(f'Spec {path} is missing fields: {missing}')
    if 'checkpoint' in spec:
        try:
            checkpoint = normalize_checkpoint_selector(spec['checkpoint'])
        except (TypeError, ValueError, KeyError) as error:
            raise ReevaluationError(f'Invalid checkpoint selector in {path}: {error}') from error
    elif 'checkpoint_step' in spec:
        try:
            checkpoint = normalize_checkpoint_selector({'selector': 'step', 'step': spec['checkpoint_step']})
        except (TypeError, ValueError, KeyError) as error:
            raise ReevaluationError(f'Invalid checkpoint_step in {path}: {error}') from error
    else:
        raise ReevaluationError(f'Spec {path} requires checkpoint or checkpoint_step')
    protocol = dict(spec['protocol'] or {})
    protocol_defaults = {
        'task_selection': 'all',
        'episodes_per_task': 100,
        'evaluation_seed': 20260819,
        'seed_scheme': COMMON_EPISODE_SEED_SCHEME,
        'eval_temperature': 0.0,
        'eval_gaussian': None,
        'video_episodes': 0,
    }
    for key, value in protocol_defaults.items():
        protocol.setdefault(key, value)
    if protocol['task_selection'] != 'all':
        raise ReevaluationError('The generic runner currently requires task_selection=all')
    if int(protocol['episodes_per_task']) <= 0:
        raise ReevaluationError('episodes_per_task must be positive')
    if int(protocol['video_episodes']) != 0:
        raise ReevaluationError('Post-hoc reevaluation requires video_episodes=0')
    if protocol['seed_scheme'] != COMMON_EPISODE_SEED_SCHEME:
        raise ReevaluationError(f'Unsupported seed_scheme: {protocol["seed_scheme"]!r}')
    spec['protocol'] = protocol
    spec['checkpoint'] = checkpoint
    spec['checkpoint_step'] = checkpoint.get('step') if checkpoint['selector'] == 'step' else None
    spec['environments'] = list(spec['environments'])
    spec['training_seeds'] = [int(seed) for seed in spec.get('training_seeds', [0, 1, 2])]
    spec['configs'] = spec.get('configs', 'all')
    spec['_spec_path'] = str(path)
    return spec


def protocol_fingerprint(protocol):
    """Fingerprint only scientific reevaluation protocol fields."""

    fields = {
        'task_selection': protocol['task_selection'],
        'episodes_per_task': int(protocol['episodes_per_task']),
        'evaluation_seed': int(protocol['evaluation_seed']),
        'seed_scheme': protocol['seed_scheme'],
        'eval_temperature': float(protocol['eval_temperature']),
        'eval_gaussian': protocol['eval_gaussian'],
        'video_episodes': int(protocol['video_episodes']),
    }
    return config_fingerprint(fields)


def campaign_root(reeval_root, spec):
    return Path(reeval_root) / spec['source_study_id'] / spec['reevaluation_id']


_SEED_COMPONENT_RE = re.compile(r'^seed_(?P<seed>\d+)(?:__attempt_(?P<attempt>\d+))?$')
_SCOPED_PROVENANCE_SCOPE_FIELDS = (
    'study_id',
    'config_id',
    'environment',
    'training_seed',
    'run_attempt',
    'git_commit',
    'checkpoint_sha256',
)


def _split_config_identity(source_run_dir):
    """Parse a canonical Run path, including an explicit nonzero attempt."""

    try:
        config_component = Path(source_run_dir).parent.parent.name
        config_id, config_slug = config_component.split('__', 1)
        environment = Path(source_run_dir).parent.name
        study_id = Path(source_run_dir).parent.parent.parent.name
        seed_component = Path(source_run_dir).name
        seed_match = _SEED_COMPONENT_RE.fullmatch(seed_component)
        if seed_match is None:
            raise ValueError
        training_seed = int(seed_match.group('seed'))
        run_attempt = int(seed_match.group('attempt') or 0)
    except (ValueError, IndexError) as error:
        raise ReevaluationError(f'Cannot parse canonical source run path: {source_run_dir}') from error
    return study_id, config_id, config_slug, environment, training_seed, run_attempt


def _scoped_provenance_exception(exception, *, identity, git_dirty):
    """Validate the narrow, auditable exception for one dirty source Run.

    Normal source validation remains clean-only.  An exception is accepted
    only when its complete identity and final checkpoint hash are declared in
    advance, so a broad ``allow_dirty`` switch cannot accidentally admit a
    different source Run.
    """

    if git_dirty is False:
        if exception is not None:
            raise ReevaluationError('A scoped provenance exception is only valid for git_dirty=true sources')
        return None
    if git_dirty is not True:
        raise ReevaluationError(f'Formal source run is not clean: {git_dirty!r}')
    if not isinstance(exception, Mapping):
        raise ReevaluationError(
            'Formal source run is dirty and has no explicit scoped provenance exception'
        )
    exception = dict(exception)
    if exception.get('provenance_status') != 'scoped_exception':
        raise ReevaluationError('Dirty source exception must declare provenance_status=scoped_exception')
    for field in ('reason', 'evidence_level'):
        if not isinstance(exception.get(field), str) or not exception[field].strip():
            raise ReevaluationError(f'Dirty source exception requires a non-empty {field!r}')
    scope = exception.get('scope')
    if not isinstance(scope, Mapping):
        raise ReevaluationError('Dirty source exception requires a mapping scope')
    missing = [field for field in _SCOPED_PROVENANCE_SCOPE_FIELDS if field not in scope]
    if missing:
        raise ReevaluationError(
            f'Dirty source exception scope is missing required fields: {missing}'
        )
    for field in _SCOPED_PROVENANCE_SCOPE_FIELDS[:-1]:
        if str(scope[field]) != str(identity[field]):
            raise ReevaluationError(
                f'Dirty source exception scope mismatch for {field}: '
                f'expected={scope[field]!r}, observed={identity[field]!r}'
            )
    checkpoint_sha256 = scope['checkpoint_sha256']
    if not isinstance(checkpoint_sha256, str) or len(checkpoint_sha256) != 64:
        raise ReevaluationError('Dirty source exception scope requires a SHA-256 checkpoint hash')
    return jsonable(exception)


def _resolved_payload(resolved):
    return {
        'study': resolved.get('study'),
        'configuration': resolved.get('configuration'),
        'algorithm_config': resolved.get('algorithm_config', {}),
    }


def validate_source_run(
    source_run_dir,
    *,
    checkpoint_step=None,
    checkpoint_selector=None,
    expected_study_id=None,
    expected_environment=None,
    check_checkpoint_metadata=True,
    allow_running_source_if_checkpoint_best=False,
    scoped_provenance_exception=None,
):
    """Validate a source run and return immutable provenance information."""

    source_run_dir = Path(source_run_dir).resolve()
    metadata_path = source_run_dir / 'runtime_metadata.json'
    resolved_path = source_run_dir / 'resolved_config.json'
    if not metadata_path.is_file():
        raise ReevaluationError(f'Missing source runtime_metadata.json: {metadata_path}')
    if not resolved_path.is_file():
        raise ReevaluationError(f'Missing source resolved_config.json: {resolved_path}')
    metadata = _read_json(metadata_path)
    resolved = _read_json(resolved_path)
    (
        study_id,
        config_id,
        config_slug,
        environment,
        training_seed,
        run_attempt,
    ) = _split_config_identity(source_run_dir)
    if checkpoint_selector is None:
        if checkpoint_step is None:
            raise ReevaluationError('A checkpoint_step or checkpoint_selector is required')
        checkpoint_selector = {'selector': 'step', 'step': int(checkpoint_step)}
    try:
        normalized_selector = normalize_checkpoint_selector(checkpoint_selector)
    except (TypeError, ValueError, KeyError) as error:
        raise ReevaluationError(f'Invalid checkpoint selector {checkpoint_selector!r}: {error}') from error
    allowed_statuses = {'completed'}
    if allow_running_source_if_checkpoint_best and normalized_selector['selector'] == 'best':
        # A semantic best artifact is complete and independently checksummed
        # before its index pointer is published.  This narrow exception is
        # intentionally unavailable to last/final or numeric checkpoints.
        allowed_statuses.add('running')
    if metadata.get('status') not in allowed_statuses:
        raise ReevaluationError(
            f'Source run status {metadata.get("status")!r} is not allowed for '
            f'checkpoint selector {normalized_selector!r}; allowed={sorted(allowed_statuses)!r}'
        )
    source_identity = {
        'study_id': study_id,
        'config_id': config_id,
        'environment': environment,
        'training_seed': training_seed,
        'run_attempt': run_attempt,
        'git_commit': metadata.get('git_commit'),
    }
    provenance_exception = _scoped_provenance_exception(
        scoped_provenance_exception,
        identity=source_identity,
        git_dirty=metadata.get('git_dirty'),
    )
    expected = {
        'study_id': expected_study_id,
        'environment': expected_environment,
        'config_id': config_id,
        'config_slug': config_slug,
        'training_seed': training_seed,
        'run_attempt': run_attempt,
    }
    observed = {
        'study_id': metadata.get('study_id'),
        'environment': metadata.get('environment'),
        'config_id': metadata.get('config_id'),
        'config_slug': metadata.get('config_slug'),
        'training_seed': metadata.get('seed'),
        # Pre-attempt artifacts did not record this field.  Their canonical
        # path is unambiguously attempt zero, so retain compatibility while
        # requiring an explicit match for nonzero attempts.
        'run_attempt': metadata.get('run_attempt', 0),
    }
    for key, wanted in expected.items():
        if wanted is not None and str(observed[key]) != str(wanted):
            raise ReevaluationError(
                f'Source path/metadata mismatch for {key}: path={wanted!r}, metadata={observed[key]!r}'
            )
    if metadata.get('run_dir') and Path(metadata['run_dir']).resolve() != source_run_dir:
        raise ReevaluationError('runtime_metadata.run_dir does not match source run directory')
    stored_fingerprint = metadata.get('resolved_config_fingerprint')
    resolved_fingerprint = resolved.get('resolved_config_fingerprint')
    calculated_fingerprint = config_fingerprint(_resolved_payload(resolved))
    if not stored_fingerprint or stored_fingerprint != resolved_fingerprint or stored_fingerprint != calculated_fingerprint:
        raise ReevaluationError(
            'Resolved config fingerprint mismatch: '
            f'metadata={stored_fingerprint!r}, file={resolved_fingerprint!r}, calculated={calculated_fingerprint!r}'
        )

    try:
        checkpoint = resolve_checkpoint(
            source_run_dir,
            normalized_selector,
            load_metadata=check_checkpoint_metadata,
        )
    except (FileNotFoundError, OSError, TypeError, ValueError, KeyError) as error:
        raise ReevaluationError(f'Cannot resolve checkpoint {normalized_selector!r}: {error}') from error
    checkpoint_path = Path(checkpoint['checkpoint_path'])
    checkpoint_step = int(checkpoint['checkpoint_step'])
    if not checkpoint_path.is_file():
        raise ReevaluationError(f'Missing requested checkpoint: {checkpoint_path}')
    if checkpoint_step == 500000 and checkpoint_path.name != 'params_500000.pkl':
        raise ReevaluationError('M10A-R001 requires exactly params_500000.pkl')
    checkpoint_metadata = checkpoint.get('checkpoint_metadata') or {}
    if check_checkpoint_metadata:
        with checkpoint_path.open('rb') as file:
            checkpoint_payload = pickle.load(file)
        if not isinstance(checkpoint_payload, Mapping) or 'agent' not in checkpoint_payload:
            raise ReevaluationError(f'Checkpoint is not a serialized RLC agent: {checkpoint_path}')
        checkpoint_metadata = checkpoint_payload.get('checkpoint_metadata') or checkpoint_metadata
        for key, source_key in (
            ('environment', 'environment'),
            ('study_id', 'study_id'),
            ('config_id', 'config_id'),
            ('config_slug', 'config_slug'),
            ('git_commit', 'git_commit'),
        ):
            if checkpoint_metadata.get(key) != metadata.get(source_key):
                raise ReevaluationError(
                    f'Checkpoint metadata mismatch for {key}: '
                    f'checkpoint={checkpoint_metadata.get(key)!r}, source={metadata.get(source_key)!r}'
                )
        if int(checkpoint_metadata.get('seed', -1)) != training_seed:
            raise ReevaluationError('Checkpoint metadata seed does not match source training seed')
    checkpoint_sha256 = sha256_file(checkpoint_path)
    if provenance_exception is not None:
        expected_hash = provenance_exception['scope']['checkpoint_sha256']
        if checkpoint_sha256 != expected_hash:
            raise ReevaluationError(
                'Dirty source exception checkpoint hash mismatch: '
                f'expected={expected_hash!r}, observed={checkpoint_sha256!r}'
            )

    return {
        'source_run_dir': str(source_run_dir),
        'source_study_id': study_id,
        'source_config_id': config_id,
        'source_config_slug': config_slug,
        'source_environment': environment,
        'source_training_seed': training_seed,
        'source_run_attempt': run_attempt,
        'source_git_commit': metadata.get('git_commit'),
        'source_git_dirty': metadata.get('git_dirty'),
        'source_provenance_status': (
            'scoped_exception' if provenance_exception is not None else 'verified_clean'
        ),
        'source_provenance_exception': provenance_exception,
        'source_run_status_at_validation': metadata.get('status'),
        'source_resolved_config_fingerprint': stored_fingerprint,
        'source_metadata': metadata,
        'resolved_config': resolved,
        'checkpoint_step': int(checkpoint_step),
        'checkpoint_path': str(checkpoint_path),
        'checkpoint_sha256': checkpoint_sha256,
        'checkpoint_metadata': jsonable(checkpoint_metadata),
        'requested_checkpoint_selector': normalized_selector,
        'resolved_checkpoint_role': checkpoint['checkpoint_role'],
        'resolved_checkpoint_step': int(checkpoint['checkpoint_step']),
    }


def _resolved_agent_config(resolved):
    algorithm_config = resolved.get('algorithm_config', {})
    if isinstance(algorithm_config, Mapping) and isinstance(algorithm_config.get('agent'), Mapping):
        return dict(algorithm_config['agent'])
    if isinstance(resolved.get('agent'), Mapping):
        return dict(resolved['agent'])
    raise ReevaluationError('resolved_config.json has no algorithm_config.agent mapping')


def _mutable_mapping(value):
    """Copy mappings without copying array leaves."""

    if not isinstance(value, Mapping):
        return value
    return {key: _mutable_mapping(item) for key, item in value.items()}


def _require_mapping(value, *, label):
    if not isinstance(value, Mapping):
        raise ReevaluationError(f'{label} must be a mapping')
    return value


def _legacy_mixer_body_to_modular(legacy_body, target_body, *, label):
    """Map the pre-M17 Puzzle Mixer ownership layout into the modular layout.

    This is an inference-only checkpoint adapter. M15/M16 checkpoints owned
    adapter, Mixer blocks, and readout under one ``PuzzleStructuredBody``;
    M17 preserved their forward semantics but separated those components into
    adapter/core/readout modules. The mapping is intentionally exact and fails
    on any unknown or missing parameter name.
    """

    legacy_body = _require_mapping(legacy_body, label=f'{label} legacy body')
    target = _mutable_mapping(_require_mapping(target_body, label=f'{label} target body'))
    adapter = _mutable_mapping(_require_mapping(target.get('adapter'), label=f'{label}.adapter'))
    core = _mutable_mapping(_require_mapping(target.get('core'), label=f'{label}.core'))
    topology = _mutable_mapping(
        _require_mapping(core.get('topology'), label=f'{label}.core.topology')
    )
    primitive = _mutable_mapping(
        _require_mapping(
            topology.get('primitive'),
            label=f'{label}.core.topology.primitive',
        )
    )
    readout = _mutable_mapping(_require_mapping(target.get('readout'), label=f'{label}.readout'))
    target_block_names = sorted(primitive)
    if any(not name.startswith('blocks_') for name in target_block_names):
        raise ReevaluationError(f'{label} target has an unsupported non-Mixer block layout')
    source_block_names = {f'mixer_blocks_{name.removeprefix("blocks_")}' for name in target_block_names}
    allowed_source_names = set(adapter) | source_block_names | set(readout)
    if set(legacy_body) != allowed_source_names:
        raise ReevaluationError(
            f'{label} legacy Mixer parameter names differ from the exact supported layout: '
            f'observed={sorted(legacy_body)!r}, expected={sorted(allowed_source_names)!r}'
        )
    for name in adapter:
        adapter[name] = legacy_body[name]
    for target_name in target_block_names:
        source_name = f'mixer_blocks_{target_name.removeprefix("blocks_")}'
        primitive[target_name] = legacy_body[source_name]
    for name in readout:
        readout[name] = legacy_body[name]
    topology['primitive'] = primitive
    core['topology'] = topology
    target['adapter'] = adapter
    target['core'] = core
    target['readout'] = readout
    return target


def _require_matching_leaf_shapes(source, target, *, label):
    source_leaves = _tree_leaves(source)
    target_leaves = _tree_leaves(target)
    source_shapes = [tuple(np.asarray(leaf).shape) for leaf in source_leaves]
    target_shapes = [tuple(np.asarray(leaf).shape) for leaf in target_leaves]
    if source_shapes != target_shapes:
        raise ReevaluationError(
            f'{label} legacy Mixer parameter shapes do not match the reconstructed network'
        )


def _legacy_mixer_params_to_modular(source_params, target_params):
    """Translate a complete pre-M17 GCIQL Puzzle-Mixer parameter tree.

    No source values are changed: each leaf is transplanted into the ownership
    location that M17 proved forward-equivalent. Non-Mixer and unknown layouts
    are rejected rather than heuristically coerced.
    """

    source = _require_mapping(source_params, label='legacy checkpoint params')
    target = _mutable_mapping(_require_mapping(target_params, label='target checkpoint params'))
    module_names = (
        'modules_actor',
        'modules_value',
        'modules_critic',
        'modules_target_critic',
    )
    if set(source) != set(module_names) or set(target) != set(module_names):
        raise ReevaluationError('Legacy Mixer adapter requires the exact GCIQL module set')

    source_actor = _require_mapping(source['modules_actor'], label='legacy modules_actor')
    target_actor = _mutable_mapping(_require_mapping(target['modules_actor'], label='target modules_actor'))
    if set(source_actor) != {'actor_net', 'mean_net'} or set(target_actor) != {'actor_net', 'mean_net'}:
        raise ReevaluationError('Legacy Mixer adapter requires the canonical GCIQL actor layout')
    source_actor_net = _require_mapping(source_actor['actor_net'], label='legacy actor_net')
    if set(source_actor_net) != {'topology'}:
        raise ReevaluationError('Checkpoint is not a pre-M17 Puzzle-Mixer actor layout')
    source_actor_topology = _require_mapping(
        source_actor_net['topology'], label='legacy actor topology'
    )
    if set(source_actor_topology) != {'primitive'}:
        raise ReevaluationError('Checkpoint actor topology is not an exact legacy Mixer layout')
    target_actor['actor_net'] = _legacy_mixer_body_to_modular(
        source_actor_topology['primitive'],
        target_actor['actor_net'],
        label='modules_actor.actor_net',
    )
    target_actor['mean_net'] = source_actor['mean_net']
    target['modules_actor'] = target_actor

    for module_name in module_names[1:]:
        source_module = _require_mapping(source[module_name], label=f'legacy {module_name}')
        target_module = _mutable_mapping(
            _require_mapping(target[module_name], label=f'target {module_name}')
        )
        if set(source_module) != {'value_net', 'value_readout'} or set(target_module) != {
            'value_net', 'value_readout'
        }:
            raise ReevaluationError(
                f'Legacy Mixer adapter requires the canonical GCIQL value layout for {module_name}'
            )
        source_value_net = _require_mapping(
            source_module['value_net'], label=f'legacy {module_name}.value_net'
        )
        if set(source_value_net) != {'core'}:
            raise ReevaluationError(
                f'Checkpoint {module_name}.value_net is not an exact legacy Mixer layout'
            )
        source_core = _require_mapping(
            source_value_net['core'], label=f'legacy {module_name}.value_net.core'
        )
        source_topology = _require_mapping(
            source_core.get('topology'), label=f'legacy {module_name}.value_net.core.topology'
        )
        if set(source_topology) != {'primitive'}:
            raise ReevaluationError(
                f'Checkpoint {module_name}.value_net topology is not an exact legacy Mixer layout'
            )
        target_value_net = _mutable_mapping(
            _require_mapping(target_module['value_net'], label=f'target {module_name}.value_net')
        )
        if set(target_value_net) != {'core'}:
            raise ReevaluationError(
                f'Target {module_name}.value_net is not the expected modular Mixer layout'
            )
        target_value_net['core'] = _legacy_mixer_body_to_modular(
            source_topology['primitive'],
            target_value_net['core'],
            label=f'{module_name}.value_net.core',
        )
        target_module['value_net'] = target_value_net
        target_module['value_readout'] = source_module['value_readout']
        target[module_name] = target_module

    from ..utils.flax_utils import _mapping_like

    converted = _mapping_like(target_params, target)
    _require_matching_leaf_shapes(converted, target_params, label='Converted')
    return converted


def _restore_legacy_mixer_agent_for_reevaluation(agent, checkpoint_path):
    """Restore a legacy Puzzle-Mixer checkpoint for inference only.

    The source optimizer state is deliberately not migrated: post-hoc
    reevaluation never calls ``update`` and retains the freshly initialized
    optimizer/model state. Source network parameters and RNG are preserved.
    """

    try:
        with Path(checkpoint_path).open('rb') as file:
            payload = pickle.load(file)
        source_agent = _require_mapping(payload.get('agent'), label='legacy checkpoint agent')
        source_network = _require_mapping(source_agent.get('network'), label='legacy checkpoint network')
        source_params = source_network.get('params')
        source_rng = source_agent.get('rng')
    except (OSError, EOFError, pickle.UnpicklingError, AttributeError) as error:
        raise ReevaluationError(f'Cannot read legacy Mixer checkpoint: {checkpoint_path}') from error
    if source_rng is None:
        raise ReevaluationError('Legacy Mixer checkpoint has no agent RNG')
    source_model_state = source_network.get('model_state', {})
    if source_model_state not in ({}, None):
        raise ReevaluationError('Legacy Mixer checkpoint has unsupported non-empty model state')
    converted_params = _legacy_mixer_params_to_modular(source_params, agent.network.params)
    restored = agent.replace(network=agent.network.replace(params=converted_params), rng=source_rng)
    print(f'Restored legacy Mixer layout for inference from {checkpoint_path}')
    return restored


def _restore_agent_for_reevaluation(agent, checkpoint_path):
    """Restore native checkpoints, with one exact legacy Mixer inference adapter."""

    from ..utils.flax_utils import restore_agent_from_checkpoint

    try:
        return restore_agent_from_checkpoint(agent, checkpoint_path), 'native_state_dict'
    except ValueError as native_error:
        try:
            restored = _restore_legacy_mixer_agent_for_reevaluation(agent, checkpoint_path)
        except ReevaluationError as compatibility_error:
            raise ReevaluationError(
                'Checkpoint cannot be restored natively and is not a supported '
                'pre-M17 Puzzle-Mixer inference layout'
            ) from native_error
        return restored, 'legacy_mixer_layout_adapter_v1'


def _make_restored_agent(provenance):
    from ..agents import agents
    from ..utils.datasets import GCDataset, HGCDataset, MultiHGCDataset
    from ..utils.env_utils import make_env_and_datasets

    metadata = provenance['source_metadata']
    resolved = provenance['resolved_config']
    config = _resolved_agent_config(resolved)
    algorithm = metadata.get('algorithm') or config.get('agent_name')
    if algorithm not in agents:
        raise ReevaluationError(f'Unsupported source algorithm for reevaluation: {algorithm!r}')
    if config.get('agent_name') not in (None, algorithm):
        raise ReevaluationError(
            f'Agent mismatch between runtime metadata and resolved config: {algorithm!r} vs {config.get("agent_name")!r}'
        )
    dataset_dir = metadata.get('dataset_dir')
    environment = provenance['source_environment']
    training_seed = provenance['source_training_seed']
    env, raw_train, _ = make_env_and_datasets(
        environment,
        frame_stack=config.get('frame_stack'),
        seed=derive_seed(training_seed, 3),
        dataset_seed=derive_seed(training_seed, 1),
        dataset_dir=dataset_dir,
    )
    dataset_classes = {
        'GCDataset': GCDataset,
        'HGCDataset': HGCDataset,
        'MultiHGCDataset': MultiHGCDataset,
    }
    dataset_name = config.get('dataset_class')
    if dataset_name not in dataset_classes:
        raise ReevaluationError(f'Unsupported source dataset_class: {dataset_name!r}')
    train_dataset = dataset_classes[dataset_name](
        raw_train,
        config,
        rng=derive_seed(training_seed, 11),
    )
    example_batch = train_dataset.sample(1)
    if config.get('discrete', False):
        example_batch['actions'] = np.full_like(
            example_batch['actions'], env.action_space.n - 1
        )
    agent = agents[algorithm].create(
        training_seed,
        example_batch['observations'],
        example_batch['actions'],
        config,
    )
    restored, restore_mode = _restore_agent_for_reevaluation(agent, provenance['checkpoint_path'])
    provenance['checkpoint_restore_mode'] = restore_mode
    leaves = []
    leaves.extend(jax_leaf for jax_leaf in _tree_leaves(restored.network.params))
    if not all(np.all(np.isfinite(np.asarray(leaf))) for leaf in leaves):
        raise ReevaluationError('Restored checkpoint contains non-finite parameters')
    return restored, env, config, example_batch


def _tree_leaves(value):
    import jax

    return jax.tree_util.tree_leaves(value)


def _restore_probe(agent, example_batch, algorithm, training_seed):
    import jax

    observations = example_batch['observations']
    goals = example_batch.get('high_actor_goals', example_batch.get('actor_goals'))
    if algorithm == 'coghp':
        observations = observations[0]
        if goals is not None and np.asarray(goals).ndim > 1:
            goals = goals[0]
    action = agent.sample_actions(
        observations,
        goals,
        seed=jax.random.PRNGKey(derive_seed(training_seed, 0xA11CE)),
    )
    if not np.all(np.isfinite(np.asarray(action))):
        raise ReevaluationError('Restored action probe is non-finite')


def _wilson_interval(success_count, episode_count, z=1.959963984540054):
    if episode_count <= 0:
        raise ValueError('episode_count must be positive')
    p = success_count / episode_count
    denominator = 1.0 + z * z / episode_count
    center = (p + z * z / (2.0 * episode_count)) / denominator
    radius = z * math.sqrt(p * (1.0 - p) / episode_count + z * z / (4.0 * episode_count**2)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def _mean_std(values):
    values = [float(value) for value in values]
    if not values:
        return None, None
    return statistics.mean(values), statistics.pstdev(values) if len(values) > 1 else 0.0


def _read_episode_rows(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline='') as file:
        reader = csv.DictReader(file)
        if tuple(reader.fieldnames or ()) != EPISODE_FIELDS:
            raise ReevaluationError(f'Unexpected episode_results.csv header: {path}')
        rows = list(reader)
    seen = set()
    for row in rows:
        key = (int(row['task_id']), int(row['episode_index']))
        if key in seen:
            raise ReevaluationError(f'Duplicate episode key in {path}: {key}')
        seen.add(key)
        if row['paired_episode_id'] != f'task{key[0]:02d}_ep{key[1]:03d}':
            raise ReevaluationError(f'Invalid paired_episode_id in {path}: {row["paired_episode_id"]}')
    return rows


def _write_task_and_overall_summaries(output_dir, rows, *, task_names, episodes_per_task, checkpoint_step, evaluation_seed):
    grouped = {}
    for row in rows:
        grouped.setdefault(int(row['task_id']), []).append(row)
    task_rows = []
    task_success = {}
    for task_id in sorted(task_names):
        task_rows_for_id = grouped.get(task_id, [])
        if len(task_rows_for_id) != episodes_per_task:
            raise ReevaluationError(
                f'Task {task_id} has {len(task_rows_for_id)} rows; expected {episodes_per_task}'
            )
        success_count = sum(float(row['success']) for row in task_rows_for_id)
        returns = [float(row['episode_return']) for row in task_rows_for_id]
        lengths = [float(row['episode_length']) for row in task_rows_for_id]
        return_mean, return_std = _mean_std(returns)
        length_mean, length_std = _mean_std(lengths)
        success_rate = success_count / episodes_per_task
        wilson_low, wilson_high = _wilson_interval(int(success_count), episodes_per_task)
        task_success[f'task{task_id}'] = success_rate
        task_rows.append({
            'task_id': task_id,
            'task_name': task_names[task_id],
            'episode_count': episodes_per_task,
            'success_count': int(success_count),
            'success_rate': success_rate,
            'success_standard_error': math.sqrt(success_rate * (1.0 - success_rate) / episodes_per_task),
            'success_wilson_low_95': wilson_low,
            'success_wilson_high_95': wilson_high,
            'return_mean': return_mean,
            'return_std': return_std,
            'episode_length_mean': length_mean,
            'episode_length_std': length_std,
        })
    with (Path(output_dir) / 'task_summary.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=TASK_SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(task_rows)
    task_count = len(task_rows)
    overall_success = sum(task_success.values()) / task_count
    overall_se = math.sqrt(
        sum(row['success_rate'] * (1.0 - row['success_rate']) / episodes_per_task for row in task_rows)
    ) / task_count
    summary = {
        'status': 'completed',
        'evaluation/overall_success': overall_success,
        'task_count': task_count,
        'total_episodes': len(rows),
        'checkpoint_step': int(checkpoint_step),
        'evaluation_seed': int(evaluation_seed),
        'episodes_per_task': int(episodes_per_task),
        'task_success': task_success,
        'overall_episode_sampling_se': overall_se,
    }
    _write_json(Path(output_dir) / 'summary.json', summary)
    return summary


def _metadata_for_reevaluation(provenance, spec, *, output_dir, assigned_gpu=None, repo_root=None):
    source_meta = provenance['source_metadata']
    protocol = dict(spec['protocol'])
    git_info = _git_metadata(repo_root)
    return {
        'reevaluation_id': spec['reevaluation_id'],
        'status': 'running',
        'start_time': _utc_now(),
        'source_run_dir': provenance['source_run_dir'],
        'source_study_id': provenance['source_study_id'],
        'source_config_id': provenance['source_config_id'],
        'source_config_slug': provenance['source_config_slug'],
        'source_environment': provenance['source_environment'],
        'source_training_seed': provenance['source_training_seed'],
        'source_git_commit': provenance['source_git_commit'],
        'source_git_dirty': provenance['source_git_dirty'],
        'source_resolved_config_fingerprint': provenance['source_resolved_config_fingerprint'],
        'requested_checkpoint_selector': spec['checkpoint'],
        'resolved_checkpoint_role': provenance.get('resolved_checkpoint_role', 'explicit'),
        'resolved_checkpoint_step': provenance.get('resolved_checkpoint_step', provenance['checkpoint_step']),
        'checkpoint_step': provenance['checkpoint_step'],
        'checkpoint_path': provenance['checkpoint_path'],
        'checkpoint_sha256': provenance['checkpoint_sha256'],
        'checkpoint_metadata': provenance['checkpoint_metadata'],
        'reevaluation_git_commit': git_info['git_commit'],
        'reevaluation_git_dirty': git_info['git_dirty'],
        'dataset_dir': source_meta.get('dataset_dir'),
        'ogbench_module': source_meta.get('ogbench_module'),
        'jax_backend': None,
        'jax_devices': [],
        'hostname': socket.gethostname(),
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'assigned_gpu': None if assigned_gpu is None else str(assigned_gpu),
        'evaluation_protocol': protocol | {
            'task_count': int(spec.get('expected_task_count', 0)),
            'total_episodes': int(spec.get('expected_task_count', 0)) * int(protocol['episodes_per_task']),
        },
        'reevaluation_protocol_fingerprint': protocol_fingerprint(protocol),
        'source_resolved_config_path': str(Path(provenance['source_run_dir']) / 'resolved_config.json'),
        'output_dir': str(Path(output_dir).resolve()),
    }


def _update_status(metadata_path, metadata, status, failure_reason=None):
    metadata = dict(metadata)
    metadata['status'] = status
    metadata['end_time'] = _utc_now()
    if failure_reason is not None:
        metadata['failure_reason'] = str(failure_reason)
    _write_json(metadata_path, metadata)


def run_checkpoint_reevaluation(
    source_run_dir,
    spec,
    *,
    reeval_root,
    resume=False,
    assigned_gpu=None,
    repo_root=None,
):
    """Evaluate one source checkpoint with safe incremental output."""

    provenance = validate_source_run(
        source_run_dir,
        checkpoint_selector=spec['checkpoint'],
        expected_study_id=spec['source_study_id'],
        expected_environment=spec['environments'][0] if len(spec['environments']) == 1 else None,
    )
    if provenance['source_environment'] not in spec['environments']:
        raise ReevaluationError(
            f'Source environment {provenance["source_environment"]!r} is not in spec environments'
        )
    output_dir = (
        campaign_root(reeval_root, spec)
        / f'{provenance["source_config_id"]}__{provenance["source_config_slug"]}'
        / provenance['source_environment']
        / f'seed_{provenance["source_training_seed"]:03d}'
    )
    metadata_path = output_dir / 'reevaluation_metadata.json'
    if output_dir.exists() and not resume:
        raise FileExistsError(
            f'Reevaluation output exists; use --resume after validating it: {output_dir}'
        )
    if output_dir.exists() and resume and not metadata_path.exists():
        raise ReevaluationError(
            f'Cannot resume an output directory without reevaluation_metadata.json: {output_dir}'
        )
    if output_dir.exists() and resume and metadata_path.exists():
        existing_metadata = _read_json(metadata_path)
        if existing_metadata.get('checkpoint_sha256') != provenance['checkpoint_sha256']:
            raise ReevaluationError('Resume checkpoint SHA256 mismatch')
        if existing_metadata.get('requested_checkpoint_selector') is not None:
            if existing_metadata['requested_checkpoint_selector'] != spec['checkpoint']:
                raise ReevaluationError('Resume checkpoint selector mismatch')
        elif spec['checkpoint']['selector'] == 'step' and existing_metadata.get('checkpoint_step') != spec['checkpoint']['step']:
            raise ReevaluationError('Resume legacy checkpoint selector mismatch')
        if existing_metadata.get('reevaluation_protocol_fingerprint') != protocol_fingerprint(spec['protocol']):
            raise ReevaluationError('Resume reevaluation protocol fingerprint mismatch')
        if existing_metadata.get('status') == 'completed':
            return _read_json(output_dir / 'summary.json')
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = _metadata_for_reevaluation(
        provenance,
        spec,
        output_dir=output_dir,
        assigned_gpu=assigned_gpu,
        repo_root=repo_root,
    )
    if metadata_path.exists():
        previous = _read_json(metadata_path)
        metadata['start_time'] = previous.get('start_time', metadata['start_time'])
    _write_json(metadata_path, metadata)

    episode_path = output_dir / 'episode_results.csv'
    try:
        rows = _read_episode_rows(episode_path)
        protocol = spec['protocol']
        expected_episodes = int(protocol['episodes_per_task'])
        restored, env, config, example_batch = _make_restored_agent(provenance)
        _restore_probe(
            restored,
            example_batch,
            provenance['source_metadata'].get('algorithm', config.get('agent_name')),
            provenance['source_training_seed'],
        )
        import jax

        metadata['jax_backend'] = jax.default_backend()
        metadata['jax_devices'] = [str(device) for device in jax.devices()]
        task_infos = getattr(env.unwrapped, 'task_infos', None)
        if task_infos is None:
            raise ReevaluationError('Expected task_infos for task_selection=all')
        task_names = {index + 1: str(item['task_name']) for index, item in enumerate(task_infos)}
        expected_task_count = int(spec.get('expected_task_count', len(task_names)))
        if len(task_names) != expected_task_count:
            raise ReevaluationError(
                f'Environment task count {len(task_names)} != expected {expected_task_count}'
            )
        metadata['evaluation_protocol']['task_count'] = len(task_names)
        metadata['evaluation_protocol']['total_episodes'] = len(task_names) * expected_episodes
        _write_json(metadata_path, metadata)

        for row in rows:
            expected_identity = {
                'study_id': provenance['source_study_id'],
                'config_id': provenance['source_config_id'],
                'config_slug': provenance['source_config_slug'],
                'environment': provenance['source_environment'],
                'training_seed': str(provenance['source_training_seed']),
                'checkpoint_step': str(provenance['checkpoint_step']),
                'evaluation_seed': str(protocol['evaluation_seed']),
            }
            for field, expected_value in expected_identity.items():
                if str(row.get(field)) != expected_value:
                    raise ReevaluationError(
                        f'Existing episode row has incompatible {field}: '
                        f'{row.get(field)!r} != {expected_value!r}'
                    )
            task_id = int(row['task_id'])
            if task_id not in task_names or row.get('task_name') != task_names[task_id]:
                raise ReevaluationError('Existing episode row has an invalid task identity')

        existing_keys = {(int(row['task_id']), int(row['episode_index'])) for row in rows}
        expected_keys = {
            (task_id, episode_index)
            for task_id in task_names
            for episode_index in range(expected_episodes)
        }
        if not existing_keys.issubset(expected_keys):
            raise ReevaluationError('Existing episode rows contain keys outside the requested protocol')
        new_file = not episode_path.exists()
        with episode_path.open('a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=EPISODE_FIELDS)
            if new_file:
                writer.writeheader()
                file.flush()
            for task_id in sorted(task_names):
                missing_indices = [
                    index for index in range(expected_episodes)
                    if (task_id, index) not in existing_keys
                ]
                from ..utils.evaluation import evaluate_episodes

                task_records = evaluate_episodes(
                    restored,
                    env,
                    task_id=task_id,
                    task_name=task_names[task_id],
                    config=config,
                    evaluation_seed=int(protocol['evaluation_seed']),
                    episode_indices=missing_indices,
                    eval_temperature=float(protocol['eval_temperature']),
                    eval_gaussian=protocol['eval_gaussian'],
                    seed_scheme=protocol['seed_scheme'],
                )
                for record in task_records:
                    output_record = {
                        'study_id': provenance['source_study_id'],
                        'config_id': provenance['source_config_id'],
                        'config_slug': provenance['source_config_slug'],
                        'environment': provenance['source_environment'],
                        'training_seed': provenance['source_training_seed'],
                        'checkpoint_step': provenance['checkpoint_step'],
                        **record,
                    }
                    writer.writerow(output_record)
                    file.flush()
                    rows.append(output_record)
                    existing_keys.add((task_id, int(record['episode_index'])))
        if len(rows) != len(expected_keys) or existing_keys != expected_keys:
            raise ReevaluationError(
                f'Completed row set has {len(rows)} rows; expected {len(expected_keys)}'
            )
        summary = _write_task_and_overall_summaries(
            output_dir,
            rows,
            task_names=task_names,
            episodes_per_task=expected_episodes,
            checkpoint_step=provenance['checkpoint_step'],
            evaluation_seed=protocol['evaluation_seed'],
        )
        metadata['status'] = 'completed'
        metadata['end_time'] = _utc_now()
        _write_json(metadata_path, metadata)
        return summary
    except KeyboardInterrupt as error:
        _update_status(metadata_path, metadata, 'aborted', error)
        raise
    except BaseException as error:
        _update_status(metadata_path, metadata, 'failed', error)
        raise


def _sample_sd(values):
    values = [float(value) for value in values]
    return statistics.stdev(values) if len(values) > 1 else 0.0


def aggregate_campaign(spec, *, reeval_root, source_runs):
    """Rebuild campaign-level summaries from per-run immutable outputs."""

    root = campaign_root(reeval_root, spec)
    root.mkdir(parents=True, exist_ok=True)
    checkpoint_spec = spec.get('checkpoint') or {
        'selector': 'step',
        'step': int(spec['checkpoint_step']),
    }
    manifest = []
    completed_task_rows = []
    for provenance in source_runs:
        output_dir = (
            root
            / f'{provenance["source_config_id"]}__{provenance["source_config_slug"]}'
            / provenance['source_environment']
            / f'seed_{provenance["source_training_seed"]:03d}'
        )
        metadata_path = output_dir / 'reevaluation_metadata.json'
        summary_path = output_dir / 'summary.json'
        status = 'planned'
        summary = {}
        if metadata_path.exists():
            metadata = _read_json(metadata_path)
            status = metadata.get('status', 'invalid')
            if metadata.get('checkpoint_sha256') != provenance['checkpoint_sha256']:
                status = 'invalid'
        if summary_path.exists():
            summary = _read_json(summary_path)
        manifest.append({
            'study_id': provenance['source_study_id'],
            'reevaluation_id': spec['reevaluation_id'],
            'config_id': provenance['source_config_id'],
            'config_slug': provenance['source_config_slug'],
            'environment': provenance['source_environment'],
            'training_seed': provenance['source_training_seed'],
            'requested_checkpoint_selector': json.dumps(checkpoint_spec, sort_keys=True),
            'resolved_checkpoint_role': provenance.get('resolved_checkpoint_role', 'explicit'),
            'resolved_checkpoint_step': provenance.get('resolved_checkpoint_step', provenance['checkpoint_step']),
            'checkpoint_step': provenance['checkpoint_step'],
            'checkpoint_sha256': provenance['checkpoint_sha256'],
            'status': status,
            'overall_success': summary.get('evaluation/overall_success'),
            'output_dir': str(output_dir),
        })
        task_path = output_dir / 'task_summary.csv'
        if status == 'completed' and task_path.exists():
            with task_path.open(newline='') as file:
                for row in csv.DictReader(file):
                    completed_task_rows.append({
                        **row,
                        'config_id': provenance['source_config_id'],
                        'config_slug': provenance['source_config_slug'],
                        'environment': provenance['source_environment'],
                        'training_seed': provenance['source_training_seed'],
                    })
    manifest_fields = (
        'study_id', 'reevaluation_id', 'config_id', 'config_slug', 'environment',
        'training_seed', 'requested_checkpoint_selector', 'resolved_checkpoint_role',
        'resolved_checkpoint_step', 'checkpoint_step', 'checkpoint_sha256', 'status',
        'overall_success', 'output_dir',
    )
    with (root / 'manifest.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=manifest_fields)
        writer.writeheader()
        writer.writerows(sorted(manifest, key=lambda row: (row['config_id'], row['training_seed'])))

    config_groups = {}
    for row in manifest:
        config_groups.setdefault((row['config_id'], row['config_slug'], row['environment']), []).append(row)
    config_fields = (
        'config_id', 'config_slug', 'environment', 'number_training_seeds',
        'overall_success_seed0', 'overall_success_seed1', 'overall_success_seed2',
        'overall_success_mean', 'overall_success_population_sd', 'overall_success_sample_sd',
    )
    with (root / 'config_summary.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=config_fields)
        writer.writeheader()
        for (config_id, slug, environment), group in sorted(config_groups.items()):
            values = {int(row['training_seed']): row['overall_success'] for row in group if row['overall_success'] not in (None, '')}
            numeric = [float(value) for value in values.values()]
            writer.writerow({
                'config_id': config_id,
                'config_slug': slug,
                'environment': environment,
                'number_training_seeds': len(numeric),
                'overall_success_seed0': values.get(0, ''),
                'overall_success_seed1': values.get(1, ''),
                'overall_success_seed2': values.get(2, ''),
                'overall_success_mean': statistics.mean(numeric) if numeric else '',
                'overall_success_population_sd': statistics.pstdev(numeric) if len(numeric) > 1 else (0.0 if numeric else ''),
                'overall_success_sample_sd': _sample_sd(numeric) if numeric else '',
            })

    task_groups = {}
    for row in completed_task_rows:
        key = (row['config_id'], row['config_slug'], row['environment'], row['task_id'], row['task_name'])
        task_groups.setdefault(key, {})[int(row['training_seed'])] = float(row['success_rate'])
    task_fields = (
        'config_id', 'config_slug', 'environment', 'task_id', 'task_name',
        'success_seed0', 'success_seed1', 'success_seed2', 'number_training_seeds',
        'success_mean', 'success_population_sd', 'success_sample_sd',
    )
    with (root / 'task_config_summary.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=task_fields)
        writer.writeheader()
        for key, values in sorted(task_groups.items()):
            numeric = list(values.values())
            writer.writerow({
                'config_id': key[0], 'config_slug': key[1], 'environment': key[2],
                'task_id': key[3], 'task_name': key[4],
                'success_seed0': values.get(0, ''), 'success_seed1': values.get(1, ''),
                'success_seed2': values.get(2, ''), 'number_training_seeds': len(numeric),
                'success_mean': statistics.mean(numeric) if numeric else '',
                'success_population_sd': statistics.pstdev(numeric) if len(numeric) > 1 else (0.0 if numeric else ''),
                'success_sample_sd': _sample_sd(numeric) if numeric else '',
            })
    campaign_metadata = {
        'reevaluation_id': spec['reevaluation_id'],
        'source_study_id': spec['source_study_id'],
        'checkpoint': checkpoint_spec,
        'checkpoint_step': checkpoint_spec.get('step'),
        'protocol': spec['protocol'],
        'protocol_fingerprint': protocol_fingerprint(spec['protocol']),
        'source_run_count': len(source_runs),
        'completed_run_count': sum(row['status'] == 'completed' for row in manifest),
        'manifest_path': str(root / 'manifest.csv'),
        'generated_at': _utc_now(),
    }
    _write_json(root / 'campaign_metadata.json', campaign_metadata)
    return root
