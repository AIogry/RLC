"""Generic checkpoint provenance and frozen-critic pairing for diagnostics."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import yaml

from impls.agents import resolve_agent_class
from impls.experiment import make_run_path, prepare_run_design
from impls.main import _make_config, _parse_args
from impls.utils.checkpointing import (
    checkpoint_module_fingerprint,
    parameter_module_key,
    resolve_checkpoint,
    tree_fingerprint,
)
from impls.utils.datasets import GCDataset
from impls.utils.flax_utils import restore_agent_from_checkpoint
from impls.utils.reproducibility import derive_seed


@dataclass(frozen=True)
class ActorSource:
    name: str
    label: str
    study: Path
    config: str
    run_attempt: int
    checkpoint_role: str = 'last'
    artifact_identity: str | None = None


@dataclass(frozen=True)
class LoadedActor:
    source: ActorSource
    seed: int
    run_dir: Path
    checkpoint: dict
    metadata: dict
    study: object
    configuration: object
    config: object
    agent: object
    actor_fingerprint: str
    critic_fingerprint: str


def load_protocol(path):
    path = Path(path).resolve()
    with path.open() as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, Mapping):
        raise ValueError(f'Protocol must be a mapping: {path}')
    data = dict(data)
    data['_protocol_path'] = str(path)
    return data


def _resolve(protocol_path, value):
    path = Path(value)
    return path if path.is_absolute() else (Path(protocol_path).parent / path).resolve()


def actor_sources(protocol):
    protocol_path = Path(protocol['_protocol_path'])
    return {
        name: ActorSource(
            name=name,
            label=str(spec.get('label', name)),
            study=_resolve(protocol_path, spec['study']),
            config=str(spec['config']),
            run_attempt=int(spec.get('run_attempt', 0)),
            checkpoint_role=str(spec.get('checkpoint_role', 'last')),
            artifact_identity=spec.get('artifact_identity'),
        )
        for name, spec in protocol['actors'].items()
    }


def actor_run_dir(source, *, run_root, environment, seed):
    study, configuration = prepare_run_design(source.study, source.config)
    return make_run_path(
        run_root, study.study_id, configuration.config_id, configuration.slug,
        environment, int(seed), run_attempt=source.run_attempt,
    )


def _read_json(path):
    with Path(path).open() as file:
        return json.load(file)


def inspect_actor_checkpoint(source, *, seed, run_root, environment,
                             expected_commit=None, checkpoint_selector=None,
                             expected_checkpoint_step=None):
    study, configuration = prepare_run_design(source.study, source.config)
    run_dir = actor_run_dir(source, run_root=run_root, environment=environment, seed=seed)
    metadata_path = run_dir / 'runtime_metadata.json'
    resolved_path = run_dir / 'resolved_config.json'
    if not metadata_path.is_file() or not resolved_path.is_file():
        raise FileNotFoundError(f'Missing actor metadata/config: {run_dir}')
    metadata = _read_json(metadata_path)
    expected = {
        'status': 'completed', 'study_id': study.study_id,
        'config_id': configuration.config_id, 'environment': environment,
        'seed': int(seed), 'run_attempt': source.run_attempt,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f'{source.name} seed={seed}: {key}={metadata.get(key)!r}, expected {value!r}')
    if metadata.get('git_dirty') is not False:
        raise ValueError(f'{source.name} seed={seed}: git_dirty is not false')
    if expected_commit is not None and metadata.get('git_commit') != expected_commit:
        raise ValueError(f'{source.name} seed={seed}: actor commit mismatch')
    selector = source.checkpoint_role if checkpoint_selector is None else checkpoint_selector
    checkpoint = resolve_checkpoint(run_dir, selector)
    required_step = 1_000_000 if expected_checkpoint_step is None else int(expected_checkpoint_step)
    if checkpoint['checkpoint_step'] != required_step:
        raise ValueError(f'{source.name} seed={seed}: expected step {required_step}, got {checkpoint["checkpoint_step"]}')
    if checkpoint_selector is None and checkpoint['checkpoint_role'] != 'last':
        raise ValueError(f'{source.name} seed={seed}: primary checkpoint is not last')
    dependency = metadata.get('frozen_dependencies', {}).get('frozen_critic')
    if not isinstance(dependency, Mapping):
        raise ValueError(f'{source.name} seed={seed}: missing frozen critic metadata')
    for key in ('checkpoint_sha256', 'module_fingerprint'):
        if not dependency.get(key):
            raise ValueError(f'{source.name} seed={seed}: missing critic {key}')
    return {
        'source': source, 'seed': int(seed), 'run_dir': run_dir,
        'study': study, 'configuration': configuration, 'metadata': metadata,
        'resolved': _read_json(resolved_path), 'checkpoint': checkpoint,
        'actor_fingerprint': checkpoint_module_fingerprint(checkpoint['checkpoint_path'], 'actor'),
        'critic_fingerprint': dependency['module_fingerprint'],
        'critic_sha256': dependency['checkpoint_sha256'],
    }


def inspect_critic_checkpoint(source, *, seed, run_root, environment,
                              expected_step=1_000_000, expected_commit=None):
    study, configuration = prepare_run_design(source.study, source.config)
    run_dir = actor_run_dir(source, run_root=run_root, environment=environment, seed=seed)
    metadata_path = run_dir / 'runtime_metadata.json'
    if not metadata_path.is_file():
        raise FileNotFoundError(f'Missing critic metadata: {run_dir}')
    metadata = _read_json(metadata_path)
    expected = {
        'status': 'completed', 'study_id': study.study_id,
        'config_id': configuration.config_id, 'environment': environment,
        'seed': int(seed), 'run_attempt': source.run_attempt,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f'critic seed={seed}: {key} mismatch')
    if metadata.get('git_dirty') is not False:
        raise ValueError(f'critic seed={seed}: git_dirty is not false')
    if expected_commit is not None and metadata.get('git_commit') != expected_commit:
        raise ValueError(f'critic seed={seed}: commit mismatch')
    checkpoint = resolve_checkpoint(run_dir, source.checkpoint_role)
    if checkpoint['checkpoint_role'] != 'last' or checkpoint['checkpoint_step'] != int(expected_step):
        raise ValueError(f'critic seed={seed}: expected last@{expected_step}')
    return {
        'source': source, 'seed': int(seed), 'run_dir': run_dir,
        'study': study, 'configuration': configuration, 'metadata': metadata,
        'checkpoint': checkpoint,
        'critic_fingerprint': checkpoint_module_fingerprint(checkpoint['checkpoint_path'], 'critic'),
    }


def load_actor(source, *, seed, run_root, environment, dataset,
               expected_commit=None, checkpoint_selector=None,
               expected_checkpoint_step=None):
    record = inspect_actor_checkpoint(
        source, seed=seed, run_root=run_root, environment=environment,
        expected_commit=expected_commit, checkpoint_selector=checkpoint_selector,
        expected_checkpoint_step=expected_checkpoint_step,
    )
    config = _make_config(_parse_args(['--agent', 'crl']), configuration=record['configuration'])
    example = GCDataset(
        dataset=dataset, config=config, rng=np.random.default_rng(derive_seed(seed, 11))
    ).sample(1)
    variant = config.get('runtime_variant', 'canonical')
    agent = resolve_agent_class('crl', variant).create(
        seed, example['observations'], example['actions'], config
    )
    loaded = restore_agent_from_checkpoint(agent, record['checkpoint']['checkpoint_path'])
    actor_key = parameter_module_key(loaded.network.params, 'actor')
    critic_key = parameter_module_key(loaded.network.params, 'critic')
    actor_fp = tree_fingerprint(loaded.network.params[actor_key])
    critic_fp = tree_fingerprint(loaded.network.params[critic_key])
    if actor_fp != record['actor_fingerprint'] or critic_fp != record['critic_fingerprint']:
        raise ValueError(f'{source.name} seed={seed}: restored module fingerprint mismatch')
    return LoadedActor(
        source=source, seed=int(seed), run_dir=record['run_dir'],
        checkpoint=record['checkpoint'], metadata=record['metadata'],
        study=record['study'], configuration=record['configuration'], config=config,
        agent=loaded, actor_fingerprint=actor_fp, critic_fingerprint=critic_fp,
    )


def _critic_source(protocol):
    spec = protocol['critic']
    return ActorSource(
        name='fixed_critic', label='fixed critic',
        study=_resolve(protocol['_protocol_path'], spec['study']),
        config=str(spec['config']), run_attempt=int(spec.get('run_attempt', 0)),
        checkpoint_role=str(spec.get('checkpoint_role', 'last')),
    )


def load_primary_actors(protocol, *, seed, run_root, dataset,
                        checkpoint_selector=None, expected_checkpoint_step=None):
    sources = actor_sources(protocol)
    expected_commit = protocol.get('source_commit')
    actors = {
        name: load_actor(
            source, seed=seed, run_root=run_root, environment=protocol['environment'],
            dataset=dataset, expected_commit=expected_commit,
            checkpoint_selector=checkpoint_selector,
            expected_checkpoint_step=expected_checkpoint_step,
        )
        for name, source in sources.items()
    }
    critic_source = _critic_source(protocol)
    critic = inspect_critic_checkpoint(
        critic_source, seed=seed, run_root=run_root, environment=protocol['environment'],
        expected_step=int(protocol['critic'].get('checkpoint_step', 1_000_000)),
    )
    critic_fps = {actor.critic_fingerprint for actor in actors.values()}
    critic_shas = {
        actor.metadata['frozen_dependencies']['frozen_critic']['checkpoint_sha256']
        for actor in actors.values()
    }
    if critic_fps != {critic['critic_fingerprint']}:
        raise ValueError(f'seed={seed}: critic subtree fingerprints do not agree')
    if critic_shas != {critic['checkpoint']['checkpoint_sha256']}:
        raise ValueError(f'seed={seed}: critic SHA256 values do not agree')
    for actor in actors.values():
        dependency = actor.metadata['frozen_dependencies']['frozen_critic']
        checks = {
            'source_study_id': critic['study'].study_id,
            'source_config_id': critic['configuration'].config_id,
            'source_run_attempt': critic_source.run_attempt,
            'checkpoint_role': 'last',
            'checkpoint_step': critic['checkpoint']['checkpoint_step'],
            'checkpoint_sha256': critic['checkpoint']['checkpoint_sha256'],
            'module_fingerprint': critic['critic_fingerprint'],
        }
        for key, value in checks.items():
            if dependency.get(key) != value:
                raise ValueError(f'{actor.source.name} seed={seed}: critic dependency {key} mismatch')
    return actors


def provenance_summary(protocol, *, seed, run_root):
    sources = actor_sources(protocol)
    actor_records = []
    for source in sources.values():
        record = inspect_actor_checkpoint(
            source, seed=seed, run_root=run_root, environment=protocol['environment'],
            expected_commit=protocol.get('source_commit'),
        )
        actor_records.append({
            'name': source.name, 'label': source.label, 'study': record['study'].study_id,
            'config': record['configuration'].config_id, 'attempt': source.run_attempt,
            'seed': int(seed), 'checkpoint_role': 'last',
            'checkpoint_step': record['checkpoint']['checkpoint_step'],
            'checkpoint_sha256': record['checkpoint']['checkpoint_sha256'],
            'actor_fingerprint': record['actor_fingerprint'],
            'critic_fingerprint': record['critic_fingerprint'],
            'critic_checkpoint_sha256': record['critic_sha256'],
            'source_commit': record['metadata'].get('git_commit'),
            'git_dirty': record['metadata'].get('git_dirty'),
            'run_dir': str(record['run_dir']),
            'resolved_computation': record['resolved'].get('algorithm_config', {}).get('agent', {}).get('compute', {}),
        })
    critic = inspect_critic_checkpoint(
        _critic_source(protocol), seed=seed, run_root=run_root,
        environment=protocol['environment'],
        expected_step=int(protocol['critic'].get('checkpoint_step', 1_000_000)),
    )
    return {
        'actors': actor_records,
        'critic': {
            'study': critic['study'].study_id, 'config': critic['configuration'].config_id,
            'attempt': _critic_source(protocol).run_attempt, 'seed': int(seed),
            'checkpoint': critic['checkpoint']['checkpoint_path'],
            'checkpoint_sha256': critic['checkpoint']['checkpoint_sha256'],
            'critic_fingerprint': critic['critic_fingerprint'],
            'source_commit': critic['metadata'].get('git_commit'),
            'git_dirty': critic['metadata'].get('git_dirty'),
        },
    }
