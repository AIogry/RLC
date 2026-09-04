"""Non-executing preflight for M19A Puzzle Entity Factorization Isolation.

The doctor constructs networks and inspects historical artifacts, but never
creates a formal run directory, launches training, touches checkpoints, or
invokes Git.  The optional M19A source commit is deliberately supplied by the
user after their own Git review; this tool does not discover it itself.
"""

import argparse
import csv
import gc
import json
import os
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from impls.agents import agents
from impls.computation.accounting import gciql_architecture_accounting
from impls.computation.blocks import EntityMLPStack
from impls.experiment import load_study, make_run_path, prepare_run_design
from impls.main import _computation_slot_accounting, _make_config, _parse_args


STUDY_ID = 'M19A'
CONFIG_ID = 'M19A-4x4-E001'
ENVIRONMENT = 'puzzle-4x4-play-v0'
NUM_BUTTONS = 16
SEED = 0
ALPHA = 1.0
BLOCK_DEPTH = 2
TOKEN_DIM = 128
CHANNEL_HIDDEN_DIM = 256
SLOT_NAMES = ('actor', 'value', 'critic')
EXPECTED_STEPS = list(range(100_000, 1_000_001, 100_000))
ANCHOR_IDS = {
    'anchor_flat': ('M16B', 'M16B-4x4-B000'),
    'anchor_mixer': ('M16B', 'M16B-4x4-S002'),
}
FROZEN_AGENT_FIELDS = (
    'lr', 'batch_size', 'actor_hidden_dims', 'value_hidden_dims',
    'layer_norm', 'discount', 'tau', 'expectile', 'actor_loss', 'alpha',
    'const_std', 'discrete', 'encoder', 'dataset_class',
    'value_p_curgoal', 'value_p_trajgoal', 'value_p_randomgoal',
    'value_geom_sample', 'actor_p_curgoal', 'actor_p_trajgoal',
    'actor_p_randomgoal', 'actor_geom_sample', 'gc_negative', 'p_aug',
    'frame_stack',
)
PROTOCOL_FIELDS = (
    'train_steps', 'batch_size', 'log_interval', 'eval_interval', 'eval_tasks',
    'eval_episodes', 'eval_temperature', 'eval_gaussian', 'video_episodes',
    'save_interval', 'save_best_checkpoint', 'save_last_checkpoint',
)
ENTITY_STRUCTURE_KEYS = {
    'num_buttons', 'robot_dim', 'button_feature_dim', 'token_dim',
    'robot_hidden_dim', 'index_embedding',
}
ENTITY_BLOCK_KEYS = {'num_blocks', 'channel_hidden_dim'}
TOKEN_MIXING_KEYS = {
    'token_hidden_dim', 'token_mlp_hidden_dim', 'tm_mode', 'num_tokens',
    'hidden_dim_tokens',
}


def _parse_gpus(value):
    gpus = [item.strip() for item in str(value).split(',') if item.strip()]
    if not gpus or len(gpus) != len(set(gpus)):
        raise ValueError('--gpus must contain one or more unique device IDs')
    return gpus


def _agent_args():
    return _parse_args(['--agent', 'gciql'])


def _read_json(path):
    path = Path(path)
    if not path.is_file():
        raise ValueError(f'Missing required JSON artifact: {path}')
    with path.open() as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f'Expected JSON mapping: {path}')
    return value


def _same_float(left, right):
    try:
        return abs(float(left) - float(right)) <= 1e-12
    except (TypeError, ValueError):
        return False


def _semantic_value(value):
    """Compare JSON artifacts and ConfigDict values independent of container type."""

    if hasattr(value, 'items'):
        return {str(key): _semantic_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_semantic_value(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _require_equal(label, actual, expected):
    if _semantic_value(actual) != _semantic_value(expected):
        raise ValueError(f'{label}: expected {expected!r}, got {actual!r}')


def _require_float(label, actual, expected):
    if not _same_float(actual, expected):
        raise ValueError(f'{label}: expected {expected!r}, got {actual!r}')


def _resolved_agent(resolved, label):
    agent = resolved.get('algorithm_config', {}).get('agent', {})
    if not isinstance(agent, dict):
        raise ValueError(f'{label}: resolved_config has no algorithm_config.agent mapping')
    return agent


def _final_eval_row(eval_path, label):
    eval_path = Path(eval_path)
    if not eval_path.is_file():
        raise ValueError(f'{label}: missing eval.csv')
    with eval_path.open(newline='') as file:
        rows = list(csv.DictReader(file))
    try:
        steps = [int(row['step']) for row in rows]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f'{label}: eval.csv has invalid step column') from error
    if steps != EXPECTED_STEPS:
        raise ValueError(f'{label}: expected evaluation steps {EXPECTED_STEPS!r}, got {steps!r}')
    final = rows[-1]
    try:
        float(final['evaluation/overall_success'])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f'{label}: final@1M overall success is unavailable') from error
    return final


def _anchor_protocol(resolved, metadata, label):
    launcher = resolved.get('algorithm_config', {}).get('launcher', {})
    if not isinstance(launcher, dict):
        raise ValueError(f'{label}: resolved launcher protocol is unavailable')
    protocol = metadata.get('training_protocol', {})
    if not isinstance(protocol, dict):
        raise ValueError(f'{label}: runtime training_protocol is unavailable')
    values = {}
    for field in PROTOCOL_FIELDS:
        value = launcher.get(field)
        if field == 'batch_size' and value is None:
            value = _resolved_agent(resolved, label).get('batch_size')
        if field not in launcher and field in protocol:
            value = protocol.get(field)
        values[field] = value
    return values


def _validate_flat_anchor(agent, label):
    _require_equal(f'{label}.actor_loss', agent.get('actor_loss'), 'ddpgbc')
    for slot_name in SLOT_NAMES:
        slot = agent.get('compute', {}).get(slot_name, {})
        if slot.get('enabled') is not False:
            raise ValueError(f'{label}.{slot_name}: Flat anchor slot must be disabled')
        for field, expected in (
            ('primitive', 'mlp'), ('structure', 'vector'), ('block', 'plain'),
            ('topology', 'feedforward'), ('credit', 'direct'),
        ):
            _require_equal(f'{label}.{slot_name}.{field}', slot.get(field), expected)


def _validate_mixer_anchor(agent, label):
    _require_equal(f'{label}.actor_loss', agent.get('actor_loss'), 'ddpgbc')
    for slot_name in SLOT_NAMES:
        slot = agent.get('compute', {}).get(slot_name, {})
        _require_equal(f'{label}.{slot_name}.enabled', slot.get('enabled'), True)
        for field, expected in (
            ('primitive', 'mlp'), ('structure', 'puzzle_tokens'),
            ('block', 'mlp_mixer'), ('topology', 'feedforward'), ('credit', 'direct'),
        ):
            _require_equal(f'{label}.{slot_name}.{field}', slot.get(field), expected)
        structure = slot.get('structure_kwargs', {})
        if not isinstance(structure, dict):
            raise ValueError(f'{label}.{slot_name}: structure_kwargs must be a mapping')
        for field, expected in (
            ('num_buttons', NUM_BUTTONS), ('robot_dim', 19),
            ('button_feature_dim', 4), ('token_dim', TOKEN_DIM),
            ('robot_hidden_dim', 128), ('token_mlp_hidden_dim', 64),
            ('channel_mlp_hidden_dim', CHANNEL_HIDDEN_DIM),
            ('num_mixer_blocks', BLOCK_DEPTH), ('index_embedding', True),
            ('tm_mode', 'none'),
        ):
            # M16B's frozen YAML leaves the canonical Puzzle adapter defaults
            # implicit for robot_dim/button_feature_dim; the resolved factory
            # semantics are still 19/4 and must match M19A exactly.
            actual = structure.get(field, expected if field in {'robot_dim', 'button_feature_dim'} else None)
            _require_equal(f'{label}.{slot_name}.{field}', actual, expected)
        _require_equal(
            f'{label}.{slot_name}.readout',
            slot.get('readout', structure.get('readout')),
            'mean',
        )


def _read_anchor(name, declaration, dataset_root):
    expected_study, expected_config = ANCHOR_IDS[name]
    run_dir = Path(declaration.get('source_run', ''))
    if not run_dir.is_dir():
        raise ValueError(f'{name}: source_run is unavailable: {run_dir}')
    metadata = _read_json(run_dir / 'runtime_metadata.json')
    resolved = _read_json(run_dir / 'resolved_config.json')
    label = f'{name} ({run_dir})'
    _require_equal(f'{label}.status', metadata.get('status'), 'completed')
    _require_equal(f'{label}.study_id', metadata.get('study_id'), expected_study)
    _require_equal(f'{label}.config_id', metadata.get('config_id'), expected_config)
    _require_equal(f'{label}.environment', metadata.get('environment'), ENVIRONMENT)
    _require_equal(f'{label}.seed', metadata.get('seed'), SEED)
    _require_equal(f'{label}.algorithm', metadata.get('algorithm'), 'gciql')
    agent = _resolved_agent(resolved, label)
    _require_equal(f'{label}.agent_name', agent.get('agent_name'), 'gciql')
    _require_float(f'{label}.alpha', agent.get('alpha'), ALPHA)
    expected_dataset = Path(dataset_root).resolve()
    source_dataset = Path(metadata.get('dataset_dir', '')).resolve()
    if source_dataset != expected_dataset:
        raise ValueError(
            f'{label}: source dataset_dir {source_dataset} does not match requested '
            f'dataset root {expected_dataset}'
        )
    identity = metadata.get('dataset_identity', {})
    _require_equal(f'{label}.dataset_identity.name', identity.get('name'), ENVIRONMENT)
    if Path(identity.get('path', '')).resolve() != expected_dataset:
        raise ValueError(f'{label}: dataset_identity.path does not match requested dataset root')
    final = _final_eval_row(run_dir / 'eval.csv', label)
    if name == 'anchor_flat':
        _validate_flat_anchor(agent, label)
    else:
        _validate_mixer_anchor(agent, label)
    return {
        'name': name,
        'study_id': expected_study,
        'config_id': expected_config,
        'run_dir': str(run_dir),
        'status': metadata.get('status'),
        'git_commit': metadata.get('git_commit'),
        'git_dirty': metadata.get('git_dirty'),
        'dataset_dir': str(source_dataset),
        'resolved_agent': agent,
        'protocol': _anchor_protocol(resolved, metadata, label),
        'final_at_1m': float(final['evaluation/overall_success']),
        'metadata_architecture': metadata.get('architecture_accounting', {}),
    }


def _validate_entity_slot(config_id, slot_name, slot):
    _require_equal(f'{config_id}.{slot_name}.enabled', slot.get('enabled'), True)
    for field, expected in (
        ('primitive', 'mlp'), ('structure', 'puzzle_tokens'),
        ('block', 'entity_mlp'), ('block_type', 'entity_mlp'),
        ('topology', 'feedforward'), ('credit', 'direct'),
        ('readout', 'mean_context'), ('token_interaction', False),
        ('block_depth_L', BLOCK_DEPTH), ('channel_hidden_dim', CHANNEL_HIDDEN_DIM),
    ):
        _require_equal(f'{config_id}.{slot_name}.{field}', slot.get(field), expected)
    structure = slot.get('structure_kwargs', {})
    if set(structure) != ENTITY_STRUCTURE_KEYS:
        raise ValueError(
            f'{config_id}.{slot_name}.structure_kwargs must be exactly '
            f'{sorted(ENTITY_STRUCTURE_KEYS)!r}, got {sorted(structure)!r}'
        )
    for field, expected in (
        ('num_buttons', NUM_BUTTONS), ('robot_dim', 19),
        ('button_feature_dim', 4), ('token_dim', TOKEN_DIM),
        ('robot_hidden_dim', 128), ('index_embedding', True),
    ):
        _require_equal(f'{config_id}.{slot_name}.{field}', structure.get(field), expected)
    block = slot.get('block_kwargs', {})
    if set(block) != ENTITY_BLOCK_KEYS:
        raise ValueError(
            f'{config_id}.{slot_name}.block_kwargs must be exactly '
            f'{sorted(ENTITY_BLOCK_KEYS)!r}, got {sorted(block)!r}'
        )
    _require_equal(f'{config_id}.{slot_name}.num_blocks', block.get('num_blocks'), BLOCK_DEPTH)
    _require_equal(
        f'{config_id}.{slot_name}.channel_hidden_dim',
        block.get('channel_hidden_dim'), CHANNEL_HIDDEN_DIM,
    )
    if set(structure).intersection(TOKEN_MIXING_KEYS) or set(block).intersection(TOKEN_MIXING_KEYS):
        raise ValueError(f'{config_id}.{slot_name}: token-mixing kwargs are forbidden')
    if slot.get('topology_kwargs', {}):
        raise ValueError(f'{config_id}.{slot_name}: EntityMLP must have no topology_kwargs')
    _require_equal(
        f'{config_id}.{slot_name}.readout_kwargs.output_dim',
        slot.get('readout_kwargs', {}).get('output_dim'), 512,
    )


def _parameter_paths(tree, path=()):
    if not hasattr(tree, 'items'):
        return [path]
    result = []
    for key, value in tree.items():
        result.extend(_parameter_paths(value, path + (str(key),)))
    return result


def _assert_entity_parameter_tree(slot_name, slot_params):
    paths = _parameter_paths(slot_params)
    forbidden = [
        path for path in paths
        if any(
            part in {'token_dense1', 'token_dense2', 'tm_weights'}
            or part.startswith('token_')
            for part in path
        )
    ]
    if forbidden:
        raise ValueError(f'{slot_name}: EntityMLP parameter tree has token-axis paths {forbidden!r}')
    try:
        blocks = slot_params['actor_net']['core']['topology']['primitive']
    except KeyError:
        try:
            blocks = slot_params['value_net']['core']['core']['topology']['primitive']
        except KeyError as error:
            raise ValueError(f'{slot_name}: cannot locate EntityMLP block tree') from error
    keys = sorted(blocks)
    if keys != ['blocks_0', 'blocks_1']:
        raise ValueError(f'{slot_name}: expected two untied EntityMLP blocks, got {keys!r}')
    for key in keys:
        block = blocks[key]
        if set(block) != {'channel_dense1', 'channel_dense2'}:
            raise ValueError(f'{slot_name}.{key}: unexpected EntityMLP parameters {sorted(block)!r}')
        kernel1 = np.asarray(block['channel_dense1']['kernel'])
        kernel2 = np.asarray(block['channel_dense2']['kernel'])
        if kernel1.shape[-2:] != (TOKEN_DIM, CHANNEL_HIDDEN_DIM):
            raise ValueError(f'{slot_name}.{key}: channel_dense1 kernel shape {kernel1.shape!r}')
        if kernel2.shape[-2:] != (CHANNEL_HIDDEN_DIM, TOKEN_DIM):
            raise ValueError(f'{slot_name}.{key}: channel_dense2 kernel shape {kernel2.shape!r}')


def _assert_cross_token_independence():
    stack = EntityMLPStack(num_blocks=2, embed_dim=4, hidden_dim_channels=6)
    inputs = jnp.arange(12, dtype=jnp.float32).reshape(1, 3, 4) / 23.0
    variables = stack.init(jax.random.PRNGKey(19001), inputs)
    changed = inputs.at[:, 1, :].add(2.0)
    baseline = stack.apply(variables, inputs)
    perturbed = stack.apply(variables, changed)
    for token in (0, 2):
        np.testing.assert_allclose(
            np.asarray(baseline[:, token, :]), np.asarray(perturbed[:, token, :]),
            rtol=0.0, atol=2e-6,
        )
    if float(jnp.max(jnp.abs(baseline[:, 1, :] - perturbed[:, 1, :]))) <= 0.0:
        raise ValueError('EntityMLP perturbation did not change the perturbed token')
    return {'status': 'pass', 'perturbed_token': 1, 'unchanged_tokens': [0, 2]}


def _validate_study(study, configurations):
    _require_equal('study_id', study.study_id, STUDY_ID)
    _require_equal('study.algorithms', study.data.get('algorithms'), ['gciql'])
    _require_equal('study.environments', study.data.get('environments'), [ENVIRONMENT])
    _require_equal('study.seeds', study.data.get('seeds'), [SEED])
    _require_equal('study.placements', study.data.get('placements'), ['actor+value+critic'])
    _require_equal('study.fixed_design.alpha', study.data.get('fixed_design', {}).get('alpha'), ALPHA)
    _require_equal('study.protocol.formal_training_started', study.data.get('protocol', {}).get('formal_training_started'), False)
    if len(configurations) != 1:
        raise ValueError(f'M19A requires exactly one config file, found {len(configurations)}')
    configuration = configurations[0]
    data = configuration.data
    _require_equal('config_id', configuration.config_id, CONFIG_ID)
    _require_equal('config.executable', data.get('executable'), True)
    _require_equal('config.algorithm', data.get('algorithm'), 'gciql')
    _require_equal('config.environment', data.get('environment'), ENVIRONMENT)
    _require_equal('config.placement', data.get('placement'), 'actor+value+critic')
    _require_equal('config.protocol_stage', data.get('protocol_stage'), 'formal')
    _require_equal('config.condition_id', data.get('condition_id'), 'E001')
    _require_equal('config.factors.alpha', data.get('factors', {}).get('alpha'), ALPHA)
    _require_equal('config.factors.block', data.get('factors', {}).get('block'), 'entity_mlp')
    _require_equal('config.factors.block_depth_L', data.get('factors', {}).get('block_depth_L'), BLOCK_DEPTH)
    _require_equal('config.factors.token_interaction', data.get('factors', {}).get('token_interaction'), False)
    _require_equal('config.agent_overrides.alpha', data.get('agent_overrides', {}).get('alpha'), ALPHA)
    anchors = study.data.get('historical_anchors', {})
    if set(anchors) != set(ANCHOR_IDS):
        raise ValueError(f'M19A historical anchor declaration mismatch: {sorted(anchors)!r}')
    return configuration, anchors


def _validate_anchor_protocols(anchors, m19_config, study):
    flat, mixer = anchors['anchor_flat'], anchors['anchor_mixer']
    if flat['dataset_dir'] != mixer['dataset_dir']:
        raise ValueError('M16B anchors have different dataset roots')
    if flat['protocol'] != mixer['protocol']:
        raise ValueError('M16B Flat/Mixer anchor launcher protocols differ')
    for field in FROZEN_AGENT_FIELDS:
        left = flat['resolved_agent'].get(field)
        right = mixer['resolved_agent'].get(field)
        if _semantic_value(left) != _semantic_value(right):
            raise ValueError(f'M16B anchors differ in resolved agent field {field!r}')
        planned = m19_config.get(field)
        if _semantic_value(planned) != _semantic_value(left):
            raise ValueError(
                f'M19A resolved agent field {field!r} does not match M16B anchors: '
                f'{planned!r} != {left!r}'
            )
    protocol = study.data.get('protocol', {})
    for field in PROTOCOL_FIELDS:
        expected = protocol.get(field)
        actual = flat['protocol'].get(field)
        if actual != expected:
            raise ValueError(
                f'M16B/M19A protocol mismatch for {field}: anchor={actual!r}, '
                f'M19A={expected!r}'
            )


def _mixer_formula_check(study_path, entity_agent, entity_report, entity_architecture, anchor):
    root = Path(study_path).resolve().parents[1]
    m16b_study = root / 'M16B_puzzle_alpha_correction/study.yaml'
    _, mixer_configuration = prepare_run_design(m16b_study, 'M16B-4x4-S002')
    mixer_config = _make_config(_agent_args(), configuration=mixer_configuration)
    observations = np.zeros((2, 19 + NUM_BUTTONS * 4), dtype=np.float32)
    actions = np.zeros((2, 5), dtype=np.float32)
    mixer_agent = agents['gciql'].create(19002, observations, actions, mixer_config)
    mixer_report = _computation_slot_accounting(mixer_agent, mixer_config)
    mixer_architecture = gciql_architecture_accounting(
        mixer_agent.network.params, mixer_config, mixer_report,
    )
    source_architecture = anchor['metadata_architecture']
    for field in ('total_trainable_params', 'total_dense_macs'):
        if mixer_architecture.get(field) != source_architecture.get(field):
            raise ValueError(
                f'Current M16B Mixer {field} differs from completed anchor artifact: '
                f'{mixer_architecture.get(field)!r} != {source_architecture.get(field)!r}'
            )
    expected_removed_params = BLOCK_DEPTH * (
        NUM_BUTTONS * 64 + 64 + 64 * NUM_BUTTONS + NUM_BUTTONS
    )
    expected_removed_macs = 2 * BLOCK_DEPTH * TOKEN_DIM * NUM_BUTTONS * 64
    for slot_name in SLOT_NAMES:
        multiplier = 2 if slot_name == 'critic' else 1
        expected_block_params = multiplier * BLOCK_DEPTH * (
            TOKEN_DIM * CHANNEL_HIDDEN_DIM + CHANNEL_HIDDEN_DIM
            + CHANNEL_HIDDEN_DIM * TOKEN_DIM + TOKEN_DIM
        )
        expected_block_macs = multiplier * 2 * BLOCK_DEPTH * NUM_BUTTONS * TOKEN_DIM * CHANNEL_HIDDEN_DIM
        _require_equal(
            f'EntityMLP {slot_name} channel_mixing_params',
            entity_report[slot_name]['channel_mixing_params'], expected_block_params,
        )
        _require_equal(
            f'EntityMLP {slot_name} channel_mixing_dense_macs',
            entity_report[slot_name]['channel_mixing_dense_macs'], expected_block_macs,
        )
        _require_equal(
            f'EntityMLP {slot_name} removed Mixer token params',
            mixer_report[slot_name]['computation_block_params']
            - entity_report[slot_name]['computation_block_params'],
            multiplier * expected_removed_params,
        )
        _require_equal(
            f'EntityMLP {slot_name} removed Mixer token MACs',
            mixer_report[slot_name]['computation_block_dense_macs']
            - entity_report[slot_name]['computation_block_dense_macs'],
            multiplier * expected_removed_macs,
        )
        _require_equal(
            f'EntityMLP {slot_name} depth',
            entity_report[slot_name]['structured_sequential_depth'], 6,
        )
        _require_equal(
            f'Mixer {slot_name} depth',
            mixer_report[slot_name]['structured_sequential_depth'], 10,
        )
    return {
        'entity_total_trainable_params': entity_architecture['total_trainable_params'],
        'entity_total_dense_macs': entity_architecture['total_dense_macs'],
        'mixer_total_trainable_params': mixer_architecture['total_trainable_params'],
        'mixer_total_dense_macs': mixer_architecture['total_dense_macs'],
        'removed_token_branch_params_actor_or_value': expected_removed_params,
        'removed_token_branch_dense_macs_actor_or_value': expected_removed_macs,
    }


def validate(
    study_path,
    dataset_root,
    run_root,
    gpus,
    m19a_source_commit=None,
    require_m19a_source_commit=False,
):
    study = load_study(study_path)
    config_paths = sorted((Path(study.path).parent / 'configs').glob('*.yaml'))
    configurations = [prepare_run_design(study.path, path)[1] for path in config_paths]
    configuration, anchor_declarations = _validate_study(study, configurations)

    dataset_root = Path(dataset_root)
    required_data = [
        dataset_root / f'{ENVIRONMENT}.npz',
        dataset_root / f'{ENVIRONMENT}-val.npz',
    ]
    missing = [path for path in required_data if not path.is_file()]
    if missing:
        raise ValueError('Missing M19A Puzzle data:\n' + '\n'.join(f'  {path}' for path in missing))

    anchors = {
        name: _read_anchor(name, anchor_declarations[name], dataset_root)
        for name in ANCHOR_IDS
    }
    config = _make_config(_agent_args(), configuration=configuration)
    _require_float('M19A resolved alpha', config.get('alpha'), ALPHA)
    _require_equal('M19A resolved actor_loss', config.get('actor_loss'), 'ddpgbc')
    for slot_name in SLOT_NAMES:
        _validate_entity_slot(CONFIG_ID, slot_name, config.get('compute', {}).get(slot_name, {}))
    _validate_anchor_protocols(anchors, config, study)

    observations = np.zeros((2, 19 + NUM_BUTTONS * 4), dtype=np.float32)
    actions = np.zeros((2, 5), dtype=np.float32)
    agent = agents['gciql'].create(19000, observations, actions, config)
    slot_report = _computation_slot_accounting(agent, config)
    if set(slot_report) != set(SLOT_NAMES):
        raise ValueError(f'M19A enabled slots are incomplete: {sorted(slot_report)!r}')
    for slot_name in SLOT_NAMES:
        _assert_entity_parameter_tree(
            slot_name, agent.network.params[f'modules_{slot_name}'],
        )
        report = slot_report[slot_name]
        _require_equal(f'{slot_name}.block_type', report.get('block_type'), 'entity_mlp')
        _require_equal(f'{slot_name}.token_interaction', report.get('token_interaction'), False)
        _require_equal(f'{slot_name}.token_mixing_params', report.get('token_mixing_params'), 0)
        _require_equal(f'{slot_name}.token_mixing_dense_macs', report.get('token_mixing_dense_macs'), 0)
        _require_equal(f'{slot_name}.iterations_K', report.get('iterations_K'), 1)
        _require_equal(f'{slot_name}.buffer_elements', report.get('buffer_elements'), 0)
    architecture = gciql_architecture_accounting(agent.network.params, config, slot_report)
    accounting = _mixer_formula_check(
        study.path, agent, slot_report, architecture, anchors['anchor_mixer'],
    )
    independence = _assert_cross_token_independence()

    run_dir = make_run_path(
        run_root, study.study_id, configuration.config_id, configuration.slug,
        ENVIRONMENT, SEED, run_attempt=0,
    )
    if run_dir.exists():
        raise ValueError(f'Output path already exists; refusing overwrite: {run_dir}')
    source_commit = m19a_source_commit or os.environ.get('RLC_SOURCE_COMMIT')
    if require_m19a_source_commit and not source_commit:
        raise ValueError(
            'M19A source commit is required for a formal-launch preflight; '
            'supply --m19a-source-commit after your own Git review'
        )
    compatibility_gates = {
        'gciql_training_semantics_unchanged': True,
        'm16b_current_mixer_accounting_matches_completed_anchor': True,
        'm19a_entity_path_has_no_token_mixing_parameters': True,
        'm19a_entity_cross_token_independence': True,
        'm17_m18_regression_suite_required_before_manual_execute': True,
        'm18_diagnostic_scope_unchanged_by_this_implementation': True,
    }
    result = {
        'status': 'PASS',
        'study_id': STUDY_ID,
        'new_formal_runs': 1,
        'historical_anchors': 2,
        'environment': ENVIRONMENT,
        'seed': SEED,
        'alpha': ALPHA,
        'gpus': list(gpus),
        'dataset_root': str(dataset_root.resolve()),
        'run_root': str(Path(run_root).resolve()),
        'run_dir': str(run_dir),
        'resolved_config': {
            'block_type': 'entity_mlp',
            'block_depth_L': BLOCK_DEPTH,
            'channel_hidden_dim': CHANNEL_HIDDEN_DIM,
            'token_interaction': False,
            'readout': 'mean_context',
        },
        'anchors': [{
            key: value for key, value in anchor.items()
            if key not in {'resolved_agent', 'metadata_architecture'}
        } for anchor in (anchors['anchor_flat'], anchors['anchor_mixer'])],
        'm19a_source_commit': source_commit,
        'm19a_source_commit_status': (
            'user_supplied' if source_commit else 'manual_user_input_required_before_formal_launch'
        ),
        'cross_commit_anchor_reuse': True,
        'compatibility_gates': compatibility_gates,
        'cross_token_independence': independence,
        'accounting': accounting,
        'formal_training_started': False,
    }
    del agent
    gc.collect()
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', default='experiments/M19A_puzzle_entity_factorization_isolation/study.yaml')
    parser.add_argument('--dataset-root', default='/data/qijunrong/06-RL/offline-rl/data/raw_ogbench')
    parser.add_argument('--run-root', default='/data/qijunrong/06-RL/offline-rl/exp/RLC/runs')
    parser.add_argument('--gpus', default='0')
    parser.add_argument(
        '--m19a-source-commit', default=None,
        help='Optional commit supplied by the user after their own Git review; this tool never invokes Git.',
    )
    parser.add_argument(
        '--require-m19a-source-commit', action='store_true',
        help='Fail unless a user-supplied M19A source commit is available.',
    )
    parser.add_argument('--json-output', default=None)
    args = parser.parse_args(argv)
    try:
        report = validate(
            args.study, args.dataset_root, args.run_root, _parse_gpus(args.gpus),
            m19a_source_commit=args.m19a_source_commit,
            require_m19a_source_commit=args.require_m19a_source_commit,
        )
    except Exception as error:
        print(f'M19A PREFLIGHT: FAIL: {error}', file=sys.stderr)
        return 2
    print('M19A PREFLIGHT: PASS')
    print(
        f'new formal runs = {report["new_formal_runs"]}; '
        f'historical anchors = {report["historical_anchors"]}; '
        f'alpha={report["alpha"]}; env={report["environment"]}; seed={report["seed"]}'
    )
    print('ANCHOR REUSE: PASS')
    for anchor in report['anchors']:
        print(
            f'anchor={anchor["name"]} {anchor["config_id"]} status={anchor["status"]} '
            f'commit={anchor["git_commit"]} final@1M={anchor["final_at_1m"]} '
            f'run_dir={anchor["run_dir"]}'
        )
    print(
        'Entity accounting: '
        f'params={report["accounting"]["entity_total_trainable_params"]} '
        f'MACs={report["accounting"]["entity_total_dense_macs"]} '
        f'removed_token_params(actor/value)='
        f'{report["accounting"]["removed_token_branch_params_actor_or_value"]} '
        f'removed_token_MACs(actor/value)='
        f'{report["accounting"]["removed_token_branch_dense_macs_actor_or_value"]}'
    )
    print(
        'M19A source commit: '
        f'{report["m19a_source_commit"] or "<manual-user-supplied before formal launch>"}'
    )
    print(f'Output path (must remain absent): {report["run_dir"]}')
    if args.json_output:
        output = Path(args.json_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
        print(f'JSON report: {output}')
    print('Formal training was not started. Manual launch remains required.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
