"""Declarative M11B design and result-aggregation helpers.

M11B deliberately keeps the scientific factors small.  Environment and
algorithm profiles are resolved here, while the individual configuration
YAMLs contain only stable identity and a computation condition.  This avoids
copying canonical task hyperparameters into 34 near-identical files.

The module has no training side effects.  It is used by configuration
resolution, the M11B doctor, and the synthetic aggregation tests.
"""

from __future__ import annotations

from collections.abc import Mapping


STUDY_ID = 'M11B'
ANCHOR_ENVIRONMENT = 'antmaze-large-navigate-v0'
NEW_ENVIRONMENTS = (
    'antmaze-giant-navigate-v0',
    'humanoidmaze-large-navigate-v0',
    'humanoidmaze-giant-navigate-v0',
    'antmaze-large-stitch-v0',
)
ALL_ENVIRONMENTS = (ANCHOR_ENVIRONMENT, *NEW_ENVIRONMENTS)

CRL_CONDITIONS = ('baseline', 'critic_ss', 'actor_ss', 'actor_critic_ss')
HIQL_CONDITIONS = ('baseline', 'high_ss', 'low_ss', 'high_low_ss')

PROTOCOL = {
    'train_steps': 1_000_000,
    'batch_size': 1024,
    'log_interval': 5_000,
    'eval_interval': 100_000,
    'eval_tasks': 'all',
    'eval_episodes': 20,
    'eval_temperature': 0.0,
    'eval_gaussian': None,
    'video_episodes': 0,
    'save_interval': 100_000,
    'save_best_checkpoint': True,
    'save_last_checkpoint': True,
    'selection_metric': 'evaluation/overall_success',
    'primary_endpoint': 'last@1M',
    'secondary_endpoints': (
        'normalized_eval_auc',
        'best_success',
        'best_step',
        'last3_mean',
    ),
    'auc_checkpoints': tuple(range(100_000, 1_000_001, 100_000)),
    'auc_interval': [100_000, 1_000_000],
    'auc_rule': 'trapezoidal_area_divided_by_900000',
}

CANONICAL_SOURCE = '/home/eai/Research/offline-rl/docs/ALGORITHM_HYPERPARAMETERS.md'


def _copy(value):
    if isinstance(value, Mapping):
        return {key: _copy(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_copy(item) for item in value]
    if isinstance(value, list):
        return [_copy(item) for item in value]
    return value


def _is_stitch(environment):
    return environment.endswith('-stitch-v0')


def _is_humanoid(environment):
    return environment.startswith('humanoidmaze-')


def _is_giant(environment):
    return '-giant-' in environment


def canonical_hyperparameters(algorithm: str, environment: str) -> dict:
    """Resolve task-context hyperparameters from the local canonical table."""

    if algorithm not in {'crl', 'hiql'}:
        raise ValueError(f'Unsupported M11B algorithm: {algorithm!r}')
    if environment not in ALL_ENVIRONMENTS:
        raise ValueError(f'Unsupported M11B environment: {environment!r}')

    discount = 0.995 if (_is_giant(environment) or _is_humanoid(environment)) else 0.99
    actor_trajgoal = 0.5 if _is_stitch(environment) else 1.0
    actor_randomgoal = 0.5 if _is_stitch(environment) else 0.0

    if algorithm == 'crl':
        return {
            'lr': 3e-4,
            'batch_size': 1024,
            'actor_hidden_dims': [512, 512, 512],
            'value_hidden_dims': [512, 512, 512],
            'latent_dim': 512,
            'layer_norm': True,
            'discount': discount,
            'actor_loss': 'ddpgbc',
            'alpha': 0.1,
            'const_std': True,
            'dataset_class': 'GCDataset',
            'value_p_curgoal': 0.0,
            'value_p_trajgoal': 1.0,
            'value_p_randomgoal': 0.0,
            'value_geom_sample': True,
            'actor_p_curgoal': 0.0,
            'actor_p_trajgoal': actor_trajgoal,
            'actor_p_randomgoal': actor_randomgoal,
            'actor_geom_sample': False,
            'gc_negative': False,
            'p_aug': 0.0,
        }

    return {
        'lr': 3e-4,
        'batch_size': 1024,
        'actor_hidden_dims': [512, 512, 512],
        'value_hidden_dims': [512, 512, 512],
        'layer_norm': True,
        'discount': discount,
        'tau': 0.005,
        'expectile': 0.7,
        'low_alpha': 3.0,
        'high_alpha': 3.0,
        'subgoal_steps': 100 if _is_humanoid(environment) else 25,
        'rep_dim': 10,
        'low_actor_rep_grad': False,
        'const_std': True,
        'dataset_class': 'HGCDataset',
        'value_p_curgoal': 0.2,
        'value_p_trajgoal': 0.5,
        'value_p_randomgoal': 0.3,
        'value_geom_sample': True,
        'actor_p_curgoal': 0.0,
        'actor_p_trajgoal': actor_trajgoal,
        'actor_p_randomgoal': actor_randomgoal,
        'actor_geom_sample': False,
        'gc_negative': True,
        'p_aug': 0.0,
    }


def single_state_spec(role: str) -> dict:
    """Return the frozen M11B actor or CRL-critic SingleState semantics."""

    if role == 'actor':
        return {
            'enabled': True,
            'primitive': 'mlp',
            'topology': 'single_state',
            'credit': 'direct',
            'topology_kwargs': {
                'iterations': 4,
                'residual': False,
                'input_injection': 'z_plus_x',
                'state_dim': 512,
                'state_init': 'normal_buffer',
                'state_init_std': 1.0,
                'update_depth': 2,
                'layer_norm': False,
                'update_activate_final': True,
            },
        }
    if role == 'critic':
        return {
            'enabled': True,
            'primitive': 'mlp',
            'topology': 'single_state',
            'credit': 'direct',
            'topology_kwargs': {
                'iterations': 4,
                'residual': False,
                'input_injection': 'z_plus_x',
                'state_dim': 512,
                'state_init': 'normal_buffer',
                'state_init_std': 1.0,
                'update_depth': 3,
                'layer_norm': True,
                'update_activate_final': False,
            },
        }
    raise ValueError(f'Unsupported SingleState role: {role!r}')


def feedforward_slot() -> dict:
    """Explicit disabled/feedforward slot semantics for M11B accounting."""

    return {
        'enabled': False,
        'primitive': 'mlp',
        'topology': 'feedforward',
        'credit': 'direct',
    }


def computation_slots(algorithm: str, condition: str) -> dict:
    """Resolve all compute slots, including disabled slots, deterministically."""

    if algorithm == 'crl':
        if condition not in CRL_CONDITIONS:
            raise ValueError(f'Unsupported CRL condition: {condition!r}')
        slots = {
            name: feedforward_slot()
            for name in ('actor', 'critic_state', 'critic_goal', 'value_state', 'value_goal')
        }
        if condition in {'actor_ss', 'actor_critic_ss'}:
            slots['actor'] = single_state_spec('actor')
        if condition in {'critic_ss', 'actor_critic_ss'}:
            slots['critic_state'] = single_state_spec('critic')
            slots['critic_goal'] = single_state_spec('critic')
        return slots

    if algorithm == 'hiql':
        if condition not in HIQL_CONDITIONS:
            raise ValueError(f'Unsupported HIQL condition: {condition!r}')
        slots = {name: feedforward_slot() for name in ('high_actor', 'low_actor', 'value')}
        if condition in {'high_ss', 'high_low_ss'}:
            slots['high_actor'] = single_state_spec('actor')
        if condition in {'low_ss', 'high_low_ss'}:
            slots['low_actor'] = single_state_spec('actor')
        return slots
    raise ValueError(f'Unsupported M11B algorithm: {algorithm!r}')


def m11b_agent_overrides(algorithm: str, environment: str, condition: str) -> dict:
    """Resolve the immutable agent configuration for one M11B condition."""

    overrides = canonical_hyperparameters(algorithm, environment)
    overrides['compute'] = computation_slots(algorithm, condition)
    return overrides


def condition_label(algorithm: str, condition: str) -> str:
    labels = {
        ('crl', 'baseline'): 'CRL-A: FF actor × FF critic',
        ('crl', 'critic_ss'): 'CRL-C: FF actor × SS critic',
        ('crl', 'actor_ss'): 'CRL-P: SS actor × FF critic',
        ('crl', 'actor_critic_ss'): 'CRL-PC: SS actor × SS critic',
        ('hiql', 'baseline'): 'HIQL-A: FF high × FF low',
        ('hiql', 'high_ss'): 'HIQL-H: SS high × FF low',
        ('hiql', 'low_ss'): 'HIQL-L: FF high × SS low',
        ('hiql', 'high_low_ss'): 'HIQL-HL: SS high × SS low',
    }
    try:
        return labels[(algorithm, condition)]
    except KeyError as error:
        raise ValueError(f'Unsupported M11B condition: {algorithm}/{condition}') from error


def config_specs() -> list[dict]:
    """Return the permanent 34-row configuration identity table."""

    rows = [
        ('M11B-C001', ANCHOR_ENVIRONMENT, 'crl', 'baseline', 'crl_anchor_ff_ff'),
        ('M11B-C002', ANCHOR_ENVIRONMENT, 'hiql', 'baseline', 'hiql_anchor_ff_ff'),
    ]
    next_id = 3
    for environment in NEW_ENVIRONMENTS:
        for algorithm, conditions in (('crl', CRL_CONDITIONS), ('hiql', HIQL_CONDITIONS)):
            for condition in conditions:
                config_id = f'M11B-C{next_id:03d}'
                rows.append((config_id, environment, algorithm, condition, f'{algorithm}_{condition}_{environment[:-3]}'))
                next_id += 1
    return [
        {
            'config_id': config_id,
            'environment': environment,
            'algorithm': algorithm,
            'condition': condition,
            'semantic_condition': condition_label(algorithm, condition),
            'semantic_label': f'{environment} | {condition_label(algorithm, condition)}',
            'slug': slug,
        }
        for config_id, environment, algorithm, condition, slug in rows
    ]


def spec_by_id(config_id: str) -> dict:
    for spec in config_specs():
        if spec['config_id'] == config_id:
            return spec
    raise KeyError(config_id)


def normalized_eval_auc(checkpoint_values: Mapping | list[tuple[int, float]]) -> float:
    """Compute the preregistered trapezoidal AUC on the success scale."""

    if isinstance(checkpoint_values, Mapping):
        points = [(int(step), float(value)) for step, value in checkpoint_values.items()]
    else:
        points = [(int(step), float(value)) for step, value in checkpoint_values]
    points.sort()
    expected = list(PROTOCOL['auc_checkpoints'])
    if [step for step, _ in points] != expected:
        raise ValueError(
            f'normalized_eval_auc requires checkpoints {expected}, got {[step for step, _ in points]}'
        )
    area = 0.0
    for (left_step, left_value), (right_step, right_value) in zip(points, points[1:]):
        area += (right_step - left_step) * (left_value + right_value) / 2.0
    return area / float(PROTOCOL['auc_interval'][1] - PROTOCOL['auc_interval'][0])


def _value_by_condition(rows: list[Mapping], algorithm: str, environment: str, metric: str):
    result = {}
    for row in rows:
        if row.get('algorithm') != algorithm or row.get('environment') != environment:
            continue
        result[row['condition']] = row.get(metric)
    return result


def aggregate_factorial_rows(rows: list[Mapping], metric: str = 'final_success') -> list[dict]:
    """Return descriptive CRL/HIQL factorial quantities without interpretation."""

    environments = sorted({row.get('environment') for row in rows if row.get('environment')})
    output = []
    for environment in environments:
        for algorithm, conditions in (('crl', CRL_CONDITIONS), ('hiql', HIQL_CONDITIONS)):
            values = _value_by_condition(rows, algorithm, environment, metric)
            if not all(condition in values for condition in conditions):
                continue
            baseline, first, second, both = (float(values[condition]) for condition in conditions)
            output.append({
                'environment': environment,
                'algorithm': algorithm,
                'metric': metric,
                'baseline': baseline,
                'primary_placement_effect': first - baseline,
                'secondary_placement_effect': second - baseline,
                'interaction': both - first - second + baseline,
            })
    return output


def resolved_fingerprint_payload(
    *,
    spec: Mapping,
    resolved_agent: Mapping,
    dataset_root: str,
    seed: int,
    source_commit: str | None,
) -> dict:
    """Build the complete M11B provenance payload used for fingerprinting."""

    return {
        'study_id': STUDY_ID,
        'config_id': spec['config_id'],
        'semantic_condition': spec['semantic_condition'],
        'algorithm': spec['algorithm'],
        'environment': spec['environment'],
        'dataset_root': dataset_root,
        'canonical_agent_config': _copy(resolved_agent),
        'training_seed': int(seed),
        'training_protocol': _copy(PROTOCOL),
        'source_commit': source_commit,
    }
