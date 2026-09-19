"""Shared network-input semantics for deterministic goal conditioning."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
from numbers import Integral

from ..representation.puzzle_algebra import operator_metadata
from ..representation.goal_coordinate_transforms import (
    GoalCoordinateTransform,
    resolve_goal_coordinate_transform,
    semantic_hash,
)
from ..representation.puzzle_conditioning import (
    PUZZLE_GOAL_CONDITIONING_MODES,
    condition_puzzle_goal,
    prepare_puzzle_goal,
)


_CONFIG_KEYS = frozenset({
    'schema_version',
    'domain',
    'mode',
    'rows',
    'cols',
    'num_buttons',
    'robot_dim',
    'button_feature_dim',
    'robot_goal_policy',
    'button_goal_transient_policy',
})
_BOARD_EQUALITY_MODES = frozenset({'board', 'residual', 'oracle_operation'})
_SLOT_LOCAL_GOAL_FIELDS = frozenset({
    'goal_conditioning',
    'goal_conditioning_mode',
    'coordinate', 'transform_id', 'distance_feature', 'token_aux_dim',
    'input_schema', 'roles', 'transforms',
})
_V2_KEYS = (_CONFIG_KEYS - {'mode'}) | {'input_schema', 'roles', 'transforms'}
_ROLE_KEYS = frozenset({'coordinate', 'transform_id', 'distance_feature'})


def _schema_version(config):
    version = config.get('schema_version', 1) if config is not None else 1
    if isinstance(version, bool) or not isinstance(version, Integral) or version not in (1, 2):
        raise ValueError(f'Unsupported goal_conditioning schema_version: {version!r}')
    return int(version)


def _positive_int(config: Mapping, name: str) -> int:
    if name not in config:
        raise ValueError(f'goal_conditioning requires {name!r}')
    value = config[name]
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
        raise ValueError(
            f'goal_conditioning.{name} must be a positive integer, got {value!r}'
        )
    return int(value)


def _validate_puzzle_slots(compute_slots, normalized) -> None:
    if compute_slots is None:
        return
    if not hasattr(compute_slots, 'items'):
        raise ValueError('compute slots must be a mapping for layout validation')
    expected = {
        'num_buttons': normalized['num_buttons'],
        'robot_dim': normalized['robot_dim'],
        'button_feature_dim': normalized['button_feature_dim'],
    }
    for slot_name, slot in compute_slots.items():
        if not hasattr(slot, 'get') or not slot.get('enabled', False):
            continue
        if slot.get('structure', 'vector') != 'puzzle_tokens':
            continue
        structure_kwargs = slot.get('structure_kwargs', {})
        if not hasattr(structure_kwargs, 'get'):
            raise ValueError(
                f'compute.{slot_name}.structure_kwargs must be a mapping'
            )
        actual = {
            'num_buttons': structure_kwargs.get('num_buttons'),
            'robot_dim': structure_kwargs.get('robot_dim', 19),
            'button_feature_dim': structure_kwargs.get('button_feature_dim', 4),
        }
        for field, expected_value in expected.items():
            if actual[field] != expected_value:
                raise ValueError(
                    'Puzzle goal-conditioning layout conflicts with structured slot: '
                    f'compute.{slot_name}.structure_kwargs.{field}={actual[field]!r}, '
                    f'goal_conditioning.{field}={expected_value!r}'
                )


def _reject_slot_local_goal_conditioning(compute_slots) -> None:
    """Keep scientific goal semantics out of computation-slot config."""

    if compute_slots is None:
        return
    if not hasattr(compute_slots, 'items'):
        raise ValueError('compute slots must be a mapping for layout validation')
    for slot_name, slot in compute_slots.items():
        if not hasattr(slot, 'get'):
            continue
        for container_name, container in (
            ('slot', slot),
            ('structure_kwargs', slot.get('structure_kwargs', {})),
            ('block_kwargs', slot.get('block_kwargs', {})),
            ('topology_kwargs', slot.get('topology_kwargs', {})),
            ('readout_kwargs', slot.get('readout_kwargs', {})),
            ('relation_kwargs', slot.get('relation_kwargs', {})),
            ('relation_augmenter_kwargs', slot.get('relation_augmenter_kwargs', {})),
        ):
            if not hasattr(container, 'keys'):
                continue
            forbidden = set(container) & _SLOT_LOCAL_GOAL_FIELDS
            if forbidden:
                raise ValueError(
                    'goal conditioning is one shared top-level network semantic; '
                    f'compute.{slot_name}.{container_name} contains forbidden fields '
                    f'{sorted(forbidden)!r}'
                )


def goal_conditioning_layout(
    config,
    *,
    compute_slots=None,
    dataset_class: str | None = None,
):
    """Validate shared Puzzle layout fields without constructing an operator."""

    _reject_slot_local_goal_conditioning(compute_slots)
    if config is None:
        if dataset_class == 'PuzzleBoardGCDataset':
            raise ValueError(
                'PuzzleBoardGCDataset requires an explicit goal_conditioning layout'
            )
        return None
    if not isinstance(config, Mapping) and not hasattr(config, 'items'):
        raise ValueError('goal_conditioning must be a mapping or None')
    if _schema_version(config) == 2:
        unexpected = set(config) - _V2_KEYS
        if unexpected:
            raise ValueError(f'Unsupported v2 goal_conditioning fields: {sorted(unexpected)!r}')
        if config.get('input_schema') != 'token_aux_v1':
            raise ValueError('v2 requires input_schema=token_aux_v1')
        # Reuse the raw layout/equality validation, without resolving any role,
        # transform, rank, inverse, or distance in Dataset construction.
        layout = goal_conditioning_layout(
            {**{key: value for key, value in config.items()
                if key in _CONFIG_KEYS and key != 'schema_version'}, 'mode': 'residual'},
            compute_slots=compute_slots, dataset_class=dataset_class,
        )
        layout.pop('mode')
        return {**layout, 'schema_version': 2, 'input_schema': 'token_aux_v1'}
    unexpected = set(config) - _CONFIG_KEYS
    if unexpected:
        raise ValueError(
            f'Unsupported goal_conditioning fields: {sorted(unexpected)!r}'
        )
    if config.get('domain') != 'puzzle':
        raise ValueError(
            f"goal_conditioning.domain must be 'puzzle', got {config.get('domain')!r}"
        )
    mode = config.get('mode')
    if mode not in PUZZLE_GOAL_CONDITIONING_MODES:
        raise ValueError(
            f'Unsupported goal_conditioning.mode {mode!r}; '
            f'expected one of {sorted(PUZZLE_GOAL_CONDITIONING_MODES)!r}'
        )
    rows = _positive_int(config, 'rows')
    cols = _positive_int(config, 'cols')
    num_buttons = _positive_int(config, 'num_buttons')
    robot_dim = _positive_int(config, 'robot_dim')
    button_feature_dim = _positive_int(config, 'button_feature_dim')
    if rows * cols != num_buttons:
        raise ValueError(
            'goal_conditioning rows*cols must equal num_buttons: '
            f'{rows}*{cols} != {num_buttons}'
        )
    if robot_dim != 19:
        raise ValueError(
            f'goal_conditioning.robot_dim must match canonical Puzzle 19D, got {robot_dim}'
        )
    if button_feature_dim != 4:
        raise ValueError(
            'goal_conditioning.button_feature_dim must match canonical Puzzle 4D, '
            f'got {button_feature_dim}'
        )
    expected_policy = 'preserve' if mode == 'canonical' else 'zero'
    robot_goal_policy = config.get('robot_goal_policy', expected_policy)
    transient_policy = config.get(
        'button_goal_transient_policy', expected_policy
    )
    if robot_goal_policy != expected_policy:
        raise ValueError(
            'goal_conditioning.robot_goal_policy must be '
            f'{expected_policy!r} for mode={mode!r}'
        )
    if transient_policy != expected_policy:
        raise ValueError(
            'goal_conditioning.button_goal_transient_policy must be '
            f'{expected_policy!r} for mode={mode!r}'
        )
    if (
        mode in _BOARD_EQUALITY_MODES
        and dataset_class is not None
        and dataset_class != 'PuzzleBoardGCDataset'
    ):
        raise ValueError(
            f'goal_conditioning.mode={mode!r} requires '
            "dataset_class='PuzzleBoardGCDataset' for board-equality success semantics; "
            f'got {dataset_class!r}'
        )
    normalized = {
        'domain': 'puzzle',
        'mode': mode,
        'rows': rows,
        'cols': cols,
        'num_buttons': num_buttons,
        'robot_dim': robot_dim,
        'button_feature_dim': button_feature_dim,
        'robot_goal_policy': robot_goal_policy,
        'button_goal_transient_policy': transient_policy,
    }
    _validate_puzzle_slots(compute_slots, normalized)
    return normalized


def validate_goal_conditioning_config(
    config,
    *,
    compute_slots=None,
    dataset_class: str | None = None,
    agent_config=None,
):
    """Validate one shared network config, including oracle invertibility."""

    normalized = goal_conditioning_layout(
        config,
        compute_slots=compute_slots,
        dataset_class=dataset_class,
    )
    if normalized is None:
        return None
    if normalized.get('schema_version') == 2:
        return resolve_goal_conditioning(
            config, compute_slots=compute_slots, dataset_class=dataset_class,
            agent_config=agent_config,
        ).to_config()
    if normalized['mode'] == 'oracle_operation':
        metadata = operator_metadata(normalized['rows'], normalized['cols'])
        if not metadata['full_rank']:
            raise ValueError(
                'oracle_operation requires a unique full-rank GF(2) operator: '
                f'layout={normalized["rows"]}x{normalized["cols"]}, '
                f'rank={metadata["rank"]}, '
                f'num_buttons={normalized["num_buttons"]}'
            )
    return normalized


@dataclass(frozen=True)
class PuzzleGoalConditioner:
    """Hashable, parameter-free callable embedded in GC network modules."""

    mode: str
    rows: int
    cols: int
    num_buttons: int
    robot_dim: int
    button_feature_dim: int
    operation_inverse: tuple[tuple[int, ...], ...] | None = None

    def __call__(self, observations, goals):
        return condition_puzzle_goal(
            observations,
            goals,
            mode=self.mode,
            rows=self.rows,
            cols=self.cols,
            num_buttons=self.num_buttons,
            robot_dim=self.robot_dim,
            button_feature_dim=self.button_feature_dim,
            operation_inverse=self.operation_inverse,
        )


def make_goal_conditioner(
    config,
    *,
    compute_slots=None,
    dataset_class: str | None = None,
):
    """Build a parameter-free conditioner from shared network semantics."""

    if config is not None and hasattr(config, 'get') and _schema_version(config) == 2:
        raise ValueError('v2 requires make_goal_conditioners and explicit role preparation; legacy factory returns arrays')
    normalized = validate_goal_conditioning_config(
        config,
        compute_slots=compute_slots,
        dataset_class=dataset_class,
    )
    if normalized is None:
        return None
    operation_inverse = None
    if normalized['mode'] == 'oracle_operation':
        inverse = operator_metadata(normalized['rows'], normalized['cols'])['inverse']
        operation_inverse = tuple(
            tuple(int(item) for item in row) for row in inverse
        )
    return PuzzleGoalConditioner(
        mode=normalized['mode'],
        rows=normalized['rows'],
        cols=normalized['cols'],
        num_buttons=normalized['num_buttons'],
        robot_dim=normalized['robot_dim'],
        button_feature_dim=normalized['button_feature_dim'],
        operation_inverse=operation_inverse,
    )


@dataclass(frozen=True)
class PuzzleRoleGoalConditioner:
    """One immutable v2 role; deliberately not a legacy array callable."""

    coordinate: str
    distance_feature: str
    num_buttons: int
    robot_dim: int
    button_feature_dim: int
    transform: GoalCoordinateTransform
    operation_inverse: tuple[tuple[int, ...], ...] | None
    fingerprint: str

    def prepare(self, observations, goals):
        if goals is None:
            raise ValueError('v2 goal conditioning requires an external raw goal')
        return prepare_puzzle_goal(
            observations, goals, coordinate=self.coordinate,
            distance_feature=self.distance_feature, transform=self.transform,
            num_buttons=self.num_buttons, robot_dim=self.robot_dim,
            button_feature_dim=self.button_feature_dim,
            operation_inverse=self.operation_inverse,
        )


@dataclass(frozen=True)
class GoalConditioningPlan:
    """Single resolved source of roles, static transforms and input semantics."""

    actor: PuzzleGoalConditioner | PuzzleRoleGoalConditioner | None
    value_side: PuzzleGoalConditioner | PuzzleRoleGoalConditioner | None
    token_aux_dim: int
    config_json: str
    fingerprint: str | None = None
    metadata_json: str | None = None

    def to_config(self):
        return json.loads(self.config_json)


def _validate_v2_support(compute_slots, agent_config):
    if agent_config is not None:
        if agent_config.get('agent_name') != 'gciql':
            raise ValueError('v2 goal conditioning supports only GCIQL')
        if agent_config.get('discrete', False) or agent_config.get('actor_loss') != 'ddpgbc':
            raise ValueError('v2 goal conditioning requires continuous GCIQL DDPG+BC')
        if agent_config.get('dataset_class') != 'PuzzleBoardGCDataset':
            raise ValueError('v2 GCIQL requires PuzzleBoardGCDataset board-equality semantics')
        if agent_config.get('encoder') is not None or agent_config.get('frame_stack') is not None:
            raise ValueError('v2 requires raw standard state observations without encoder/frame_stack')
        if compute_slots is None:
            raise ValueError('v2 GCIQL requires structured compute slots')
    if compute_slots is None:
        return  # Pure representation/factory callers need no algorithm config.
    for name in ('actor', 'value', 'critic'):
        slot = compute_slots.get(name, {})
        if not slot.get('enabled', False):
            raise ValueError(f'v2 requires enabled Puzzle computation in {name}')
        if (slot.get('structure', 'vector') != 'puzzle_tokens'
                or slot.get('topology', 'feedforward') != 'feedforward'
                or slot.get('block', 'plain') != 'mlp_mixer'
                or slot.get('credit', 'direct') != 'direct'
                or slot.get('relation_mode', 'legacy_none') != 'legacy_none'
                or slot.get('relation_augmenter', 'none') != 'none'):
            raise ValueError('v2 requires relation-free Puzzle feedforward MLP-Mixer with direct credit')
        if slot.get('readout', slot.get('structure_kwargs', {}).get('readout', 'mean_context')) not in ('mean', 'mean_context'):
            raise ValueError('v2 supports only the existing mean_context readout')


def resolve_goal_conditioning(config, *, compute_slots=None, dataset_class=None, agent_config=None):
    """Resolve both versions at setup; JSON roundtrips retain actual P/B data."""
    layout = goal_conditioning_layout(config, compute_slots=compute_slots, dataset_class=dataset_class)
    if layout is None or layout.get('schema_version', 1) == 1:
        conditioner = make_goal_conditioner(config, compute_slots=compute_slots, dataset_class=dataset_class)
        return GoalConditioningPlan(conditioner, conditioner, 0, json.dumps(layout, sort_keys=True))
    _validate_v2_support(compute_slots, agent_config)
    roles = config.get('roles')
    if not hasattr(roles, 'items') or set(roles) != {'actor', 'value_side'}:
        raise ValueError('v2 roles must contain exactly actor and value_side; no independent Q/target overrides')
    payloads = config.get('transforms')
    if not hasattr(payloads, 'items'):
        raise ValueError('v2 transforms must be an explicit mapping (empty is allowed for identity)')
    transforms = {'identity': resolve_goal_coordinate_transform(
        {'kind': 'identity'}, num_coordinates=layout['num_buttons'],
    )}
    for name, payload in payloads.items():
        if not isinstance(name, str) or not name:
            raise ValueError('transform_id must be a nonempty string')
        transform = resolve_goal_coordinate_transform(payload, num_coordinates=layout['num_buttons'])
        if name == 'identity' and transform.kind != 'identity':
            raise ValueError('The reserved identity transform cannot be redefined')
        transforms[name] = transform
    normalized_roles = {}
    for name, role in roles.items():
        if not hasattr(role, 'items') or set(role) != _ROLE_KEYS:
            raise ValueError(f'roles.{name} requires exactly {sorted(_ROLE_KEYS)!r}')
        coordinate, distance, transform_id = role['coordinate'], role['distance_feature'], role['transform_id']
        if coordinate not in ('residual', 'operation'):
            raise ValueError(f'Unsupported role coordinate: {coordinate!r}')
        if distance not in ('zero', 'exact_press_fraction'):
            raise ValueError(f'Unsupported role distance_feature: {distance!r}')
        if not isinstance(transform_id, str) or transform_id not in transforms:
            raise ValueError(f'Unknown transform_id: {transform_id!r}')
        if coordinate == 'residual' and transform_id != 'identity':
            raise ValueError('Non-identity transforms are allowed only for operation coordinates')
        normalized_roles[name] = dict(role)
    uses_oracle = any(r['coordinate'] == 'operation' or r['distance_feature'] != 'zero'
                      for r in normalized_roles.values())
    operator = operator_metadata(layout['rows'], layout['cols']) if uses_oracle else None
    if operator is not None and not operator['full_rank']:
        raise ValueError('operation/exact distance requires a unique full-rank GF(2) Puzzle operator')
    inverse = (tuple(tuple(int(v) for v in row) for row in operator['inverse'])
               if operator is not None else None)
    operator_facts = None if operator is None else {
        'source': operator['operator_matrix_source'],
        'orientation': operator['operator_orientation'],
        'rank': operator['rank'],
        'matrix_sha256': semantic_hash(operator['matrix'].tolist()),
    }
    role_metadata, conditioners = {}, {}
    for name, role in normalized_roles.items():
        transform = transforms[role['transform_id']]
        role_oracle = role['coordinate'] == 'operation' or role['distance_feature'] != 'zero'
        semantics = {
            'layout': layout, 'token_aux_dim': 1, 'coordinate': role['coordinate'],
            'distance_feature': role['distance_feature'],
            'transform_sha256': transform.content_sha256,
            'operator': operator_facts if role_oracle else None,
            'goal_success_semantics': 'board_equality',
        }
        fingerprint = semantic_hash(semantics)
        role_metadata[name] = {**role, 'fingerprint': fingerprint,
                               'uses_oracle_preprocessing': role_oracle,
                               'transform_sha256': transform.content_sha256}
        conditioners[name] = PuzzleRoleGoalConditioner(
            coordinate=role['coordinate'], distance_feature=role['distance_feature'],
            num_buttons=layout['num_buttons'], robot_dim=layout['robot_dim'],
            button_feature_dim=layout['button_feature_dim'], transform=transform,
            operation_inverse=inverse if role_oracle else None, fingerprint=fingerprint,
        )
    normalized = {**layout, 'roles': normalized_roles,
                  'transforms': {name: item.to_config() for name, item in sorted(transforms.items())}}
    fingerprint = semantic_hash({name: role_metadata[name]['fingerprint'] for name in ('actor', 'value_side')})
    metadata = {
        **normalized, 'token_aux_dim': 1, 'roles': role_metadata,
        'semantic_fingerprint': fingerprint, 'operator': operator_facts,
        'uses_oracle_preprocessing': uses_oracle,
        'preprocessing_accounting': 'GF(2), XOR, gather, sum and broadcast excluded from network Dense MACs',
    }
    return GoalConditioningPlan(
        actor=conditioners['actor'], value_side=conditioners['value_side'], token_aux_dim=1,
        config_json=json.dumps(normalized, sort_keys=True, allow_nan=False), fingerprint=fingerprint,
        metadata_json=json.dumps(metadata, sort_keys=True, allow_nan=False),
    )


def make_goal_conditioners(config, *, compute_slots=None, dataset_class=None, agent_config=None):
    """Role bundle factory; the legacy single-conditioner API stays unchanged."""
    return resolve_goal_conditioning(config, compute_slots=compute_slots,
                                    dataset_class=dataset_class, agent_config=agent_config)


def goal_conditioning_runtime_metadata(
    config,
    *,
    compute_slots=None,
    dataset_class: str | None = None,
):
    """Return JSON-ready derived provenance without a parallel config system."""

    if config is not None and _schema_version(config) == 2:
        plan = resolve_goal_conditioning(config, compute_slots=compute_slots, dataset_class=dataset_class)
        return {
            'goal_conditioning': json.loads(plan.metadata_json),
            'dataset_class': dataset_class,
            'goal_success_semantics': 'board_equality',
        }
    normalized = validate_goal_conditioning_config(
        config,
        compute_slots=compute_slots,
        dataset_class=dataset_class,
    )
    if normalized is None:
        return {}
    operator = operator_metadata(normalized['rows'], normalized['cols'])
    return {
        'goal_conditioning': {
            **normalized,
            'operator_matrix_source': operator['operator_matrix_source'],
            'operator_orientation': operator['operator_orientation'],
            'operator_rank': operator['rank'],
            'operator_full_rank': operator['full_rank'],
        },
        'dataset_class': dataset_class,
        'goal_success_semantics': (
            'board_equality'
            if dataset_class == 'PuzzleBoardGCDataset'
            else 'timestep_index_equality'
        ),
    }


__all__ = [
    'GoalConditioningPlan',
    'PuzzleRoleGoalConditioner',
    'PuzzleGoalConditioner',
    'goal_conditioning_layout',
    'goal_conditioning_runtime_metadata',
    'make_goal_conditioner',
    'make_goal_conditioners',
    'resolve_goal_conditioning',
    'validate_goal_conditioning_config',
]
