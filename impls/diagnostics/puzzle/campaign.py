"""Generic declarative campaigns for paired Puzzle rollout diagnostics."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path

from ...experiment import (
    load_configuration,
    load_study,
    resolve_run_dependency,
)
from ...experiment.management import jsonable
from ...experiment.reevaluation import ReevaluationError, validate_source_run
from .replay import (
    ControlledGoalReplay,
    ControlledGoalReplayError,
    verify_paired_fingerprint_groups,
)


class PuzzleDiagnosticCampaignError(ValueError):
    """Raised when a declarative Puzzle diagnostic campaign is invalid."""


_REQUIRED_PROTOCOL_FIELDS = (
    'campaign_id',
    'task_ids',
    'episodes_per_task',
    'evaluation_seed',
    'eval_temperature',
    'eval_gaussian',
    'checkpoint_role',
    'checkpoint_step',
    'controlled_goal_replay',
)
_REQUIRED_SOURCE_EXPECTATIONS = (
    'source_study_id',
    'source_config_id',
    'environment',
    'training_seed',
    'run_attempt',
    'git_commit',
    'git_dirty',
    'checkpoint_sha256',
    'resolved_config_fingerprint',
    'alpha',
    'condition_id',
    'architecture',
)


def _json_write(path, value):
    path = Path(path)
    with path.open('w') as file:
        json.dump(jsonable(value), file, indent=2, sort_keys=True, ensure_ascii=False)
        file.write('\n')


def _protocol(study):
    protocol = study.data.get('diagnostic_protocol')
    if not isinstance(protocol, Mapping):
        raise PuzzleDiagnosticCampaignError(
            f'{study.study_id} requires a diagnostic_protocol mapping'
        )
    protocol = dict(protocol)
    missing = [field for field in _REQUIRED_PROTOCOL_FIELDS if field not in protocol]
    if missing:
        raise PuzzleDiagnosticCampaignError(
            f'{study.study_id} diagnostic_protocol is missing {missing}'
        )
    task_ids = tuple(int(task_id) for task_id in protocol['task_ids'])
    if task_ids != tuple(sorted(set(task_ids))) or not task_ids:
        raise PuzzleDiagnosticCampaignError('diagnostic_protocol.task_ids must be sorted unique integers')
    if int(protocol['episodes_per_task']) <= 0:
        raise PuzzleDiagnosticCampaignError('diagnostic_protocol.episodes_per_task must be positive')
    if float(protocol['eval_temperature']) != 0.0:
        raise PuzzleDiagnosticCampaignError('Puzzle diagnostic protocol requires eval_temperature=0')
    if protocol['eval_gaussian'] is not None:
        raise PuzzleDiagnosticCampaignError('Puzzle diagnostic protocol requires eval_gaussian=None')
    if protocol['checkpoint_role'] != 'last' or int(protocol['checkpoint_step']) != 1_000_000:
        raise PuzzleDiagnosticCampaignError('Primary Puzzle diagnostic requires checkpoint=final@1M')
    if protocol['controlled_goal_replay'] is not True:
        raise PuzzleDiagnosticCampaignError('Paired Puzzle diagnostic requires controlled_goal_replay=true')
    protocol['task_ids'] = task_ids
    protocol['episodes_per_task'] = int(protocol['episodes_per_task'])
    protocol['evaluation_seed'] = int(protocol['evaluation_seed'])
    return protocol


def load_campaign(study_path, *, include_configs=None, allow_partial=False):
    """Load a Study and its selected declarative diagnostic configurations."""

    study = load_study(study_path)
    protocol = _protocol(study)
    config_paths = sorted((study.path.parent / 'configs').glob('*.yaml'))
    if not config_paths:
        raise PuzzleDiagnosticCampaignError(f'{study.study_id} has no configuration files')
    configurations = [load_configuration(study, path) for path in config_paths]
    known = {configuration.config_id for configuration in configurations}
    if include_configs is not None:
        requested = {str(config_id) for config_id in include_configs}
        unknown = requested - known
        if unknown:
            raise PuzzleDiagnosticCampaignError(f'Unknown diagnostic configs: {sorted(unknown)}')
        configurations = [
            configuration for configuration in configurations if configuration.config_id in requested
        ]
    _validate_configuration_matrix(
        study,
        configurations,
        protocol,
        require_all_environments=not allow_partial,
    )
    return study, tuple(configurations), protocol


def _validate_configuration_matrix(study, configurations, protocol, *, require_all_environments):
    if len(study.data.get('seeds', ())) != 1 or int(study.data['seeds'][0]) != 0:
        raise PuzzleDiagnosticCampaignError('Paired source-policy campaign requires exactly source seed 0')
    environments = defaultdict(list)
    for configuration in configurations:
        data = configuration.data
        if data.get('environment') not in study.data['environments']:
            raise PuzzleDiagnosticCampaignError(
                f'{configuration.config_id} lacks a declared Puzzle environment'
            )
        source_policy = data.get('source_policy')
        if not isinstance(source_policy, Mapping):
            raise PuzzleDiagnosticCampaignError(f'{configuration.config_id} lacks source_policy')
        expected = source_policy.get('expected')
        if not isinstance(expected, Mapping):
            raise PuzzleDiagnosticCampaignError(f'{configuration.config_id} source_policy lacks expected')
        missing = [field for field in _REQUIRED_SOURCE_EXPECTATIONS if field not in expected]
        if missing:
            raise PuzzleDiagnosticCampaignError(
                f'{configuration.config_id} source expectations are missing {missing}'
            )
        dependency_name = source_policy.get('dependency')
        dependencies = data.get('dependencies')
        if not isinstance(dependencies, Mapping) or dependency_name not in dependencies:
            raise PuzzleDiagnosticCampaignError(
                f'{configuration.config_id} source dependency is not declared'
            )
        environments[data['environment']].append(configuration)
    for environment, members in environments.items():
        labels = [member.data['source_policy'].get('label') for member in members]
        if len(members) != 3 or len(labels) != len(set(labels)):
            raise PuzzleDiagnosticCampaignError(
                f'{environment} must contain exactly three distinct source-policy cells'
            )
    if require_all_environments and set(environments) != set(study.data['environments']):
        raise PuzzleDiagnosticCampaignError(
            'Full paired campaign must declare three source policies for every Study environment'
        )
    if tuple(protocol['task_ids']) != (1, 2, 3, 4, 5):
        raise PuzzleDiagnosticCampaignError('M23A protocol requires all five canonical Puzzle tasks')


def _source_exception(source_policy):
    provenance = source_policy.get('provenance')
    if not isinstance(provenance, Mapping):
        raise PuzzleDiagnosticCampaignError('source_policy.provenance must be a mapping')
    status = provenance.get('provenance_status')
    if status == 'verified_clean':
        return None
    if status != 'scoped_exception':
        raise PuzzleDiagnosticCampaignError(f'Unsupported source provenance status: {status!r}')
    return dict(provenance)


def _agent_config(provenance):
    resolved = provenance['resolved_config']
    algorithm_config = resolved.get('algorithm_config', {})
    agent = algorithm_config.get('agent') if isinstance(algorithm_config, Mapping) else None
    if not isinstance(agent, Mapping):
        agent = resolved.get('agent')
    if not isinstance(agent, Mapping):
        raise PuzzleDiagnosticCampaignError('Source resolved config has no agent mapping')
    return dict(agent)


def _require_architecture(agent, expected_architecture):
    if expected_architecture not in {'flat', 'mixer_l2'}:
        raise PuzzleDiagnosticCampaignError(
            f'Unsupported Puzzle source architecture declaration: {expected_architecture!r}'
        )
    compute = agent.get('compute')
    if not isinstance(compute, Mapping):
        raise PuzzleDiagnosticCampaignError('Source agent has no compute mapping')
    for slot_name in ('actor', 'value', 'critic'):
        slot = compute.get(slot_name)
        if not isinstance(slot, Mapping):
            raise PuzzleDiagnosticCampaignError(f'Source agent has no {slot_name} compute slot')
        if expected_architecture == 'flat':
            wanted = {
                'enabled': False,
                'primitive': 'mlp',
                'topology': 'feedforward',
                'block': 'plain',
                'structure': 'vector',
            }
            for key, value in wanted.items():
                if slot.get(key) != value:
                    raise PuzzleDiagnosticCampaignError(
                        f'Flat source {slot_name}.{key}={slot.get(key)!r}, expected={value!r}'
                    )
        else:
            wanted = {
                'enabled': True,
                'primitive': 'mlp',
                'topology': 'feedforward',
                'block': 'mlp_mixer',
                'structure': 'puzzle_tokens',
            }
            for key, value in wanted.items():
                if slot.get(key) != value:
                    raise PuzzleDiagnosticCampaignError(
                        f'Mixer-L2 source {slot_name}.{key}={slot.get(key)!r}, expected={value!r}'
                    )
            structure_kwargs = slot.get('structure_kwargs')
            if not isinstance(structure_kwargs, Mapping) or int(
                structure_kwargs.get('num_mixer_blocks', -1)
            ) != 2:
                raise PuzzleDiagnosticCampaignError(
                    f'Mixer-L2 source {slot_name} must declare num_mixer_blocks=2'
                )


def validate_source_policy(study, configuration, *, source_run_root):
    """Resolve and machine-validate one declared immutable source policy."""

    source_policy = dict(configuration.data['source_policy'])
    expected = dict(source_policy['expected'])
    dependency = resolve_run_dependency(
        study,
        configuration,
        source_policy['dependency'],
        seed=int(study.data['seeds'][0]),
        run_root=source_run_root,
    )
    if dependency['checkpoint_role'] != 'last' or int(dependency['checkpoint_step']) != 1_000_000:
        raise PuzzleDiagnosticCampaignError(
            f'{configuration.config_id} does not declare source final@1M'
        )
    exception = _source_exception(source_policy)
    try:
        provenance = validate_source_run(
            dependency['source_run_dir'],
            checkpoint_selector={'selector': 'last'},
            expected_study_id=str(expected['source_study_id']),
            expected_environment=str(expected['environment']),
            check_checkpoint_metadata=True,
            scoped_provenance_exception=exception,
        )
    except (ReevaluationError, OSError, ValueError) as error:
        raise PuzzleDiagnosticCampaignError(
            f'{configuration.config_id} source validation failed: {error}'
        ) from error
    actual = {
        'source_study_id': provenance['source_study_id'],
        'source_config_id': provenance['source_config_id'],
        'environment': provenance['source_environment'],
        'training_seed': provenance['source_training_seed'],
        'run_attempt': provenance['source_run_attempt'],
        'git_commit': provenance['source_git_commit'],
        'git_dirty': provenance['source_git_dirty'],
        'checkpoint_sha256': provenance['checkpoint_sha256'],
        'resolved_config_fingerprint': provenance['source_resolved_config_fingerprint'],
    }
    for field, value in actual.items():
        if str(expected[field]) != str(value):
            raise PuzzleDiagnosticCampaignError(
                f'{configuration.config_id} source {field} mismatch: '
                f'expected={expected[field]!r}, observed={value!r}'
            )
    if provenance['resolved_checkpoint_role'] != 'last' or provenance['checkpoint_step'] != 1_000_000:
        raise PuzzleDiagnosticCampaignError(
            f'{configuration.config_id} did not resolve final@1M checkpoint'
        )
    resolved_configuration = provenance['resolved_config'].get('configuration')
    if not isinstance(resolved_configuration, Mapping):
        raise PuzzleDiagnosticCampaignError('Source resolved config has no configuration mapping')
    if resolved_configuration.get('condition_id') != expected['condition_id']:
        raise PuzzleDiagnosticCampaignError(
            f'{configuration.config_id} source condition mismatch: '
            f'{resolved_configuration.get("condition_id")!r}'
        )
    agent = _agent_config(provenance)
    if agent.get('agent_name') != 'gciql':
        raise PuzzleDiagnosticCampaignError(
            f'{configuration.config_id} source algorithm is not gciql'
        )
    try:
        alpha_matches = math.isclose(float(agent.get('alpha')), float(expected['alpha']), rel_tol=0.0, abs_tol=0.0)
    except (TypeError, ValueError) as error:
        raise PuzzleDiagnosticCampaignError('Source alpha is missing or invalid') from error
    if not alpha_matches:
        raise PuzzleDiagnosticCampaignError(
            f'{configuration.config_id} alpha mismatch: '
            f'expected={expected["alpha"]!r}, observed={agent.get("alpha")!r}'
        )
    _require_architecture(agent, expected['architecture'])
    return {
        'configuration_id': configuration.config_id,
        'policy_label': source_policy['label'],
        'source_provenance': dict(source_policy['provenance']),
        'provenance': provenance,
    }


def validate_campaign_sources(study, configurations, *, source_run_root):
    return tuple(
        validate_source_policy(study, configuration, source_run_root=source_run_root)
        for configuration in configurations
    )


def campaign_output_root(diagnostic_root, study, protocol, *, campaign_id=None):
    return (
        Path(diagnostic_root).resolve()
        / study.study_id
        / str(protocol['campaign_id'] if campaign_id is None else campaign_id)
    )


def diagnostic_output_dir(campaign_root, configuration):
    return (
        Path(campaign_root)
        / 'runs'
        / f'{configuration.config_id}__{configuration.slug}'
        / configuration.data['environment']
        / 'seed_000'
    )


def create_controlled_replays(campaign_root, configurations, protocol):
    """Create one real-reset replay artifact per environment before rollouts."""

    import ogbench

    campaign_root = Path(campaign_root)
    result = {}
    by_environment = defaultdict(list)
    for configuration in configurations:
        by_environment[configuration.data['environment']].append(configuration)
    for environment in sorted(by_environment):
        replay_root = campaign_root / 'controlled_goal_replay' / environment
        env = ogbench.make_env_and_datasets(environment, env_only=True)
        try:
            result[environment] = ControlledGoalReplay.create(
                replay_root,
                env,
                environment=environment,
                evaluation_seed=protocol['evaluation_seed'],
                task_ids=protocol['task_ids'],
                episodes_per_task=protocol['episodes_per_task'],
            )
        finally:
            env.close()
    return result


def pairing_invariants_from_outputs(campaign_root, configurations):
    """Check all persisted per-policy episode rows against the paired contract."""

    campaign_root = Path(campaign_root)
    by_environment = defaultdict(list)
    records = []
    for configuration in configurations:
        environment = configuration.data['environment']
        by_environment[environment].append(configuration)
        episode_path = diagnostic_output_dir(campaign_root, configuration) / 'episodes.csv'
        if not episode_path.is_file():
            raise PuzzleDiagnosticCampaignError(f'Missing diagnostic episode output: {episode_path}')
        with episode_path.open(newline='') as file:
            for row in csv.DictReader(file):
                records.append({
                    'environment': row.get('environment'),
                    'paired_episode_id': row.get('paired_episode_id'),
                    'goal_fingerprint': row.get('goal_fingerprint'),
                    'board_goal_fingerprint': row.get('board_goal_fingerprint'),
                    'initial_observation_fingerprint': row.get('initial_observation_fingerprint'),
                })
    summaries = {}
    for environment, members in by_environment.items():
        environment_records = [record for record in records if record['environment'] == environment]
        try:
            summaries[environment] = verify_paired_fingerprint_groups(
                environment_records,
                expected_members=len(members),
            )
        except ControlledGoalReplayError as error:
            raise PuzzleDiagnosticCampaignError(
                f'Pairing invariant failed for {environment}: {error}'
            ) from error
    result = {
        'status': 'passed',
        'environments': summaries,
        'records': len(records),
    }
    _json_write(campaign_root / 'pairing_invariants.json', result)
    return result


__all__ = [
    'PuzzleDiagnosticCampaignError',
    'campaign_output_root',
    'create_controlled_replays',
    'diagnostic_output_dir',
    'load_campaign',
    'pairing_invariants_from_outputs',
    'validate_campaign_sources',
    'validate_source_policy',
]
