"""Study-specific aggregation skeleton for M12B's nine-condition table."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from impls.experiment import load_study, make_run_path, prepare_run_design

from .preflight import ENVIRONMENT, SEEDS, STUDY_PATH, _source_study_path


RESULT_FIELDS = (
    'condition_id', 'structure', 'topology', 'block', 'state_init',
    'parameter_sharing', 'input_reinjection', 'residual',
    'sequential_depth', 'unique_dense_layers', 'executed_dense_layers',
    'actor_body_params', 'actor_total_params', 'buffer_elements',
    'MACs', 'FLOPs', 'seed', 'final_success', 'AUC', 'best_success',
    'best_step', 'last3_mean', 'source_study', 'source_config',
    'source_attempt', 'source_commit', 'source_checkpoint', 'critic_sha',
    'critic_fingerprint', 'status',
)


def _read_json(path):
    with Path(path).open() as file:
        return json.load(file)


def _metrics(run_dir):
    path = Path(run_dir) / 'eval.csv'
    if not path.is_file():
        return {
            'final_success': None, 'AUC': None, 'best_success': None,
            'best_step': None, 'last3_mean': None,
        }
    with path.open(newline='') as file:
        rows = list(csv.DictReader(file))
    points = [(int(row['step']), float(row['evaluation/overall_success'])) for row in rows]
    if not points:
        return {
            'final_success': None, 'AUC': None, 'best_success': None,
            'best_step': None, 'last3_mean': None,
        }
    best_step, best_success = max(points, key=lambda item: item[1])
    denominator = float(points[-1][0] - points[0][0])
    auc = None if denominator <= 0 else sum(
        (right_step - left_step) * (left_value + right_value) / 2.0
        for (left_step, left_value), (right_step, right_value)
        in zip(points, points[1:])
    ) / denominator
    return {
        'final_success': points[-1][1],
        'AUC': auc,
        'best_success': best_success,
        'best_step': best_step,
        'last3_mean': sum(value for _, value in points[-3:]) / min(3, len(points)),
    }


def _row(condition_id, spec, seed, run_dir, metadata, *, source_study,
         source_config, source_attempt, source_checkpoint, critic_sha,
         critic_fingerprint, status):
    accounting = metadata.get('actor_parameter_accounting', {}).get('actor', {})
    macs = accounting.get('full_actor_forward_dense_macs')
    return {
        'condition_id': condition_id,
        'structure': spec.get('structure'),
        'topology': spec.get('topology'),
        'block': spec.get('block'),
        'state_init': spec.get('state_init'),
        'parameter_sharing': spec.get('parameter_sharing'),
        'input_reinjection': spec.get('input_reinjection'),
        'residual': spec.get('residual'),
        'sequential_depth': accounting.get('executed_dense_layers', spec.get('sequential_depth')),
        'unique_dense_layers': accounting.get('unique_dense_layers'),
        'executed_dense_layers': accounting.get('executed_dense_layers'),
        'actor_body_params': accounting.get('core_trainable_params'),
        'actor_total_params': accounting.get('trainable_params'),
        'buffer_elements': accounting.get('buffer_elements'),
        'MACs': macs,
        'FLOPs': None if macs is None else 2 * macs,
        'seed': seed,
        **_metrics(run_dir),
        'source_study': source_study,
        'source_config': source_config,
        'source_attempt': source_attempt,
        'source_commit': metadata.get('git_commit'),
        'source_checkpoint': source_checkpoint,
        'critic_sha': critic_sha,
        'critic_fingerprint': critic_fingerprint,
        'status': status,
    }


def aggregate_rows(study_path=STUDY_PATH, run_root='/data/qijunrong/06-RL/offline-rl/exp/RLC/runs'):
    """Return the nine-condition × three-seed table without fabricating metrics."""

    study = load_study(study_path)
    run_root = Path(run_root)
    rows = []
    for condition_id, spec in study.data['conceptual_conditions'].items():
        if spec['source'] == 'external_anchor':
            anchor = study.data['external_anchors'][condition_id]
            source_study, source_config = prepare_run_design(
                _source_study_path(study, anchor), anchor['source_config_id']
            )
            for seed in SEEDS:
                run_dir = make_run_path(
                    run_root, source_study.study_id, source_config.config_id,
                    source_config.slug, ENVIRONMENT, seed,
                    run_attempt=int(anchor['source_run_attempt']),
                )
                metadata = _read_json(run_dir / 'runtime_metadata.json') if (run_dir / 'runtime_metadata.json').is_file() else {}
                dependency = metadata.get('frozen_dependencies', {}).get('frozen_critic', {})
                rows.append(_row(
                    condition_id, spec, seed, run_dir, metadata,
                    source_study=source_study.study_id,
                    source_config=source_config.config_id,
                    source_attempt=anchor['source_run_attempt'],
                    source_checkpoint=dependency.get('checkpoint_path'),
                    critic_sha=dependency.get('checkpoint_sha256'),
                    critic_fingerprint=dependency.get('module_fingerprint'),
                    status=metadata.get('status', 'missing'),
                ))
        else:
            _, configuration = prepare_run_design(
                study.path, spec['source_config_id']
            )
            for seed in SEEDS:
                run_dir = make_run_path(
                    run_root, study.study_id, configuration.config_id,
                    configuration.slug, ENVIRONMENT, seed, run_attempt=0,
                )
                metadata = _read_json(run_dir / 'runtime_metadata.json') if (run_dir / 'runtime_metadata.json').is_file() else {}
                dependency = metadata.get('frozen_dependencies', {}).get('frozen_critic', {})
                rows.append(_row(
                    condition_id, spec, seed, run_dir, metadata,
                    source_study=dependency.get('source_study_id'),
                    source_config=dependency.get('source_config_id'),
                    source_attempt=dependency.get('source_run_attempt'),
                    source_checkpoint=dependency.get('checkpoint_path'),
                    critic_sha=dependency.get('checkpoint_sha256'),
                    critic_fingerprint=dependency.get('module_fingerprint'),
                    status=metadata.get('status', 'not_started'),
                ))
    return rows


def write_csv(rows, path):
    """Write the aggregation schema; absent results remain empty fields."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in RESULT_FIELDS} for row in rows)
    return path


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, default=STUDY_PATH)
    parser.add_argument('--run-root', type=Path, default='/data/qijunrong/06-RL/offline-rl/exp/RLC/runs')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    write_csv(aggregate_rows(args.study, args.run_root), args.output)
