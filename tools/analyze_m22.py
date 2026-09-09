"""Analyze M22 Puzzle baseline Runs with an explicit paper-readiness gate.

The analyzer is deliberately independent of training.  It consumes the
deterministic Study Run identities and per-Run ``runtime_metadata.json`` /
``eval.csv`` artifacts, preserves raw success values in [0, 1], and only
publishes a paper-ready primary table when all 72 Runs satisfy the frozen
protocol.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from impls.experiment import load_study, make_run_path, prepare_run_design  # noqa: E402


STUDY_ID = 'M22'
ALGORITHMS = ('gcbc', 'gcivl', 'gciql', 'qrl', 'crl', 'hiql')
ENVIRONMENTS = (
    'puzzle-3x3-play-v0',
    'puzzle-4x4-play-v0',
    'puzzle-4x5-play-v0',
    'puzzle-4x6-play-v0',
)
SEEDS = (0, 1, 2)
EXPECTED_EVAL_STEPS = tuple(range(100_000, 1_000_001, 100_000))
PRIMARY_METRIC = 'evaluation/overall_success'
DISPLAY_NAMES = {
    'gcbc': 'GCBC',
    'gcivl': 'GCIVL',
    'gciql': 'GCIQL',
    'qrl': 'QRL',
    'crl': 'CRL',
    'hiql': 'HIQL',
}
ENV_DISPLAY_NAMES = {
    'puzzle-3x3-play-v0': 'Puzzle 3×3',
    'puzzle-4x4-play-v0': 'Puzzle 4×4',
    'puzzle-4x5-play-v0': 'Puzzle 4×5',
    'puzzle-4x6-play-v0': 'Puzzle 4×6',
}
OFFICIAL_OVERRIDES = {
    'gcbc': 'none',
    'gcivl': 'alpha=10',
    'gciql': 'alpha=1',
    'qrl': 'alpha=0.3',
    'crl': 'alpha=3',
    'hiql': 'high_alpha=3, low_alpha=3, subgoal_steps=10',
}
SEMANTICS = {
    'gcbc': ('upstream_semantic_match', '—'),
    'gcivl': ('rlc_variant_documented', 'post-gradient target Polyak update'),
    'gciql': ('rlc_variant_documented', 'post-gradient target Polyak update'),
    'qrl': ('upstream_semantic_match', '—'),
    'crl': ('upstream_semantic_match', '—'),
    'hiql': ('upstream_semantic_match', '—'),
}
DEFAULT_STUDY = REPO_ROOT / 'experiments/M22_puzzle_baselines_unified_rlc/study.yaml'
DEFAULT_RUN_PARENT = Path('/data/qijunrong/06-RL/offline-rl/exp/RLC/runs')
DEFAULT_OUTPUT = REPO_ROOT / 'docs/9-9/M22_results'
DEFAULT_DATASET_AUDIT = REPO_ROOT / 'docs/9-9/M22_dataset_audit.json'


def _float(value: Any) -> float | None:
    if value in (None, ''):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _int(value: Any) -> int | None:
    result = _float(value)
    return None if result is None else int(result)


def _mean_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    mean = float(np.mean(values))
    std = 0.0 if len(values) == 1 else float(np.std(values, ddof=1))
    return mean, std


def _trapezoid(values, x):
    """Use the NumPy 2 name while retaining compatibility with NumPy 1."""

    integrate = getattr(np, 'trapezoid', None) or np.trapz
    return integrate(values, x)


def _read_json(path: Path) -> dict[str, Any]:
    with Path(path).open() as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f'Expected JSON object: {path}')
    return value


def expected_runs(study_path: Path = DEFAULT_STUDY, run_root: Path = DEFAULT_RUN_PARENT) -> list[dict[str, Any]]:
    """Expand the Study into deterministic run identities without filesystem writes."""

    study = load_study(study_path)
    configurations = [
        prepare_run_design(study_path, path)[1]
        for path in sorted((study.path.parent / 'configs').glob('*.yaml'))
    ]
    runs = []
    for configuration in configurations:
        environments = [configuration.data['environment']] if 'environment' in configuration.data else study.data['environments']
        for environment in environments:
            for seed in study.data['seeds']:
                run_dir = make_run_path(
                    run_root,
                    study.study_id,
                    configuration.config_id,
                    configuration.slug,
                    environment,
                    int(seed),
                )
                runs.append({
                    'study_id': study.study_id,
                    'config_id': configuration.config_id,
                    'slug': configuration.slug,
                    'algorithm': configuration.data['algorithm'],
                    'environment': environment,
                    'seed': int(seed),
                    'run_attempt': 0,
                    'run_dir': run_dir,
                    'configuration': configuration,
                })
    return runs


def _eval_columns(fieldnames: list[str]) -> list[str]:
    return [
        field for field in fieldnames
        if field.startswith('evaluation/')
        and field.endswith('_success')
        and field != PRIMARY_METRIC
    ]


def _parse_eval(eval_path: Path) -> dict[str, Any]:
    """Parse one eval.csv without interpolating missing evaluations."""

    records = []
    with Path(eval_path).open(newline='') as file:
        reader = csv.DictReader(file)
        fields = reader.fieldnames or []
        if 'step' not in fields or PRIMARY_METRIC not in fields:
            raise ValueError(
                f'eval.csv must contain step and {PRIMARY_METRIC}: {eval_path}'
            )
        task_columns = _eval_columns(fields)
        for row in reader:
            step = _int(row.get('step'))
            value = _float(row.get(PRIMARY_METRIC))
            if step is None or value is None:
                continue
            if not 0.0 <= value <= 1.0:
                raise ValueError(f'primary success outside [0,1] at step {step}: {value}')
            task_values = {
                column: _float(row.get(column))
                for column in task_columns
                if _float(row.get(column)) is not None
            }
            records.append({'step': step, 'success': value, 'tasks': task_values})
    records.sort(key=lambda item: item['step'])
    if not records:
        raise ValueError(f'eval.csv contains no finite primary metric rows: {eval_path}')
    by_step = {}
    for record in records:
        if record['step'] in by_step:
            raise ValueError(f'duplicate evaluation step {record["step"]}: {eval_path}')
        by_step[record['step']] = record
    final = by_step.get(1_000_000)
    values = [(record['success'], record['step']) for record in records]
    best_success, best_step = max(values, key=lambda pair: (pair[0], -pair[1]))
    last_three = [record['success'] for record in records[-3:]]
    missing_steps = [step for step in EXPECTED_EVAL_STEPS if step not in by_step]
    if not missing_steps:
        auc = float(_trapezoid(
            [by_step[step]['success'] for step in EXPECTED_EVAL_STEPS],
            EXPECTED_EVAL_STEPS,
        ) / 900_000.0)
        auc_status = 'complete'
    else:
        auc = None
        auc_status = 'incomplete_curve'
    final_tasks = {} if final is None else final['tasks']
    return {
        'records': records,
        'evaluation_steps': [record['step'] for record in records],
        'eval_points': len(records),
        'final_success': None if final is None else final['success'],
        'final_step': None if final is None else final['step'],
        'best_success': best_success,
        'best_step': best_step,
        'last3_mean': float(np.mean(last_three)) if last_three else None,
        'normalized_auc': auc,
        'auc_status': auc_status,
        'auc_missing_steps': missing_steps,
        'final_tasks': final_tasks,
    }


def _dataset_sha_map(dataset_audit_path: Path) -> dict[str, str]:
    try:
        audit = _read_json(dataset_audit_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return {}
    return {
        environment: record.get('train', {}).get('sha256')
        for environment, record in audit.get('environments', {}).items()
        if record.get('train', {}).get('sha256')
    }


def _run_record(run: Mapping[str, Any], dataset_shas: Mapping[str, str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    run_dir = Path(run['run_dir'])
    metadata_path = run_dir / 'runtime_metadata.json'
    base = {
        'study_id': run['study_id'],
        'config_id': run['config_id'],
        'algorithm': run['algorithm'],
        'environment': run['environment'],
        'seed': run['seed'],
        'run_attempt': run['run_attempt'],
        'run_dir': str(run_dir),
        'dataset_sha': dataset_shas.get(run['environment']),
        'eval_points': 0,
        'evaluation_steps': [],
        'final_success': None,
        'final_step': None,
        'best_success': None,
        'best_step': None,
        'last3_mean': None,
        'normalized_auc': None,
        'auc_status': 'not_available',
        'auc_missing_steps': list(EXPECTED_EVAL_STEPS),
        'source_commit': None,
        'eval_path': str(run_dir / 'eval.csv'),
        'failure_reason': None,
    }
    if not metadata_path.is_file():
        return base | {'status': 'missing'}, []
    try:
        metadata = _read_json(metadata_path)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return base | {'status': 'invalid', 'failure_reason': f'metadata: {error}'}, []
    base['source_commit'] = metadata.get('git_commit')
    base['run_attempt'] = metadata.get('run_attempt', run['run_attempt'])
    observed_identity = {
        'study_id': metadata.get('study_id'),
        'config_id': metadata.get('config_id'),
        'algorithm': metadata.get('algorithm'),
        'environment': metadata.get('environment'),
        'seed': metadata.get('seed'),
        'run_attempt': metadata.get('run_attempt', 0),
    }
    expected_identity = {
        key: run[key]
        for key in ('study_id', 'config_id', 'algorithm', 'environment', 'seed', 'run_attempt')
    }
    if any(observed_identity.get(key) != value for key, value in expected_identity.items()):
        base['failure_reason'] = f'identity mismatch: expected={expected_identity}, observed={observed_identity}'
        return base | {'status': 'invalid'}, []
    runtime_status = metadata.get('status', 'invalid')
    if runtime_status != 'completed':
        return base | {'status': runtime_status if runtime_status in {'running', 'failed', 'aborted', 'invalid'} else 'invalid', 'failure_reason': metadata.get('failure_reason')}, []
    eval_path = run_dir / 'eval.csv'
    if not eval_path.is_file():
        return base | {'status': 'incomplete', 'failure_reason': 'completed Run has no eval.csv'}, []
    try:
        parsed = _parse_eval(eval_path)
    except (OSError, ValueError, csv.Error) as error:
        return base | {'status': 'invalid', 'failure_reason': f'eval.csv: {error}'}, []
    base.update({
        key: parsed[key]
        for key in (
            'eval_points', 'evaluation_steps', 'final_success', 'final_step',
            'best_success', 'best_step', 'last3_mean', 'normalized_auc',
            'auc_status', 'auc_missing_steps',
        )
    })
    task_rows = [
        {
            'study_id': run['study_id'],
            'config_id': run['config_id'],
            'algorithm': run['algorithm'],
            'environment': run['environment'],
            'seed': run['seed'],
            'task_metric': metric,
            'final_step': parsed['final_step'],
            'final_success': value,
            'status': 'complete' if parsed['final_step'] == 1_000_000 else 'missing_final',
            'run_dir': str(run_dir),
        }
        for metric, value in sorted(parsed['final_tasks'].items())
    ]
    if parsed['final_step'] != 1_000_000:
        base['status'] = 'incomplete'
        base['failure_reason'] = 'final@1M primary metric is missing'
    else:
        base['status'] = 'completed'
    return base, task_rows


def _cell_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    complete = [record for record in records if record['status'] == 'completed']
    final_values = [record['final_success'] for record in complete if record['final_success'] is not None]
    mean, std = _mean_std(final_values)
    commits = sorted({record['source_commit'] for record in complete if record.get('source_commit')})
    return {
        'algorithm': records[0]['algorithm'] if records else None,
        'environment': records[0]['environment'] if records else None,
        'expected_seeds': list(SEEDS),
        'observed_seeds': sorted(record['seed'] for record in complete),
        'seed_count': len({record['seed'] for record in complete}),
        'mean': mean,
        'std': std,
        'source_commits': commits,
        'ready': (
            len(complete) == len(SEEDS)
            and {record['seed'] for record in complete} == set(SEEDS)
            and all(record['final_step'] == 1_000_000 for record in complete)
        ),
    }


def _paper_ready(records: list[dict[str, Any]], cells: Mapping[tuple[str, str], Mapping[str, Any]]) -> tuple[bool, list[str]]:
    reasons = []
    if len(records) != 72:
        reasons.append(f'Run count {len(records)}/72')
    if any(record['status'] != 'completed' for record in records):
        counts = defaultdict(int)
        for record in records:
            counts[record['status']] += 1
        reasons.append('Run statuses: ' + ', '.join(f'{key}={value}' for key, value in sorted(counts.items())))
    for key in [(algorithm, environment) for algorithm in ALGORITHMS for environment in ENVIRONMENTS]:
        cell = cells.get(key, {})
        if not cell.get('ready', False):
            reasons.append(f'cell {key[0]} × {key[1]} is not 3/3 with final@1M')
    commits = {record['source_commit'] for record in records if record.get('source_commit')}
    if len(commits) != 1 or any(not record.get('source_commit') for record in records):
        reasons.append(f'source commit is not consistent across completed Runs: {sorted(commits)}')
    return not reasons, reasons


def build_analysis(
    *,
    study_path: Path = DEFAULT_STUDY,
    run_root: Path = DEFAULT_RUN_PARENT,
    dataset_audit_path: Path = DEFAULT_DATASET_AUDIT,
) -> dict[str, Any]:
    """Collect raw Run records and readiness facts without writing outputs."""

    runs = expected_runs(study_path, run_root)
    dataset_shas = _dataset_sha_map(dataset_audit_path)
    records = []
    task_rows = []
    for run in runs:
        record, run_task_rows = _run_record(run, dataset_shas)
        records.append(record)
        task_rows.extend(run_task_rows)
    cells = {
        (algorithm, environment): _cell_summary([
            record for record in records
            if record['algorithm'] == algorithm and record['environment'] == environment
        ])
        for algorithm in ALGORITHMS
        for environment in ENVIRONMENTS
    }
    paper_ready, reasons = _paper_ready(records, cells)
    return {
        'records': records,
        'task_rows': task_rows,
        'cells': cells,
        'paper_ready': paper_ready,
        'paper_ready_reasons': reasons,
        'expected_eval_steps': list(EXPECTED_EVAL_STEPS),
        'study_path': str(Path(study_path).resolve()),
        'run_root': str(Path(run_root).resolve()),
    }


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _write_csv(path: Path, rows: list[Mapping[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def _write_raw_outputs(output_root: Path, analysis: Mapping[str, Any]) -> None:
    records = list(analysis['records'])
    raw_fields = [
        'study_id', 'config_id', 'algorithm', 'environment', 'seed', 'run_attempt',
        'status', 'source_commit', 'dataset_sha', 'eval_points', 'evaluation_steps',
        'final_step', 'final_success', 'best_success', 'best_step', 'last3_mean',
        'normalized_auc', 'auc_status', 'auc_missing_steps', 'run_dir', 'eval_path',
        'failure_reason',
    ]
    raw_rows = []
    for record in records:
        row = dict(record)
        row['evaluation_steps'] = json.dumps(row['evaluation_steps'], separators=(',', ':'))
        row['auc_missing_steps'] = json.dumps(row['auc_missing_steps'], separators=(',', ':'))
        raw_rows.append(row)
    _write_csv(output_root / 'raw_seed_results.csv', raw_rows, raw_fields)
    (output_root / 'raw_seed_results.json').write_text(
        json.dumps(records, indent=2, sort_keys=True, default=_json_default) + '\n'
    )
    _write_csv(
        output_root / 'per_task_results.csv',
        list(analysis['task_rows']),
        ['study_id', 'config_id', 'algorithm', 'environment', 'seed', 'task_metric', 'final_step', 'final_success', 'status', 'run_dir'],
    )
    secondary_rows = []
    for record in records:
        secondary_rows.append({
            key: record.get(key)
            for key in ('study_id', 'config_id', 'algorithm', 'environment', 'seed', 'status', 'best_success', 'best_step', 'last3_mean', 'normalized_auc', 'auc_status')
        })
    _write_csv(
        output_root / 'secondary_metrics.csv',
        secondary_rows,
        ['study_id', 'config_id', 'algorithm', 'environment', 'seed', 'status', 'best_success', 'best_step', 'last3_mean', 'normalized_auc', 'auc_status'],
    )


def _format_percent(mean: float | None, std: float | None) -> str:
    if mean is None or std is None:
        return '—'
    return f'{100.0 * mean:.3f} ± {100.0 * std:.3f}'


def _write_paper_outputs(output_root: Path, analysis: Mapping[str, Any], allow_partial: bool) -> None:
    paper_ready = bool(analysis['paper_ready'])
    cells = analysis['cells']
    csv_rows = []
    markdown_rows = []
    for algorithm in ALGORITHMS:
        row = {'Algorithm': DISPLAY_NAMES[algorithm]}
        for environment in ENVIRONMENTS:
            cell = cells[(algorithm, environment)]
            value = _format_percent(cell.get('mean'), cell.get('std')) if cell.get('ready') else '—'
            row[environment] = value if (paper_ready or allow_partial) else ''
        csv_rows.append(row)
        markdown_rows.append(
            '| ' + ' | '.join([row['Algorithm']] + [row[environment] for environment in ENVIRONMENTS]) + ' |'
        )
    csv_path = output_root / 'paper_table_final.csv'
    if not paper_ready and not allow_partial:
        _write_csv(csv_path, [], ['Algorithm', *ENVIRONMENTS])
        text = (
            '# M22 primary table\n\n'
            '**PAPER TABLE NOT GENERATED: paper-ready gate failed.**\n\n'
            + '\n'.join(f'- {reason}' for reason in analysis['paper_ready_reasons'])
            + '\n'
        )
        (output_root / 'paper_table_final.md').write_text(text)
        return
    _write_csv(csv_path, csv_rows, ['Algorithm', *ENVIRONMENTS])
    label = 'PAPER READY' if paper_ready else 'PARTIAL / NOT PAPER READY'
    text = (
        '# M22 primary table\n\n'
        f'**{label}**\n\n'
        '| Algorithm | Puzzle 3×3 | Puzzle 4×4 | Puzzle 4×5 | Puzzle 4×6 |\n'
        '|---|---:|---:|---:|---:|\n'
        + '\n'.join(markdown_rows)
        + '\n\nValues are 100 × raw success; mean and sample standard deviation use the three seed values before display rounding.\n'
    )
    (output_root / 'paper_table_final.md').write_text(text)


def _write_protocol_table(output_root: Path, study_path: Path) -> None:
    study = load_study(study_path)
    configs = {}
    for path in sorted((study.path.parent / 'configs').glob('*.yaml')):
        configuration = prepare_run_design(study_path, path)[1]
        configs.setdefault(configuration.data['algorithm'], configuration.data)
    protocol = study.data.get('protocol', {})
    training = '1,000,000 steps; batch=1,024; log=5,000; save=100,000'
    evaluation = 'all tasks; 50 episodes per task; temperature=0; video=0; primary=final@1M'
    lines = [
        '# M22 baseline protocol',
        '',
        '| Algorithm | Official Puzzle override | Implementation semantics | Documented deviations | Dataset provenance | Training protocol | Evaluation protocol |',
        '|---|---|---|---|---|---|---|',
    ]
    for algorithm in ALGORITHMS:
        data = configs[algorithm]
        semantics, difference = SEMANTICS[algorithm]
        lines.append(
            f'| {DISPLAY_NAMES[algorithm]} | {OFFICIAL_OVERRIDES[algorithm]} | '
            f'{semantics} | {difference} | official OGBench dataset + audited SHA-256 | '
            f'{training} | {evaluation} |'
        )
    lines.extend([
        '',
        f'Protocol fields are declared in `{study.path}` and were not inferred from result files.',
        f'`eval_episodes={protocol.get("eval_episodes")}` means 50 episodes for each task in the canonical evaluator.',
        '',
    ])
    (output_root / 'baseline_protocol_table.md').write_text('\n'.join(lines))


def _write_curves(output_root: Path, analysis: Mapping[str, Any]) -> None:
    curve_root = output_root / 'curves'
    curve_root.mkdir(parents=True, exist_ok=True)
    for environment in ENVIRONMENTS:
        rows = []
        for step in EXPECTED_EVAL_STEPS:
            values = []
            for record in analysis['records']:
                if record['environment'] != environment or record['status'] != 'completed':
                    continue
                eval_path = Path(record['eval_path'])
                if not eval_path.is_file():
                    continue
                try:
                    parsed = _parse_eval(eval_path)
                except (OSError, ValueError, csv.Error):
                    continue
                values_by_step = {item['step']: item['success'] for item in parsed['records']}
                if step in values_by_step:
                    values.append(values_by_step[step])
            mean, std = _mean_std(values)
            rows.append({'step': step, 'mean_success': mean, 'std_success': std, 'n': len(values)})
        _write_csv(curve_root / f'{environment}.csv', rows, ['step', 'mean_success', 'std_success', 'n'])


def _write_manifest(output_root: Path, analysis: Mapping[str, Any], allow_partial: bool) -> None:
    records = list(analysis['records'])
    counts = defaultdict(int)
    for record in records:
        counts[record['status']] += 1
    cells = {
        f'{algorithm}::{environment}': value
        for (algorithm, environment), value in analysis['cells'].items()
    }
    manifest = {
        'schema': 'm22_completion_manifest_v1',
        'study_id': STUDY_ID,
        'study_path': analysis['study_path'],
        'run_root': analysis['run_root'],
        'expected_runs': 72,
        'observed_runs': len(records),
        'counts': {
            'completed': counts['completed'],
            'running': counts['running'],
            'failed': counts['failed'],
            'aborted': counts['aborted'],
            'missing': counts['missing'],
            'incomplete': counts['incomplete'],
            'invalid': counts['invalid'],
        },
        'cells': cells,
        'source_commits': sorted({record['source_commit'] for record in records if record.get('source_commit')}),
        'expected_eval_steps': list(EXPECTED_EVAL_STEPS),
        'paper_ready': bool(analysis['paper_ready']),
        'paper_ready_reasons': list(analysis['paper_ready_reasons']),
        'mode': 'partial' if allow_partial else 'final',
        'primary_metric': PRIMARY_METRIC,
        'primary_endpoint': 'final@1M',
        'final_table_generated': bool(analysis['paper_ready'] or allow_partial),
    }
    (output_root / 'completion_manifest.json').write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + '\n'
    )


def analyze_and_write(
    *,
    study_path: Path = DEFAULT_STUDY,
    run_root: Path = DEFAULT_RUN_PARENT,
    output_root: Path = DEFAULT_OUTPUT,
    dataset_audit_path: Path = DEFAULT_DATASET_AUDIT,
    allow_partial: bool = False,
) -> dict[str, Any]:
    analysis = build_analysis(
        study_path=study_path,
        run_root=run_root,
        dataset_audit_path=dataset_audit_path,
    )
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    _write_raw_outputs(output_root, analysis)
    _write_paper_outputs(output_root, analysis, allow_partial)
    _write_protocol_table(output_root, Path(study_path))
    _write_curves(output_root, analysis)
    _write_manifest(output_root, analysis, allow_partial)
    return analysis


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, default=DEFAULT_STUDY)
    parser.add_argument('--run-root', type=Path, default=DEFAULT_RUN_PARENT)
    parser.add_argument('--output-root', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--dataset-audit', type=Path, default=DEFAULT_DATASET_AUDIT)
    parser.add_argument('--allow-partial', action='store_true')
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    analysis = analyze_and_write(
        study_path=args.study,
        run_root=args.run_root,
        output_root=args.output_root,
        dataset_audit_path=args.dataset_audit,
        allow_partial=args.allow_partial,
    )
    records = analysis['records']
    counts = defaultdict(int)
    for record in records:
        counts[record['status']] += 1
    print(
        f'M22 ANALYSIS: {"PAPER READY" if analysis["paper_ready"] else "PARTIAL / NOT PAPER READY"}'
    )
    print(
        f'completed={counts["completed"]} running={counts["running"]} '
        f'failed={counts["failed"]} missing={counts["missing"]} '
        f'incomplete={counts["incomplete"]} invalid={counts["invalid"]}'
    )
    if not analysis['paper_ready']:
        for reason in analysis['paper_ready_reasons']:
            print(f'  - {reason}')
    if analysis['paper_ready'] or args.allow_partial:
        return 0
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
