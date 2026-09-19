#!/usr/bin/env python3
"""Read-only audit of Study agent/protocol inheritance against exact run files."""

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def differences(left, right, prefix=''):
    if isinstance(left, dict) and isinstance(right, dict):
        result = {}
        for key in sorted(set(left) | set(right)):
            name = f'{prefix}.{key}' if prefix else key
            if key not in left or key not in right:
                result[name] = {'reference': left.get(key), 'candidate': right.get(key), 'missing_key': True}
            else:
                result.update(differences(left[key], right[key], name))
        return result
    return {} if left == right else {prefix: {'reference': left, 'candidate': right}}


def audit(study_path, reference_study_path, reference_ids, reference_runs, allowed_agent_fields):
    from tools.audit_goal_conditioning_study import resolved_configurations
    from impls.experiment.management import config_fingerprint, jsonable, git_metadata
    from impls.utils.checkpointing import sha256_file
    study, candidates = resolved_configurations(study_path)
    reference_study, references = resolved_configurations(reference_study_path, set(reference_ids))
    report = {'source': git_metadata(ROOT), 'study_id': study.study_id,
              'allowed_agent_differences': sorted(allowed_agent_fields), 'declared': [], 'runtime': [],
              'evidence_level': 'runtime_verified' if reference_runs else 'declared_only',
              'formal_preflight_ready': False, 'errors': []}
    for reference, config in references:
        baseline = jsonable(config)
        for candidate, candidate_config in candidates:
            changes = differences(baseline, jsonable(candidate_config))
            forbidden = {k: v for k, v in changes.items() if k.split('.')[0] not in allowed_agent_fields}
            report['declared'].append({'reference': reference.config_id, 'candidate': candidate.config_id,
                                       'changed_fields': list(changes), 'forbidden_differences': forbidden})
            if forbidden:
                report['errors'].append({'config': candidate.config_id, 'agent': forbidden})
    reference_protocol = {k: v for k, v in reference_study.data['protocol'].items() if k != 'formal_training_started'}
    protocol_changes = differences(reference_protocol, study.data['protocol'])
    report['declared_protocol_differences'] = protocol_changes
    if protocol_changes:
        report['errors'].append({'protocol': protocol_changes})
    refs = {c.config_id: (c, jsonable(cfg)) for c, cfg in references}
    seen = set()
    for root in reference_runs:
        root = Path(root)
        resolved_path, metadata_path = root / 'resolved_config.json', root / 'runtime_metadata.json'
        resolved, metadata = json.loads(resolved_path.read_text()), json.loads(metadata_path.read_text())
        fingerprint = config_fingerprint({k: v for k, v in resolved.items() if k != 'resolved_config_fingerprint'})
        if fingerprint != resolved['resolved_config_fingerprint'] or fingerprint != metadata['resolved_config_fingerprint']:
            raise ValueError(f'Resolved config fingerprint mismatch: {root}')
        config_id = resolved['configuration']['config_id']
        if config_id not in refs or config_id in seen:
            raise ValueError(f'Unknown/duplicate reference run: {config_id}')
        seen.add(config_id)
        config, declared = refs[config_id]
        actual = resolved['algorithm_config']
        agent_changes = differences(declared, actual['agent'])
        # Every effective launcher protocol value is required, not intersection-only.
        launcher_keys = ('train_steps', 'batch_size', 'log_interval', 'eval_interval', 'eval_tasks',
                         'eval_episodes', 'eval_temperature', 'eval_gaussian', 'video_episodes',
                         'save_interval', 'save_best_checkpoint', 'save_last_checkpoint')
        launcher_expected = {k: reference_protocol[k] for k in launcher_keys}
        launcher_actual = {k: actual['launcher'][k] for k in launcher_keys if k in actual['launcher']}
        launcher_changes = differences(launcher_expected, launcher_actual)
        expected_selection = {k: reference_protocol[k] for k in ('selection_metric', 'selection_rule')}
        selection_changes = differences(expected_selection, {k: metadata.get('checkpoint_lifecycle', {}).get(k) for k in expected_selection})
        if agent_changes or launcher_changes or selection_changes:
            report['errors'].append({'run': str(root), 'agent': agent_changes,
                                     'protocol': launcher_changes, 'checkpoint_selection': selection_changes})
        report['runtime'].append({
            'run': str(root), 'config_id': config_id, 'environment': actual['environment'],
            'training_seed': actual['training_seed'], 'run_attempt': actual['run_attempt'],
            'recorded_source_sha': metadata['git_commit'], 'recorded_git_dirty': metadata['git_dirty'],
            'recorded_status': metadata.get('status'), 'resolved_config_fingerprint': fingerprint,
            'resolved_file_sha256': sha256_file(resolved_path), 'runtime_file_sha256': sha256_file(metadata_path),
            'agent_field_count': len(declared), 'agent_field_names': sorted(declared),
            'agent_differences': agent_changes, 'launcher_protocol': launcher_actual,
            'protocol_differences': launcher_changes, 'checkpoint_selection_differences': selection_changes,
            'recorded_computation_readouts': {k: v.get('readout') for k, v in metadata.get('architecture_accounting', {}).get('slots', {}).items()},
        })
    if reference_runs and seen != set(reference_ids):
        report['errors'].append({'missing_runtime_references': sorted(set(reference_ids) - seen)})
    report['status'] = 'pass' if not report['errors'] else 'fail'
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--reference-study', type=Path, required=True)
    parser.add_argument('--reference-configs', required=True)
    parser.add_argument('--reference-run', type=Path, action='append', default=[])
    parser.add_argument('--allow-agent-field', action='append', default=[])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        parser.error('Output exists; select a new evidence path')
    os.environ['JAX_PLATFORMS'] = 'cpu'
    report = audit(args.study, args.reference_study, args.reference_configs.split(','),
                   args.reference_run, set(args.allow_agent_field))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as file:
        json.dump(report, file, indent=2, sort_keys=True, allow_nan=False)
        file.write('\n')
    print(json.dumps({'status': report['status'], 'evidence_level': report['evidence_level'],
                      'output': str(args.output), 'errors': report['errors']}))
    return 0 if report['status'] == 'pass' else 1


if __name__ == '__main__':
    raise SystemExit(main())
