#!/usr/bin/env python3
"""Plan or execute a declarative paired Puzzle rollout diagnostic campaign."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from impls.diagnostics.puzzle.campaign import (  # noqa: E402
    PuzzleDiagnosticCampaignError,
    campaign_output_root,
    create_controlled_replays,
    diagnostic_output_dir,
    load_campaign,
    pairing_invariants_from_outputs,
    validate_campaign_sources,
)
from tools import run_puzzle_event_diagnosis  # noqa: E402


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', required=True, type=Path)
    parser.add_argument('--source-run-root', required=True, type=Path)
    parser.add_argument('--diagnostic-root', required=True, type=Path)
    parser.add_argument('--configs', default=None, help='Comma-separated diagnostic config IDs.')
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--dry-run', action='store_true')
    mode.add_argument('--execute', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--task-ids', default=None, help='Smoke-only comma-separated task IDs.')
    parser.add_argument('--episodes-per-task', type=int, default=None, help='Smoke-only override.')
    parser.add_argument('--campaign-id', default=None, help='Output namespace override; required for smoke.')
    parser.add_argument('--jax-platform', choices=('cpu', 'gpu'), default=None)
    parser.add_argument('--cuda-visible-devices', default=None)
    parser.add_argument('--repo-root', type=Path, default=REPO_ROOT)
    return parser


def _parse_csv(value, option):
    if value is None:
        return None
    result = tuple(item.strip() for item in str(value).split(',') if item.strip())
    if not result:
        raise PuzzleDiagnosticCampaignError(f'{option} must contain at least one value')
    return result


def _parse_task_ids(value):
    parsed = _parse_csv(value, '--task-ids')
    if parsed is None:
        return None
    try:
        task_ids = tuple(sorted({int(item) for item in parsed}))
    except ValueError as error:
        raise PuzzleDiagnosticCampaignError('--task-ids must be integers') from error
    if not task_ids or any(task_id <= 0 for task_id in task_ids):
        raise PuzzleDiagnosticCampaignError('--task-ids must contain positive integers')
    return task_ids


def _git_state(repo_root):
    repo_root = Path(repo_root).resolve()
    commit = subprocess.run(
        ['git', 'rev-parse', 'HEAD'], cwd=repo_root, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    dirty = bool(subprocess.run(
        ['git', 'status', '--porcelain', '--untracked-files=all'], cwd=repo_root,
        check=True, capture_output=True, text=True,
    ).stdout)
    return {'commit': commit, 'dirty': dirty}


def _effective_protocol(protocol, args):
    task_ids = _parse_task_ids(args.task_ids)
    episodes_per_task = args.episodes_per_task
    if not args.smoke and (task_ids is not None or episodes_per_task is not None or args.campaign_id is not None):
        raise PuzzleDiagnosticCampaignError(
            'Protocol/output overrides require --smoke; full execution uses the frozen Study protocol'
        )
    if args.smoke and args.campaign_id is None:
        raise PuzzleDiagnosticCampaignError('--smoke requires an explicit --campaign-id')
    effective = dict(protocol)
    if task_ids is not None:
        if not set(task_ids).issubset(set(protocol['task_ids'])):
            raise PuzzleDiagnosticCampaignError('--task-ids must be a subset of frozen task IDs')
        effective['task_ids'] = task_ids
    if episodes_per_task is not None:
        if int(episodes_per_task) <= 0:
            raise PuzzleDiagnosticCampaignError('--episodes-per-task must be positive')
        effective['episodes_per_task'] = int(episodes_per_task)
    return effective


def _source_summary(item):
    provenance = item['provenance']
    return {
        'diagnostic_config_id': item['configuration_id'],
        'policy_label': item['policy_label'],
        'source_study_id': provenance['source_study_id'],
        'source_config_id': provenance['source_config_id'],
        'source_environment': provenance['source_environment'],
        'source_training_seed': provenance['source_training_seed'],
        'source_run_attempt': provenance['source_run_attempt'],
        'source_run_dir': provenance['source_run_dir'],
        'source_git_commit': provenance['source_git_commit'],
        'source_git_dirty': provenance['source_git_dirty'],
        'source_provenance_status': provenance['source_provenance_status'],
        'source_provenance_exception': provenance['source_provenance_exception'],
        'resolved_config_fingerprint': provenance['source_resolved_config_fingerprint'],
        'checkpoint_role': provenance['resolved_checkpoint_role'],
        'checkpoint_step': provenance['checkpoint_step'],
        'checkpoint_sha256': provenance['checkpoint_sha256'],
    }


def _run_one(configuration, source, replay, effective_protocol, campaign_root, args):
    source_policy = configuration.data['source_policy']
    output_root = diagnostic_output_dir(campaign_root, configuration)
    runner_args = SimpleNamespace(
        source_run=Path(source['provenance']['source_run_dir']),
        checkpoint_step=None,
        checkpoint_role='last',
        output_root=output_root,
        evaluation_seed=effective_protocol['evaluation_seed'],
        episodes_per_task=effective_protocol['episodes_per_task'],
        task_ids=list(effective_protocol['task_ids']),
        eval_temperature=effective_protocol['eval_temperature'],
        eval_gaussian=effective_protocol['eval_gaussian'],
        controlled_goal_replay=replay.root,
        source_provenance_exception=(
            source_policy['provenance']
            if source_policy['provenance']['provenance_status'] == 'scoped_exception'
            else None
        ),
        jax_platform=args.jax_platform,
        cuda_visible_devices=args.cuda_visible_devices,
        repo_root=args.repo_root,
    )
    return run_puzzle_event_diagnosis._run(runner_args)


def _print_plan(study, configurations, effective_protocol, sources, root):
    print(f'study              = {study.study_id}')
    print(f'campaign root      = {root}')
    print(f'cells              = {len(configurations)}')
    print(f'tasks/cell         = {len(effective_protocol["task_ids"])}')
    print(f'episodes/task      = {effective_protocol["episodes_per_task"]}')
    print(f'total episodes     = {len(configurations) * len(effective_protocol["task_ids"]) * effective_protocol["episodes_per_task"]}')
    for configuration, source in zip(configurations, sources, strict=True):
        provenance = source['provenance']
        print(
            f'[PLANNED] {configuration.config_id} <- '
            f'{provenance["source_config_id"]} '
            f'{provenance["source_environment"]} '
            f'attempt={provenance["source_run_attempt"]} '
            f'provenance={provenance["source_provenance_status"]}'
        )


def main(argv=None):
    args = _parser().parse_args(argv)
    try:
        selected = _parse_csv(args.configs, '--configs')
        study, configurations, protocol = load_campaign(
            args.study,
            include_configs=selected,
            allow_partial=bool(args.smoke),
        )
        effective_protocol = _effective_protocol(protocol, args)
        sources = validate_campaign_sources(
            study,
            configurations,
            source_run_root=args.source_run_root,
        )
        root = campaign_output_root(
            args.diagnostic_root,
            study,
            protocol,
            campaign_id=args.campaign_id,
        )
        _print_plan(study, configurations, effective_protocol, sources, root)
        if args.dry_run:
            return 0
        if root.exists() and any(root.iterdir()):
            raise PuzzleDiagnosticCampaignError(f'Campaign output root is not empty: {root}')
        git = _git_state(args.repo_root)
        if not args.smoke and git['dirty']:
            raise PuzzleDiagnosticCampaignError(
                'Full diagnostic execution requires a clean frozen worktree; use --smoke for a non-formal check'
            )
        root.mkdir(parents=True, exist_ok=True)
        campaign_manifest = {
            'status': 'running',
            'study_id': study.study_id,
            'campaign_id': root.name,
            'diagnostic_protocol': effective_protocol,
            'frozen_protocol': protocol,
            'smoke': bool(args.smoke),
            'diagnostic_code_commit': git['commit'],
            'diagnostic_git_dirty': git['dirty'],
            'sources': [_source_summary(source) for source in sources],
            'output_root': str(root),
        }
        (root / 'campaign_manifest.json').write_text(
            json.dumps(campaign_manifest, indent=2, sort_keys=True) + '\n'
        )
        replays = create_controlled_replays(root, configurations, effective_protocol)
        manifests = []
        for configuration, source in zip(configurations, sources, strict=True):
            manifests.append(_run_one(
                configuration,
                source,
                replays[configuration.data['environment']],
                effective_protocol,
                root,
                args,
            ))
        pairing = pairing_invariants_from_outputs(root, configurations)
        campaign_manifest['status'] = 'completed'
        campaign_manifest['pairing_invariants'] = pairing
        campaign_manifest['diagnostic_manifests'] = [
            str(diagnostic_output_dir(root, configuration) / 'manifest.json')
            for configuration in configurations
        ]
        (root / 'campaign_manifest.json').write_text(
            json.dumps(campaign_manifest, indent=2, sort_keys=True) + '\n'
        )
        print(json.dumps({
            'status': 'completed',
            'output_root': str(root),
            'cells': len(manifests),
            'pairing_invariants': pairing,
        }, sort_keys=True))
        return 0
    except (PuzzleDiagnosticCampaignError, OSError, ValueError) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
