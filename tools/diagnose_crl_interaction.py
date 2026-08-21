#!/usr/bin/env python3
"""Doctor, build, audit, and score the M11A CRL interaction diagnostics.

Every stage is post-hoc except ``doctor``.  No stage starts formal training.
The bank, candidate pools, scores, and reports are written below the explicit
diagnostic root and never under a source run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from impls.analysis.crl_interaction import (
    aggregate_interaction_metrics,
    generate_candidate_pools,
    generate_diagnostic_bank,
    load_interaction_spec,
    run_critic_identity_audit,
    score_diagnostics,
    _source_validation_payload,
    smoke_m11a_configs,
    validate_source_set,
)


def _args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--stage',
        choices=('doctor', 'bank', 'candidates', 'audit', 'score', 'aggregate'),
        required=True,
    )
    parser.add_argument('--study', default='experiments/M11A_crl_computation_interaction/study.yaml')
    parser.add_argument('--spec', default='experiments/M11A_crl_computation_interaction/diagnostic.yaml')
    parser.add_argument('--diagnostic-root', default='/data/qijunrong/06-RL/offline-rl/exp/RLC/diagnostics')
    parser.add_argument('--identity-tolerance', type=float, default=1e-6)
    return parser.parse_args(argv)


def main(argv=None):
    args = _args(argv)
    try:
        spec = load_interaction_spec(args.spec)
        if args.stage == 'doctor':
            sources = validate_source_set(spec)
            result = {
                'status': 'passed',
                'source_validation': _source_validation_payload(spec, sources),
                'planned_artifact_root': str(Path(args.diagnostic_root).resolve() / spec['diagnostic_id']),
                'planned_artifact_paths': {
                    'bank': 'bank/diagnostic_bank.npz + bank_metadata.json',
                    'candidates': 'candidates/single_state_candidates.npz + two_state_candidates.npz + candidate_metadata.json',
                    'audits': 'audits/source_validation.json + critic_identity.json',
                    'scores': 'scores/evaluator_scores.npz + extraction_scores.npz + score_metadata.json',
                    'metrics': 'metrics/evaluator_metrics.csv + extraction_metrics.csv + mechanism_deltas.csv + interaction_metrics.csv',
                    'reports': 'reports/diagnostic_summary.json + diagnostic_summary.md',
                },
                'pair_count': 'runtime-dependent; reported by bank stage after environment rollout',
                'config_smoke': smoke_m11a_configs(args.study),
            }
        else:
            if args.stage == 'bank':
                result = generate_diagnostic_bank(spec, args.diagnostic_root)
            elif args.stage == 'candidates':
                result = generate_candidate_pools(spec, args.diagnostic_root)
            elif args.stage == 'audit':
                result = run_critic_identity_audit(spec, args.diagnostic_root, args.identity_tolerance)
            elif args.stage == 'score':
                result = score_diagnostics(spec, args.diagnostic_root)
            else:
                result = aggregate_interaction_metrics(spec, args.diagnostic_root)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f'M11A diagnostic failed: {error}', file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
