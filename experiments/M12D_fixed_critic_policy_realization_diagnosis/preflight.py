from __future__ import annotations

import argparse
import json
from pathlib import Path

from impls.diagnostics.checkpoints import load_primary_actors

from common import load_env_dataset, protocol_from_arg, provenance_for_seed, write_json

EXPECTED_PRIMARY = ('K1SN', 'K4SN', 'K4SZ', 'D9', 'Residual')


def run(protocol, seeds, temporal=False):
    names = tuple(protocol['primary_actor_names'])
    if names != EXPECTED_PRIMARY:
        raise ValueError(f'Unexpected M12-D primary set: {names}')
    if protocol['actors']['K4SN'].get('config') != 'M12A-C003' or protocol['actors']['K4SN'].get('run_attempt') != 2:
        raise ValueError('K4SN is not M12B-R M12A-C003 attempt2')
    if protocol['actors']['D9'].get('config') != 'M12B-C006' or protocol['actors']['Residual'].get('config') != 'M12B-C007':
        raise ValueError('D9/Residual source identity mismatch')
    results = []
    for seed in seeds:
        _, dataset, _ = load_env_dataset(protocol, seed)
        actors = load_primary_actors(
            protocol, seed=seed, run_root=protocol['run_root'], dataset=dataset
        )
        if tuple(actors) != names:
            raise ValueError(f'Loaded actor order mismatch: {tuple(actors)}')
        actor_rows = []
        for name in names:
            actor = actors[name]
            actor_rows.append({
                'name': name, 'label': actor.source.label, 'run_dir': str(actor.run_dir),
                'checkpoint': actor.checkpoint, 'actor_fingerprint': actor.actor_fingerprint,
                'critic_fingerprint': actor.critic_fingerprint,
                'metadata_commit': actor.metadata.get('git_commit'),
                'metadata_dirty': actor.metadata.get('git_dirty'),
            })
        results.append({
            'seed': int(seed), 'status': 'PASS', 'actor_count': len(actor_rows),
            'actors': actor_rows, 'provenance': provenance_for_seed(protocol, seed),
        })
    temporal_results = []
    if temporal:
        for step in protocol['temporal_checkpoints']:
            for seed in seeds:
                _, dataset, _ = load_env_dataset(protocol, seed)
                actors = load_primary_actors(
                    protocol, seed=seed, run_root=protocol['run_root'], dataset=dataset,
                    checkpoint_selector=int(step), expected_checkpoint_step=int(step),
                )
                temporal_results.append({'seed': int(seed), 'checkpoint_step': int(step),
                                         'actor_count': len(actors), 'status': 'PASS'})
    return {
        'study_id': protocol['study_id'], 'source_commit': protocol['source_commit'],
        'primary_actor_names': list(names), 'seeds': list(map(int, seeds)),
        'checkpoint_rule': 'last@1M', 'status': 'PASS', 'results': results,
        'temporal_results': temporal_results,
        'formal_training_started': False, 'formal_diagnostic_started': False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', default=str(Path(__file__).with_name('protocol.yaml')))
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument('--temporal', action='store_true')
    parser.add_argument('--output', default=None)
    args = parser.parse_args(argv)
    result = run(protocol_from_arg(args.protocol), args.seeds, temporal=args.temporal)
    if args.output:
        write_json(args.output, result)
    else:
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()

