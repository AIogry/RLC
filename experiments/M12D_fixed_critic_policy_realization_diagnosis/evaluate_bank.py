from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from impls.diagnostics.banks import load_bank
from impls.diagnostics.checkpoints import load_primary_actors
from impls.diagnostics.metrics import aggregate_rows, evaluate_bank, evaluate_bank_samples, pairwise_contrasts
from impls.diagnostics.support import support_distance

from common import load_env_dataset, protocol_from_arg, write_json


def write_csv(path, rows):
    fields = sorted({key for row in rows for key in row})
    with Path(path).open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', default=str(Path(__file__).with_name('protocol.yaml')))
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--bank', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--support-bank', default=None)
    parser.add_argument('--checkpoint-step', type=int, default=None)
    args = parser.parse_args(argv)
    protocol = protocol_from_arg(args.protocol)
    bank = load_bank(args.bank)
    _, dataset, _ = load_env_dataset(protocol, args.seed)
    actors = load_primary_actors(
        protocol, seed=args.seed, run_root=protocol['run_root'], dataset=dataset,
        checkpoint_selector=args.checkpoint_step,
        expected_checkpoint_step=args.checkpoint_step,
    )
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite evaluation output: {output}')
    output.mkdir(parents=True)
    support = load_bank(args.support_bank) if args.support_bank else None
    support_values = None
    if support is not None:
        support_values = support_distance(bank.arrays['observations'], support)
        np.save(output / 'support_proxy.npy', support_values)
    actor_results, summary_rows = {}, []
    for actor_name in protocol['primary_actor_names']:
        actor = actors[actor_name]
        actor_dir = output / actor_name
        actor_dir.mkdir()
        rows = evaluate_bank(actor, bank)
        samples = evaluate_bank_samples(actor, bank)
        np.savez_compressed(actor_dir / 'metrics_raw.npz', **samples)
        write_csv(actor_dir / 'metrics_rows.csv', rows)
        summary = aggregate_rows(rows)
        if support_values is not None:
            summary.update({
                'support_proxy_mean': float(support_values.mean()),
                'support_proxy_p90': float(np.quantile(support_values, 0.90)),
                'support_proxy_max': float(support_values.max()),
            })
        write_csv(actor_dir / 'metrics_summary.csv', [summary])
        actor_results[actor_name] = samples
        summary_rows.append({'actor': actor_name, **summary})
        write_json(actor_dir / 'evaluation_manifest.json', {
            'study_id': protocol['study_id'], 'seed': int(args.seed),
            'bank_root': str(Path(args.bank).resolve()), 'bank_hash': bank.manifest['bank_hash'],
            'bank_type': bank.manifest['bank_type'],
            'checkpoint_selector': 'last' if args.checkpoint_step is None else int(args.checkpoint_step),
            'actor': {
                'name': actor_name, 'run_dir': str(actor.run_dir),
                'checkpoint': actor.checkpoint,
                'actor_fingerprint': actor.actor_fingerprint,
                'critic_fingerprint': actor.critic_fingerprint,
                'source_commit': actor.metadata.get('git_commit'),
                'git_dirty': actor.metadata.get('git_dirty'),
            },
        })
    write_csv(output / 'summary.csv', summary_rows)
    write_json(output / 'pairwise_contrasts.json', pairwise_contrasts(actor_results, protocol['primary_actor_names']))
    write_json(output / 'evaluation_manifest.json', {
        'study_id': protocol['study_id'], 'seed': int(args.seed),
        'bank_manifest': bank.manifest, 'bank_hash': bank.manifest['bank_hash'],
        'primary_actor_names': list(protocol['primary_actor_names']),
        'checkpoint_selector': 'last' if args.checkpoint_step is None else int(args.checkpoint_step),
        'no_training': True, 'no_bank_resampling': True,
    })
    print(f'Evaluation written: {output}')


if __name__ == '__main__':
    main()

