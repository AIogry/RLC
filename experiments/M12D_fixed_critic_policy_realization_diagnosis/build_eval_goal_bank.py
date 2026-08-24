from __future__ import annotations

import argparse
from pathlib import Path

from impls.diagnostics.banks import build_eval_goal_bank, load_bank, save_bank
from impls.diagnostics.rollout import environment_task_ids, eval_goals_from_resets

from common import load_env_dataset, protocol_from_arg


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', default=str(Path(__file__).with_name('protocol.yaml')))
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--training-bank', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    protocol = protocol_from_arg(args.protocol)
    env, _, _ = load_env_dataset(protocol, args.seed)
    training_bank = load_bank(args.training_bank)
    task_ids = environment_task_ids(env)
    goals, names = eval_goals_from_resets(
        env, task_ids=task_ids, evaluation_seed=protocol['evaluation_seed']
    )
    arrays, manifest, rows = build_eval_goal_bank(
        training_bank, eval_goals=goals, task_names=names,
        environment=protocol['environment'], source_commit=protocol['source_commit'],
        dataset_root=protocol['dataset_root'], evaluation_seed=protocol['evaluation_seed'],
        provenance=training_bank.manifest.get('provenance', {}),
    )
    artifact = save_bank(args.output, arrays, manifest, rows)
    print(f'B_DE written: {artifact.root} hash={artifact.manifest["bank_hash"]}')


if __name__ == '__main__':
    main()

