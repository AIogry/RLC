from __future__ import annotations

import argparse
from pathlib import Path

from impls.diagnostics.banks import save_bank
from impls.diagnostics.checkpoints import load_primary_actors
from impls.diagnostics.rollout import (
    build_rollout_bank,
    collect_rollout_records,
    environment_task_ids,
    eval_goals_from_resets,
)

from common import load_env_dataset, protocol_from_arg, provenance_for_seed


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', default=str(Path(__file__).with_name('protocol.yaml')))
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--episodes', type=int, default=None)
    parser.add_argument('--evaluation-seed', type=int, default=None)
    args = parser.parse_args(argv)
    protocol = protocol_from_arg(args.protocol)
    env, dataset, _ = load_env_dataset(protocol, args.seed)
    actors = load_primary_actors(
        protocol, seed=args.seed, run_root=protocol['run_root'], dataset=dataset
    )
    actor_names = list(protocol['primary_actor_names'])
    task_ids = environment_task_ids(env)
    evaluation_seed = protocol['evaluation_seed'] if args.evaluation_seed is None else args.evaluation_seed
    _, task_names = eval_goals_from_resets(
        env, task_ids=task_ids, evaluation_seed=evaluation_seed
    )
    episodes = protocol['banks']['B_R']['episodes_per_actor_task'] if args.episodes is None else args.episodes
    records = []
    for actor_name in actor_names:
        records.extend(collect_rollout_records(
            actors[actor_name], env, actor_name=actor_name,
            task_ids=task_ids, episodes=episodes, evaluation_seed=evaluation_seed,
        ))
    arrays, manifest, rows = build_rollout_bank(
        records, actor_names=actor_names, task_names=task_names,
        bins=protocol['banks']['B_R']['progress_bins'], environment=protocol['environment'],
        source_commit=protocol['source_commit'], evaluation_seed=evaluation_seed,
        episodes_per_task=episodes, provenance=provenance_for_seed(protocol, args.seed),
    )
    artifact = save_bank(args.output, arrays, manifest, rows)
    print(f'B_R written: {artifact.root} hash={artifact.manifest["bank_hash"]}')


if __name__ == '__main__':
    main()

