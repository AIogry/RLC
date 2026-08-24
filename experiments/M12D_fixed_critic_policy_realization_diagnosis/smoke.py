from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from impls.diagnostics.banks import (
    arrays_hash,
    build_eval_goal_bank,
    build_training_support_bank,
    save_bank,
)
from impls.diagnostics.checkpoints import load_primary_actors
from impls.diagnostics.metrics import evaluate_bank, evaluate_bank_samples, pairwise_contrasts
from impls.diagnostics.rollout import (
    build_rollout_bank,
    collect_rollout_records,
    environment_task_ids,
    eval_goals_from_resets,
)
from impls.diagnostics.support import build_support_reference, support_distance
from impls.utils.checkpointing import tree_fingerprint

from common import load_env_dataset, protocol_from_arg, provenance_for_seed, source_config, write_json


def assert_finite(samples, label):
    for key, value in samples.items():
        if np.issubdtype(np.asarray(value).dtype, np.number) and not np.all(np.isfinite(value)):
            raise AssertionError(f'{label}: non-finite {key}')


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', default=str(Path(__file__).with_name('protocol.yaml')))
    parser.add_argument('--output', default='/tmp/m12d_diagnostic_smoke')
    args = parser.parse_args(argv)
    protocol = protocol_from_arg(args.protocol)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f'Smoke path exists; refusing overwrite: {output}')
    output.mkdir(parents=True)
    seed = 0
    env, dataset, _ = load_env_dataset(protocol, seed)
    config, source, _ = source_config(protocol, 'K1SN')
    provenance = provenance_for_seed(protocol, seed)
    arrays_a, manifest_a, rows_a = build_training_support_bank(
        dataset, config, seed=seed, batches=2, batch_size=32,
        environment=protocol['environment'], dataset_root=protocol['dataset_root'],
        source_commit=protocol['source_commit'], provenance=provenance,
    )
    arrays_b, _, _ = build_training_support_bank(
        dataset, config, seed=seed, batches=2, batch_size=32,
        environment=protocol['environment'], dataset_root=protocol['dataset_root'],
        source_commit=protocol['source_commit'], provenance=provenance,
    )
    if arrays_hash(arrays_a) != arrays_hash(arrays_b):
        raise AssertionError('B_T deterministic hash failure')
    bt = save_bank(output / 'B_T', arrays_a, manifest_a, rows_a)
    task_ids = environment_task_ids(env)
    goals, task_names = eval_goals_from_resets(
        env, task_ids=task_ids, evaluation_seed=protocol['evaluation_seed']
    )
    de_arrays, de_manifest, de_rows = build_eval_goal_bank(
        bt, eval_goals=goals, task_names=task_names, environment=protocol['environment'],
        source_commit=protocol['source_commit'], dataset_root=protocol['dataset_root'],
        evaluation_seed=protocol['evaluation_seed'], provenance=provenance,
    )
    de = save_bank(output / 'B_DE', de_arrays, de_manifest, de_rows)
    if not np.array_equal(
        de.arrays['dataset_indices'][:len(bt.arrays['dataset_indices'])],
        bt.arrays['dataset_indices'],
    ):
        raise AssertionError('B_DE state identity failure')
    actors = load_primary_actors(
        protocol, seed=seed, run_root=protocol['run_root'], dataset=dataset
    )
    names = list(protocol['primary_actor_names'])
    before = {name: tree_fingerprint(actors[name].agent.network.params) for name in names}
    records = []
    for name in names:
        records.extend(collect_rollout_records(
            actors[name], env, actor_name=name, task_ids=task_ids, episodes=1,
            evaluation_seed=protocol['evaluation_seed'],
        ))
    r_arrays, r_manifest, r_rows = build_rollout_bank(
        records, actor_names=names, task_names=task_names,
        bins=protocol['banks']['B_R']['progress_bins'], environment=protocol['environment'],
        source_commit=protocol['source_commit'], evaluation_seed=protocol['evaluation_seed'],
        episodes_per_task=1, provenance=provenance,
    )
    br = save_bank(output / 'B_R', r_arrays, r_manifest, r_rows)
    support_arrays, support_manifest = build_support_reference(
        dataset['observations'], max_states=256, environment=protocol['environment'],
        dataset_root=protocol['dataset_root'], source_commit=protocol['source_commit'],
    )
    support = save_bank(output / 'support_reference', support_arrays, support_manifest)
    support_a = support_distance(br.arrays['observations'], support)
    support_b = support_distance(br.arrays['observations'], support)
    if not np.array_equal(support_a, support_b):
        raise AssertionError('Support proxy is not deterministic')
    actor_results = {}
    summaries = {}
    for name in names:
        bt_rows = evaluate_bank(actors[name], bt)
        direct_loss, _ = actors[name].agent.policy_extraction_loss(
            {key: bt.arrays[key][:32] for key in ('observations', 'actions', 'actor_goals', 'value_goals')},
            actors[name].agent.network.params, rng=actors[name].agent.rng,
        )
        if not np.allclose(bt_rows[0]['actor_loss_return'], np.asarray(direct_loss), rtol=1e-5, atol=1e-6):
            raise AssertionError(f'{name}: exact objective parity failure')
        samples = evaluate_bank_samples(actors[name], br)
        assert_finite(samples, name)
        if not np.array_equal(samples['q_min'], np.minimum(samples['q1'], samples['q2'])):
            raise AssertionError(f'{name}: Qmin semantics failure')
        actor_results[name] = samples
        summaries[name] = {
            'B_T_rows': len(bt_rows), 'B_DE_rows': len(evaluate_bank(actors[name], de)),
            'B_R_rows': len(evaluate_bank(actors[name], br)),
        }
    after = {name: tree_fingerprint(actors[name].agent.network.params) for name in names}
    if before != after:
        raise AssertionError('Diagnostic mutated actor or critic parameters')
    result = {
        'status': 'PASS', 'seed': seed, 'formal_training_started': False,
        'formal_diagnostic_started': False, 'primary_actor_names': names,
        'banks': {'B_T': bt.manifest, 'B_DE': de.manifest, 'B_R': br.manifest, 'support': support.manifest},
        'summaries': summaries, 'pairwise_contrast_count': len(pairwise_contrasts(actor_results, names)),
        'checks': ['checkpoint pairing', 'K4SN attempt2', 'D9 loading', 'Residual loading',
                   'B_T deterministic exact sampler', 'B_DE state identity', 'B_R balancing',
                   'cross-evaluation', 'Qmin semantics', 'exact objective parity',
                   'finite metrics', 'support determinism', 'no parameter mutation'],
    }
    write_json(output / 'smoke_summary.json', result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()

