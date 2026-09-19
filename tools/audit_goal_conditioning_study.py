#!/usr/bin/env python3
"""CPU-only Study input/init audit and optional tiny real-data smoke; no rollout.

Production forward uses unchanged widths. The optional two-update CPU check
explicitly derives a tiny engineering model and is NOT a GPU workload gate.
"""

import argparse
import copy
import gc
import json
import os
from pathlib import Path
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def resolved_configurations(study_path, config_ids=None):
    from impls.experiment import load_study, prepare_run_design
    from impls.main import _make_config, _parse_args
    study = load_study(study_path)
    result = []
    for path in sorted((study.path.parent / 'configs').glob('*.yaml')):
        _, configuration = prepare_run_design(study.path, path)
        if config_ids is None or configuration.config_id in config_ids:
            config = _make_config(_parse_args(['--agent', configuration.data['algorithm']]), configuration)
            result.append((configuration, config))
    if config_ids is not None and set(config_ids) != {c.config_id for c, _ in result}:
        raise ValueError('Unknown or missing configuration IDs')
    if not result:
        raise ValueError('No configurations selected')
    return study, result


def dataset_metadata(path):
    import numpy as np
    from impls.utils.checkpointing import sha256_file
    path = Path(path)
    with zipfile.ZipFile(path) as archive:
        arrays = {}
        for item in archive.infolist():
            if not item.filename.endswith('.npy'):
                continue
            with archive.open(item) as file:
                version = np.lib.format.read_magic(file)
                reader = {(1, 0): np.lib.format.read_array_header_1_0,
                          (2, 0): np.lib.format.read_array_header_2_0}.get(version)
                if reader is None:
                    raise ValueError(f'Unsupported npy header: {version}')
                shape, fortran, dtype = reader(file)
            arrays[item.filename[:-4]] = {
                'shape': list(shape), 'dtype': str(dtype), 'fortran_order': fortran,
                'crc32': item.CRC, 'uncompressed_bytes': item.file_size,
            }
    return {'path': str(path.resolve()), 'bytes': path.stat().st_size,
            'mtime_ns': path.stat().st_mtime_ns, 'sha256': sha256_file(path), 'arrays': arrays}


def tree_equal(left, right):
    import jax
    import numpy as np
    if jax.tree_util.tree_structure(left) != jax.tree_util.tree_structure(right):
        raise AssertionError('Tree structures differ')
    for a, b in zip(jax.tree_util.tree_leaves(left), jax.tree_util.tree_leaves(right)):
        np.testing.assert_array_equal(a, b)


def state_fingerprint(value):
    from flax.serialization import to_state_dict
    from impls.utils.checkpointing import tree_fingerprint
    return tree_fingerprint(to_state_dict(value))


def production_audit(configurations):
    import jax
    import numpy as np
    from impls.agents import agents
    from impls.diagnostics.goal_conditioning import audit_goal_conditioning
    from impls.utils.checkpointing import tree_fingerprint
    result, reference = {}, None
    for configuration, config in configurations:
        n = config['goal_conditioning']['num_buttons']
        rng = np.random.default_rng(26019)
        def observation():
            bits = rng.integers(0, 2, (2, n))
            buttons = np.stack([1-bits, bits, rng.normal(size=(2, n)), rng.normal(size=(2, n))], -1)
            return np.concatenate([rng.normal(size=(2, 19)), buttons.reshape(2, -1)], -1).astype(np.float32)
        states, goals = observation(), observation()
        actions = rng.uniform(-1, 1, (2, 5)).astype(np.float32)
        agent = agents[configuration.data['algorithm']].create(0, states[:1], actions[:1], config)
        state = jax.device_get((agent.network.params, agent.network.opt_state, agent.rng))
        if reference is None:
            reference = state
        else:
            tree_equal(reference, state)
        audit = audit_goal_conditioning(agent, states, goals, actions)
        audit['initial_parameter_optimizer_rng_fingerprint'] = state_fingerprint(state)
        audit['parameter_elements_including_target'] = sum(x.size for x in jax.tree_util.tree_leaves(agent.network.params))
        audit['production_widths_unchanged'] = True
        result[configuration.config_id] = audit
        del agent, state
        gc.collect()
    return result


def real_data_smoke(configurations, dataset_root, representative_id):
    import jax
    import numpy as np
    from impls.agents import agents
    from impls.utils.checkpointing import tree_fingerprint
    from impls.utils.env_utils import make_env_and_datasets
    from impls.utils.flax_utils import save_agent, restore_agent_from_checkpoint
    from impls.utils.puzzle_datasets import PuzzleBoardGCDataset
    from impls.utils.reproducibility import derive_seed
    environments = {c.data['environment'] for c, _ in configurations}
    if len(environments) != 1:
        raise ValueError('Paired real-data audit requires exactly one environment')
    environment = next(iter(environments))
    for suffix in ('', '-val'):
        if not (Path(dataset_root) / f'{environment}{suffix}.npz').is_file():
            raise FileNotFoundError(f'Missing {environment}{suffix}.npz; no download allowed')
    env, train, val = make_env_and_datasets(environment, seed=derive_seed(0, 3),
                                         dataset_seed=derive_seed(0, 1), dataset_dir=str(dataset_root))
    try:
        if val is None:
            raise ValueError('Validation data are required')
        traces = {}
        for label, raw, stream in (('train', train, 11), ('validation', val, 12)):
            reference = None
            for configuration, config in configurations:
                dataset = PuzzleBoardGCDataset(raw, config, rng=derive_seed(0, stream))
                samples = [dataset.sample(4, evaluation=label == 'validation', return_sampling_trace=True)
                           for _ in range(2)]
                if reference is None:
                    reference = samples
                else:
                    tree_equal(reference, samples)
            traces[label] = {'paired_configurations': len(configurations), 'draws': 2,
                             'batch_size': 4, 'batch_and_trace_fingerprint': state_fingerprint(reference)}
        configuration, original = next((c, cfg) for c, cfg in configurations if c.config_id == representative_id)
        tiny = copy.deepcopy(original.to_dict())
        tiny.update(actor_hidden_dims=[8], value_hidden_dims=[8], batch_size=2)
        for slot in tiny['compute'].values():
            slot['structure_kwargs'].update(token_dim=8, robot_hidden_dim=8,
                                           token_mlp_hidden_dim=4, channel_mlp_hidden_dim=8,
                                           num_mixer_blocks=1)
        dataset = PuzzleBoardGCDataset(train, tiny, rng=derive_seed(0, 11))
        example = dataset.sample(1)
        agent = agents[configuration.data['algorithm']].create(0, example['observations'], example['actions'], tiny)
        losses = []
        for _ in range(2):
            batch = dataset.sample(2)
            agent, info = agent.update(batch)
            for value in jax.tree_util.tree_leaves(info):
                if not np.all(np.isfinite(value)):
                    raise AssertionError('Non-finite real-data update')
            losses.append({k: float(v) for k, v in info.items()})
        with tempfile.TemporaryDirectory(prefix='goal_input_cpu_smoke_') as temporary:
            path = save_agent(agent, temporary, 2)
            fresh = agents[configuration.data['algorithm']].create(991, example['observations'], example['actions'], json.loads(json.dumps(tiny)))
            restored = restore_agent_from_checkpoint(fresh, path)
            tree_equal((agent.network.params, agent.network.opt_state, agent.rng),
                       (restored.network.params, restored.network.opt_state, restored.rng))
            for name in ('value', 'critic', 'target_critic'):
                args = [batch['observations'], batch['value_goals']]
                if name != 'value':
                    args.append(batch['actions'])
                tree_equal(agent.network.select(name)(*args), restored.network.select(name)(*args))
            key = jax.random.PRNGKey(42)
            tree_equal(agent.sample_actions(batch['observations'], batch['actor_goals'], seed=key),
                       restored.sample_actions(batch['observations'], batch['actor_goals'], seed=key))
        return {'sampling_pairing': traces, 'tiny_update_config_id': representative_id,
                'updates': 2, 'batch_size': 2, 'losses': losses, 'fresh_restore': 'pass',
                'tiny_config': tiny, 'production_gpu_workload_test': False, 'rollouts': 0}
    finally:
        env.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--dataset-root', type=Path)
    parser.add_argument('--real-data-smoke-config', help='Opt in to two tiny CPU updates for this config')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        parser.error('Select a new output file')
    if args.real_data_smoke_config and not args.dataset_root:
        parser.error('--real-data-smoke-config requires --dataset-root')
    os.environ['JAX_PLATFORMS'] = 'cpu'
    import impls
    import jax
    import ogbench
    from impls.experiment.management import git_metadata, config_fingerprint
    if jax.default_backend() != 'cpu':
        raise RuntimeError('This audit must run on CPU')
    study, configs = resolved_configurations(args.study)
    report = {'engineering_only': True, 'backend': jax.default_backend(), 'source': git_metadata(ROOT),
              'study_id': study.study_id, 'resolved_config_hashes': {c.config_id: config_fingerprint(cfg) for c, cfg in configs},
              'production_forward': production_audit(configs), 'gpu_smoke_executed': False,
              'real_data_smoke': 'not_run'}
    if args.dataset_root:
        report['datasets'] = [dataset_metadata(args.dataset_root / f'{env}{suffix}.npz')
                              for env in study.data['environments'] for suffix in ('', '-val')]
    if args.real_data_smoke_config:
        report['real_data_smoke'] = real_data_smoke(configs, args.dataset_root, args.real_data_smoke_config)
    imports = {name: str(Path(module.__file__).resolve()) for name, module in sys.modules.items()
               if (name == 'impls' or name.startswith('impls.') or name == 'ogbench' or name.startswith('ogbench.'))
               and getattr(module, '__file__', None)}
    if any(not Path(path).is_relative_to(ROOT) for path in imports.values()):
        raise RuntimeError('Cross-worktree import detected')
    report['imports'] = imports
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as file:
        json.dump(report, file, indent=2, sort_keys=True, allow_nan=False)
        file.write('\n')
    print(json.dumps({'output': str(args.output), 'production_configurations': len(configs),
                      'real_data_smoke': bool(args.real_data_smoke_config), 'backend': 'cpu'}))


if __name__ == '__main__':
    main()
