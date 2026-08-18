"""Print a compact M10A parameter and Dense-MAC budget table.

The network shapes are obtained from a real AntMaze dataset sample.  The
accounting itself is performed from the initialized Flax parameter/buffer
trees, so generic accounting does not contain AntMaze dimensions.
"""

import argparse

import jax.numpy as jnp

from impls.agents.hiql import HIQLAgent
from impls.computation.accounting import hiql_policy_accounting
from impls.experiment import prepare_run_design
from impls.main import _make_config, _parse_args
from impls.utils.env_utils import make_env_and_datasets


STUDY = 'experiments/M10A_fixed_budget_placement/study.yaml'


def _make_agent(config_id, observations, actions):
    _, configuration = prepare_run_design(STUDY, config_id)
    config = _make_config(_parse_args(['--agent', 'hiql']), configuration=configuration)
    return HIQLAgent.create(0, observations, actions, config), config


def build_rows(dataset_root=None, env_name='antmaze-large-navigate-v0'):
    env, raw_train, _ = make_env_and_datasets(env_name, dataset_dir=dataset_root)
    try:
        observations = jnp.asarray(raw_train['observations'][:1])
        actions = jnp.asarray(raw_train['actions'][:1])
        rows = []
        for index in range(1, 12):
            config_id = f'M10A-C{index:03d}'
            agent, config = _make_agent(config_id, observations, actions)
            audit = hiql_policy_accounting(
                agent.network.params,
                agent.network.model_state.get('buffers', {}),
                config.get('compute', {}),
            )
            high = audit['slots']['high_actor']
            low = audit['slots']['low_actor']
            factors = prepare_run_design(STUDY, config_id)[1].data['factors']
            rows.append({
                'config': config_id,
                'K_H': factors.get('high_iterations_K'),
                'K_L': factors.get('low_iterations_K'),
                'B_body': factors.get('body_compute_budget'),
                'C_update': high['update_module_dense_macs_per_execution'] or low['update_module_dense_macs_per_execution'],
                'input_M_high': high['input_mapping_dense_macs'],
                'input_M_low': low['input_mapping_dense_macs'],
                'update_M_high': high['total_update_module_dense_macs'],
                'update_M_low': low['total_update_module_dense_macs'],
                'core_M_high': high['computation_core_dense_macs'],
                'core_M_low': low['computation_core_dense_macs'],
                'core_M_combined': audit['combined_high_low_computation_core_dense_macs'],
                'full_actor_M_combined': audit['combined_high_low_full_actor_dense_macs'],
                'trainable_high_low': audit['combined_high_low_trainable_params'],
                'network_trainable': audit['network_total_trainable_params'],
                'buffers_high_low': audit['combined_high_low_buffer_elements'],
            })
        return rows
    finally:
        env.close()


def _fmt(value):
    return '-' if value is None else str(value)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', default=None)
    parser.add_argument('--env-name', default='antmaze-large-navigate-v0')
    args = parser.parse_args(argv)
    rows = build_rows(args.dataset_root, args.env_name)
    columns = (
        ('config', 'config'), ('K_H', 'K_H'), ('K_L', 'K_L'), ('B_body', 'B'),
        ('C_update', 'C_upd'), ('input_M_high', 'M_in,H'), ('input_M_low', 'M_in,L'),
        ('update_M_high', 'M_upd,H'), ('update_M_low', 'M_upd,L'),
        ('core_M_high', 'M_core,H'), ('core_M_low', 'M_core,L'),
        ('core_M_combined', 'M_core,total'), ('full_actor_M_combined', 'M_actor,total'),
        ('trainable_high_low', 'P_H+L'), ('network_trainable', 'P_net'),
        ('buffers_high_low', 'Bf_H+L'),
    )
    print('M10A Dense-MAC and parameter audit (MAC = Dense kernel multiply-accumulates)')
    print('dataset environment:', args.env_name)
    print(' | '.join(title for _, title in columns))
    print(' | '.join('---' for _ in columns))
    for row in rows:
        print(' | '.join(_fmt(row[key]) for key, _ in columns))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
