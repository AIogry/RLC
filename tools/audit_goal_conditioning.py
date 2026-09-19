#!/usr/bin/env python3
"""CPU synthetic input audit. No data loading, rollout, update, or Study."""

import argparse
import json
import os
from pathlib import Path
import sys

os.environ['JAX_PLATFORMS'] = 'cpu'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import impls
import jax
import numpy as np
import ogbench

from impls.agents import agent_configs, agents
from impls.diagnostics.goal_conditioning import audit_goal_conditioning
from impls.networks.goal_conditioning import goal_conditioning_layout


def synthetic_config():
    """Small interface fixture, deliberately not a formal hyperparameter set."""
    config = agent_configs['gciql']().to_dict()
    config.update(actor_hidden_dims=(8,), value_hidden_dims=(8,), batch_size=2,
                  dataset_class='PuzzleBoardGCDataset')
    config['goal_conditioning'] = {
        'schema_version': 2, 'domain': 'puzzle', 'rows': 3, 'cols': 3,
        'num_buttons': 9, 'robot_dim': 19, 'button_feature_dim': 4,
        'robot_goal_policy': 'zero', 'button_goal_transient_policy': 'zero',
        'input_schema': 'token_aux_v1', 'transforms': {},
        'roles': {
            'actor': {'coordinate': 'residual', 'transform_id': 'identity', 'distance_feature': 'zero'},
            'value_side': {'coordinate': 'operation', 'transform_id': 'identity', 'distance_feature': 'exact_press_fraction'},
        },
    }
    for name in ('actor', 'value', 'critic'):
        config['compute'][name] = {
            'enabled': True, 'structure': 'puzzle_tokens', 'primitive': 'mlp',
            'block': 'mlp_mixer', 'topology': 'feedforward', 'credit': 'direct',
            'structure_kwargs': {
                'num_buttons': 9, 'token_dim': 7, 'robot_hidden_dim': 8,
                'token_mlp_hidden_dim': 5, 'channel_mlp_hidden_dim': 11,
                'num_mixer_blocks': 1, 'index_embedding': True,
                'readout': 'mean', 'tm_mode': 'none',
            },
        }
    return config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--agent-config', type=Path, help='Optional JSON agent config; default is a tiny engineering fixture')
    parser.add_argument('--fixture-seed', type=int, default=0)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if args.output and (args.output.exists() or args.output.is_symlink()):
        parser.error('output already exists; select a new audit path')
    if jax.default_backend() != 'cpu':
        raise RuntimeError('Synthetic goal-input audit requires CPU')
    for module in (impls, ogbench):
        if not Path(module.__file__).resolve().is_relative_to(ROOT):
            raise RuntimeError(f'Wrong worktree import: {module.__file__}')
    config = json.loads(args.agent_config.read_text()) if args.agent_config else synthetic_config()
    layout = goal_conditioning_layout(config.get('goal_conditioning'))
    if layout is None:
        parser.error('audit requires an explicit Puzzle layout')
    n = layout['num_buttons']
    rng = np.random.default_rng(args.fixture_seed)
    def observation():
        bits = rng.integers(0, 2, (2, n))
        buttons = np.stack([1 - bits, bits, np.zeros_like(bits), np.zeros_like(bits)], axis=-1)
        return np.concatenate([rng.normal(size=(2, 19)), buttons.reshape(2, -1)], axis=-1).astype(np.float32)
    states, goals = observation(), observation()
    actions = rng.uniform(-1, 1, size=(2, 5)).astype(np.float32)
    agent = agents['gciql'].create(args.fixture_seed, states, actions, config)
    report = audit_goal_conditioning(agent, states, goals, actions)
    report.update(engineering_only=True, backend=jax.default_backend(),
                  imports={'impls': impls.__file__, 'ogbench': ogbench.__file__},
                  fixture_seed=args.fixture_seed, real_data_used=False, optimizer_updates=0)
    encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as file:
            file.write(encoded + '\n')
    print(encoded)


if __name__ == '__main__':
    main()
