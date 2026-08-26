"""Canonical OGBench goal-conditioned implicit Q-learning."""

import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from ..computation.factory import resolve_slot_spec
from ..computation.slots import validate_compute_slots
from ..networks.common import GCActor, GCDiscreteActor, GCDiscreteCritic, GCValue
from ..utils.encoders import GCEncoder, encoder_modules
from ..utils.flax_utils import (
    ModuleDict,
    TrainState,
    nonpytree_field,
    synchronize_target_module,
)


class GCIQLAgent(flax.struct.PyTreeNode):
    """Goal-conditioned implicit Q-learning (GCIQL) agent."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    @staticmethod
    def expectile_loss(adv, diff, expectile):
        weight = jnp.where(adv >= 0, expectile, 1 - expectile)
        return weight * (diff ** 2)

    def value_loss(self, batch, grad_params):
        q1, q2 = self.network.select('target_critic')(
            batch['observations'], batch['value_goals'], batch['actions']
        )
        q = jnp.minimum(q1, q2)
        v = self.network.select('value')(
            batch['observations'], batch['value_goals'], params=grad_params
        )
        value_loss = self.expectile_loss(
            q - v, q - v, self.config['expectile']
        ).mean()
        return value_loss, {
            'value_loss': value_loss,
            'v_mean': v.mean(),
            'v_max': v.max(),
            'v_min': v.min(),
        }

    def critic_loss(self, batch, grad_params):
        next_v = self.network.select('value')(
            batch['next_observations'], batch['value_goals']
        )
        target = (
            batch['rewards']
            + self.config['discount'] * batch['masks'] * next_v
        )
        q1, q2 = self.network.select('critic')(
            batch['observations'],
            batch['value_goals'],
            batch['actions'],
            params=grad_params,
        )
        critic_loss = ((q1 - target) ** 2 + (q2 - target) ** 2).mean()
        return critic_loss, {
            'critic_loss': critic_loss,
            'q_mean': target.mean(),
            'q_max': target.max(),
            'q_min': target.min(),
        }

    def actor_loss(self, batch, grad_params, rng=None):
        if self.config['actor_loss'] == 'awr':
            v = self.network.select('value')(
                batch['observations'], batch['actor_goals']
            )
            q1, q2 = self.network.select('critic')(
                batch['observations'], batch['actor_goals'], batch['actions']
            )
            q = jnp.minimum(q1, q2)
            adv = q - v
            exp_a = jnp.minimum(
                jnp.exp(adv * self.config['alpha']), 100.0
            )
            dist = self.network.select('actor')(
                batch['observations'], batch['actor_goals'], params=grad_params
            )
            log_prob = dist.log_prob(batch['actions'])
            actor_loss = -(exp_a * log_prob).mean()
            actor_info = {
                'actor_loss': actor_loss,
                'adv': adv.mean(),
                'bc_log_prob': log_prob.mean(),
            }
            if not self.config['discrete']:
                actor_info.update({
                    'mse': jnp.mean((dist.mode() - batch['actions']) ** 2),
                    'std': jnp.mean(dist.scale_diag),
                })
            return actor_loss, actor_info

        if self.config['actor_loss'] == 'ddpgbc':
            assert not self.config['discrete']
            dist = self.network.select('actor')(
                batch['observations'], batch['actor_goals'], params=grad_params
            )
            if self.config['const_std']:
                q_actions = jnp.clip(dist.mode(), -1, 1)
            else:
                q_actions = jnp.clip(dist.sample(seed=rng), -1, 1)
            q1, q2 = self.network.select('critic')(
                batch['observations'], batch['actor_goals'], q_actions
            )
            q = jnp.minimum(q1, q2)
            q_loss = -q.mean() / jax.lax.stop_gradient(
                jnp.abs(q).mean() + 1e-6
            )
            log_prob = dist.log_prob(batch['actions'])
            bc_loss = -(self.config['alpha'] * log_prob).mean()
            actor_loss = q_loss + bc_loss
            return actor_loss, {
                'actor_loss': actor_loss,
                'q_loss': q_loss,
                'bc_loss': bc_loss,
                'q_mean': q.mean(),
                'q_abs_mean': jnp.abs(q).mean(),
                'bc_log_prob': log_prob.mean(),
                'mse': jnp.mean((dist.mode() - batch['actions']) ** 2),
                'std': jnp.mean(dist.scale_diag),
            }

        raise ValueError(f'Unsupported actor loss: {self.config["actor_loss"]}')

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        rng = self.rng if rng is None else rng
        value_loss, value_info = self.value_loss(batch, grad_params)
        critic_loss, critic_info = self.critic_loss(batch, grad_params)
        rng, actor_rng = jax.random.split(rng)
        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        info = {}
        info.update({f'value/{key}': value for key, value in value_info.items()})
        info.update({f'critic/{key}': value for key, value in critic_info.items()})
        info.update({f'actor/{key}': value for key, value in actor_info.items()})
        return value_loss + critic_loss + actor_loss, info

    def target_update(self, network, module_name):
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @jax.jit
    def update(self, batch):
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, 'critic')
        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(self, observations, goals=None, seed=None, temperature=1.0):
        dist = self.network.select('actor')(observations, goals, temperature=temperature)
        actions = dist.sample(seed=seed)
        if not self.config['discrete']:
            actions = jnp.clip(actions, -1, 1)
        return actions

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        validate_compute_slots('gciql', config)
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)
        ex_goals = ex_observations
        if config['discrete']:
            action_dim = ex_actions.max() + 1
        else:
            action_dim = ex_actions.shape[-1]

        encoders = {}
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['value'] = GCEncoder(concat_encoder=encoder_module())
            encoders['critic'] = GCEncoder(concat_encoder=encoder_module())
            encoders['actor'] = GCEncoder(concat_encoder=encoder_module())

        value_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            ensemble=False,
            gc_encoder=encoders.get('value'),
            computation_spec=resolve_slot_spec(config, 'value'),
        )
        if config['discrete']:
            critic_def = GCDiscreteCritic(
                hidden_dims=config['value_hidden_dims'],
                layer_norm=config['layer_norm'],
                ensemble=True,
                gc_encoder=encoders.get('critic'),
                action_dim=action_dim,
                computation_spec=resolve_slot_spec(config, 'critic'),
            )
        else:
            critic_def = GCValue(
                hidden_dims=config['value_hidden_dims'],
                layer_norm=config['layer_norm'],
                ensemble=True,
                gc_encoder=encoders.get('critic'),
                computation_spec=resolve_slot_spec(config, 'critic'),
            )
        if config['discrete']:
            actor_def = GCDiscreteActor(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=action_dim,
                gc_encoder=encoders.get('actor'),
                computation_spec=resolve_slot_spec(config, 'actor'),
            )
        else:
            actor_def = GCActor(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=action_dim,
                state_dependent_std=False,
                const_std=config['const_std'],
                gc_encoder=encoders.get('actor'),
                computation_spec=resolve_slot_spec(config, 'actor'),
            )

        network_info = {
            'value': (value_def, (ex_observations, ex_goals)),
            'critic': (critic_def, (ex_observations, ex_goals, ex_actions)),
            'target_critic': (
                copy.deepcopy(critic_def),
                (ex_observations, ex_goals, ex_actions),
            ),
            'actor': (actor_def, (ex_observations, ex_goals)),
        }
        network_def = ModuleDict({key: value[0] for key, value in network_info.items()})
        variables = network_def.init(
            {'params': init_rng, 'buffers': jax.random.fold_in(rng, 0x4D39)},
            **{key: value[1] for key, value in network_info.items()},
        )
        network_params = variables['params']
        model_state = {key: value for key, value in variables.items() if key != 'params'}
        network = TrainState.create(
            network_def,
            network_params,
            model_state=model_state,
            tx=optax.adam(learning_rate=config['lr']),
        )
        network = synchronize_target_module(network, 'critic')
        return cls(
            rng,
            network=network,
            config=flax.core.FrozenDict(dict(config)),
        )


def get_config():
    return ml_collections.ConfigDict(
        dict(
            agent_name='gciql',
            lr=3e-4,
            batch_size=1024,
            actor_hidden_dims=(512, 512, 512),
            value_hidden_dims=(512, 512, 512),
            layer_norm=True,
            discount=0.99,
            tau=0.005,
            expectile=0.9,
            actor_loss='ddpgbc',
            alpha=0.3,
            const_std=True,
            discrete=False,
            encoder=None,
            dataset_class='GCDataset',
            value_p_curgoal=0.2,
            value_p_trajgoal=0.5,
            value_p_randomgoal=0.3,
            value_geom_sample=True,
            actor_p_curgoal=0.0,
            actor_p_trajgoal=1.0,
            actor_p_randomgoal=0.0,
            actor_geom_sample=False,
            gc_negative=True,
            p_aug=0.0,
            frame_stack=None,
            compute=ml_collections.ConfigDict(
                dict(
                    actor=ml_collections.ConfigDict(
                        dict(enabled=False, primitive='mlp', topology='feedforward', credit='direct')
                    ),
                    value=ml_collections.ConfigDict(
                        dict(enabled=False, primitive='mlp', topology='feedforward', credit='direct')
                    ),
                    critic=ml_collections.ConfigDict(
                        dict(enabled=False, primitive='mlp', topology='feedforward', credit='direct')
                    ),
                )
            ),
        )
    )
