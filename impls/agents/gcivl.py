"""Canonical OGBench goal-conditioned implicit V-learning."""

import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from ..computation.factory import resolve_slot_spec
from ..computation.slots import validate_compute_slots
from ..networks.common import GCActor, GCDiscreteActor, GCValue
from ..utils.encoders import GCEncoder, encoder_modules
from ..utils.flax_utils import (
    ModuleDict,
    TrainState,
    nonpytree_field,
    synchronize_target_module,
)


class GCIVLAgent(flax.struct.PyTreeNode):
    """Goal-conditioned implicit V-learning (GCIVL) agent."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    @staticmethod
    def expectile_loss(adv, diff, expectile):
        weight = jnp.where(adv >= 0, expectile, 1 - expectile)
        return weight * (diff ** 2)

    def value_loss(self, batch, grad_params):
        next_v1_t, next_v2_t = self.network.select('target_value')(
            batch['next_observations'], batch['value_goals']
        )
        next_v_t = jnp.minimum(next_v1_t, next_v2_t)
        q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v_t
        v1_t, v2_t = self.network.select('target_value')(
            batch['observations'], batch['value_goals']
        )
        v_t = (v1_t + v2_t) / 2
        adv = q - v_t

        q1 = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v1_t
        q2 = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v2_t
        v1, v2 = self.network.select('value')(
            batch['observations'], batch['value_goals'], params=grad_params
        )
        v = (v1 + v2) / 2
        value_loss1 = self.expectile_loss(
            adv, q1 - v1, self.config['expectile']
        ).mean()
        value_loss2 = self.expectile_loss(
            adv, q2 - v2, self.config['expectile']
        ).mean()
        value_loss = value_loss1 + value_loss2
        return value_loss, {
            'value_loss': value_loss,
            'v_mean': v.mean(),
            'v_max': v.max(),
            'v_min': v.min(),
        }

    def actor_loss(self, batch, grad_params, rng=None):
        del rng
        v1, v2 = self.network.select('value')(
            batch['observations'], batch['actor_goals']
        )
        nv1, nv2 = self.network.select('value')(
            batch['next_observations'], batch['actor_goals']
        )
        v = (v1 + v2) / 2
        nv = (nv1 + nv2) / 2
        adv = nv - v
        exp_a = jnp.minimum(jnp.exp(adv * self.config['alpha']), 100.0)
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

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        rng = self.rng if rng is None else rng
        value_loss, value_info = self.value_loss(batch, grad_params)
        rng, actor_rng = jax.random.split(rng)
        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        info = {}
        info.update({f'value/{key}': value for key, value in value_info.items()})
        info.update({f'actor/{key}': value for key, value in actor_info.items()})
        return value_loss + actor_loss, info

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
        self.target_update(new_network, 'value')
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
        validate_compute_slots('gcivl', config)
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
            encoders['actor'] = GCEncoder(concat_encoder=encoder_module())

        value_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            ensemble=True,
            gc_encoder=encoders.get('value'),
            computation_spec=resolve_slot_spec(config, 'value'),
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
            'target_value': (
                copy.deepcopy(value_def),
                (ex_observations, ex_goals),
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
        network = synchronize_target_module(network, 'value')
        return cls(
            rng,
            network=network,
            config=flax.core.FrozenDict(dict(config)),
        )


def get_config():
    return ml_collections.ConfigDict(
        dict(
            agent_name='gcivl',
            lr=3e-4,
            batch_size=1024,
            actor_hidden_dims=(512, 512, 512),
            value_hidden_dims=(512, 512, 512),
            layer_norm=True,
            discount=0.99,
            tau=0.005,
            expectile=0.9,
            alpha=10.0,
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
                )
            ),
        )
    )
