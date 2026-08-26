"""Canonical OGBench goal-conditioned behavioral cloning."""

from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from ..networks.common import GCActor, GCDiscreteActor
from ..utils.encoders import GCEncoder, encoder_modules
from ..utils.flax_utils import ModuleDict, TrainState, nonpytree_field


class GCBCAgent(flax.struct.PyTreeNode):
    """Goal-conditioned behavioral cloning (GCBC) agent."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def actor_loss(self, batch, grad_params, rng=None):
        del rng
        dist = self.network.select('actor')(
            batch['observations'], batch['actor_goals'], params=grad_params
        )
        log_prob = dist.log_prob(batch['actions'])
        actor_loss = -log_prob.mean()
        actor_info = {
            'actor_loss': actor_loss,
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
        rng, actor_rng = jax.random.split(rng)
        del rng
        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        info = {f'actor/{key}': value for key, value in actor_info.items()}
        return actor_loss, info

    @jax.jit
    def update(self, batch):
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
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
            encoders['actor'] = GCEncoder(concat_encoder=encoder_module())

        if config['discrete']:
            actor_def = GCDiscreteActor(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=action_dim,
                gc_encoder=encoders.get('actor'),
            )
        else:
            actor_def = GCActor(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=action_dim,
                state_dependent_std=False,
                const_std=config['const_std'],
                gc_encoder=encoders.get('actor'),
            )

        network_info = {'actor': (actor_def, (ex_observations, ex_goals))}
        network_def = ModuleDict({key: value[0] for key, value in network_info.items()})
        network_params = network_def.init(
            init_rng,
            **{key: value[1] for key, value in network_info.items()},
        )['params']
        network = TrainState.create(
            network_def,
            network_params,
            tx=optax.adam(learning_rate=config['lr']),
        )
        return cls(
            rng,
            network=network,
            config=flax.core.FrozenDict(dict(config)),
        )


def get_config():
    return ml_collections.ConfigDict(
        dict(
            agent_name='gcbc',
            lr=3e-4,
            batch_size=1024,
            actor_hidden_dims=(512, 512, 512),
            discount=0.99,
            const_std=True,
            discrete=False,
            encoder=None,
            dataset_class='GCDataset',
            value_p_curgoal=0.0,
            value_p_trajgoal=1.0,
            value_p_randomgoal=0.0,
            value_geom_sample=False,
            actor_p_curgoal=0.0,
            actor_p_trajgoal=1.0,
            actor_p_randomgoal=0.0,
            actor_geom_sample=False,
            gc_negative=True,
            p_aug=0.0,
            frame_stack=None,
        )
    )
