"""Contrastive reinforcement learning with independent computation slots."""

from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from ..computation.factory import resolve_slot_spec
from ..networks.common import GCActor, GCDiscreteActor, GCBilinearValue, GCDiscreteBilinearCritic
from ..utils.encoders import GCEncoder, encoder_modules
from ..utils.flax_utils import ModuleDict, TrainState, nonpytree_field


class CRLAgent(flax.struct.PyTreeNode):
    """OGBench CRL with configurable actor, critic, and AWR value bodies."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def contrastive_loss(self, batch, grad_params, module_name='critic'):
        batch_size = batch['observations'].shape[0]
        actions = batch['actions'] if module_name == 'critic' else None
        value, phi, psi = self.network.select(module_name)(
            batch['observations'],
            batch['value_goals'],
            actions=actions,
            info=True,
            params=grad_params,
        )
        if len(phi.shape) == 2:
            phi = phi[None, ...]
            psi = psi[None, ...]
        logits = jnp.einsum('eik,ejk->ije', phi, psi) / jnp.sqrt(phi.shape[-1])
        identity = jnp.eye(batch_size)
        contrastive_loss = jax.vmap(
            lambda ensemble_logits: optax.sigmoid_binary_cross_entropy(
                logits=ensemble_logits,
                labels=identity,
            ),
            in_axes=-1,
            out_axes=-1,
        )(logits)
        contrastive_loss = jnp.mean(contrastive_loss)

        value = jnp.exp(value)
        mean_logits = jnp.mean(logits, axis=-1)
        correct = jnp.argmax(mean_logits, axis=1) == jnp.argmax(identity, axis=1)
        logits_pos = jnp.sum(mean_logits * identity) / jnp.sum(identity)
        logits_neg = jnp.sum(mean_logits * (1 - identity)) / jnp.sum(1 - identity)
        return contrastive_loss, {
            'contrastive_loss': contrastive_loss,
            'v_mean': value.mean(),
            'v_max': value.max(),
            'v_min': value.min(),
            'binary_accuracy': jnp.mean((mean_logits > 0) == identity),
            'categorical_accuracy': jnp.mean(correct),
            'logits_pos': logits_pos,
            'logits_neg': logits_neg,
            'logits': mean_logits.mean(),
        }

    def actor_loss(self, batch, grad_params, rng=None):
        if self.config['actor_loss'] == 'awr':
            value = self.network.select('value')(batch['observations'], batch['actor_goals'])
            q1, q2 = self.network.select('critic')(
                batch['observations'], batch['actor_goals'], batch['actions']
            )
            advantage = jnp.minimum(q1, q2) - value
            exp_a = jnp.minimum(jnp.exp(advantage * self.config['alpha']), 100.0)
            dist = self.network.select('actor')(
                batch['observations'], batch['actor_goals'], params=grad_params
            )
            log_prob = dist.log_prob(batch['actions'])
            actor_loss = -(exp_a * log_prob).mean()
            info = {
                'actor_loss': actor_loss,
                'adv': advantage.mean(),
                'bc_log_prob': log_prob.mean(),
            }
            if not self.config['discrete']:
                info.update({
                    'mse': jnp.mean((dist.mode() - batch['actions']) ** 2),
                    'std': jnp.mean(dist.scale_diag),
                })
            return actor_loss, info

        if self.config['actor_loss'] == 'ddpgbc':
            assert not self.config['discrete']
            dist = self.network.select('actor')(
                batch['observations'], batch['actor_goals'], params=grad_params
            )
            if self.config['const_std']:
                q_actions = jnp.clip(dist.mode(), -1, 1)
            else:
                q_actions = jnp.clip(dist.sample(seed=rng), -1, 1)
            # DDPG+BC uses the critic as a fixed differentiable evaluator for
            # the actor update.  Preserve gradients through q_actions while
            # preventing the actor-loss branch from updating critic params;
            # the contrastive critic loss remains the sole critic update path.
            frozen_params = jax.tree_util.tree_map(jax.lax.stop_gradient, grad_params)
            q1, q2 = self.network.select('critic')(
                batch['observations'], batch['actor_goals'], q_actions,
                params=frozen_params,
            )
            q = jnp.minimum(q1, q2)
            q_loss = -q.mean() / jax.lax.stop_gradient(jnp.abs(q).mean() + 1e-6)
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
    def critic_only_loss(self, batch, grad_params):
        """Return the canonical critic component without an actor objective."""

        return self.contrastive_loss(batch, grad_params, module_name='critic')

    @jax.jit
    def critic_only_update(self, batch):
        """Update only the canonical critic objective.

        The CRL actor parameters remain part of the ordinary TrainState for
        checkpoint compatibility, but the loss has no actor dependency.  With
        Adam's default no-weight-decay transform, their zero gradients produce
        an exactly unchanged actor parameter tree.
        """

        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.critic_only_loss(batch, grad_params)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        info = {}
        rng = self.rng if rng is None else rng
        critic_loss, critic_info = self.contrastive_loss(batch, grad_params, 'critic')
        for key, value in critic_info.items():
            info[f'critic/{key}'] = value

        if self.config['actor_loss'] == 'awr':
            value_loss, value_info = self.contrastive_loss(batch, grad_params, 'value')
            for key, value in value_info.items():
                info[f'value/{key}'] = value
        else:
            value_loss = 0.0

        rng, actor_rng = jax.random.split(rng)
        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for key, value in actor_info.items():
            info[f'actor/{key}'] = value
        return critic_loss + value_loss + actor_loss, info

    @jax.jit
    def update(self, batch):
        new_rng, rng = jax.random.split(self.rng)
        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        # Canonical CRL update: differentiate the joint total_loss once and
        # apply the configured optimizer once.  The DDPG+BC Q branch freezes
        # critic parameters inside actor_loss while retaining dQ/da.
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
        action_dim = ex_actions.max() + 1 if config['discrete'] else ex_actions.shape[-1]

        encoders = {}
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['critic_state'] = encoder_module()
            encoders['critic_goal'] = encoder_module()
            encoders['actor'] = GCEncoder(concat_encoder=encoder_module())
            if config['actor_loss'] == 'awr':
                encoders['value_state'] = encoder_module()
                encoders['value_goal'] = encoder_module()

        critic_state_spec = resolve_slot_spec(config, 'critic_state')
        critic_goal_spec = resolve_slot_spec(config, 'critic_goal')

        if config['discrete']:
            critic_def = GCDiscreteBilinearCritic(
                hidden_dims=config['value_hidden_dims'],
                latent_dim=config['latent_dim'],
                layer_norm=config['layer_norm'],
                ensemble=True,
                value_exp=False,
                state_encoder=encoders.get('critic_state'),
                goal_encoder=encoders.get('critic_goal'),
                state_computation_spec=critic_state_spec,
                goal_computation_spec=critic_goal_spec,
                action_dim=action_dim,
            )
        else:
            critic_def = GCBilinearValue(
                hidden_dims=config['value_hidden_dims'],
                latent_dim=config['latent_dim'],
                layer_norm=config['layer_norm'],
                ensemble=True,
                value_exp=False,
                state_encoder=encoders.get('critic_state'),
                goal_encoder=encoders.get('critic_goal'),
                state_computation_spec=critic_state_spec,
                goal_computation_spec=critic_goal_spec,
            )

        network_info = {
            'critic': (critic_def, (ex_observations, ex_goals, ex_actions)),
        }
        if config['actor_loss'] == 'awr':
            value_state_spec = resolve_slot_spec(config, 'value_state')
            value_goal_spec = resolve_slot_spec(config, 'value_goal')
            network_info['value'] = (
                GCBilinearValue(
                    hidden_dims=config['value_hidden_dims'],
                    latent_dim=config['latent_dim'],
                    layer_norm=config['layer_norm'],
                    ensemble=False,
                    value_exp=False,
                    state_encoder=encoders.get('value_state'),
                    goal_encoder=encoders.get('value_goal'),
                    state_computation_spec=value_state_spec,
                    goal_computation_spec=value_goal_spec,
                ),
                (ex_observations, ex_goals),
            )

        actor_spec = resolve_slot_spec(config, 'actor')
        if config['discrete']:
            actor_def = GCDiscreteActor(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=action_dim,
                gc_encoder=encoders.get('actor'),
                computation_spec=actor_spec,
            )
        else:
            actor_def = GCActor(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=action_dim,
                state_dependent_std=False,
                const_std=config['const_std'],
                gc_encoder=encoders.get('actor'),
                computation_spec=actor_spec,
            )
        network_info['actor'] = (actor_def, (ex_observations, ex_goals))

        network_def = ModuleDict({key: value[0] for key, value in network_info.items()})
        # Preserve the baseline params RNG while giving non-param buffers an
        # independent deterministic stream.
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
        return cls(rng, network=network, config=flax.core.FrozenDict(dict(config)))


def get_config():
    return ml_collections.ConfigDict(
        dict(
            agent_name='crl', lr=3e-4, batch_size=1024,
            actor_hidden_dims=(512, 512, 512), value_hidden_dims=(512, 512, 512),
            latent_dim=512, layer_norm=True, discount=0.99,
            actor_loss='ddpgbc', alpha=0.1, const_std=True, discrete=False,
            encoder=None, compute=ml_collections.ConfigDict(
                dict(
                    actor=ml_collections.ConfigDict(
                        dict(enabled=False, primitive='mlp', topology='feedforward', credit='direct')
                    ),
                    critic_state=ml_collections.ConfigDict(
                        dict(enabled=False, primitive='mlp', topology='feedforward', credit='direct')
                    ),
                    critic_goal=ml_collections.ConfigDict(
                        dict(enabled=False, primitive='mlp', topology='feedforward', credit='direct')
                    ),
                    value_state=ml_collections.ConfigDict(
                        dict(enabled=False, primitive='mlp', topology='feedforward', credit='direct')
                    ),
                    value_goal=ml_collections.ConfigDict(
                        dict(enabled=False, primitive='mlp', topology='feedforward', credit='direct')
                    ),
                )
            ),
            dataset_class='GCDataset',
            value_p_curgoal=0.0, value_p_trajgoal=1.0, value_p_randomgoal=0.0,
            value_geom_sample=True, actor_p_curgoal=0.0, actor_p_trajgoal=1.0,
            actor_p_randomgoal=0.0, actor_geom_sample=False, gc_negative=False,
            p_aug=0.0, frame_stack=None,
        )
    )
