"""Official vanilla CoGHP agent migrated into the RLC runtime."""

from typing import Any

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import optax

from ..networks.coghp import HierarchicalPolicyNetwork
from ..networks.common import GCActor, GCDiscreteActor, GCValue, Identity, LengthNormalize, MLP
from ..utils.encoders import GCEncoder, encoder_modules
from ..utils.flax_utils import ModuleDict, TrainState, nonpytree_field


class CoGHPAgent(flax.struct.PyTreeNode):
    """Chain of goals with one shared autoregressive Mixer core."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()
    all_time_actions: Any = None
    goal_temp: Any = None

    @staticmethod
    def expectile_loss(adv, diff, expectile):
        weight = jnp.where(adv >= 0, expectile, 1 - expectile)
        return weight * (diff**2)

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
        value_loss = (
            self.expectile_loss(adv, q1 - v1, self.config['expectile']).mean()
            + self.expectile_loss(adv, q2 - v2, self.config['expectile']).mean()
        )
        return value_loss, {
            'value_loss': value_loss,
            'v_mean': v.mean(),
            'v_max': v.max(),
            'v_min': v.min(),
        }

    def actor_loss(self, batch, grad_params, rng):
        observations = batch['observations']
        actions = batch['actions']
        goals = batch['high_actor_goals']
        obs_expand = jnp.repeat(
            jnp.expand_dims(observations, axis=1),
            self.config['num_subgoals'],
            axis=1,
        )
        subgoals_reps = self.network.select('goal_rep')(
            jnp.concatenate([obs_expand, batch['high_actor_targets']], axis=-1),
            params=grad_params,
        )
        high_dist, low_dist, _ = self.network.select('actor_mixer')(
            observations,
            goals,
            rng,
            subgoal_reps=subgoals_reps,
            action_seq=actions,
            params=grad_params,
        )

        if self.config['num_subgoals'] > 0:
            high_loss, high_info = self.multi_high_actor_loss(
                batch, high_dist, obs_expand, grad_params
            )
        else:
            high_loss, high_info = 0.0, {}
        low_loss, low_info = self.low_actor_loss(batch, low_dist)
        return high_loss + low_loss, high_info, low_info

    def multi_high_actor_loss(self, batch, dist_list, obs_expand, grad_params):
        del grad_params  # official target path intentionally uses stopped params
        multi_targets = self.network.select('goal_rep')(
            jnp.concatenate([obs_expand, batch['high_actor_targets']], axis=-1)
        )
        actor_loss = adv_mean = bc_log_prob = mse = std = 0.0
        for i in range(self.config['num_subgoals']):
            v1, v2 = self.network.select('value')(
                batch['observations'], batch['high_actor_goals']
            )
            nv1, nv2 = self.network.select('value')(
                batch['high_actor_targets'][:, i, :], batch['high_actor_goals']
            )
            adv = (nv1 + nv2) / 2 - (v1 + v2) / 2
            exp_a = jnp.minimum(jnp.exp(adv * self.config['high_alpha']), 100.0)
            target = multi_targets[:, i, :]
            log_prob = dist_list[i].log_prob(target)
            actor_loss += (
                -((exp_a * log_prob).mean() / self.config['subgoal_steps'])
                * self.config['high_discount'] ** (self.config['num_subgoals'] - i - 1)
            )
            adv_mean += adv.mean()
            bc_log_prob += log_prob.mean()
            mse += jnp.mean((dist_list[i].mode() - target) ** 2)
            std += jnp.mean(dist_list[i].scale_diag)
        return actor_loss / self.config['num_subgoals'], {
            'actor_loss': actor_loss,
            'adv': adv_mean / self.config['num_subgoals'],
            'bc_log_prob': bc_log_prob / self.config['num_subgoals'],
            'mse': mse / self.config['num_subgoals'],
            'std': std / self.config['num_subgoals'],
        }

    def low_actor_loss(self, batch, dist):
        target = (
            batch['high_actor_targets'][:, -1, :]
            if self.config['num_subgoals'] > 0
            else batch['high_actor_goals']
        )
        v1, v2 = self.network.select('value')(batch['observations'], target)
        nv1, nv2 = self.network.select('value')(batch['next_observations'], target)
        adv = (nv1 + nv2) / 2 - (v1 + v2) / 2
        exp_a = jnp.minimum(jnp.exp(adv * self.config['low_alpha']), 100.0)
        log_prob = dist.log_prob(batch['actions'])
        actor_loss = -(exp_a * log_prob).mean()
        info = {
            'actor_loss': actor_loss,
            'adv': adv.mean(),
            'bc_log_prob': log_prob.mean(),
        }
        if not self.config['discrete']:
            info.update(
                {
                    'mse': jnp.mean((dist.mode() - batch['actions']) ** 2),
                    'std': jnp.mean(dist.scale_diag),
                }
            )
        return actor_loss, info

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        rng = self.rng if rng is None else rng
        info = {}
        value_loss, value_info = self.value_loss(batch, grad_params)
        info.update({f'value/{key}': value for key, value in value_info.items()})
        actor_loss, high_info, low_info = self.actor_loss(batch, grad_params, rng)
        info.update({f'high_actor/{key}': value for key, value in high_info.items()})
        info.update({f'low_actor/{key}': value for key, value in low_info.items()})
        return value_loss + actor_loss, info

    def target_update(self, network, module_name):
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
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
        high_dist, _, predicted_actions = self.network.select('actor_mixer')(
            jnp.expand_dims(observations, axis=0),
            None if goals is None else jnp.expand_dims(goals, axis=0),
            seed,
            subgoal_reps=None,
            action_seq=None,
            temperature=temperature,
        )
        del high_dist
        actions = predicted_actions[0]
        return actions if self.config['discrete'] else jnp.clip(actions, -1, 1)

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)
        ex_goals = ex_observations
        action_dim = ex_actions.max() + 1 if config['discrete'] else ex_actions.shape[-1]

        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            goal_rep_seq = [encoder_module()]
        else:
            goal_rep_seq = []
        goal_rep_seq.extend(
            [
                MLP(
                    hidden_dims=(*config['enc_hidden_dims'], config['feature_dim']),
                    activate_final=False,
                    layer_norm=config['layer_norm'],
                ),
                LengthNormalize(),
            ]
        )
        goal_rep_def = nn.Sequential(goal_rep_seq)

        if config['encoder'] is not None:
            value_encoder_def = GCEncoder(state_encoder=encoder_module(), concat_encoder=goal_rep_def)
            target_value_encoder_def = GCEncoder(state_encoder=encoder_module(), concat_encoder=goal_rep_def)
            low_actor_encoder_def = GCEncoder(state_encoder=encoder_module(), concat_encoder=goal_rep_def)
            high_actor_encoder_def = GCEncoder(concat_encoder=encoder_module())
        else:
            value_encoder_def = GCEncoder(state_encoder=Identity(), concat_encoder=goal_rep_def)
            target_value_encoder_def = GCEncoder(state_encoder=Identity(), concat_encoder=goal_rep_def)
            low_actor_encoder_def = GCEncoder(state_encoder=Identity(), concat_encoder=goal_rep_def)
            high_actor_encoder_def = None

        value_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            ensemble=True,
            gc_encoder=value_encoder_def,
        )
        target_value_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            ensemble=True,
            gc_encoder=target_value_encoder_def,
        )
        if config['discrete']:
            raise NotImplementedError('Discrete actions not supported yet.')
        low_actor_def = GCActor(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            state_dependent_std=False,
            const_std=config['low_const_std'],
            gc_encoder=None,
        )
        high_actor_def = GCActor(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=config['feature_dim'],
            state_dependent_std=False,
            const_std=config['high_const_std'],
            gc_encoder=None,
        )
        gc_encoder = low_actor_encoder_def if config['gc_enc'] == 'concat' else high_actor_encoder_def
        actor_mixer_def = HierarchicalPolicyNetwork(
            num_tokens=1,
            state_dim=config['feature_dim'],
            num_action_dims=action_dim,
            joint_embed_dim=config['feature_dim'],
            num_mixer_blocks=config['num_mixer_blocks'],
            mixer_token_hidden=config['mixer_hidden'],
            mixer_channel_hidden=config['mixer_hidden'],
            gc_encoder=gc_encoder,
            layer_norm=config['layer_norm'],
            high_actor_head=high_actor_def,
            low_actor_head=low_actor_def,
            enc_hidden=config['enc_hidden_dims'],
            num_subgoals=config['num_subgoals'],
        )

        network_info = {
            'goal_rep': (goal_rep_def, jnp.concatenate([ex_observations, ex_goals], axis=-1)),
            'value': (value_def, (ex_observations, ex_goals)),
            'target_value': (target_value_def, (ex_observations, ex_goals)),
            'actor_mixer': (actor_mixer_def, (ex_observations, ex_goals, rng)),
        }
        network_def = ModuleDict({key: value[0] for key, value in network_info.items()})
        network_params = network_def.init(
            init_rng,
            **{key: value[1] for key, value in network_info.items()},
        )['params']
        network = TrainState.create(network_def, network_params, tx=optax.adam(learning_rate=config['lr']))
        network.params['modules_target_value'] = network.params['modules_value']
        return cls(rng, network=network, config=flax.core.freeze(dict(config)))


def get_config():
    return ml_collections.ConfigDict(
        dict(
            agent_name='coghp', lr=3e-4, batch_size=256, discount=0.99,
            actor_hidden_dims=(512, 512, 512), value_hidden_dims=(512, 512, 512),
            alpha=1.0, tau=0.005, expectile=0.7, low_alpha=3.0, high_alpha=3.0,
            subgoal_steps=25, low_actor_rep_grad=False, high_const_std=True,
            low_const_std=True, discrete=False, encoder=None,
            dataset_class='MultiHGCDataset', value_p_curgoal=0.2,
            value_p_trajgoal=0.5, value_p_randomgoal=0.3, value_geom_sample=True,
            actor_p_curgoal=0.0, actor_p_trajgoal=1.0, actor_p_randomgoal=0.0,
            actor_geom_sample=False, gc_negative=True, p_aug=0.0, frame_stack=None,
            feature_dim=32, mixer_hidden=32, num_mixer_blocks=1,
            enc_hidden_dims=(512, 512, 512), layer_norm=True, action_chunk=None,
            num_subgoals=1, gc_enc='concat', high_discount=0.8,
        )
    )
