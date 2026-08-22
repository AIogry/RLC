"""CRL actor extraction against a structurally frozen critic.

This is a runtime training variant of CRL.  It reuses ``CRLAgent``'s network
construction and DDPG+BC actor objective while replacing the optimizer and
update dispatch so the critic is not an optimizer target.
"""

import jax
import jax.numpy as jnp
import optax
from flax.traverse_util import flatten_dict, unflatten_dict

from .crl import CRLAgent


def _optimizer_labels(params):
    labels = {}
    for path in flatten_dict(params):
        labels[path] = (
            'frozen'
            if path and path[0] in {'critic', 'modules_critic'}
            else 'actor'
        )
    return unflatten_dict(labels)


class CRLPolicyExtractorAgent(CRLAgent):
    """Trainable CRL actor with an immutable critic evaluator."""

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        if config.get('actor_loss') != 'ddpgbc':
            raise ValueError('CRLPolicyExtractorAgent requires actor_loss=ddpgbc')
        agent = super().create(seed, ex_observations, ex_actions, config)
        labels = _optimizer_labels(agent.network.params)
        tx = optax.multi_transform(
            {
                'actor': optax.adam(learning_rate=config['lr']),
                'frozen': optax.set_to_zero(),
            },
            labels,
        )
        network = agent.network.replace(tx=tx, opt_state=tx.init(agent.network.params))
        return agent.replace(network=network)

    @jax.jit
    def policy_extraction_loss(self, batch, grad_params, rng=None):
        """Return canonical DDPG+BC actor loss plus frozen-Q diagnostics."""

        rng = self.rng if rng is None else rng
        actor_loss, actor_info = self.actor_loss(batch, grad_params, rng=rng)
        frozen_params = jax.tree_util.tree_map(jax.lax.stop_gradient, grad_params)
        data_q1, data_q2 = self.network.select('critic')(
            batch['observations'], batch['actor_goals'], batch['actions'],
            params=frozen_params,
        )
        dist = self.network.select('actor')(
            batch['observations'], batch['actor_goals'], params=grad_params,
        )
        policy_actions = jnp.clip(dist.mode(), -1, 1)
        policy_q1, policy_q2 = self.network.select('critic')(
            batch['observations'], batch['actor_goals'], policy_actions,
            params=frozen_params,
        )
        q_data = jnp.minimum(data_q1, data_q2)
        q_policy = jnp.minimum(policy_q1, policy_q2)
        info = {f'actor/{key}': value for key, value in actor_info.items()}
        info.update({
            'frozen/q_data_mean': q_data.mean(),
            'frozen/q_policy_mean': q_policy.mean(),
            'frozen/q_delta': q_policy.mean() - q_data.mean(),
        })
        return actor_loss, info

    @jax.jit
    def update(self, batch):
        """Apply a fresh actor-only optimizer update."""

        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.policy_extraction_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        return self.replace(network=new_network, rng=new_rng), info
