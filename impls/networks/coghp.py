"""Official vanilla CoGHP autoregressive policy network.

The implementation intentionally mirrors the official CoGHP MixerBlock and
HierarchicalPolicyNetwork.  In particular, one physical list of MixerBlocks
is reused for every autoregressive token step; high and low distributions only
have independent heads.
"""

from typing import Optional, Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp

from .common import LengthNormalize, MLP, default_init


class MixerBlock(nn.Module):
    """Official CoGHP token/channel mixer with a causal token mask."""

    num_tokens: int
    embed_dim: int
    hidden_dim_tokens: int
    hidden_dim_channels: int
    init_scale: float = 1e-2
    decay_alpha: float = 0.9

    def setup(self):
        self.token_dense1 = nn.Dense(self.hidden_dim_tokens, kernel_init=default_init())
        self.token_dense2 = nn.Dense(self.num_tokens, kernel_init=default_init())
        self.channel_dense1 = nn.Dense(self.hidden_dim_channels, kernel_init=default_init())
        self.channel_dense2 = nn.Dense(self.embed_dim, kernel_init=default_init())
        self.tm_weights = self.param(
            'tm_weights',
            nn.initializers.normal(stddev=0.02),
            (self.num_tokens, self.num_tokens),
        )
        self.tm_weights = jnp.tril(self.tm_weights)

    def __call__(self, x):
        y = jnp.transpose(x, (0, 2, 1))
        y = self.token_dense1(y)
        y = nn.gelu(y)
        y = self.token_dense2(y)
        y = jnp.transpose(y, (0, 2, 1))
        y = jnp.einsum('btd,ts->bsd', y, self.tm_weights)
        x = x + y

        z = self.channel_dense1(x)
        z = nn.gelu(z)
        z = self.channel_dense2(z)
        return x + z


class HierarchicalPolicyNetwork(nn.Module):
    """Official CoGHP autoregressive high-subgoal/low-action policy."""

    num_tokens: int
    state_dim: int
    num_action_dims: int
    joint_embed_dim: int = 128
    num_mixer_blocks: int = 2
    mixer_token_hidden: int = 64
    mixer_channel_hidden: int = 64
    gc_encoder: nn.Module = None
    layer_norm: bool = True
    final_fc_init_scale: float = 1e-2
    high_actor_head: nn.Module = None
    low_actor_head: nn.Module = None
    enc_hidden: Sequence[int] = (128, 128)
    num_subgoals: int = 1

    def setup(self):
        self.prev_tokens = self.param(
            'prev_tokens',
            nn.initializers.normal(stddev=0.1),
            (1, self.num_subgoals + 1, self.state_dim),
        )
        # This list is deliberately created once.  The autoregressive loop
        # below reuses these same physical modules for all token steps.
        self.mixer_blocks = [
            MixerBlock(
                num_tokens=self.num_subgoals + 3,
                embed_dim=self.state_dim,
                hidden_dim_tokens=self.mixer_token_hidden,
                hidden_dim_channels=self.mixer_channel_hidden,
            )
            for _ in range(self.num_mixer_blocks)
        ]
        self.feature_embed = nn.Sequential(
            [
                MLP(
                    hidden_dims=(*self.enc_hidden, self.state_dim),
                    activate_final=False,
                    layer_norm=True,
                ),
                LengthNormalize(),
            ]
        )

    def __call__(
        self,
        observations,
        goals,
        seed,
        subgoal_reps: Optional[jnp.ndarray] = None,
        action_seq: Optional[jnp.ndarray] = None,
        temperature: float = 1.0,
    ):
        del action_seq  # accepted for official training-call compatibility
        high_seed, low_seed = jax.random.split(seed)

        observations = jnp.expand_dims(observations, axis=1)
        if goals is not None:
            goals = jnp.expand_dims(goals, axis=1)

        if self.gc_encoder is not None:
            features = self.gc_encoder(observations, goals, goal_encoded=False, listwise=True)
            obs_feature = self.feature_embed(features[0])
            goal_feature = features[1]
            features = jnp.concatenate([obs_feature, goal_feature], axis=1)
        else:
            features = [self.feature_embed(observations)]
            if goals is not None:
                features.append(self.feature_embed(goals))
            features = jnp.concatenate(features, axis=1)

        batch_size, _, _ = features.shape
        predicted_subgoals = jnp.zeros(
            (batch_size, self.num_subgoals, self.joint_embed_dim),
            dtype=jnp.float32,
        )
        prev_embed_tokens = jnp.tile(self.prev_tokens, (batch_size, 1, 1))

        high_dist_list = []
        for token_dim in range(self.num_subgoals + 1):
            if token_dim == 0:
                prev_embeds = prev_embed_tokens
            elif subgoal_reps is not None:
                prev_embeds = jnp.concatenate(
                    [subgoal_reps[:, :token_dim, :], prev_embed_tokens[:, token_dim:, :]],
                    axis=1,
                )
            else:
                prev_embeds = jnp.concatenate(
                    [predicted_subgoals[:, :token_dim, :], prev_embed_tokens[:, token_dim:, :]],
                    axis=1,
                )

            x = jnp.concatenate([features, prev_embeds], axis=1)
            target_dim = features.shape[1] + token_dim + 1
            for mixer_block in self.mixer_blocks:
                x = mixer_block(x)
            target_token = x[:, target_dim - 1, :]

            if token_dim < self.num_subgoals:
                high_dist = self.high_actor_head(target_token, temperature=temperature)
                high_dist_list.append(high_dist)
                goal_reps = high_dist.sample(seed=high_seed)
                predicted_subgoals = predicted_subgoals.at[:, token_dim, :].set(goal_reps)
            else:
                low_dist = self.low_actor_head(target_token, temperature=temperature)
                predicted_actions = low_dist.sample(seed=low_seed)

        return high_dist_list, low_dist, predicted_actions
