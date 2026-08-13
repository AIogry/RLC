"""OGBench-compatible network semantics used by the first migrated slot."""

from typing import Optional, Sequence

import distrax
import flax.linen as nn
import jax.numpy as jnp

from ..computation.factory import ComputationSpec, make_computation_core
from ..computation.interfaces import ComputationOutput
from ..computation.primitives.mlp import MLP, default_init


def ensemblize(cls, num_qs, out_axes=0, in_axes=None, **kwargs):
    return nn.vmap(
        cls,
        variable_axes={'params': 0},
        split_rngs={'params': True},
        in_axes=in_axes,
        out_axes=out_axes,
        axis_size=num_qs,
        **kwargs,
    )


class _ComputationValueBody(nn.Module):
    """One value estimator body for the computationized ensemble."""

    hidden_dims: Sequence[int]
    layer_norm: bool
    computation_spec: ComputationSpec

    def setup(self):
        self.core = make_computation_core(
            self.computation_spec,
            hidden_dims=self.hidden_dims,
            activate_final=True,
            layer_norm=self.layer_norm,
        )

    def __call__(self, x):
        output = self.core(x)
        if isinstance(output, ComputationOutput):
            output = output.representation
        return output


class _ComputationBilinearBody(nn.Module):
    """One computationized branch of a CRL bilinear representation."""

    hidden_dims: Sequence[int]
    layer_norm: bool
    computation_spec: ComputationSpec

    def setup(self):
        self.core = make_computation_core(
            self.computation_spec,
            hidden_dims=self.hidden_dims,
            activate_final=False,
            layer_norm=self.layer_norm,
        )

    def __call__(self, x):
        output = self.core(x)
        if isinstance(output, ComputationOutput):
            output = output.representation
        return output


class Identity(nn.Module):
    def __call__(self, x):
        return x


class LengthNormalize(nn.Module):
    @nn.compact
    def __call__(self, x):
        return x / jnp.linalg.norm(x, axis=-1, keepdims=True) * jnp.sqrt(x.shape[-1])


class TransformedWithMode(distrax.Transformed):
    """Transformed distribution with the mode API used by OGBench losses."""

    def mode(self):
        return self.bijector.forward(self.distribution.mode())


class GCActor(nn.Module):
    """Goal-conditioned actor with an optional replaceable body."""

    hidden_dims: Sequence[int]
    action_dim: int
    log_std_min: Optional[float] = -5
    log_std_max: Optional[float] = 2
    tanh_squash: bool = False
    state_dependent_std: bool = False
    const_std: bool = True
    final_fc_init_scale: float = 1e-2
    gc_encoder: nn.Module = None
    computation_spec: Optional[ComputationSpec] = None

    def setup(self):
        if self.computation_spec is None:
            self.actor_net = MLP(self.hidden_dims, activate_final=True)
        else:
            self.actor_net = make_computation_core(self.computation_spec, hidden_dims=self.hidden_dims, activate_final=True)
        self.mean_net = nn.Dense(self.action_dim, kernel_init=default_init(self.final_fc_init_scale))
        if self.state_dependent_std:
            self.log_std_net = nn.Dense(self.action_dim, kernel_init=default_init(self.final_fc_init_scale))
        elif not self.const_std:
            self.log_stds = self.param('log_stds', nn.initializers.zeros, (self.action_dim,))

    def __call__(self, observations, goals=None, goal_encoded=False, temperature=1.0):
        if self.gc_encoder is not None:
            inputs = self.gc_encoder(observations, goals, goal_encoded=goal_encoded)
        else:
            inputs = [observations]
            if goals is not None:
                inputs.append(goals)
            inputs = jnp.concatenate(inputs, axis=-1)
        outputs = self.actor_net(inputs)
        if isinstance(outputs, ComputationOutput):
            outputs = outputs.representation

        means = self.mean_net(outputs)
        if self.state_dependent_std:
            log_stds = self.log_std_net(outputs)
        elif self.const_std:
            log_stds = jnp.zeros_like(means)
        else:
            log_stds = self.log_stds
        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        distribution = distrax.MultivariateNormalDiag(loc=means, scale_diag=jnp.exp(log_stds) * temperature)
        if self.tanh_squash:
            distribution = TransformedWithMode(distribution, distrax.Block(distrax.Tanh(), ndims=1))
        return distribution


class GCDiscreteActor(nn.Module):
    """Goal-conditioned categorical actor with an optional replaceable body."""

    hidden_dims: Sequence[int]
    action_dim: int
    final_fc_init_scale: float = 1e-2
    gc_encoder: nn.Module = None
    computation_spec: Optional[ComputationSpec] = None

    def setup(self):
        if self.computation_spec is None:
            self.actor_net = MLP(self.hidden_dims, activate_final=True)
        else:
            self.actor_net = make_computation_core(self.computation_spec, hidden_dims=self.hidden_dims, activate_final=True)
        self.logit_net = nn.Dense(self.action_dim, kernel_init=default_init(self.final_fc_init_scale))

    def __call__(self, observations, goals=None, goal_encoded=False, temperature=1.0):
        if self.gc_encoder is not None:
            inputs = self.gc_encoder(observations, goals, goal_encoded=goal_encoded)
        else:
            inputs = [observations]
            if goals is not None:
                inputs.append(goals)
            inputs = jnp.concatenate(inputs, axis=-1)
        outputs = self.actor_net(inputs)
        if isinstance(outputs, ComputationOutput):
            outputs = outputs.representation
        logits = self.logit_net(outputs)
        return distrax.Categorical(logits=logits / jnp.maximum(1e-6, temperature))


class GCValue(nn.Module):
    """Original OGBench goal-conditioned scalar value/critic network."""

    hidden_dims: Sequence[int]
    layer_norm: bool = True
    ensemble: bool = True
    gc_encoder: nn.Module = None
    computation_spec: Optional[ComputationSpec] = None

    def setup(self):
        if self.computation_spec is None:
            # Preserve the complete legacy MLP, including its final Dense(1)
            # path, for checkpoint and baseline compatibility.
            if self.ensemble:
                mlp_module = ensemblize(MLP, 2)
                self.value_net = mlp_module(
                    hidden_dims=(*self.hidden_dims, 1),
                    activate_final=False,
                    layer_norm=self.layer_norm,
                )
            else:
                self.value_net = MLP((*self.hidden_dims, 1), activate_final=False, layer_norm=self.layer_norm)
            self.value_readout = None
            return

        if self.ensemble:
            body_module = ensemblize(_ComputationValueBody, 2)
            readout_module = ensemblize(nn.Dense, 2, in_axes=0)
            self.value_net = body_module(
                hidden_dims=self.hidden_dims,
                layer_norm=self.layer_norm,
                computation_spec=self.computation_spec,
            )
            self.value_readout = readout_module(1, kernel_init=default_init())
        else:
            self.value_net = _ComputationValueBody(
                hidden_dims=self.hidden_dims,
                layer_norm=self.layer_norm,
                computation_spec=self.computation_spec,
            )
            self.value_readout = nn.Dense(1, kernel_init=default_init())

    def __call__(self, observations, goals=None, actions=None):
        if self.gc_encoder is not None:
            inputs = [self.gc_encoder(observations, goals)]
        else:
            inputs = [observations]
            if goals is not None:
                inputs.append(goals)
        if actions is not None:
            inputs.append(actions)
        outputs = self.value_net(jnp.concatenate(inputs, axis=-1))
        if isinstance(outputs, ComputationOutput):
            outputs = outputs.representation
        if self.value_readout is not None:
            outputs = self.value_readout(outputs)
        return outputs.squeeze(-1)


class GCBilinearValue(nn.Module):
    """CRL bilinear value/critic with independently replaceable branches.

    ``state_computation_spec`` and ``goal_computation_spec`` only replace the
    MLPs that produce ``phi`` and ``psi``. The bilinear interaction remains
    here so CRL's ensemble and contrastive semantics stay unchanged. Equal
    specs still create independent Flax submodules and parameters.
    """

    hidden_dims: Sequence[int]
    latent_dim: int
    layer_norm: bool = True
    ensemble: bool = True
    value_exp: bool = False
    state_encoder: nn.Module = None
    goal_encoder: nn.Module = None
    state_computation_spec: Optional[ComputationSpec] = None
    goal_computation_spec: Optional[ComputationSpec] = None

    def setup(self):
        branch_dims = (*self.hidden_dims, self.latent_dim)

        def make_branch(computation_spec):
            if computation_spec is None:
                branch_cls = MLP
                branch_kwargs = dict(
                    hidden_dims=branch_dims,
                    activate_final=False,
                    layer_norm=self.layer_norm,
                )
            else:
                branch_cls = _ComputationBilinearBody
                branch_kwargs = dict(
                    hidden_dims=branch_dims,
                    layer_norm=self.layer_norm,
                    computation_spec=computation_spec,
                )
            if self.ensemble:
                branch_cls = ensemblize(branch_cls, 2)
            return branch_cls(**branch_kwargs)

        self.phi = make_branch(self.state_computation_spec)
        self.psi = make_branch(self.goal_computation_spec)

    def __call__(self, observations, goals, actions=None, info=False):
        if self.state_encoder is not None:
            observations = self.state_encoder(observations)
        if self.goal_encoder is not None:
            goals = self.goal_encoder(goals)

        phi_inputs = observations if actions is None else jnp.concatenate([observations, actions], axis=-1)
        phi = self.phi(phi_inputs)
        psi = self.psi(goals)
        value = (phi * psi / jnp.sqrt(self.latent_dim)).sum(axis=-1)
        if self.value_exp:
            value = jnp.exp(value)
        if info:
            return value, phi, psi
        return value


class GCDiscreteBilinearCritic(GCBilinearValue):
    """Bilinear CRL critic with one-hot discrete actions."""

    action_dim: int = None

    def __call__(self, observations, goals=None, actions=None, info=False):
        actions = jnp.eye(self.action_dim)[actions]
        return super().__call__(observations, goals, actions, info)
