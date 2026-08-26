"""OGBench-compatible network semantics used by the first migrated slot."""

from typing import Any, Optional, Sequence

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp

from ..computation.factory import ComputationSpec, make_computation_core
from ..computation.interfaces import ComputationOutput
from ..computation.primitives.mlp import MLP, default_init


def ensemblize(cls, num_qs, out_axes=0, in_axes=None, **kwargs):
    # Computationized CRL branches may own non-trainable recurrent state
    # buffers.  Map and split the buffer collection exactly like parameters so
    # ensemble members remain independent modules rather than sharing one
    # hidden initialization buffer.  Legacy MLP branches do not create this
    # collection, so their parameter structure is unchanged.
    return nn.vmap(
        cls,
        variable_axes={'params': 0, 'buffers': 0},
        split_rngs={'params': True, 'buffers': True},
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


class ComputationVectorBody(nn.Module):
    """Vector-valued body used by QRL phi and latent dynamics.

    The algorithm sees only the vector representation. Topology state and
    non-trainable buffers remain inside the computation core, which keeps the
    network contract stable for future tokenized bodies.
    """

    hidden_dims: Sequence[int]
    layer_norm: bool
    computation_spec: ComputationSpec
    activate_final: bool = False

    def setup(self):
        self.core = make_computation_core(
            self.computation_spec,
            hidden_dims=self.hidden_dims,
            activate_final=self.activate_final,
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
        if self.computation_spec is not None and self.computation_spec.structure == 'puzzle_tokens' and self.gc_encoder is not None:
            raise ValueError('Puzzle structured computation requires raw standard Puzzle observations; encoder is unsupported')
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
        if self.computation_spec is not None and self.computation_spec.structure == 'puzzle_tokens' and self.gc_encoder is not None:
            raise ValueError('Puzzle structured computation requires raw standard Puzzle observations; encoder is unsupported')
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
        if self.computation_spec is not None and self.computation_spec.structure == 'puzzle_tokens' and self.gc_encoder is not None:
            raise ValueError('Puzzle structured computation requires raw standard Puzzle observations; encoder is unsupported')
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


class GCDiscreteCritic(GCValue):
    """Goal-conditioned critic for discrete actions."""

    action_dim: int = None

    def __call__(self, observations, goals=None, actions=None):
        actions = jnp.eye(self.action_dim)[actions]
        return super().__call__(observations, goals, actions)


class Param(nn.Module):
    """Scalar parameter module used by the canonical QRL value models."""

    init_value: float = 0.0

    @nn.compact
    def __call__(self):
        return self.param('value', init_fn=lambda key: jnp.full((), self.init_value))


class LogParam(nn.Module):
    """Positive scalar parameter represented in log space."""

    init_value: float = 1.0

    @nn.compact
    def __call__(self):
        log_value = self.param(
            'log_value',
            init_fn=lambda key: jnp.full((), jnp.log(self.init_value)),
        )
        return jnp.exp(log_value)


class GCMRNValue(nn.Module):
    """Metric Residual Network (MRN) quasimetric value function."""

    hidden_dims: Sequence[int]
    latent_dim: int
    layer_norm: bool = True
    encoder: nn.Module = None
    computation_spec: Optional[ComputationSpec] = None

    def setup(self):
        if self.computation_spec is not None and self.computation_spec.structure == 'puzzle_tokens' and self.encoder is not None:
            raise ValueError('Puzzle structured computation requires raw standard Puzzle observations; encoder is unsupported')
        phi_dims = (*self.hidden_dims, self.latent_dim)
        if self.computation_spec is None:
            self.phi = MLP(
                phi_dims,
                activate_final=False,
                layer_norm=self.layer_norm,
            )
        else:
            self.phi = ComputationVectorBody(
                hidden_dims=phi_dims,
                layer_norm=self.layer_norm,
                computation_spec=self.computation_spec,
                activate_final=False,
            )

    def __call__(self, observations, goals, is_phi=False, info=False):
        if is_phi:
            phi_s = observations
            phi_g = goals
        else:
            if self.encoder is not None:
                observations = self.encoder(observations)
                goals = self.encoder(goals)
            phi_s = self.phi(observations)
            phi_g = self.phi(goals)

        half_dim = self.latent_dim // 2
        sym_s = phi_s[..., :half_dim]
        sym_g = phi_g[..., :half_dim]
        asym_s = phi_s[..., half_dim:]
        asym_g = phi_g[..., half_dim:]
        squared_dist = ((sym_s - sym_g) ** 2).sum(axis=-1)
        quasi = jax.nn.relu((asym_s - asym_g).max(axis=-1))
        value = jnp.sqrt(jnp.maximum(squared_dist, 1e-12)) + quasi

        if info:
            return value, phi_s, phi_g
        return value


class GCIQEValue(nn.Module):
    """Interval Quasimetric Embedding (IQE) value function."""

    hidden_dims: Sequence[int]
    latent_dim: int
    dim_per_component: int
    layer_norm: bool = True
    encoder: nn.Module = None
    computation_spec: Optional[ComputationSpec] = None

    def setup(self):
        if self.computation_spec is not None and self.computation_spec.structure == 'puzzle_tokens' and self.encoder is not None:
            raise ValueError('Puzzle structured computation requires raw standard Puzzle observations; encoder is unsupported')
        phi_dims = (*self.hidden_dims, self.latent_dim)
        if self.computation_spec is None:
            self.phi = MLP(
                phi_dims,
                activate_final=False,
                layer_norm=self.layer_norm,
            )
        else:
            self.phi = ComputationVectorBody(
                hidden_dims=phi_dims,
                layer_norm=self.layer_norm,
                computation_spec=self.computation_spec,
                activate_final=False,
            )
        self.alpha = Param()

    def __call__(self, observations, goals, is_phi=False, info=False):
        alpha = jax.nn.sigmoid(self.alpha())
        if is_phi:
            phi_s = observations
            phi_g = goals
        else:
            if self.encoder is not None:
                observations = self.encoder(observations)
                goals = self.encoder(goals)
            phi_s = self.phi(observations)
            phi_g = self.phi(goals)

        x = jnp.reshape(
            phi_s, (*phi_s.shape[:-1], -1, self.dim_per_component)
        )
        y = jnp.reshape(
            phi_g, (*phi_g.shape[:-1], -1, self.dim_per_component)
        )
        valid = x < y
        xy = jnp.concatenate(jnp.broadcast_arrays(x, y), axis=-1)
        ixy = xy.argsort(axis=-1)
        neg_inc_copies = jnp.take_along_axis(
            valid, ixy % self.dim_per_component, axis=-1
        ) * jnp.where(ixy < self.dim_per_component, -1, 1)
        neg_inp_copies = jnp.cumsum(neg_inc_copies, axis=-1)
        neg_f = -1.0 * (neg_inp_copies < 0)
        neg_incf = jnp.concatenate(
            [neg_f[..., :1], neg_f[..., 1:] - neg_f[..., :-1]], axis=-1
        )
        sxy = jnp.take_along_axis(xy, ixy, axis=-1)
        components = (sxy * neg_incf).sum(axis=-1)
        value = alpha * components.mean(axis=-1) + (1 - alpha) * components.max(axis=-1)

        if info:
            return value, phi_s, phi_g
        return value


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
