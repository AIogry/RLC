# OGBench baseline compute slots

本表基于 `offline_rl_baselines/ogbench/impls/agents/{hiql,crl}.py`、`utils/networks.py` 和 `utils/encoders.py`。这里的“可替换”只针对 representation-transforming computation body；不包含最终分布、归一化、readout 或 RL loss。

## Agent: HIQL

### `value`

- current network: `GCValue`, `ensemble=True`；target copy 为 `target_value`。
- current implementation: `GCEncoder(state_encoder=Identity, concat_encoder=goal_rep_def)` 后接 `MLP(value_hidden_dims, ..., 1, activate_final=False, layer_norm=True)`；每个 ensemble member 独立。
- input shape: state `[B, obs_dim]` 和 goal `[B, obs_dim]`；视觉输入先经过 encoder。
- output shape: `[B]` per ensemble member，通常整体为 `[2, B]`。
- underlying computation: goal representation `MLP -> LengthNormalize`，再是 value MLP。
- replaceable computation body? **是，分两层**：后续可分别替换 `goal_rep` 与 value body；本轮不迁移。
- final readout: `Dense(1)` 和 `squeeze(-1)`，不应放入 computation core。
- candidate slot name: `value`（另有 `target_value`，必须保持 target-update 语义）。

### `high_actor`

- current network: `GCActor`，action dimension 为 `rep_dim`。
- current implementation: state-based时输入为 `[observations, high_actor_goals]`；视觉时由 `GCEncoder(concat_encoder=encoder_module())` 编码；之后 `MLP(actor_hidden_dims, activate_final=True)`。
- input shape: state/goal `[B, obs_dim]`；输出 head 前为 `[B, actor_hidden_dims[-1]]`。
- output shape: Gaussian distribution over `[B, rep_dim]`，训练 target 是 `goal_rep([s, high_actor_targets])`。
- underlying computation: actor representation MLP。
- replaceable computation body? **是**。
- final readout: `Dense(rep_dim, kernel_init=default_init(1e-2))`、std 参数和 Gaussian distribution，不应替换。
- candidate slot name: `high_actor`。

### `low_actor`

- current network: `GCActor` 或离散版本 `GCDiscreteActor`。
- current implementation: `GCEncoder(state_encoder=Identity, concat_encoder=goal_rep_def)`；训练时输入 state 和 raw low-level goal，`goal_rep_def` 产生 length-normalized subgoal representation；运行时 `goal_encoded=True` 直接接收该 representation；之后是 `MLP(actor_hidden_dims, activate_final=True)`。
- input shape: observations `[B, obs_dim]`，encoded goal `[B, rep_dim]`；body 输入为 `[B, obs_dim + rep_dim]`。
- output shape: continuous Gaussian over `[B, action_dim]`，或 discrete categorical over `[B, action_dim]`。
- underlying computation: actor body MLP；`goal_rep_def` 是输入语义/representation encoder，不是 action computation body。
- replaceable computation body? **是，本轮首个迁移 slot**。
- final readout: continuous `Dense(action_dim)` + constant/state-dependent std + optional tanh；discrete `Dense(action_dim)` + categorical distribution；这些保持原样。
- candidate slot name: `low_actor`。

## Agent: CRL

### `actor`

- current network: `GCActor` 或 `GCDiscreteActor`。
- current implementation: state-based时 concat observations/goals；视觉时 `GCEncoder(concat_encoder=encoder_module())`；之后 actor MLP。
- input shape: observations/goals `[B, ...]`。
- output shape: action distribution over `[B, action_dim]`。
- underlying computation: actor representation MLP。
- replaceable computation body? **是，后续迁移**。
- final readout: action mean/logit、std、distribution semantics，不替换。
- candidate slot name: `actor`。

### `critic_state`

- current network: `GCBilinearValue.phi` 的 state branch；critic ensemble 时有两个 member。
- current implementation: 可选 state encoder 后，`MLP(value_hidden_dims + latent_dim, activate_final=False, layer_norm=True)`；若有 action，action 与 state representation 在 phi 输入前拼接。
- input shape: state `[B, obs_dim]`，连续 action 时 phi 输入为 state representation 与 action 的拼接。
- output shape: phi `[B, latent_dim]`（ensemble 时 `[2, B, latent_dim]`）。
- underlying computation: phi representation MLP。
- replaceable computation body? **是，后续迁移**。
- final readout: 与 goal branch 的 dot product、`/sqrt(latent_dim)` 和 optional `exp`，不替换。
- candidate slot name: `critic_state`。

### `critic_goal`

- current network: `GCBilinearValue.psi` 的 goal branch。
- current implementation: 可选 goal encoder 后，独立的 goal MLP。
- input shape: goal `[B, obs_dim]` 或 encoded goal。
- output shape: psi `[B, latent_dim]`（ensemble 时 `[2, B, latent_dim]`）。
- underlying computation: psi representation MLP。
- replaceable computation body? **是，后续迁移**。
- final readout: 与 phi 的 bilinear product，不替换。
- candidate slot name: `critic_goal`。

### `value_state` / `value_goal`（仅 AWR）

- current network: AWR 的独立 `GCBilinearValue(ensemble=False)` 的 phi/psi branch。
- input/output: 与 `critic_state`/`critic_goal` 相同，但没有 critic ensemble。
- replaceable computation body? **是**。
- candidate slot names: `value_state`、`value_goal`；不要误并入 critic。

## 关键边界

OGBench 中的 MLP 出现在三个层次：encoder 内部的视觉投影、actor/value 的 representation body、以及 CRL bilinear branch。`GCEncoder` 负责 state/goal/concat 的输入组合；`GCActor`/`GCValue` 负责 task-specific 输入输出语义；MLP body 才是第一阶段的 computation slot。Gaussian/Categorical construction、`LengthNormalize`、bilinear readout 和 agent loss 不属于可替换 body。
