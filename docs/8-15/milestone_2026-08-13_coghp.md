# Milestone：CRL 迁移、CRL Computationization 与 Vanilla CoGHP M7

日期：2026-08-13

## Milestone 结论

今天完成了 CRL baseline 迁移、CRL actor/critic/AWR value computationization，
以及 Vanilla CoGHP 从官方仓库到 RLC 的迁移、shared runtime 接入、官方实现
审计和 M7 cleanup。所有迁移都保持对应 baseline 的网络结构、loss、RNG、
target/update 和参数边界。

当前结论：

> CRL 的 baseline、actor、critic、AWR value 迁移，以及 Vanilla CoGHP，均已
> 达到 local integration validated。官方/RLC synthetic parity、真实 OGBench
> N=20 parity、完整 CPU regression 和 CUDA 1000-step smoke 均通过。

这不是长时间 baseline reproduction，也没有开始多 seed 或成功率对比实验。

## 一、CRL 迁移与 Computationization

### 1. CRL baseline 迁移（M4）

以 `/home/eai/Research/offline_rl_baselines/ogbench` 中的 OGBench reference
CRL 为算法来源，将完整 CRL 接入 RLC shared runtime，
同时保留两个 actor loss 分支：

```text
actor_loss = ddpgbc
actor_loss = awr
```

保持的 CRL semantics 包括：

- ensemble bilinear critic：`Q(s,a,g) = phi(s,a)^T psi(g) / sqrt(latent_dim)`；
- AWR 使用的独立、非 ensemble bilinear value；
- contrastive binary/categorical objective 和 diagnostics；
- DDPG+BC 的 Q normalization 与 behavior-cloning term；
- AWR advantage weighting、log-probability loss 和 clipping；
- actor distribution、temperature、action clipping、`sample_actions`；
- reference initialization、optimizer、update 和 agent RNG。

新增/迁移了当前 RLC 所需的 `GCBilinearValue` 和
`GCDiscreteBilinearCritic`，没有无差别复制旧 OGBench `utils/networks.py`。
CRL critic 的 bilinear interaction、contrastive logits、normalization、
ensemble 和 latent scaling 保持在 network/agent semantics 中。

### 2. CRL actor computationization

CRL 先增加独立的：

```yaml
compute:
  actor:
    enabled: true
    primitive: mlp
    topology: feedforward
    credit: direct
```

只替换 actor representation body：

```text
GCActor
  -> ComputationCore(MLP + FeedForward + Direct)
  -> same action readout/distribution
```

actor input、state/goal concatenation、action readout、std、Gaussian/Categorical
distribution、DDPG+BC/AWR loss 和 critic-to-actor interaction 均未改变。
CRL 仍然只有一个 `CRLAgent`，计算方式由 config 控制。

### 3. CRL critic computationization（M5）

在 actor 迁移基础上，新增两个独立 computation slots：

```yaml
compute:
  critic_state:
    enabled: true
    primitive: mlp
    topology: feedforward
    credit: direct
  critic_goal:
    enabled: true
    primitive: mlp
    topology: feedforward
    credit: direct
```

分别对应：

```text
critic_state -> phi(s, a)
critic_goal  -> psi(g)
```

两个 slot 可以使用相同 computation specification，但参数完全独立。计算
只替换 representation branch；bilinear/dot product、ensemble、normalization、
contrastive logits、`sqrt(latent_dim)` scaling 和 actor interaction 保持
legacy CRL semantics。

AWR value branches 在 M5 保持 legacy，没有与 critic branches 合并。

### 4. CRL AWR value computationization（M6）

在 `actor_loss='awr'` 时新增两个独立 slots：

```yaml
compute:
  value_state:
    enabled: true
    primitive: mlp
    topology: feedforward
    credit: direct
  value_goal:
    enabled: true
    primitive: mlp
    topology: feedforward
    credit: direct
```

分别对应：

```text
value_state -> phi_V(s)
value_goal  -> psi_V(g)
```

AWR value 保持 `GCBilinearValue(ensemble=False)`，并保持：

```text
Q = min(Q1, Q2)
A = Q - V
weight = min(exp(alpha * A), 100)
actor_loss = -(weight * log_prob).mean()
```

`value_state`、`value_goal` 与 `critic_state`、`critic_goal` 是四组完全独立
的参数。DDPG+BC 模式不实例化 `modules_value`，即使配置中存在 value slot
也不会增加参数。

### 5. CRL runtime integration

CRL 复用同一套 RLC runtime：

```text
RLC/ogbench
  -> Dataset / GCDataset
  -> CRLAgent
  -> optional actor computation
  -> optional critic_state/critic_goal computation
  -> optional AWR value_state/value_goal computation
  -> bilinear CRL semantics
  -> update
  -> evaluation
  -> logging/checkpoint
```

通过 `--agent=crl` 选择 CRL，通过 `--computation` 启用对应 computation
branches；没有新增第二套 trainer 或 main。

### 6. CRL 参数统计

小型配置为 `obs_dim=29`、`action_dim=8`、hidden dimensions `(6,6)`、
`latent_dim=3`：

| Component | Parameters |
|---|---:|
| DDPG+BC CRL total | 1616 |
| actor | 452 |
| critic | 1164 |
| critic state core `phi` | 630 |
| critic goal core `psi` | 534 |
| AWR value total | 534 |
| AWR value state core `phi_V` | 267 |
| AWR value goal core `psi_V` | 267 |
| AWR CRL total | 2150 |
| bilinear/readout parameters | 0 |

bilinear/readout 参数为 0 是预期结果，因为该 interaction 是无参数 dot
product，而不是额外 Dense readout。

## CRL 验证结果

### Reference migration 与 actor parity

固定 synthetic batch 上，`ddpgbc` 和 `awr` 均通过：

- critic output/logits；
- actor distribution；
- critic/actor/total loss；
- full semantic gradient；
- one-step optimizer update；
- optimizer state 与 agent RNG。

actor computation 在 test-only semantic parameter mapping 后通过：

- actor mean/mode/std/log-probability；
- actor loss 与 total loss；
- semantic full gradient；
- one-step parameter update；
- critic structure 与 trainable parameter count 保持一致。

### Critic slot wiring 与 parity

以下组合均完成 initialize/forward wiring 验证：

```text
critic_state ON / critic_goal OFF
critic_state OFF / critic_goal ON
critic_state ON / critic_goal ON
```

`critic_state` 与 `critic_goal` 的 computation parameter subtrees 独立。
两者同时启用时，legacy 与 computationized CRL 在两种 actor loss 分支上通过：

- 每个 ensemble member 的 `phi`、`psi`、final bilinear value/logits；
- critic/actor/total loss 与 contrastive diagnostics；
- critic 两个 branches 和 actor 的 semantic gradients；
- `agent.update(batch)`、optimizer state、agent RNG。

### AWR value parity

AWR value 的隔离与完整 five-slot parity 通过：

- `phi_V`、`psi_V`、`V(s,g)` forward；
- production contrastive value loss；
- `Q1`、`Q2`、`V`、advantage、AWR weight；
- AWR actor loss 与相关 gradients；
- one-step update、optimizer state、agent RNG；
- value/critic 四组参数独立性；
- DDPG+BC 不实例化 `modules_value`。

### Real OGBench N=20

环境：`antmaze-medium-navigate-v0`，真实 `GCDataset` batch sequence。

- CRL critic/actor computation N=20 strict parity：`first_divergence=None`，
  所有 tracked maximum errors `0.0`；
- 完整 AWR five-slot N=20 strict parity：`first_divergence=None`，所有
  tracked maximum errors `0.0`。

比较项包括 critic/value/actor/total losses、semantic parameters、optimizer
state 和 agent RNG。

### CRL CPU regression

CRL M5 regression：`32/32 PASS`。  
CRL M6 full AWR regression：`38/38 PASS`。  
加入 CoGHP 后今天完整项目回归：`41/41 PASS`。

### CRL GPU 1000-step smokes

DDPG+BC：

| Mode | Final aggregate loss | Result |
|---|---:|---|
| Legacy CRL | `2.308672` | finite/evaluation/logging/checkpoint PASS |
| Actor + critic-state + critic-goal computation | `2.302007` | finite/evaluation/logging/checkpoint PASS |

AWR：

| Mode | Critic | Value | Actor | Total |
|---|---:|---:|---:|---:|
| Legacy AWR | `0.384524` | `0.383256` | `8.873828` | `9.641608` |
| Full five-slot computation AWR | `0.377802` | `0.407084` | `8.666108` | `9.450995` |

两种 AWR 模式均完成 finite updates、evaluation、CSV logging 和 checkpoint
save/load action/value probe。短 smoke 的 task success 为 `0.0`，不作为
算法性能结论。

## 二、Vanilla CoGHP 官方来源

唯一 source of truth 是：

- 仓库：[`wlsdn9350/CoGHP`](https://github.com/wlsdn9350/CoGHP)
- owner：`wlsdn9350/CoGHP`
- branch：`main`
- 审计 commit：`8f362e9f86bf97fdbc9ce36d1b7b73b024e18b36`

官方审计覆盖：

- `impls/agents/coghp.py`
- `impls/utils/coghp_network.py`
- `impls/utils/datasets.py`
- `impls/utils/networks.py`
- `impls/utils/encoders.py`
- `impls/main.py`

详细审计见 [`docs/coghp_official_migration_audit.md`](coghp_official_migration_audit.md)。

## 三、Vanilla CoGHP 迁移与 M7 收尾

### 1. Vanilla CoGHP agent 迁移

新增 `impls/agents/coghp.py`，接入官方：

- `goal_rep`；
- 双头 `value` 与独立 `target_value`；
- `actor_mixer`；
- value expectile loss；
- high actor 多 subgoal AWR loss；
- low actor AWR loss；
- `high_discount`、`subgoal_steps`、`num_subgoals`；
- teacher forcing；
- autoregressive inference；
- Polyak target update；
- checkpoint/evaluation 所需的 `sample_actions` 接口。

官方模块名称和职责保持不变：

```text
goal_rep
value
target_value
actor_mixer
```

### 2. 官方 Mixer network 迁移

新增 `impls/networks/coghp.py`，将官方原始文件
`impls/utils/coghp_network.py` 在 RLC 中归类到更合理的 `networks` 层。

`MixerBlock` 保持官方实现：

- token mixing：`token_dense1 -> GELU -> token_dense2`；
- trainable `tm_weights`；
- lower-triangular causal mask；
- token residual；
- channel mixing：`channel_dense1 -> GELU -> channel_dense2`；
- channel residual；
- 无 LayerNorm、无额外 pre/post normalization。

当前 effective token count 为：

```text
num_tokens = num_subgoals + 3
```

当前 token replacement 依赖隐含 invariant：

```text
joint_embed_dim == state_dim
```

### 3. 参数共享关系确认并保留

官方 `actor_mixer` 只创建一份 `mixer_blocks` 列表，在所有
autoregressive token steps 中重复使用。因此：

- subgoal prediction 和 final action 共享同一物理 Mixer core；
- high actor 与 low actor 只有独立 readout heads；
- `prev_tokens` 只有一份；
- `feature_embed` 只有一份；
- 不存在 high/low 两套 Mixer core；
- autoregressive loop 不会复制参数。

high-level sampling 复用同一个 `high_seed`，保持官方行为，未改为每一步独立
seed。

官方 forward 接口事实也已记录：

- `subgoal_reps` 实际参与 teacher forcing；
- `action_seq` 仅为接口兼容参数，当前 forward 不使用；
- `init_scale`、`decay_alpha` 为 inactive/compatibility fields，当前不参与计算。

### 4. MultiHGCDataset 与 runtime 接入

在 `impls/utils/datasets.py` 新增 `MultiHGCDataset`，保持官方字段和采样语义：

- `value_goals`；
- `low_actor_goals`；
- `high_actor_goals`；
- `high_actor_targets`，形状为 `(B, num_subgoals, state_dim)`；
- `rewards`、`masks`、`next_observations`；
- trajectory boundary 截断；
- future/random goal 分支；
- `subgoal_steps` 和 `num_subgoals` target offsets。

同时保留 RLC 已有的显式 NumPy `Generator` RNG contract，没有回退到官方的
process-global NumPy RNG。

CoGHP 已注册到 shared runtime：

```text
impls.main
  -> agent registry
  -> MultiHGCDataset
  -> CoGHPAgent
  -> update
  -> evaluation
  -> CSV logging
  -> checkpoint save/load
```

没有新增单独 trainer。`--computation` 对 vanilla CoGHP 明确不适用，避免把
官方 Mixer core 误解释为 computation slot。

### 5. Encoder、配置与 checkpoint 边界

完成了：

- `GCEncoder(listwise=True)` 支持 actor mixer 的官方输入路径；
- `configs/agents/coghp.py`；
- agent/config registry；
- shared loss metric aggregation；
- CoGHP checkpoint probe 的 batch-to-unbatched observation 处理。

## 参数审计

使用小型配置：

```text
feature_dim=4
num_subgoals=2
num_mixer_blocks=2
actor_hidden_dims=(6,)
value_hidden_dims=(6,)
enc_hidden_dims=(5,)
```

参数统计：

| 模块 | 参数标量 |
|---|---:|
| `modules_goal_rep` | 79 |
| `modules_value` | 146 |
| `modules_target_value` | 146 |
| `modules_actor_mixer` | 361 |
| 总计 | 732 |

`actor_mixer` 内部：

| 子模块 | 参数标量 |
|---|---:|
| `prev_tokens` | 12 |
| `feature_embed` | 59 |
| `mixer_blocks_0` | 94 |
| `mixer_blocks_1` | 94 |
| `high_actor_head` | 58 |
| `low_actor_head` | 44 |
| 合计 | 361 |

autoregressive 重复调用没有被错误计为额外参数。

## Parity 与 regression 验证

### Synthetic official/RLC parity

同一 seed、同一配置和同一 batch：

- 初始化 parameter tree：51/51 leaves，最大误差 `0.0`；
- `goal_rep`、`value`、`target_value`：最大误差 `0.0`；
- high/low distribution、mode、sample：最大误差 `0.0`；
- value/high actor/low actor loss：最大误差 `0.0`；
- diagnostics：最大误差 `0.0`；
- online/target parameter update：语义一致；
- optimizer state 与 agent RNG：一致。

已有 RLC `TrainState` 和官方 `TrainState` 对 `grad/norm` 的聚合顺序不同，
因此该非语义诊断字段不作为严格算法 parity 条件。

### Real OGBench N=20

环境：

```text
antmaze-medium-navigate-v0
```

使用真实 OGBench `MultiHGCDataset`、匹配的参数初始化、target、Adam、batch
和 RNG：

- required dataset fields：20/20 bitwise parity；
- loss 最大误差：`7.62939453125e-06`；
- online parameters 最大误差：`1.4901161193847656e-08`；
- target value 最大误差：`2.9103830456733704e-11`；
- optimizer state 最大误差：`1.1920928955078125e-07`；
- agent RNG 最大误差：`0.0`；
- 最终参数保持 finite。

差异属于 float32/reduction 数值噪声，没有观察到网络、loss、dataset、target
update 或 RNG 的语义 divergence。

### CoGHP 专项 CPU tests

```text
tests/integration/test_coghp_runtime.py: 3/3 PASS
```

覆盖：

- 官方 Mixer 参数共享与参数计数；
- high/low head 独立性；
- `MultiHGCDataset` 20 次同 seed deterministic sampling；
- autoregressive policy finite；
- update finite；
- 参数树 finite。

### 完整 CPU regression

```text
41/41 PASS
```

既有 HIQL/CRL regression 全部保持通过，新增 CoGHP regression 也全部通过。

### Shared runtime smoke

真实数据 shared runtime smoke 完成：

- update；
- evaluation；
- CSV logging；
- checkpoint save/load；
- action/value restore equality probe。

### GPU 1000-step smoke

配置：

```text
CUDA backend
antmaze-medium-navigate-v0
batch_size=256
actor/value hidden width=512
depth=3
train_steps=1000
```

结果：

- 1000 steps 完成；
- 10 个 logging rows 的 training/validation/gradient diagnostics 全部 finite；
- step 1000 value loss：`0.3899533`；
- step 1000 high actor loss：`24.7590027`；
- step 1000 low actor loss：`75.8899841`；
- evaluation 完成，`task1_success=0.0`；
- checkpoint save/restore action/value probe：PASS；
- 未发现 NaN/Inf。

该 smoke 只验证运行稳定性和 runtime 闭环，不解释短训练 success rate。

## 相关文件

### Production code

- `impls/agents/crl.py`
- `impls/agents/coghp.py`
- `impls/agents/__init__.py`
- `impls/networks/common.py`
- `impls/networks/coghp.py`
- `impls/utils/datasets.py`
- `impls/utils/encoders.py`
- `impls/main.py`
- `configs/agents/coghp.py`

### Tests

- `tests/integration/test_crl_runtime.py`
- `tests/integration/test_coghp_runtime.py`

### Documentation

- `docs/coghp_official_migration_audit.md`
- `docs/coghp_vanilla_migration.md`
- `docs/crl_migration.md`
- `docs/crl_critic_migration.md`
- `docs/crl_awr_value_migration.md`
- `docs/architecture_decisions.md`
- `docs/status.md`
- `README.md`

## 明确未完成事项

- 没有进行长时间训练；
- 没有进行多 seed 实验；
- 没有进行正式 success-rate baseline comparison；
- 没有将 `MixerBlock` 移入 `impls/computation/`；
- 没有新增 Mixer primitive/topology；
- 没有开始 HRM、SwiGLU/RMS 或其他新实验；
- 没有改变任何现有 HIQL/CRL 算法行为。

## 下一步边界

如果未来正式将 MLP-Mixer 纳入统一 computation framework，届时再评估将
`MixerBlock` 从 `impls/networks/coghp.py` 移入
`impls/computation/` 下通用 primitive/block 层。本 milestone 不提前进行
该结构调整。
