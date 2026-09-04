# M19A：Puzzle Entity Factorization Isolation

## 科学问题

M19A 补齐 M16B 的 Flat MLP 与 Puzzle-token MLP-Mixer L2 之间缺失的中间对照：

1. Flat MLP（M16B historical anchor）；
2. Entity Token + Entity-wise MLP（M19A 唯一新 formal condition）；
3. Entity Token + MLP-Mixer（M16B historical anchor）。

问题不是“MLP 与 Mixer 哪个更强”，而是在固定 button-token representation、absolute index embedding、robot/context branch、mean-context readout 与 GCIQL 语义后，Mixer 的显式 pre-pooling cross-token interaction 是否提供了 Entity-wise computation 之外的描述性收益。

## 为什么只有一个新 formal run

新 run 仅为 `M19A-4x4-E001`：Puzzle-4×4、seed 0、GCIQL、alpha=1.0、actor+value+critic、feedforward、EntityMLP L=2。Flat 与 Mixer 不重新训练，而是复用完成的 M16B alpha=1.0 产物；这避免把新的 seed、训练时间或代码变动混入本轮的中间对照。

不在 M19A 中新增其他 Puzzle 尺寸、seed、alpha、Flat/Mixer rerun、parameter/MAC-matched EntityMLP、alternate readout 或 recurrent variant。

## M16B 锚点身份

| Method | Study / config | 实际 run path | 状态 | source commit |
|---|---|---|---|---|
| Flat MLP | `M16B-4x4-B000` | `/data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M16B/M16B-4x4-B000__4x4_flat_alpha1p0/puzzle-4x4-play-v0/seed_000` | `completed` | `1eb3ac0f7ef40773bad5f0015d4fe4f490d4de6b` |
| Entity Token + MLP-Mixer | `M16B-4x4-S002` | `/data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M16B/M16B-4x4-S002__4x4_mixer_l2_alpha1p0/puzzle-4x4-play-v0/seed_000` | `completed` | `1eb3ac0f7ef40773bad5f0015d4fe4f490d4de6b` |

两者均为 Puzzle-4×4、seed 0、GCIQL、alpha=1.0、1M steps、batch 1024、每 100k 评估一次、5 个任务各 20 episode。M19A E001 严格匹配这些非-block 条件。

## EntityMLP 的精确定义

对 token tensor `H ∈ R^(B×T×D)`，每个 block 仅在最终 channel axis 上做共享变换：

```text
U_i = W1 H_i + b1
V_i = GELU(U_i)
R_i = W2 V_i + b2
H_next_i = H_i + R_i
```

其中 `W1 ∈ R^(D×H_D)`、`W2 ∈ R^(H_D×D)` 在所有 entity/token 位置 `i` 间共享。Dense initializer 复用当前 `MLPMixerBlock` channel branch 的 `default_init()`；第二个 Dense 后没有额外 activation，block 内没有 LayerNorm。

`EntityMLPStack(num_blocks=L)` 顺序含有 L 个彼此 **untied** 的 `EntityMLPBlock`。M19A 固定 `L=2`；stack 不知道 Puzzle/button/robot/GCIQL，也不持有 `num_tokens` 或 token-axis 参数。

## 与 Mixer 的精确差异

当前 Mixer block 为：

```text
H' = H + TokenMLP(H)
H_next = H' + ChannelMLP(H')
```

EntityMLP 是其 channel branch 的 surgical removal control：

```text
H_next = H + ChannelMLP(H)
```

因此它没有 `token_dense1`、`token_dense2`、transpose、token-axis einsum、`tm_weights`、attention、token convolution 或 token-specific parameter copy。它保留 channel Dense dimensions、GELU、residual 和 L 的 stack 语义。

没有直接复用 canonical vector `MLP`，因为 canonical MLP 的层数/激活/残差语义并不等于 Mixer 的 channel branch；将其作为 token block会引入额外结构差异，不能隔离 token branch。

## Representation、context 与 readout

M19A 不修改 `PuzzleTokenAdapter` 与 `MeanContextReadout`：

- button token 输入仍为 `[b_i^state, b_i^goal]`，经 shared button projection 加 absolute index embedding；
- actor/value context 仍为 `[robot_state, robot_goal]`；
- critic action 仍只追加到 robot/context branch，绝不 broadcast 到 tokens；
- token 仍做 mean pooling，再与 context concat 后经过既有 fusion Dense、activation/LayerNorm 语义。

因此 EntityMLP 不是严格 permutation-invariant DeepSets：absolute index embedding 保留 entity identity。合适描述是“带 absolute entity identity embedding、无 token interaction 的 shared entity-wise MLP”。

## 工厂接入与限制

`structure=puzzle_tokens` 现保留原有 `block=mlp_mixer` 路径，并新增严格的 `block=entity_mlp` 分支：

```text
PuzzleTokenAdapter
→ FeedForward(EntityMLPStack(L))
→ MeanContextReadout
```

EntityMLP 只允许：

- `structure=puzzle_tokens`；
- `topology=feedforward`；
- `credit=direct`；
- `readout=mean_context`；
- `block_kwargs={num_blocks: 2, channel_hidden_dim: 256}`（M19A exact config）。

`single_state`、`two_state`、`one_step`、alternate readout、token-mixing kwargs、token-axis parameters 都 fail fast。原始 `MLPMixerBlock` 文件、其 normal forward 和 parameter tree 均未修改。

## Accounting

新增 generic fields：`block_type`、`block_depth_L`、`iterations_K`、`token_interaction`、token/channel mixing params 与 MAC、computation block params/MAC、unique/executed block Dense layers，以及 adapter/readout totals。旧 Mixer accounting fields 仍保持旧语义；EntityMLP 不把 channel 参数伪标为 `mixer_params`。

对 `T=16, D=128, H_D=256, L=2`：

```text
per EntityMLP block params = D·H_D + H_D + H_D·D + D = 65,920
EntityMLP L2 params (actor/value) = 131,840
per-sample L2 block MAC = 2·L·T·D·H_D = 2,097,152
structured body Dense depth = 1 + 2L + 1 = 6
```

相对同一 M16B Mixer L2 (`H_T=64`) 删除的 token branch 为：

```text
params = L·[(T·H_T + H_T) + (H_T·T + T)] = 4,256
MACs   = 2·L·D·T·H_T = 524,288
```

上述差异对于 critic ensemble 的物理树乘以 2。整个 GCIQL actor/value/critic accounting：Entity 为 1,095,688 params、9,003,264 Dense MAC；Mixer 为 1,112,712 params、11,100,416 Dense MAC，差值分别为 17,024 与 2,097,152。它们不是 parameter/MAC-matched comparison。

## 正确性 gates

1. pooling 前 cross-token perturbation：修改 token `j` 后，所有 `i≠j` 的 EntityMLPStack output 在 float32 tolerance 内不变；
2. JAX Jacobian：EntityMLP token-to-token Jacobian 的 off-diagonal block 为零；
3. Mixer positive control：确定性非退化 token branch 下至少一个 off-diagonal derivative 非零；
4. surgical parity：将 Mixer token branch 置零、transplant channel params 后，EntityMLP forward 和对应 channel gradients 一致；
5. L=2 的两个 blocks untied、token positions 共享同一组 channel parameters，且没有 16 份 token copies；
6. GCIQL actor/value/critic、target update、action sampling、真实 Puzzle two-update lifecycle 均通过。

## Anchor-reuse compatibility audit

`tools/m19a_doctor.py` 检查两个 source run 的 completed status、env/seed/alpha/algorithm/dataset、resolved training/evaluation protocol、B000 Flat 语义、S002 Mixer L2 语义、final@1M row、E001 的非-block 匹配、当前 Mixer accounting 与历史 artifact 一致、Entity token independence 与实际树的 parameter/MAC/depth 公式。

锚点与 M19A formal worktree 可以跨 commit 复用，记录为 `cross_commit_anchor_reuse=true`；这不是 same-commit 声明。复用条件是：GCIQL 训练语义未改、M17 legacy→modular Mixer parity 回归通过、M18-D 仅为 diagnostics、M19A 未改 Flat/Mixer normal path，且 M16B/M17/M18 回归通过。

## Formal config 与指标

`M19A-4x4-E001` 的 resolved semantics：alpha=1.0、GCIQL DDPG+BC、Puzzle-4×4、seed 0、`actor/value/critic` 全部 `puzzle_tokens + entity_mlp + feedforward + direct + mean_context`，token dim 128、channel hidden 256、L=2、index embedding=true、output dim 512。

主终点为 `evaluation/overall_success` 的 final@1M；secondary 为 best、best step、last-3 mean、100k–1M 10 个点的 normalized trapezoidal AUC。分析器还保留 5 个 task 的 final success，并定义：

```text
Delta_structured_factorization_package = J(EntityMLP) - J(FlatMLP)
Delta_added_token_mixing_branch        = J(MLPMixer) - J(EntityMLP)
```

前者不是 pure causal factorization effect，因为 Flat→Entity 还同时改变 token encoder、index embedding、robot/entity separation、mean pooling 与 fusion geometry；后者仍包含额外参数、MAC 与 depth。

## 解释边界与延后工作

本轮仅 seed 0、使用 historical cross-commit anchors、且预算不匹配。因此不得声称统计显著性、因果证明、通用规律、compute efficiency、SOTA 或跨 Puzzle size 泛化。

如 M19A 显示需要进一步确认，后续候选为 pre-registered parameter-matched 或 MAC-matched EntityMLP control；但它们明确不属于本轮，且不得根据 E001 中途调 alpha/width/depth/readout。
