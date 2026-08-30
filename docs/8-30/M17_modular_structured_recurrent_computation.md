# M17：模块化 Structured Recurrent Computation Framework

日期：2026-08-30  
性质：computation framework 与 correctness milestone；不是性能实验。  
状态：实现、专项 parity、真实 Puzzle tiny lifecycle 和规定的回归测试已完成；未启动任何正式 M17 training，未执行 Git 操作。

## 1. 修改前审计结论

M17 实现前实际审计了 `impls/computation/`、`impls/representation/`、`impls/networks/common.py`、slot registry、GCIQL construction、runtime/config/checkpoint/accounting，以及 M15/M16A/M16B/M16C 的 tests、Study 和文档。

当时的代码事实是：

- `PuzzleStructuredBody` 同时实现 raw input split、button token projection/index embedding、Mixer stack、mean readout、robot projection 和 fusion；
- `factory.py` 将 `structure=puzzle_tokens` 硬编码为 `feedforward + mlp_mixer`，拒绝 token SingleState；
- `SingleState` 只创建内建 MLP `input_mapping` 与 MLP `update_module`；
- M15/M16 的 `num_mixer_blocks` 已经表示 Mixer stack depth，但没有与 recurrent execution budget 分离；
- legacy structured accounting 只描述这个 monolithic FF body；
- M16 当前主线配置均为 `FeedForward`，`L=num_mixer_blocks`，是 M17 legacy→modular FF parity 的适当 oracle。

当前实现以这些实际事实为基础。没有改动 M15/M16 的历史 run、checkpoint 或冻结 worktree。

## 2. 最终架构

```text
raw flat Puzzle input
        │
        ▼
PuzzleTokenAdapter
  └─ StructuredRepresentation(tokens, context, mask, auxiliary)
        │                       canonical internal tokens: [B,T,D]
        ▼
StructuredComputationBody
  ├─ ComputationCore
  │    ├─ FeedForward(MLPMixerStack(L))
  │    └─ SingleState(update_block=MLPMixerStack(L), K)
  └─ MeanContextReadout
        │
        ▼
algorithm-compatible vector
        │
        ▼
existing GCActor / GCValue head and unchanged algorithm loss
```

`StructuredComputationBody` 是 domain-agnostic orchestration layer，不是 Puzzle-specific topology：adapter 决定 raw input 的 representation geometry；block 决定单个 transformation unit；topology 决定执行方式；readout 决定 token state 如何成为算法向量。

它也是唯一的 batching normalization 边界：单条 observation 时 adapter 产生 `[T,D]` / `[C]`，body 临时提升为 `[1,T,D]` / `[1,C]`，完成 core/readout 后恢复 `[H]`。Mixer、SingleState、adapter 和 readout 内部均不再各自 expand/squeeze。

## 3. 数学 contract：L 与 K

令 adapter 为

```text
(X, c) = A_phi(x)
X in R^(B×T×D)
```

单个 Mixer layer 为 `M_l`，深度为 `L` 的 block unit 是

```text
B_theta^(L)(X) = M_theta_L(...M_theta_2(M_theta_1(X)))
```

其中一个 block unit 内的 L 层参数互不共享。

FeedForward：

```text
Z = B_theta^(L)(X)                  # K = 1
h = Readout(Z, c)
```

structured SingleState：

```text
Z^0 = 0
Z^(k+1) = B_theta^(L)(Z^k + X)      # k = 0, ..., K-1
h = Readout(Z^K, c)
```

M17 canonical structured SingleState 固定为：

- `input_mapping=identity`；
- `state_init=zero_buffer`（也保留 `normal_buffer` 能力）；
- `input_injection=z_plus_x`；
- `residual=false`；
- `parameter_sharing=shared`；
- `credit=direct`。

因此 `SingleState(L, K=1) == FeedForward(L)`。K>1 时是 repeated input injection，而不是简单的 `B(B(...B(X)))`，也不能解释为纯 parameter-sharing effect。

`residual=false` 只指 topology transition 不做 `Z^k + update`。每一个 `MLPMixerBlock` 内部仍保留 token-mixing residual 与 channel-mixing residual；两种 residual 的语义完全不同。

## 4. 参数 ownership

| 层 | 拥有的参数 | 明确不拥有 |
|---|---|---|
| `PuzzleTokenAdapter` | shared button projection、absolute index embedding、robot/context projection、可选 context LayerNorm | Mixer、iteration、token readout、fusion |
| `MLPMixerStack(L)` | L 个互不共享的 `MLPMixerBlock` | Puzzle parsing、context、readout、K copies |
| `FeedForward` / `SingleState` | topology execution逻辑；SingleState 的 non-trainable `z_init` buffer | domain-specific tokenization、fusion |
| `MeanContextReadout` | fusion Dense、可选 final LayerNorm | Puzzle parse、Mixer、recurrent state |
| GCIQL head | action mean 或 scalar readout | structured body internals |

对 critic，action 仍只拼入 adapter 的 robot/context branch；button tokens 只来自 state/goal button features，绝不从 trailing action dimension 猜测 token 输入。

## 5. 配置语义

当前 `ComputationSpec` 保留历史字段 `primitive`、`block`、`topology`、`credit`、`structure` 等，并新增 `readout` / `readout_kwargs`。M15/M16 的 `structure_kwargs.num_mixer_blocks` 仍兼容；新配置也可将 L 写在 `block_kwargs.num_blocks`。

FF-L2 的解析语义示例：

```yaml
enabled: true
primitive: mlp
structure: puzzle_tokens
structure_kwargs:
  num_buttons: 16
  robot_dim: 19
  button_feature_dim: 4
  token_dim: 128
  robot_hidden_dim: 128
  index_embedding: true
block: mlp_mixer
block_kwargs:
  num_blocks: 2             # L
  token_hidden_dim: 64
  channel_hidden_dim: 256
  tm_mode: none
topology: feedforward        # implied K=1
credit: direct
readout: mean_context
readout_kwargs:
  output_dim: 512
```

SS-L2-K4：

```yaml
enabled: true
primitive: mlp
structure: puzzle_tokens
structure_kwargs: {num_buttons: 16, token_dim: 128, robot_hidden_dim: 128, index_embedding: true}
block: mlp_mixer
block_kwargs: {num_blocks: 2, token_hidden_dim: 64, channel_hidden_dim: 256, tm_mode: none}
topology: single_state
topology_kwargs:
  iterations: 4             # K
  input_mapping: identity
  state_dim: 128
  state_init: zero_buffer
  state_init_std: 1.0
  input_injection: z_plus_x
  residual: false
  parameter_sharing: shared
parameter_sharing: shared
credit: direct
readout: mean_context
readout_kwargs: {output_dim: 512}
```

`_make_config` 会将 active structured slot 的 readout 和 SingleState defaults materialize 到 resolved runtime config；runtime metadata 还记录 structure、block、L、topology、K、mapping、init、injection、residual、readout 和 credit。

## 6. Accounting

M17 的 modular accounting 从真实 parameter tree 分别报告：

- adapter：token/context projection params 与 MAC、index embedding params；
- computation block：`L`、physical/unique Mixer layers、每次 execution 的 Mixer MAC；
- topology：`K`、executed Mixer layers 和 MAC；
- readout：fusion params/MAC；
- total：trainable params、per-sample Dense MAC、physical/执行 sequential depth、non-trainable buffers。

一个 Mixer layer 有四个 Dense transformations，所以每个单路径：

```text
unique Mixer Dense layers   = 4L
executed Mixer Dense layers = 4LK
```

以完整 M16A Puzzle-4x4 S002 actor-sized structured body（`T=16,D=128,H_T=64,H_D=256,L=2`，output width 512）为例。`params` 是完整 actor slot（含 action head）；MAC/depth 是 structured body 的 per-sample 值，buffer 不计入 trainable params：

| configuration | params | zero/normal buffer elems | unique Mixer layers | executed Mixer layers | body Dense MAC | executed depth |
|---|---:|---:|---:|---:|---:|---:|
| FF-L2 / K=1 | 278,437 | 0 | 2 | 2 | 2,773,760 | 10 |
| SS-L2 / K=1 | 278,437 | 128 | 2 | 2 | 2,773,760 | 10 |
| SS-L2 / K=2 | 278,437 | 128 | 2 | 4 | 5,395,200 | 18 |
| SS-L2 / K=4 | 278,437 | 128 | 2 | 8 | 10,638,080 | 34 |

这验证参数量不随 K 增长，而 executed Mixer MAC/depth 随 K 增长。critic ensemble 的 physical counts 会包含其两个独立成员；每个成员的 L/K 语义不变。

## 7. Legacy migration 与 parity

legacy `PuzzleStructuredBody` 被**保留为 reference oracle，但不再是 factory 的 active production branch**。历史 M15/M16 frozen commits 和 run artifacts 仍是历史 checkpoint 的复现路径；M17 不承诺将新的 module tree 直接 restore 到旧 structured checkpoint。

专项 parity 已对 legacy body 与 modular FF 做 semantic parameter transplant：

```text
legacy.button_projection        -> modular.adapter.button_projection
legacy.index_embedding          -> modular.adapter.index_embedding
legacy.robot_projection         -> modular.adapter.robot_projection
legacy.mixer_blocks_i           -> modular.core.FeedForward.stack.blocks_i
legacy.fusion                   -> modular.readout.fusion
```

结果：

- actor/value 和 critic action-context input 的 output shape 一致；
- legacy 与 modular FF parameter count 相等；
- transplant 后 forward max difference 在 float32 tolerance 内；
- 对 button projection、index embedding、context projection、每层 Mixer 和 fusion 的梯度逐语义组件一致；
- `f(x) == f(x[None])[0]` 对 actor/value/critic 成立；
- critic 改变 action 时 tokens 不变、context 改变；
- FF(L) 与 SS(L,K=1) 对 `L=1,2,4` 的 computed tokens、最终输出、梯度、trainable params、Mixer MAC 和 executed depth 都一致。

## 8. 验证结果

| 验证项 | 结果 | 覆盖 |
|---|---:|---|
| M17 synthetic/integration | PASS，7 tests | representation ownership、legacy transplant、batching、critic action separation、FF↔SS parity、K invariance、state init、direct gradient、GCIQL FF/SS update |
| M17 real Puzzle smoke | PASS，1 test（FF/SS 两个 subtests） | `puzzle-4x4-play-v0`，GCIQL，真实 batch、2 updates、finite metrics、action、checkpoint save/restore、1 eval episode |
| M15 structured integration | PASS，12 tests | legacy oracle tests、all base-agent structured construction、token accounting |
| M15 real smoke | PASS，1 test（四算法 subtests） | Puzzle-3x3 GCIQL/GCBC/GCIVL/QRL lifecycle |
| computation discovery | PASS，53 tests | vector FeedForward、SingleState、TwoState、M12B structure/credit regressions |
| M13/M14/CRL/HIQL | PASS，38 tests | canonical agent, slot, checkpoint and runtime regression |
| CoGHP runtime | PASS，3 tests | 以 EGL 环境运行 |
| checkpoint / management / computation provenance | PASS，18 tests | semantic best/last、restore、Run metadata、provenance regression |
| M16A/B/C Study configs | PASS，4 tests | 矩阵、alpha runtime resolution、endpoint contract |
| `compileall` | PASS | `impls` 与新增 M17 tests |

terra 上无可用 X11 display；一次混合 regression batch 在 CoGHP 初始化阶段触发 GLFW/X11 错误。拆分后以 `MUJOCO_GL=egl PYOPENGL_PLATFORM=egl` 运行，CoGHP 及所有需要 environment 的 suites 均通过。这是 host display configuration 要求，不是 M17 assertion failure。

## 9. Adaptive test-time computation readiness

1. **固定 L 时 K 不改变 parameter shape**：SS K1/K2/K4 test 已验证相同 trainable parameter count 和单个 `update_module` subtree。
2. **单步与总 K 解耦**：`SingleState.step(z, x_hidden, update_module=None)` 不读取最终 iteration budget。
3. **state shape invariant**：identity mapping、`z_init:[D]` broadcast 后每步均为 `[B,T,D]`。
4. **任意 intermediate state 可 readout**：`MeanContextReadout(tokens, context, mask)` 接受每一个合法 `[B,T,D]` state；没有依赖 final K。
5. **future scheduler 的替换位置明确**：只需控制 `step` 是否再次执行，不需改 adapter、Mixer stack、readout 或 algorithm head。
6. **parameter shapes 不依赖 K**：block unit 在 setup 中只构建一次，structured path 强制 shared；不会构建 `update_block_0`, `update_block_1`, ...。

本轮没有实现 ACT、halting、动态 K、test-time extrapolation 或任何 adaptive loss。

## 10. 修改文件与兼容性

| 文件 | 修改 | 旧行为影响 |
|---|---|---|
| `impls/representation/interfaces.py` | 新增 parameter-free `StructuredRepresentation` | 新文件 |
| `impls/representation/puzzle.py` | 新增 `PuzzleTokenAdapter`；parser 保持 pure | parser 语义不变 |
| `impls/computation/readouts.py` | 新增 `MeanContextReadout` | 新文件 |
| `impls/computation/structured.py` | 新增 generic `StructuredComputationBody`；保留 legacy body | legacy class 未删除 |
| `impls/computation/blocks/mlp_mixer.py` | 新增 `MLPMixerStack(L)` | `MLPMixerBlock` 数学不变 |
| `impls/computation/topologies/single_state.py` | external generic update block、identity mapping、`step` abstraction | 默认 vector MLP tree/semantics 经 regression 保持 |
| `impls/computation/factory.py` / `slots.py` | modular Puzzle FF/SS construction 与 validation | active Puzzle factory 由 legacy body 切换为 modular path |
| `impls/computation/accounting.py` | modular adapter/block/topology/readout L/K accounting | legacy accounting helper 保留 |
| `impls/main.py` | materialize structured resolved defaults，record L/K/readout | 仅 active structured slots 增加 provenance fields |
| `tests/integration/test_m17_*` | 新增 parity/real smoke | 新测试 |
| `tests/integration/test_m15_puzzle_structured_computation.py` | 更新 active module-tree expectation；保留 legacy oracle tests | M15 historical assertion改为 M17 production layout |

## 11. 明确未做与剩余风险

- 未启动 M17 正式训练，未选择 L/K/alpha/task/seed matrix；
- 未改变 GCIQL objective、DDPG+BC alpha、value/critic target、Polyak、dataset 或 goal sampling；
- 未重构 frozen primitive MLP，也未迁移 generic TwoState Mixer；
- 未实现 Cube/Scene representation、attention/CLS/robot-query readout、HRM、ACT 或 adaptive halting；
- 新 structured parameter tree 与历史 M15/M16 checkpoint tree 不同；历史应由 frozen commit 复现，而非跨架构 restore；
- pytest-specific suite 仍不在本次验证范围内；没有安装新依赖；
- future adaptive K 仍需要独立的 scheduler/halting scientific design、loss and evaluation protocol，M17 只提供正确的 architectural seam。

M17 的结论仅限于 framework correctness：representation geometry、block、topology、budget 和 readout 已解耦，并在严格 parity、accounting、checkpoint 和 real-data lifecycle gate 下验证。它不提供任何性能 superiority 结论。
