# M15：Puzzle Entity-Structured Token Computation with MLP-Mixer

日期：2026-08-26  
仓库：`/home/eai/Research/RLC`  
性质：结构化 computation infrastructure milestone，不是正式科研实验。  
状态：M15 结构化实现、专项测试、历史回归和真实数据 tiny smoke 已完成；Git 审查、commit 和 push 仍由用户手动完成。

## 1. Goal

M15 在 M13/M14 的 canonical base algorithms 和 computation-slot abstraction 之上，为标准 OGBench Puzzle state observation 增加 entity-structured token computation：

```text
flat Puzzle observation
    -> robot/button split
    -> shared button projection + absolute button index embedding
    -> [B, T, D] MLP-Mixer stack
    -> mean token readout
    -> robot/button fusion
    -> slot-required output width
```

M15 没有改变 GCBC、GCIQL、GCIVL、QRL 的 loss、target update、operator 或 dynamics equation。

## 2. Scope boundary

本轮实现了：

- 严格标准 Puzzle flat observation parser；
- robot/button 分离；
- shared button-local projection；
- learned absolute button index embedding；
- `[B,T,D]` token contract；
- feed-forward token/channel MLP-Mixer；
- 可配置 Mixer depth、mean readout、robot/button fusion；
- 统一 `ComputationSpec.structure='puzzle_tokens'` factory 路径；
- GCIQL actor/value/critic primary integration；
- GCBC actor、GCIVL actor/value、QRL actor/phi integration；
- token-aware parameter/MAC/depth accounting；
- synthetic tests、四种 Puzzle 尺寸 initialization smoke、真实 Puzzle-3x3 tiny lifecycle smoke。

明确没有实现：

- HRM、token recurrence、recurrent Mixer；
- attention、Transformer、RMSNorm、SwiGLU、ACT；
- grid/row/column embedding、邻接矩阵、GNN、XOR 或任何显式 Puzzle geometry prior；
- robot token、global token、action token、privileged button state 或 oracle input；
- 1M training、seed sweep、parameter-matched formal experiment 或 success-rate scientific claim。

## 3. Files

新增：

- `impls/representation/__init__.py`：representation public exports；
- `impls/representation/puzzle.py`：严格 Puzzle observation parser；
- `impls/computation/structured.py`：`PuzzleStructuredBody`；
- `tests/integration/test_m15_puzzle_structured_computation.py`：M15 synthetic/integration tests；
- `tests/integration/test_m15_puzzle_real_smoke.py`：真实 OGBench tiny lifecycle smoke；
- `tools/m15_static_accounting.py`：默认候选结构的静态 Flat/Mixer accounting 生成器；
- `docs/milestones/M15_puzzle_token_mixer.md`：本报告。

修改：

- `impls/computation/blocks/mlp_mixer.py`：增加 `tm_mode`，默认保持 `lower_triangular`；
- `impls/computation/factory.py`：扩展 `ComputationSpec` 与结构化 factory；
- `impls/computation/slots.py`：增加 input/action semantics 和 Puzzle 结构约束；
- `impls/computation/accounting.py`：增加 token-aware accounting；
- `impls/computation/__init__.py`：导出 structured body/accounting；
- `impls/networks/common.py`：对 structured raw-observation 要求做 construction-time 校验；
- `impls/main.py`：记录 structure、semantics 和结构化 accounting metadata。

## 4. Puzzle observation layout and parser

只接受标准 state observation：

```text
x = [robot_state (19), button_0 (4), ..., button_(N-1) (4)]
```

`parse_puzzle_observation(x, num_buttons=N, robot_dim=19, button_feature_dim=4)` 检查：

```text
x.shape[-1] == 19 + 4N
```

并返回：

```text
robot:   [..., 19]
buttons: [..., N, 4]
```

对应尺寸为：

| 环境 | N | flat observation | buttons |
|---|---:|---:|---|
| Puzzle-3x3 | 9 | 55 | `[..., 9, 4]` |
| Puzzle-4x4 | 16 | 83 | `[..., 16, 4]` |
| Puzzle-4x5 | 20 | 99 | `[..., 20, 4]` |
| Puzzle-4x6 | 24 | 115 | `[..., 24, 4]` |

parser 不读取 `dataset['button_states']`、环境 private field、oracle observation 或其他 privileged field；不做 malformed-dimension guessing、frame stacking 或 pixel parsing。

## 5. Token contract and index embedding

对 actor/value 的 paired goal mode：

```text
state buttons: [..., T, 4]
goal buttons:  [..., T, 4]
concat:        [..., T, 8]
projection:    shared Dense(8, token_dim)
tokens:        [B, T, token_dim]
```

对 QRL phi 的 single-observation mode：

```text
buttons:       [..., T, 4]
projection:    shared Dense(4, token_dim)
tokens:        [B, T, token_dim]
```

button projection 是一个共享的 Dense，不会为每个 button 创建 N 个 projection。每个 token 加上同一个 body 内的 learned absolute index parameter：

```text
E_index: [T, token_dim]
tokens = tokens + E_index
```

M15 scientific configuration 的 index embedding 为 `True`。实现也测试了 `index_embedding=False` 时不会产生该参数。

外部 structured contract 是 `[B,T,D]`。为了兼容现有评估 API 传入单条 `[D]` observation，body 只在 Mixer 边界临时增加 batch 轴，再恢复单条输出；没有改变 Mixer 的 rank-3 内部 contract。

## 6. Algorithm-aware input semantics

语义由 `ComputationSlotDescriptor` 注入，不作为用户自由拼接规则：

| Algorithm/slot | input semantics | action semantics | structured behavior |
|---|---|---|---|
| GCBC actor | `goal_pair` | `none` | state/goal button pair + robot pair |
| GCIQL actor/value | `goal_pair` | `none` | state/goal button pair + robot pair |
| GCIQL critic | `goal_pair` | `robot_context` | action 进入 robot context，不成为 token |
| GCIVL actor/value | `goal_pair` | `none` | state/goal button pair + robot pair |
| QRL actor | `goal_pair` | `none` | state/goal button pair + robot pair |
| QRL value/phi | `single_observation` | `none` | 同一个 phi module 分别处理 `s` 和 `g` |
| QRL dynamics | `latent_vector` | `latent_dynamics` | 保持 M14 canonical vector path |

因此 paired actor/value 的 robot input 为 `38 = 19 + 19`；GCIQL critic 的 robot input 为 `38 + action_dim`。Puzzle 环境的 action dimension 为 5，静态报告中 critic robot input 为 43。QRL single phi 的 robot input 为 19。

`is_phi=True` 在 QRL IQE/MRN operator 内仍然先 bypass phi module，因此不会把 latent vector 再解析成 Puzzle observation，也不会再次经过 tokenizer/Mixer。

## 7. MLP-Mixer computation

默认候选超参数为：

```text
token_dim              = 128
robot_hidden_dim       = 128
token_mlp_hidden_dim   = 64
channel_mlp_hidden_dim = 256
num_mixer_blocks       = 1
readout                = mean
tm_mode                = none
index_embedding        = true
```

对 `X in [B,T,D]`，每个 block 为：

```text
Y = TokenDense2(GELU(TokenDense1(Transpose(X))))
Y = Transpose(Y)
X = X + Y                         # token mixing residual

Z = ChannelDense2(GELU(ChannelDense1(X)))
X = X + Z                         # channel mixing residual
```

token mixing 沿 T 轴执行 `T -> H_T -> T`，channel mixing 沿 D 轴执行 `D -> H_D -> D`。每个 block 是独立参数，L 个 block 不共享参数。

### `tm_mode`

`MLPMixerBlock` 新增：

- `tm_mode='none'`：Puzzle 路径使用；不创建 `tm_weights`，token MLP 输出直接 residual-add；
- `tm_mode='lower_triangular'`：默认历史模式；保持 M14/M8 旧参数树和输出公式，包括 `tm_weights` 的 lower-triangular output masking。

M15 没有引入 token recurrence；`puzzle_tokens + single_state/two_state` 会立即报错，并明确提示 token recurrence deferred。

## 8. Readout, robot branch, and fusion

Mixer 后使用无参数 mean readout：

```text
[B,T,D] -> mean(axis=T) -> [B,D]
```

robot 分支为独立的：

```text
robot -> Dense(robot_hidden_dim) -> GELU -> optional LayerNorm
```

button summary 与 robot representation concat 后通过一次 fusion Dense 到 slot-required output width。其 final GELU/LayerNorm 由 M14 descriptor semantics 传入：

- actor：final activation true、layer norm false；
- GCIQL/GCIVL value/critic：final activation true、继承 `config.layer_norm`；
- QRL phi：final activation false、继承 `config.layer_norm`。

structured body 是 slot body 的替换，不是在旧的三层 canonical MLP 前面再叠加一个 Mixer。

## 9. Unified ComputationSpec and factory

M15 扩展统一 `ComputationSpec`：

```yaml
structure: puzzle_tokens
topology: feedforward
block: mlp_mixer
credit: direct
structure_kwargs:
  num_buttons: 24
  robot_dim: 19
  button_feature_dim: 4
  token_dim: 128
  robot_hidden_dim: 128
  token_mlp_hidden_dim: 64
  channel_mlp_hidden_dim: 256
  num_mixer_blocks: 1
  index_embedding: true
  readout: mean
  tm_mode: none
```

默认 `structure='vector'`，所以既有 M14 config 不受影响。M15 只允许 `puzzle_tokens + feedforward + mlp_mixer + direct`；向 Puzzle token body 请求 recurrent topology 会 fail loudly。

## 10. Algorithm integration

### GCIQL：primary integration gate

GCIQL actor、value、critic 均可使用 Puzzle structured body；三者的 Mixer/index/robot/fusion 参数各自独立。critic 是两成员 ensemble，每个成员独立拥有 structured parameters。`target_critic` 由 online critic definition 深拷贝生成，初始化时 params 完整相等。

value scalar readout 和 critic scalar readout 仍由原有 `GCValue` 逻辑维护，structured body 只替换 representation body。GCIQL loss、AWR/DDPGBC、critic target 和 actor gradient ownership 没有改变。

### GCBC

`compute.actor` 支持 structured actor；仍只有原有 behavior-cloning actor loss。

### GCIVL

`compute.actor` 和 `compute.value` 支持 structured body；target value 继承同一 structured architecture，原有 double-value expectile 与 actor advantage 不变。

### QRL

- actor：paired Puzzle structured body，输出 actor width；
- value：IQE/MRN 的 phi mapping 使用 single-observation Puzzle structured body，输出 `latent_dim`；
- `is_phi=True`：直接 bypass；
- dynamics：保持 canonical latent vector computation，不做 tokenization、不把 latent reshape 成 Puzzle tokens。

测试使用 `token_dim=7`、`latent_dim=6`、actor width=8，确认没有 hidden-512 假设。

## 11. Parameter and MAC accounting

`count_dense_macs(parameter_tree)` 不能单独代表 structured forward MAC，因为 rank-3 Dense 要在 token/channel 位置重复执行。M15 新增 token-aware accounting，记录：

- `button_projection_params`；
- `index_embedding_params`；
- `robot_projection_params`；
- `mixer_params`；
- `fusion_params`；
- `structured_body_params`；
- `total_per_sample_dense_macs`；
- `num_tokens`、`token_dim`、`token_hidden_dim`、`channel_hidden_dim`、`num_mixer_blocks`；
- `sequential_depth`、`unique_dense_layers`、`executed_dense_layers`。

忽略 bias 时，单 block 的主要计算为：

```text
MAC_token   = 2 * D * T * H_T
MAC_channel = 2 * T * D * H_D
MAC_tm      = T^2 * D       # 只有 lower_triangular 时
```

M15 Puzzle 默认 `tm_mode=none`，因此 `MAC_tm=0` 且 parameter tree 不含 `tm_weights`。报告使用真实初始化 parameter tree；Dense MAC 忽略 bias add、GELU、LayerNorm 和 environment/evaluation work。

## 12. Default candidate static comparison

生成命令：

```bash
PYTHONPATH=. JAX_PLATFORMS=cpu \
/home/eai/Tools/miniforge3/envs/brain_nav/bin/python \
tools/m15_static_accounting.py
```

候选为 `token_dim=128, robot_hidden_dim=128, H_T=64, H_D=256, L=1`，GCIQL actor/value/critic output width 均为 512，Puzzle critic action dimension 使用 5。这里的 body accounting 不含 actor action mean head，也不含 value/critic scalar readout；critic 数值包含两个独立 ensemble body。value/critic 的 canonical flat body 参数包含三层 hidden MLP 的 LayerNorm 参数，structured body 参数包含 robot/fusion LayerNorm 参数。

### GCIQL actor

| Puzzle | Flat params | Mixer params | Flat MACs | Mixer MACs | Flat depth | Mixer depth |
|---|---:|---:|---:|---:|---:|---:|
| 3x3 (T=9) | 582,144 | 206,025 | 580,608 | 882,432 | 3 | 6 |
| 4x4 (T=16) | 610,816 | 207,824 | 609,280 | 1,463,040 | 3 | 6 |
| 4x5 (T=20) | 627,200 | 208,852 | 625,664 | 1,794,816 | 3 | 6 |
| 4x6 (T=24) | 643,584 | 209,880 | 642,048 | 2,126,592 | 3 | 6 |

### GCIQL value

| Puzzle | Flat params | Mixer params | Flat MACs | Mixer MACs | Flat depth | Mixer depth |
|---|---:|---:|---:|---:|---:|---:|
| 3x3 (T=9) | 585,216 | 207,305 | 580,608 | 882,432 | 3 | 6 |
| 4x4 (T=16) | 613,888 | 209,104 | 609,280 | 1,463,040 | 3 | 6 |
| 4x5 (T=20) | 630,272 | 210,132 | 625,664 | 1,794,816 | 3 | 6 |
| 4x6 (T=24) | 646,656 | 211,160 | 642,048 | 2,126,592 | 3 | 6 |

### GCIQL critic（two independent ensemble members）

| Puzzle | Flat params | Mixer params | Flat MACs | Mixer MACs | Flat depth | Mixer depth |
|---|---:|---:|---:|---:|---:|---:|
| 3x3 (T=9) | 1,170,432 | 415,890 | 1,161,216 | 1,766,144 | 3 | 6 |
| 4x4 (T=16) | 1,227,776 | 419,488 | 1,218,560 | 2,927,360 | 3 | 6 |
| 4x5 (T=20) | 1,260,544 | 421,544 | 1,251,328 | 3,590,912 | 3 | 6 |
| 4x6 (T=24) | 1,293,312 | 423,600 | 1,284,096 | 4,254,464 | 3 | 6 |

说明：

- structured body 参数因 token projection/mixer 宽度和 robot/fusion 设计明显不同于 512-wide Flat MLP；M15 没有自动调参做 parameter matching；
- structured body MAC 随 T 增长，尤其 token/channel Mixer 执行 multiplicity 需要明确计入；
- Mixer sequential depth 的 6 是 button path 的 `projection + 4 Mixer Dense + fusion`；robot branch 的最长路径为 2；critic 的 `unique_dense_layers=14` 是两个独立 ensemble member 的物理层数；
- 这些是架构 accounting，不是 superiority、scaling advantage 或性能预测。

## 13. Tests

### M15 synthetic/integration

```bash
JAX_PLATFORMS=cpu \
/home/eai/Tools/miniforge3/envs/brain_nav/bin/python \
-m unittest tests.integration.test_m15_puzzle_structured_computation -q
```

结果：`12/12 PASS`。

覆盖 parser、四种尺寸、token shapes、shared projection、index embedding 开关、Mixer none/no-`tm_weights`、Mixer gradient、L=1/2/4 independent blocks、robot/action context、fusion output、GCIQL 三槽 target copy、GCBC/GCIVL/QRL integration、QRL latent dimension 解耦、非法 topology fail-loudly 和 token-aware accounting。

### M15 real-data tiny smoke

```bash
JAX_PLATFORMS=cpu MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
/home/eai/Tools/miniforge3/envs/brain_nav/bin/python \
-m unittest tests.integration.test_m15_puzzle_real_smoke -v
```

结果：`1/1 PASS`，其中包含 GCIQL、GCBC、GCIVL、QRL 四个 subtest。数据根目录为：

```text
/data/qijunrong/06-RL/offline-rl/data/raw_ogbench
```

每个算法使用真实 `puzzle-3x3-play-v0` 数据，完成 batch sampling、init、2 次 update、finite loss、checkpoint save/restore、单条 observation action path 和 1 个 evaluation episode，`video_episodes=0`。该 smoke 只验证工程生命周期，不提供成功率结论。

### Historical regression

| Suite | Result |
|---|---:|
| `tests.computation.test_foundation` | 2/2 PASS |
| `tests.computation` discovery | 53/53 PASS |
| `tests.integration.test_m14_base_algorithm_computation` | 11/11 PASS |
| `tests.integration.test_m13_canonical_agents` | 12/12 PASS |
| `tests.integration.test_crl_runtime` | 13/13 PASS |
| `tests.integration.test_hiql_accounting_compatibility` | 2/2 PASS |
| `tests.integration.test_coghp_runtime` | 3/3 PASS |
| `compileall` | PASS |

历史回归确认：默认 `structure=vector` 时 M14/M13 behavior 没有被改变；旧 `lower_triangular` Mixer regression 通过；CoGHP 使用的历史 Mixer 路径仍保持不变。

当前环境没有安装 pytest：

```text
No module named pytest
```

因此没有执行 pytest-specific analysis suite，也没有擅自安装依赖。上述 unittest targeted suites 是本轮可复现的验证范围。

## 14. Provenance and Git workflow

- M15 在 M14 主线状态上实现；M14 报告记录的 M13 starting point 为 `a3b9035300b986411d159b65e88854d991a8bc7f`，本轮没有重新读取或推断最终 Git HEAD；
- 本轮没有执行任何 Git 操作，包括 `git status`、`git diff`、`git log`、`git rev-parse`、commit、merge、checkout、reset 和 push；
- 没有修改 frozen worktree；所有代码修改均位于主线 `/home/eai/Research/RLC`；
- 用户需要自行检查主线 status/diff、确认与自身并行修改的边界，并手动完成 commit/push。

## 15. Known issues and next milestone

1. pytest 当前不可用；需要在用户指定环境补跑 pytest-specific tests，才能完成更广泛的 suite verification。
2. M14 已记录的 Polyak target semantics 疑点没有在 M15 改动；M15 target 只验证结构化参数继承和初始化复制，不重新解释该既有算法选择。
3. 当前候选不是 512-wide Flat 的 parameter/MAC-matched design；后续 milestone 才应冻结 canonical Flat、parameter/depth-matched Flat 和 structured Mixer 三组。
4. M15 只做 infrastructure tiny smoke；下一步 scientific milestone 才能在用户手动确认代码版本、Git 状态和实验设计后启动正式 1M/seed-sweep/scaling study。

## 16. Explicit non-results statement

本轮没有正式 1M run，没有 seed sweep，没有 Puzzle scaling experiment，也没有 success-rate scientific result。不能根据 tiny smoke 或静态 accounting 声称：

- MLP-Mixer 提升了 Puzzle performance；
- tokenization 提升了 combinatorial generalization；
- structured computation 具有更好的 scaling 或效率；
- 任一算法在 Puzzle benchmark 上优于 baseline。

M15 的结论仅限于：标准 Puzzle observation 能被严格解析为结构化 robot/button 表示，structured `[B,T,D]` computation 能接入指定 base-algorithm slots，token-aware accounting 能正确纳入 token/channel execution multiplicity，且既有 vector/historical 路径在 targeted regression 中保持通过。

## 17. Acceptance status

在本轮可执行的 targeted acceptance 范围内，M15 infrastructure gate 已通过：解析、token layout、shared projection、index embedding、robot/action separation、Mixer axes、`tm_mode` 双模式、readout、fusion、四类算法集成、target inheritance、QRL bypass/dynamics、vector compatibility、accounting、四种尺寸 shape smoke 和真实 tiny lifecycle 均已验证。

限制是 pytest suite 未安装，且 formal scientific training 按要求没有启动；因此这不是科研结果验收，也不是允许自动启动下一轮正式实验的授权。
