# M13：Canonical OGBench Base Algorithm Migration

日期：2026-08-26  
仓库：`/home/eai/Research/RLC`  
状态：实现、针对性测试、现有计算/集成回归和四个真实短 smoke 已完成；M13 尚未作最终验收标记。最终 Git commit、最终 HEAD/working-tree 状态仍由用户手动完成和确认。

## A. Provenance

### RLC

- 起始 RLC HEAD：`a3b9035300b986411d159b65e88854d991a8bc7f`。
  这是本轮开始前用户提供的最后一个主线 HEAD；本轮遵守“所有 Git 操作由用户完成”的要求，没有执行 `git status`、`git rev-parse`、`git diff`、`git commit` 或 `git push`，因此请用户在提交前自行确认该起点。
- 最终 RLC HEAD：未查询。本轮代码保持未提交状态，最终 HEAD 只有在用户手动 commit 后才确定。
- working tree：已知包含本轮实现和测试文档的未提交修改；未通过 Git 命令读取最终 dirty 状态。

### OGBench

本轮审计的官方参考为 `master` 最新提交：

`1d4140997f60c52c6fb0702ec100dc988b18c548`

来源：

- [OGBench commit 1d41409](https://github.com/seohongpark/ogbench/commit/1d4140997f60c52c6fb0702ec100dc988b18c548)
- [official gcbc.py](https://raw.githubusercontent.com/seohongpark/ogbench/1d4140997f60c52c6fb0702ec100dc988b18c548/impls/agents/gcbc.py)
- [official gciql.py](https://raw.githubusercontent.com/seohongpark/ogbench/1d4140997f60c52c6fb0702ec100dc988b18c548/impls/agents/gciql.py)
- [official gcivl.py](https://raw.githubusercontent.com/seohongpark/ogbench/1d4140997f60c52c6fb0702ec100dc988b18c548/impls/agents/gcivl.py)
- [official qrl.py](https://raw.githubusercontent.com/seohongpark/ogbench/1d4140997f60c52c6fb0702ec100dc988b18c548/impls/agents/qrl.py)
- [official networks.py](https://raw.githubusercontent.com/seohongpark/ogbench/1d4140997f60c52c6fb0702ec100dc988b18c548/impls/utils/networks.py)
- [official datasets.py](https://raw.githubusercontent.com/seohongpark/ogbench/1d4140997f60c52c6fb0702ec100dc988b18c548/impls/utils/datasets.py)
- [official hyperparameters.sh](https://raw.githubusercontent.com/seohongpark/ogbench/1d4140997f60c52c6fb0702ec100dc988b18c548/impls/hyperparameters.sh)

本轮没有把 OGBench checkout、源码副本或网络依赖引入 RLC 运行时。

## B. Files

### 新增

| 文件 | 作用 |
|---|---|
| `impls/agents/gcbc.py` | 官方 GCBC actor-only 迁移 |
| `impls/agents/gciql.py` | 官方 GCIQL value/critic/target critic/actor 迁移 |
| `impls/agents/gcivl.py` | 官方 GCIVL double-value/target-value/actor 迁移 |
| `impls/agents/qrl.py` | 官方 QRL IQE/MRN、lambda、latent dynamics、actor 迁移 |
| `tests/integration/test_m13_canonical_agents.py` | M13 配置、方程、目标更新、参数树、checkpoint、离散路径和 RNG 测试 |
| `docs/milestones/M13_base_algorithm_migration.md` | 本里程碑审计、验证和阻塞项报告 |

### 修改

| 文件 | 修改原因 |
|---|---|
| `impls/agents/__init__.py` | 注册 `gcbc/gciql/gcivl/qrl`，保留 CRL/HIQL/CoGHP 与 CRL policy-extractor variant |
| `impls/networks/common.py` | 增加官方缺失的 `GCDiscreteCritic`、`Param`、`LogParam`、`GCIQEValue`、`GCMRNValue` |
| `impls/networks/__init__.py` | 导出新增可复用网络类 |
| `impls/main.py` | width/depth 字段泛化、M13 computation 明确拒绝、loss total 去重、actor-only/多模块 checkpoint validation 泛化 |
| `impls/utils/datasets.py` | 保留 RLC 显式 RNG，同时修正官方 `p_curgoal == 1` 边界和轨迹概率分母语义 |

没有修改 `impls/experiment/`，没有增加 M13 algorithm-specific experiment shim，也没有修改 Puzzle observation parsing、MLP-Mixer 或 HRM。

## C. Algorithm migration

### GCBC

模块树：

```text
modules_actor
```

科学语义：

```text
L_actor = - E_D[ log pi(a_dataset | s, g) ]
```

- 仅 actor 更新；没有 value、critic、target 或 dynamics。
- 使用已有 RLC `GCActor`/`GCDiscreteActor`、`ModuleDict` 和 `TrainState`。
- 连续动作诊断保留 `actor_loss`、`bc_log_prob`、`mse`、`std`。
- 支持官方同样支持的离散路径。
- 默认值：`lr=3e-4`、`batch_size=1024`、actor `(512,512,512)`、`const_std=True`、`GCDataset`。
- 官方 GCBC 的 goal sampling/reward defaults 已逐字段迁移。

状态：PASS（配置、actor 方程、连续/离散初始化与更新、checkpoint、真实短 smoke）。

### GCIQL

模块树：

```text
modules_value
modules_critic
modules_target_critic
modules_actor
```

科学语义：

```text
q = min(Q1_target(s,a,g), Q2_target(s,a,g))
L_V = E[ expectile(q - V, q - V) ]
target_Q = r + gamma * mask * V(s', g)
L_Q = E[(Q1-target_Q)^2 + (Q2-target_Q)^2]
```

actor 支持：

- AWR：`min(exp(alpha * (q-v)), 100)` 加权行为克隆；
- DDPG+BC：`q_action=mode`（`const_std=True`），或在非 constant-std 时按官方路径采样；
- `q_loss = -mean(q) / stop_gradient(mean(abs(q)) + 1e-6)`；
- `bc_loss = -alpha * mean(log_prob(data_action))`。

默认值：`expectile=0.9`、`tau=0.005`、`alpha=0.3`、`actor_loss='ddpgbc'`，其余网络、goal sampling、reward、layer norm 与官方一致。

目标 critic 初始化时逐树等于 online critic；更新后显式满足：

```text
target_new = tau * online_new + (1 - tau) * target_old
```

状态：PASS（配置、expectile/TD/AWR/DDPGBC 分解、target 初始化和 Polyak、checkpoint、真实短 smoke）。

### GCIVL

模块树：

```text
modules_value
modules_target_value
modules_actor
```

没有 Q 网络。value 目标严格保留官方的 double-value 稳定化路径：

```text
next_v_t = min(V1_target(s',g), V2_target(s',g))
q       = r + gamma * mask * next_v_t
v_t     = (V1_target(s,g) + V2_target(s,g)) / 2
adv     = q - v_t
```

当前 `V1/V2` 分别用同一 target-derived advantage 做 expectile regression。actor 使用：

```text
adv    = mean(V(s',g)) - mean(V(s,g))
weight = min(exp(alpha * adv), 100)
L_pi   = -E[weight * log pi(a_dataset | s,g)]
```

默认值：`expectile=0.9`、`alpha=10.0`、`tau=0.005`，其余官方 config 字段逐项保留。

状态：PASS（double-value 方程、target 初始化和 Polyak、checkpoint、真实短 smoke）。

### QRL

模块树：

```text
modules_value
modules_actor
modules_dynamics       # DDPG+BC 时
modules_lam
```

QRL 没有被简化成 `GCValue + actor`：

- `GCIQEValue` 保留 latent grouping、interval sorting、valid interval event、component aggregation 和 learned sigmoid alpha；
- `GCMRNValue` 保留 symmetric Euclidean component 与 asymmetric L-infinity component；
- `LogParam` 保留正值 dual lambda 的 log parameterization；
- `d_neg = d(s,g)`，`d_pos = d(s,s')`；
- `d_neg_loss = mean(100 * softplus(5 - d_neg/100))`；
- `d_pos_loss = mean(relu(d_pos - 1)^2)`；
- `value_loss = d_neg_loss + stop_gradient(lambda) * d_pos_loss`；
- `lambda_loss = lambda * (eps - stop_gradient(d_pos_loss))`；
- `total value loss = value_loss + lambda_loss`；
- DDPG+BC 时保留 `pred_next = phi(s) + dynamics(phi(s), action)`、quasimetric dynamics loss 和 latent reparameterized actor objective；没有替换成 GCIQL critic Q-gradient；
- AWR 仍支持并使用 `-V(s,g)` 到 `-V(s',g)` 的官方优势定义。

默认值：IQE、`latent_dim=512`、`dim_per_component=8`、`eps=0.05`、`alpha=0.003`、`actor_loss='ddpgbc'`，以及官方 random-value-goal、trajectory-actor-goal、reward defaults。

状态：PASS（IQE/MRN forward、lambda/value 方程、dynamics/DDPGBC 一次更新、AWR/离散路径、checkpoint、真实短 smoke）。

## D. Shared infrastructure changes

### Agent registry

`--agent` 现在暴露：

```text
coghp, crl, gcbc, gciql, gcivl, hiql, qrl
```

`CRLPolicyExtractorAgent` 和已有 runtime variant 没有移除。

### Networks

已有 `MLP`、`GCActor`、`GCDiscreteActor`、`GCValue` 被复用。只增加官方确实缺少且跨 agent 需要的网络类，没有增加 Puzzle token、attention 或 Mixer 组件。

### `main.py`

- width/depth override 只更新 config 中实际存在的字段，因此 GCBC 不会被制造出无意义的 `value_hidden_dims`；
- M13 四个 agent 使用 canonical flat path；显式传入 `--computation` 会清晰报错，不会静默启用结构化 computation；
- `_loss_metric` 的新路径只统计完整目标：GCBC actor、GCIQL value+critic+actor、GCIVL value+actor、QRL value/total+dynamics+actor；QRL 的 `value/value_loss` 和 `lam_loss` 不会再次计入；
- unchanged CRL/HIQL/CoGHP update-info 仍走原有 legacy monitoring sum；
- checkpoint action parity 对所有 agent 必查，value/critic 按模块存在性检查，target critic/target value/lam/dynamics 逐树检查；
- evaluation、best/last lifecycle、restore、provenance 和 Study 入口保持共享 runtime。

### Dataset

四个新 agent 全部使用现有 RLC `GCDataset`，没有复制官方 dataset，也没有增加 `GCIQLDataset`/`QRLDataset`。

审计并修正了两个官方采样语义点：

1. `p_curgoal == 1.0` 时直接返回当前 transition index；
2. 非边界情况下 trajectory/random mixture 使用官方的 `p_trajgoal / (1-p_curgoal)`，删除了原 RLC 中额外的 `+1e-6`。

RLC 的显式 `numpy.random.Generator` 仍保留，详见下一节。

## E. Intentional deviations from upstream

| 项目 | upstream | RLC | 判断 |
|---|---|---|---|
| Dataset RNG | process-global `numpy.random` | `GCDataset(..., rng=derived_seed)` 的显式 Generator | infrastructure-only；goal equations 和 draw order 保持一致，且增加同 seed reproducibility test |
| Dataset API | 官方 `Dataset/GCDataset` 直接使用全局 RNG | RLC `Dataset` 带显式 seed、`next_observations` 兼容和共享 `GCDataset` wrapper | infrastructure-only |
| Module/runtime | 官方 `ModuleDict/TrainState` 包路径 | RLC 相同语义的本地 `ModuleDict/TrainState`，带 model_state/buffers 支持 | infrastructure-only；新 flat agents 未增加 buffers |
| Checkpoint/provenance | 官方基础 numeric checkpoint/logging | RLC best/last、checkpoint metadata、source provenance、action parity validation | infrastructure-only |
| `encoder` placeholder | upstream config 用 ml-collections placeholder | RLC state-based canonical runtime 将默认解析为 `None` | infrastructure-only；没有改变网络 equations |
| QRL `dim_per_component` | upstream `qrl.py` 在 create 中使用硬编码 `8` | RLC 将该官方常数显式记录为 config 字段，默认仍为 `8`，并用于同一 IQE reshape | 透明化同一科学常数，不改变 canonical default；环境特定值未加入 |
| target Polyak | 官方源码 helper 文字上读取 `self.network.params` 的 online subtree；M13 验收方程明确要求更新后的 online subtree | GCIQL/GCIVL 使用 optimizer 后 `network.params` 与旧 target 计算 `tau*online_new+(1-tau)*target_old` | 这是唯一需要用户科研确认的非纯 infrastructure 差异；它可能影响训练轨迹，因此没有隐藏，且已由显式 target invariant test 锁定 |

没有进行任何 silent “优化”、Puzzle-specific 改造、MLP-Mixer 改造、HRM/attention 改造或 computation scaling 改造。

## F. Tests

### M13 专项测试

命令：

```bash
JAX_PLATFORMS=cpu \
/home/eai/Tools/miniforge3/envs/brain_nav/bin/python \
-m unittest tests.integration.test_m13_canonical_agents -q
```

结果：`12/12 PASS`。

覆盖：

- 四个官方 config 的 scientific fields；
- registry；
- GCBC `-mean(log_prob)`；
- GCIQL expectile、TD/actor total monitoring；
- GCIVL target-value 相关路径；
- QRL IQE、MRN、`d_neg/d_pos/lambda`；
- QRL loss 不重复计数；
- GCIQL/GCIVL target 初始化和更新后 Polyak；
- 四个参数树结构；
- continuous 与 discrete runtime；
- checkpoint round-trip/action parity/auxiliary module parity；
- explicit dataset RNG 和 official current-goal boundary。

### 语法、CLI 和既有回归

- `compileall`：通过。
- `python -m impls.main --help`：通过，`--agent` 显示七个 canonical agents。
- 既有计算测试：`53/53 PASS`。

```bash
JAX_PLATFORMS=cpu \
/home/eai/Tools/miniforge3/envs/brain_nav/bin/python \
-m unittest discover -s tests/computation -p 'test*.py' -v
```

- 既有 + M13 integration discovery：`94/94 PASS`（其中 M13 专项测试 11 个在当时版本；之后补充离散测试使 M13 专项成为 12 个）。这批测试覆盖 CRL、HIQL、CoGHP、M9/M10/M11/M12 lifecycle。

```bash
JAX_PLATFORMS=cpu \
OGBENCH_DATASET_DIR=/data/qijunrong/06-RL/offline-rl/data/raw_ogbench \
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
/home/eai/Tools/miniforge3/envs/brain_nav/bin/python \
-m unittest discover -s tests/integration -p 'test*.py' -v
```

- 最后一次共享 GCDataset 修正后的重点回归：`36/36 PASS`。

```bash
JAX_PLATFORMS=cpu \
OGBENCH_DATASET_DIR=/data/qijunrong/06-RL/offline-rl/data/raw_ogbench \
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
/home/eai/Tools/miniforge3/envs/brain_nav/bin/python \
-m unittest tests.integration.test_hiql_real_runtime \
          tests.integration.test_crl_runtime \
          tests.integration.test_m11b_study \
          tests.integration.test_m13_canonical_agents -q
```

### 未执行测试

分析目录中有 16 个 pytest 风格测试；`brain_nav` 环境没有安装 pytest，系统 Python 的 pytest 又与该环境的 Python/NumPy ABI 不兼容，因此没有擅自安装依赖，也没有把这 16 个测试伪报为通过。该项是最终验收前的环境 blocker。

## G. Real Puzzle-3x3 smoke

数据根目录：

```text
/data/qijunrong/06-RL/offline-rl/data/raw_ogbench
```

环境变量：

```text
OGBENCH_DATASET_DIR=/data/qijunrong/06-RL/offline-rl/data/raw_ogbench
MUJOCO_GL=egl
PYOPENGL_PLATFORM=egl
```

每个 agent 均为真实 `puzzle-3x3-play-v0`、`train_steps=2`、`batch_size=4`、`eval_tasks=1`、`eval_episodes=1`、`video_episodes=0`。这是工程 smoke，不是科学实验；输出的 success rate 不得解释为 benchmark 结果。

最终 smoke 根目录：`/tmp/m13_puzzle_smoke_final_rg8ohqmv`。

| Agent | 初始化 | update/finite metrics | eval action | checkpoint round-trip | 结果 |
|---|---:|---:|---:|---:|---:|
| GCBC | PASS | PASS | PASS | PASS，actor-only | PASS |
| GCIQL | PASS | PASS | PASS | PASS，value/target_critic | PASS |
| GCIVL | PASS | PASS | PASS | PASS，value/target_value | PASS |
| QRL | PASS | PASS | PASS | PASS，value/lam/dynamics | PASS |

四个 smoke 的 `evaluation/overall_success` 都是 0.0；由于每个只跑一个 episode、两次更新，这个数字只是 smoke 记录，不能形成任何科学结论。

## H. Remaining issues and acceptance status

### 已知问题

1. **Git 操作未由 agent 完成。** 这是用户明确要求的工作流，不是遗漏。用户需要自行检查主线 diff/status，确认起点和无关修改后手动 commit；本轮没有 commit/push。
2. **最终 HEAD/working-tree 尚未由用户确认。** 因为本轮不执行 Git 命令，报告无法填入 final SHA 和 clean/dirty 的机器读数。
3. **Pytest 分析测试 16 个尚未执行。** 需要用户在合适环境安装/提供 pytest 后运行；当前 unittest/JAX 回归不能替代这 16 个测试。
4. **target Polyak 的官方源码文字与 M13 验收方程存在歧义。** RLC 已选择并测试 `online_new` 方程，但在正式 benchmark 前建议用户确认这一选择；不要在未确认前将新算法结果与官方数字作强结论比较。
5. **没有正式 benchmark。** 本轮没有启动任何 1M GCBC/GCIQL/GCIVL/QRL，也没有启动 Puzzle scaling/computation/placement 实验。

### M13 acceptance table

| 条目 | 状态 |
|---|---|
| 四个 agent 一等注册 | PASS |
| canonical flat scientific implementation | PASS，target Polyak 选择需确认 |
| canonical config | PASS |
| QRL IQE/MRN/lambda/dynamics | PASS |
| GCIQL target critic | PASS |
| GCIVL target value | PASS |
| reuse RLC GCDataset | PASS |
| CRL/HIQL/CoGHP regression | PASS（53 computation + 既有 integration） |
| generic checkpoint validation | PASS |
| QRL loss no double-count | PASS |
| executed unit/integration tests | PASS for executed suites |
| full test suite including pytest analysis | BLOCKED：pytest unavailable |
| four real Puzzle tiny smoke | PASS |
| no Puzzle token/MLP-Mixer/HRM added | PASS |
| no formal 1M training | PASS |
| code committed | BLOCKED by user-controlled Git workflow |

因此，本报告不把 M13 标记为“最终完成”；准确结论是：**代码迁移和可执行验证已完成，但正式验收仍等待用户手动 Git 检查/提交、pytest 16 项测试环境补齐，以及对 target Polyak 语义的科研确认。**
