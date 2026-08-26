# M14：Canonical Base Algorithm Computation Integration

日期：2026-08-26  
仓库：`/home/eai/Research/RLC`  
性质：computation infrastructure milestone，不是 scientific parameter experiment  
状态：M14 代码、专门测试、legacy 回归和 real-data tiny smoke 已完成；最终 Git 验收仍由用户手动完成。

## 1. 目标与边界

M13 已将 GCBC、GCIQL、GCIVL、QRL 迁移为 canonical OGBench base algorithms。M14 的唯一目标是把这些算法接入 RLC 的 computation-slot abstraction，使算法 loss/update ownership 保持不变，而网络中的 vector body 可以由 FeedForward、SingleState 或 TwoState 替换。

本轮明确没有做以下工作：

- 没有实现 Puzzle tokenizer、button/index parsing 或 `[B,T,D]` token contract；
- 没有修改 `impls/computation/blocks/mlp_mixer.py`；
- 没有实现 HRM、attention、RMSNorm、SwiGLU、grid/adjacency 或 `tm_weight`；
- 没有做 matched parameter/MAC scientific experiment；
- 没有启动 1M Puzzle、AntMaze 或 computation scaling 正式训练。

## 2. Provenance

### RLC

- M14 起点使用 M13 报告中记录的最后一个用户提供主线 HEAD：`a3b9035300b986411d159b65e88854d991a8bc7f`。
- 本轮严格遵守用户要求：没有执行任何 Git 操作，包括 `git status`、`git diff`、`git rev-parse`、`git commit` 和 `git push`。
- 因此最终 HEAD、是否 clean、是否存在用户并行修改，均未由 agent 读取；请用户自行审查主线 `/home/eai/Research/RLC` 后手动完成 Git 操作。

### OGBench reference

M13 已审计的官方参考 commit 为：

`1d4140997f60c52c6fb0702ec100dc988b18c548`

M14 没有引入新的 OGBench checkout 或生产运行时网络依赖。

## 3. Slot ontology

声明式 registry 位于 `impls/computation/slots.py`。descriptor 至少记录：`slot_name`、`module_path`、`core_path`、`role`、`hidden_dims_source`、`state_dim_source`、`layer_norm_semantics`、`activate_final_semantics` 和 `output_dim_source`。

| Algorithm | Slot | Canonical module/core path | Role | Output/state dim | Target inheritance |
|---|---|---|---|---:|---|
| GCBC | actor | `modules_actor/actor_net/topology` | actor | `actor_hidden_dims[-1]` | none |
| GCIQL | actor | `modules_actor/actor_net/topology` | actor | `actor_hidden_dims[-1]` | none |
| GCIQL | value | `modules_value/value_net/core/topology` | value | `value_hidden_dims[-1]` | none |
| GCIQL | critic | `modules_critic/value_net/core/topology` | critic | `value_hidden_dims[-1]` | `target_critic` inherits the online critic spec |
| GCIVL | actor | `modules_actor/actor_net/topology` | actor | `actor_hidden_dims[-1]` | none |
| GCIVL | value | `modules_value/value_net/core/topology` | value | `value_hidden_dims[-1]` | `target_value` inherits the online value spec |
| QRL | actor | `modules_actor/actor_net/topology` | actor | `actor_hidden_dims[-1]` | none |
| QRL | value | `modules_value/phi/core/topology` | value/phi | `latent_dim` | none |
| QRL | dynamics | `modules_dynamics/core/topology` | dynamics | `latent_dim` | none |

`target_critic` 和 `target_value` 不是独立 computation slots；它们由 computationized online definition `deepcopy` 得到，避免出现 online/target architecture drift。

## 4. Algorithm integration

### GCBC

- `compute.actor` 已解析为 `GCActor`/`GCDiscreteActor` 的 body；disabled 或缺省时传入 `None`，保留 M13 的原始 MLP。
- 仍只有 actor loss：`-mean(log pi(a|s,g))`。
- 没有引入 value、critic 或 dynamics。

### GCIQL

- `compute.actor` 接入 actor body；`compute.value` 接入 scalar value readout 前的 representation body；`compute.critic` 接入 ensemble critic body。
- target critic 由 computationized `critic_def` 深拷贝后创建；初始化时 params 和 buffers 都复制 online critic。
- value、critic、actor 的 loss 计算和梯度 ownership 未重写；target network 不进入 optimizer gradient。

### GCIVL

- `compute.actor` 与 `compute.value` 已接入；value body 后仍保留 scalar readout/ensemble 结构。
- target value 由 computationized online value definition 深拷贝创建，并在初始化时同步 params 和 buffers。
- canonical double-value expectile 和 actor advantage 语义保持 M13 不变。

### QRL

- `compute.actor` 接入 actor body。
- `compute.value` 只替换 IQE/MRN 的 `phi` mapping body；IQE grouping、sorting、interval event、component aggregation 和 learned alpha 没有改动；MRN 的 symmetric/asymmetric quasimetric operator 没有改动。
- `is_phi=True` 直接绕过 representation computation，仍将输入当作 latent representation 使用；测试使用 latent 维度输入验证了该 bypass。
- `compute.dynamics` 使用可复用的 `ComputationVectorBody`，输入仍为 `[phi(s), action]`，输出为 `latent_dim`；没有改变 QRL dynamics loss 或 actor loss。
- `latent_dim` 与 actor width 解耦。测试中 actor width 为 8、QRL latent 为 6，runtime/accounting/forward 均按 6 处理，没有硬编码 512。

## 5. Buffer、target 与 checkpoint 语义

对于 recurrent computation，四个新 agent 统一使用：

```text
network_def.init(
    {'params': init_rng, 'buffers': fold_in(rng, 0x4D39)}, ...
)
TrainState.create(..., model_state=model_state)
```

其中 `buffers` 是 persistent、non-trainable 的初始化状态；每次 forward 从 buffer broadcast fresh local state，不写回 buffer。target 初始化通过 `synchronize_target_module` 同时复制 online params 和 online buffers。

更新阶段只对 target params 做 Polyak，target buffers 不做 Polyak，也不被 optimizer 更新。checkpoint 使用既有完整 agent serialization，因此包含 `model_state` 和 recurrent buffers；restore 测试逐 leaf 比较 params/model_state。

## 6. Output dimension and primitive semantics

所有 computationized body 都显式使用 descriptor 对应的 branch dimensions：

- actor：`actor_hidden_dims`，body 最终 representation 是 `actor_hidden_dims[-1]`，再进入 action mean/logit readout；
- GCIQL/GCIVL value/critic：`value_hidden_dims`，body 最终 representation 是 `value_hidden_dims[-1]`，再进入 scalar readout；
- QRL value/dynamics：`(*value_hidden_dims, latent_dim)`，最终 representation 是 `latent_dim`。

默认 primitive semantics：

- actor body：`layer_norm=False`、`activate_final=True`；
- scalar value/critic body：继承 `config.layer_norm`、`activate_final=True`；
- QRL phi/dynamics：继承 `config.layer_norm`、`activate_final=False`；
- CRL/HIQL 既有 slot 也纳入 registry，保持既有 actor/value/critic branch semantics。

如果 recurrent `state_dim` 与 body 最终 branch width 不一致，factory 会立即抛出异常；不会回退到 512 或静默改变维度。

## 7. Generic accounting and runtime metadata

`impls/main.py` 不再通过 `critic_`/`value_` 前缀推断 role。它通过 descriptor 解析：

- module/core path；
- role；
- hidden/state/output dimensions；
- layer norm 和 final activation semantics；
- topology、primitive、credit。

所有 enabled slot 的 generic accounting 都记录：

`topology`、`primitive`、`credit`、`trainable_params`、`buffer_elements`、`state_dim`、`iterations`、`h_cycles`、`l_cycles`、`total_update_executions`、`unique_dense_layers`、`executed_dense_layers`、`sequential_depth`、`dense_macs`，并保留历史 accounting aliases。

新 agent 的 legacy actor accounting 状态显式为 `not_applicable`；HIQL/CRL 仍执行 shared legacy/generic fields 的严格一致性审计。这样 CRL 的 critic/value generic-only slots 不会被错误要求出现在 actor-only legacy report 中。

## 8. Canonical-disabled parity

M14 专门测试将“没有 `compute` 字段的 M13 配置”作为 baseline，将“所有 `compute.*.enabled=false` 的 M14 配置”作为 disabled path，在同一 seed、同一 synthetic batch 下比较：

- 完整 parameter tree（paths/shapes/value）；
- model_state；
- trainable parameter tree；
- actor forward/output；
- loss decomposition/info；
- one-update 后 params、model_state 和 metrics。

GCBC、GCIQL、GCIVL、QRL 均通过 exact array parity。说明 M14 的 disabled compute path 没有改变 M13 canonical behavior。

## 9. Tests

### M14 dedicated tests

命令：

```bash
JAX_PLATFORMS=cpu \
/home/eai/Tools/miniforge3/envs/brain_nav/bin/python \
-m unittest tests.integration.test_m14_base_algorithm_computation -v
```

结果：`11/11 PASS`。

覆盖：

- exact slot ontology 和 unsupported-slot fail-loudly；
- 四算法 canonical-disabled exact parity；
- 四算法每个 slot 的 SingleState K1；
- 四算法每个 slot 的 FeedForward；
- QRL IQE/MRN phi operator 与 `is_phi` bypass；
- GCBC actor、GCIQL critic、GCIVL value、QRL value/dynamics 的 TwoState H2L1 full-BPTT smoke；
- GCIQL/GCIVL target params/buffers equality、online/target initial output equality、params-only Polyak；
- 四算法 recurrent checkpoint/model_state roundtrip；
- descriptor accounting、dense MACs、latent_dim != actor width；
- 新 agent legacy accounting `not_applicable`。

### M13 and existing computation regression

- M13 dedicated：`12/12 PASS`。
- Existing computation suites：`53/53 PASS`。
- CRL runtime（含 real GCDataset strict parity，EGL 环境）：`13/13 PASS`。
- HIQL accounting/lifecycle（EGL 环境）：`2/2 PASS`。
- `compileall`：通过。

使用过的代表性命令：

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
JAX_PLATFORMS=cpu JAX_NUM_THREADS=1 \
OMP_NUM_THREADS=1 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
/home/eai/Tools/miniforge3/envs/brain_nav/bin/python \
-m unittest tests.integration.test_crl_runtime -v
```

结果为 `13/13 PASS`。

### Full discovery / pytest status

`brain_nav` 环境没有安装 pytest：

```text
/home/eai/Tools/miniforge3/envs/brain_nav/bin/python: No module named pytest
```

因此没有声称 pytest 风格 analysis tests 已通过，也没有擅自安装依赖。使用 EGL/单线程运行 full unittest discovery 时，绝大多数既有测试已打印 `ok`，但该长进程在后续 M14 编译阶段因 LLVM `Cannot allocate memory` 退出（code 134）；同时 analysis 目录的 pytest-style 模块在 unittest discovery 中产生 import errors。这个 full discovery 结果不能作为完整 suite PASS，应与上面的隔离、可复现 targeted suites 区分。

## 10. Puzzle real-data tiny smoke

数据：`/data/qijunrong/06-RL/offline-rl/data/raw_ogbench`  
环境：`puzzle-3x3-play-v0`  
配置：真实数据、`train_steps=2`、`batch_size=4`、`eval_tasks=1`、`eval_episodes=1`、`video_episodes=0`，每个 agent 的所有声明 slot 使用 SingleState K1 normal buffer。

| Agent | data/init | 2-step update finite | eval | checkpoint restore | 结果 |
|---|---|---|---|---|---|
| GCBC | PASS | PASS | PASS，success=0.0 | PASS | PASS |
| GCIQL | PASS | PASS | PASS，success=0.0 | PASS | PASS |
| GCIVL | PASS | PASS | PASS，success=0.0 | PASS | PASS |
| QRL | PASS | PASS | PASS，success=0.0 | PASS | PASS |

四个 smoke 均返回 `renders=0`，符合无视频要求。`success=0.0` 只来自单 task、单 episode、随机初始化和两次更新，不能解释为任何 scientific result，也没有启动正式训练。

## 11. Known blockers and interpretation

1. **Git provenance 尚未由 agent machine-check。** 这是用户明确指定的 workflow。用户需自行检查主线 diff/status，确认 M13/M14 边界后手动 commit/push。
2. **pytest analysis tests 未执行。** 当前环境缺 pytest；这不是 M14 代码逻辑失败，但在声明完整测试闭环前需在合适环境补跑。
3. **full unittest discovery 不能作为 PASS。** 长进程有 analysis import errors，并在大量 JAX/LLVM 编译后出现内存不足；隔离 targeted suites 已通过。
4. **M13 pre-existing Polyak semantics blocker 保留。** GCIQL/GCIVL 当前使用已冻结的 `target_new = tau * online_new + (1-tau) * target_old` 语义。它与官方 helper 文字上读取 `self.network.params[online]` 的实现存在差异；M14 没有改变这个选择，也没有把它隐藏成新的 M14 结论。正式 benchmark 前仍需用户科研确认。

## 12. M14 acceptance status

| Acceptance item | Status |
|---|---|
| GCBC actor computation slot | PASS |
| GCIQL actor/value/critic slots | PASS |
| GCIVL actor/value slots | PASS |
| QRL actor/value/dynamics slots | PASS |
| target architecture inheritance | PASS |
| recurrent target params + buffers equality | PASS |
| disabled path reproduces M13 | PASS，四算法 exact parity |
| FeedForward/SingleState/TwoState support | PASS |
| QRL IQE/MRN operators unchanged | PASS |
| QRL `is_phi=True` bypass | PASS |
| generic accounting for all new slots | PASS |
| no prefix-based role inference for new slots | PASS |
| CRL/HIQL regression | PASS for isolated executable suites |
| checkpoint/model_state restore | PASS |
| no Puzzle token/Mixer/HRM code | PASS |
| no formal scientific experiment | PASS |
| pytest analysis tests | BLOCKED：pytest unavailable |
| full discovery as one process | BLOCKED：analysis imports/LLVM memory |
| final Git commit/push | 由用户手动完成 |

因此，M14 的 computation integration 本身已达到可执行 targeted acceptance；在用户完成 Git 审查、补跑 pytest analysis tests 并确认 M13 Polyak 语义前，不应把整个仓库标记为无条件最终验收，也不应据此启动正式 scientific scaling experiment。

## 13. Next milestone boundary

M15 可以基于本轮 accounting 的 params、MACs、sequential depth 和 execution counts 设计 Flat matched vs Structured Mixer，但应在单独 milestone 中处理。Puzzle tokenization、MLP-Mixer token computation、index embedding 和任何 formal parameter-matched experiment 都不属于 M14。
