# M11B HIQL Accounting Bugfix and Rerun Preflight Report

日期：2026-08-22  
范围：M11B HIQL accounting compatibility 修复、测试、doctor、四配置 dry-run preflight。  
边界：本轮没有启动正式 1M training；所有 Git 操作仍由用户手动完成。

## 1. Exact root cause

旧 M11B 运行在 agent 已成功创建之后、第一次 optimizer update 之前失败。`_actor_parameter_accounting()` 已经能识别 HIQL 的 `high_actor` 与 `low_actor`，但随后调用的 `_computation_slot_accounting()` 的 `slot_paths` 只登记了 CRL `actor`、critic/value 分支，没有登记：

```text
high_actor -> modules_high_actor -> actor_net/topology
low_actor  -> modules_low_actor  -> actor_net/topology
```

因此只要 M11B 启用任一 HIQL actor computation slot，generic accounting 就抛出：

```text
ValueError: Unsupported computation slot for accounting: 'high_actor'
```

旧的 12 个失败目录均保留了这一 failure reason；失败发生在训练前，不是 HIQL loss、梯度或环境交互失败。

## 2. Why M9A/M10A could run

M9A/M10A 的 HIQL high/low actor 参数树和 `hiql_policy_accounting()` 本身是合法的；对应测试已经直接创建 high-only、low-only、high+low agent，并检查了初始化、独立 buffer、输出、参数/MAC invariant。它们没有被 M11B 新入口中“legacy accounting 后再调用 generic accounting”的缺失 slot path 阻断。

本次保留并运行了 legacy `hiql_policy_accounting()` 相关回归，说明修复没有替换或破坏旧 accounting API。

## 3. Why M11B failed

M11B 的正式入口在 `run()` 中固定执行：

```text
agent creation -> _actor_parameter_accounting() -> _computation_slot_accounting() -> first update
```

HIQL baseline 没有启用 `high_actor`/`low_actor`，所以能通过；HIQL high-only、low-only、high+low 都在 generic accounting 处失败。该失败与四个环境的 canonical task context 无关。

## 4. Changed files

- `impls/main.py`：补齐 HIQL generic accounting slot path；增加 legacy-vs-generic consistency audit；接入 `run_attempt` 元数据与入口参数。
- `impls/experiment/management.py`：增加通用 rerun-attempt 路径、运行元数据和聚合状态保护。
- `tools/sweep.py`：透传并解析 `--run-attempt`，为新 attempt 生成独立 job path。
- `tools/run.py`：透传通用 `--run-attempt`。
- `scripts/run_study.sh`：透传并展示 `--run-attempt`，保持 dry-run/execute 两种模式。
- `tools/m11b_doctor.py`：新增 HIQL high-only、low-only、high+low accounting smoke，并联合检查 resolved spec 与 accounting artifact。
- `tests/integration/test_hiql_accounting_compatibility.py`：新增 accounting 与真实 `run()` lifecycle smoke。
- `tests/experiment/test_run_attempt.py`：新增 canonical/attempt 路径测试。
- `tests/experiment/test_management.py`：新增非 completed attempt 不进入 aggregate 的测试。

machine-readable provenance 见：[m11b_hiql_accounting_bugfix_provenance.json](./m11b_hiql_accounting_bugfix_provenance.json)。

## 5. Exact code-level fix

在 `_computation_slot_accounting()` 中增加：

```python
'high_actor': (('modules_high_actor',), ('actor_net', 'topology')),
'low_actor': (('modules_low_actor',), ('actor_net', 'topology')),
```

这使 HIQL high/low 使用已有的通用 `computation_slot_accounting()`，从实际参数和 buffer 树计算 accounting，而不是伪造字段。

同时，入口在任何 update 前比较两个 accounting 报告的 `topology`、`trainable_params`、`buffer_elements`、`state_dim`、`iterations`；发现不一致即失败，并写入 `accounting_consistency` runtime metadata。

## 6. Whether HIQL training math changed

没有改变。未修改 HIQL 的 `total_loss`、value loss、high actor loss、low actor loss、目标网络、optimizer、梯度路径或 update 公式。修复只改变运行前 accounting 的兼容性检查。

## 7. Whether SingleState changed

没有改变。未修改 SingleState factory、module topology、循环执行、buffer 初始化、residual、input injection 或 forward/update 实现。M11B 仍然是 K=4、non-residual、`z_plus_x`、state_dim=512。

## 8. Whether scientific config changed

没有改变。没有修改 M11B study、34 个 config YAML、canonical hyperparameter resolver、dataset class、evaluation protocol 或 M11B factor definitions。新增的 `run_attempt` 是运行实例元数据，不是 scientific factor。

## 9. high_actor accounting result

在真实 `antmaze-giant-navigate-v0` 数据形状上，M11B-C008 high-only probe 通过：

| 字段 | legacy | generic |
|---|---:|---:|
| topology | single_state | single_state |
| trainable_params | 560650 | 560650 |
| buffer_elements | 512 | 512 |
| state_dim | 512 | 512 |
| iterations | 4 | 4 |

resolved spec 同时确认：`residual=false`、`update_depth=2`、`layer_norm=false`、`update_activate_final=true`。

## 10. low_actor accounting result

M11B-C009 low-only probe 通过：

| 字段 | legacy | generic |
|---|---:|---:|
| topology | single_state | single_state |
| trainable_params | 549896 | 549896 |
| buffer_elements | 512 | 512 |
| state_dim | 512 | 512 |
| iterations | 4 | 4 |

resolved spec 同样确认 K=4、non-residual、state_dim=512、`update_depth=2`、`layer_norm=false`、`update_activate_final=true`。

## 11. high+low accounting result

M11B-C010 high+low probe 通过。两个 slot 均分别通过 consistency audit；合计 trainable parameters 为 `1,110,546`，合计 buffer elements 为 `1,024`。high 与 low 仍是独立参数/buffer subtree，没有发生共享或重构。

## 12. Legacy-vs-generic accounting consistency

三种 HIQL 配置均返回 `status=pass`，且 `mismatches=[]`：

```text
high_ss:     checked_slots=[high_actor]
low_ss:      checked_slots=[low_actor]
high_low_ss: checked_slots=[high_actor, low_actor]
```

此外 baseline 也通过兼容性路径：没有启用 slot 时 generic report 为空，不会误报。

## 13. Formal-entrypoint smoke results

新增测试实际调用 `impls.main.run()`，而不是只调用 helper。使用真实 M11B AntMaze-Giant dataset、临时 run root、`train_steps=1`、最小评估协议，并走过配置解析、dataset fixture、agent creation、accounting、runtime metadata、一次 finite update、评估、checkpoint lifecycle 和 finalize。

| config | condition | result |
|---|---|---|
| M11B-C008 | high-only | PASS |
| M11B-C009 | low-only | PASS |
| M11B-C010 | high+low | PASS |

三者均生成 `status=completed` 的临时 run，`train.csv` 恰有一次训练记录，无 `failure.json`；没有写入 formal M11B run root。

## 14. M9A regression

PASS。M9A 单状态配置矩阵、high/low slot resolution、独立 buffer、canonical run path 均通过。没有修改 M9A scientific semantics。

## 15. M10A regression

PASS。M10A fixed-budget placement 的 11 个配置、high/low 独立 allocation、参数/MAC invariant、vanilla value 分支均通过；legacy `hiql_policy_accounting()` 测试继续通过。

## 16. M11A regression

PASS。M11A CRL interaction 的 7 配置结构、M9 legacy actor topology、CRL critic 两分支 update/restore、diagnostic formula checks 均通过。未改 CRL math。

## 17. M11B doctor result

PASS：`M11B doctor: 34/34 PASS`，`go=true`，`formal_training_started=false`。

doctor 还确认五个 M11B environment reference 的 train/val 数据和 gym registration 均存在；新增的三种 HIQL accounting probe 全部通过。常规 runtime probe 使用 4 个代表性 probe，均满足 finite update、action shape、action bounds、determinism。

## 18. Resolved C010 config

`M11B-C010`：`antmaze-giant-navigate-v0`，`hiql`，`high_low_ss`。

```text
discount=0.995
subgoal_steps=25
actor_p_trajgoal=1.0
actor_p_randomgoal=0.0
high_actor=SingleState K4, non-residual, state_dim=512, update_depth=2
low_actor =SingleState K4, non-residual, state_dim=512, update_depth=2
```

其余 canonical HIQL fields 保持原值：`lr=3e-4`、`batch_size=1024`、actor/value hidden dims 均为 `(512,512,512)`，goal/value sampling 不变。

## 19. Resolved C018 config

`M11B-C018`：`humanoidmaze-large-navigate-v0`，`hiql`，`high_low_ss`。

```text
discount=0.995
subgoal_steps=100
actor_p_trajgoal=1.0
actor_p_randomgoal=0.0
high_actor/low_actor=SingleState K4, non-residual, state_dim=512, update_depth=2
```

Humanoid-specific `subgoal_steps=100` 被保留，没有使用 AntMaze 的 context。

## 20. Resolved C026 config

`M11B-C026`：`humanoidmaze-giant-navigate-v0`，`hiql`，`high_low_ss`。

```text
discount=0.995
subgoal_steps=100
actor_p_trajgoal=1.0
actor_p_randomgoal=0.0
high_actor/low_actor=SingleState K4, non-residual, state_dim=512, update_depth=2
```

Giant/Humanoid canonical discount 和 Humanoid subgoal context 均保持不变。

## 21. Resolved C034 config

`M11B-C034`：`antmaze-large-stitch-v0`，`hiql`，`high_low_ss`。

```text
discount=0.99
subgoal_steps=25
actor_p_trajgoal=0.5
actor_p_randomgoal=0.5
high_actor/low_actor=SingleState K4, non-residual, state_dim=512, update_depth=2
```

Stitch-specific goal sampling 保持 `0.5/0.5`，没有泄漏 Navigate 的 `1.0/0.0`。

## 22. Old/new scientific-config parity

以四个旧失败目录中的 `resolved_config.json` 为 old artifact，与当前修复后 resolver 生成的 resolved agent/configuration 做字段级比较，排除 source commit、fingerprint、run directory、runtime accounting 等 metadata-only 字段。

| config | configuration | resolved agent/scientific config |
|---|---|---|
| C010 | exact equal | exact equal |
| C018 | exact equal | exact equal |
| C026 | exact equal | exact equal |
| C034 | exact equal | exact equal |

比较覆盖 algorithm、environment、dataset semantics、canonical hyperparameters、compute spec、SingleState topology、seed，以及 study 中的 1M/评估 protocol。四项均为 parity PASS。当前 bugfix commit 尚未由本 agent 创建；新 commit 应由用户手动提交并记录到正式 provenance。

## 23. Old failed run preservation strategy

旧目录没有删除、覆盖、复用或篡改。当前检查仍发现 12 个 failed pretraining directories；C010/C018/C026/C034 的旧 `runtime_metadata.json` 状态仍为 `failed`，并保留原始 `Unsupported computation slot` failure reason。已完成的 22 个 CRL/HIQL baseline run 没有被重新计划。

## 24. New rerun directory strategy

新增通用而非 M11B-specific 的 `run_attempt`：

```text
run_attempt=0 -> seed_000
run_attempt=1 -> seed_000__attempt_001
```

`make_run_path`、`create_run_context`、`sweep`、`tools/run.py` 和 `scripts/run_study.sh` 全部贯通。dry-run 已证明四个新实例均指向 `seed_000__attempt_001`，且这些目录尚未创建。failed/aborted/invalid 状态也不会进入 primary aggregate；只有 `completed` 行参与正式结果聚合。

## 25. Exact dry-run command

以下命令已经实际执行；它只做 preflight，不启动训练：

```bash
RLC_PYTHON=/home/eai/Tools/miniforge3/envs/brain_nav/bin/python \
OGBENCH_DATASET_DIR=/data/qijunrong/06-RL/offline-rl/data/raw_ogbench \
bash scripts/run_study.sh \
  --study experiments/M11B_cross_task_computation/study.yaml \
  --configs M11B-C010,M11B-C018,M11B-C026,M11B-C034 \
  --run-attempt 1 \
  --gpus 0,1 \
  --run-root /data/qijunrong/06-RL/offline-rl/exp/RLC/runs \
  --dataset-root /data/qijunrong/06-RL/offline-rl/data/raw_ogbench \
  --train-steps 1000000 \
  --batch-size 1024 \
  --log-interval 5000 \
  --eval-interval 100000 \
  --eval-tasks all \
  --eval-episodes 20 \
  --save-interval 100000 \
  --eval-temperature 0 \
  --dry-run
```

## 26. Expected planned runs = 4

实际输出为：

```text
planned=4 completed=0 failed=0 running=0 retained=0 remaining=4
```

四个 planned IDs 恰为 C010、C018、C026、C034；不是 34、12 或 26。

## 27. Exact future execute command

以下是用户在手动 review、commit，并准备好新的 frozen worktree 后才可执行的命令。当前没有执行它：

```bash
cd /home/eai/Research/RLC-M11B-bugfix-frozen
RLC_PYTHON=/home/eai/Tools/miniforge3/envs/brain_nav/bin/python \
OGBENCH_DATASET_DIR=/data/qijunrong/06-RL/offline-rl/data/raw_ogbench \
bash scripts/run_study.sh \
  --study experiments/M11B_cross_task_computation/study.yaml \
  --configs M11B-C010,M11B-C018,M11B-C026,M11B-C034 \
  --run-attempt 1 \
  --gpus 0,1 \
  --run-root /data/qijunrong/06-RL/offline-rl/exp/RLC/runs \
  --dataset-root /data/qijunrong/06-RL/offline-rl/data/raw_ogbench \
  --train-steps 1000000 \
  --batch-size 1024 \
  --log-interval 5000 \
  --eval-interval 100000 \
  --eval-tasks all \
  --eval-episodes 20 \
  --save-interval 100000 \
  --eval-temperature 0 \
  --execute
```

该命令要求用户自行完成 frozen worktree、Git cleanliness、source commit 记录与最终执行确认。

## 28. Explicit statement: no formal rerun was started

本轮没有执行任何 `--execute`、1M training 或正式补跑。实际执行的只有代码测试、临时目录的一步 lifecycle smoke、doctor 和 `--dry-run`。用户要求的 Git commit、push、worktree 创建和正式实验启动均未由本 agent 完成。

## 29. GO / NO-GO recommendation

结论：`GO for manual review and the four-run formal rerun`；不是“实验结果已经完成”。

GO 依据：root cause 已由旧 failure artifact 定位；high-only、low-only、high+low generic accounting 均通过；legacy/generic 一致；entrypoint 三种 smoke 通过；M9A、M10A、M11A 回归通过；M11B doctor 34/34 PASS；四个目标 scientific config parity exact equal；旧失败目录保留；新 attempt 路径已验证；dry-run 恰为 4；正式 training 未启动。

machine-readable audit 中明确记录了：

```yaml
scientific_semantics_unchanged: true
scientific_behavior_change: false
```

因此可以在用户手动完成 review、commit、frozen worktree 和最终命令确认后，只补跑 C010/C018/C026/C034。不得借本修复重跑已完成 22 个 M11B run、high-only/low-only 本轮不补跑的 8 个配置、M9A、M9B、M10A、M11A 或 M11A-D001。
