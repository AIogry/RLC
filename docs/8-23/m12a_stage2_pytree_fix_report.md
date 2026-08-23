# M12A Stage-2 PyTree Fix Report

## 结论

M12A Stage 2 的失败原因已修复并验证。根因是 frozen critic restore 后，把 target agent 的参数树从普通 `dict` 改成了 `FrozenDict`，导致 `optax.multi_transform` 在第一次 update 时无法匹配参数、gradient、mask 和 optimizer state 的 PyTree 结构。

本轮完成了代码修复、真实 C001 source checkpoint smoke、FF/SS-K4 first-update regression、10-step smoke、protocol propagation 修复以及 Stage 2 `run_attempt=1` dry-run。没有启动正式 rerun，也没有执行 Git 操作。

## 1. Exact code fix

修改 [`impls/utils/flax_utils.py`](../../RLC-M12A-final/impls/utils/flax_utils.py)：

1. 新增 `_mapping_like(template, values)`，按照 target mapping 的容器类型重建参数树；
2. 新增 `_coerce_subtree_like(source, template)`，递归地将 source critic subtree 转换为 target subtree 的嵌套 mapping 结构；
3. `restore_module_from_checkpoint()` 不再无条件执行 `freeze(target_params)`；
4. restore 前后显式比较 `jax.tree_util.tree_structure(params_before)` 和 `tree_structure(params_after)`；
5. 只替换参数 subtree，不接触 target `TrainState` 的 optimizer 或 `opt_state`。

核心逻辑现在是：

```python
params_before = agent.network.params
target_module = params_before[target_key]
source_module = _coerce_subtree_like(source_module, target_module)
values = dict(params_before)
values[target_key] = source_module
params_after = _mapping_like(params_before, values)
assert tree_structure(params_before) == tree_structure(params_after)
network = agent.network.replace(params=params_after)
```

同一修复已同步到当前工作区 `/home/eai/Research/RLC` 和实际失败的 frozen worktree `/home/eai/Research/RLC-M12A-final`。未修改 CRL、SingleState 或 DDPG+BC 数学。

## 2. Before/after PyTree evidence

正式失败前，实际 C002/C003 target agent 的参数树结构为：

```text
before restore:
  params root = dict
  modules_actor = dict
  modules_critic = dict

after old restore_module_from_checkpoint:
  params root = FrozenDict
  modules_actor = FrozenDict
  modules_critic = FrozenDict
```

而 `CRLPolicyExtractorAgent.create()` 中的 optimizer labels 和 optimizer state 是按照 restore 前的普通 `dict` PyTree 创建的。于是第一次 update 在 `optax.mask_pytree()` 处出现：

```text
prefix subtree: dict
full pytree: flax.core.FrozenDict
```

修复后：

```text
params root before = dict
params root after  = dict
tree_structure(before) == tree_structure(after) = True
```

source subtree 也会递归匹配 target subtree 的 mapping 容器类型，而不是只检查 root 类型。

## 3. Mandatory FF and SS-K4 regression

新增测试：

```text
tests.integration.test_m12a_frozen_critic
  test_restore_then_real_update_preserves_ff_and_ss_optimizer_pytrees
```

该测试对 FF 与 SS-K4 均执行完整路径：

```text
create target agent
create completed source last@1M checkpoint
validate_source_run_dependency
restore_module_from_checkpoint
agent.update(batch)
```

两种 actor 均满足：

- first update succeeds；
- restore 前后 parameter PyTree structure 相同；
- target optimizer state 在 restore 前后 exact unchanged；
- actor 参数发生变化；
- critic 参数 exact unchanged；
- critic fingerprint 不变；
- 10 次 update 后 optimizer step 为 11；
- 所有 loss/metrics finite。

测试结果：

```text
FF      PASS
SS-K4   PASS
```

## 4. Real production-source smoke

使用实际 C001 formal source checkpoint，而非伪造 checkpoint：

- `M12A-C001 seed_000 last@1M`；
- source status `completed`；
- SHA 和 critic fingerprint 通过；
- 使用真实 AntMaze-Large dataset；
- 未创建新的 formal run directory。

结果：

| configuration | first update | optimizer step | metrics | actor | critic |
|---|---|---:|---:|---|---|
| C002 FF | PASS | 2 | 14 finite | changed | exact frozen |
| C003 SS-K4 | PASS | 2 | 14 finite | changed | exact frozen |

此外，专项测试与 sweep 测试合计 20 项全部通过；compileall 也通过。

## 5. Stage-1 provenance mismatch

已有 C001 的 `runtime_metadata.json` 记录了：

```text
save_best_checkpoint = true
```

这与 M12A Stage-1 protocol 的 `false` 不一致。但三个 C001 的 checkpoint index 均显示：

```text
best = null
last_step = 1000000
```

并且 source checkpoint 均是正确的 `last@1M`，没有 best checkpoint 被创建或选用。因此该问题只影响 launcher provenance 记录，不影响 frozen critic 的 scientific identity 或 source selection。

结论：C001 不需要重跑。

## 6. Protocol propagation fix

修改 [`tools/sweep.py`](../../RLC-M12A-final/tools/sweep.py)，增加 generic Study protocol defaults：

- Configuration 可声明 `protocol_stage`；
- sweep 根据 Study 的对应 protocol section 填充未显式提供的 checkpoint flags；
- 显式命令行 flag 优先于 Study 默认值；
- 不包含 hard-coded M12A、AntMaze 或 config ID。

M12A configs 现在声明：

```text
C001 -> protocol_stage: stage1
C002 -> protocol_stage: stage2
C003 -> protocol_stage: stage2
```

实际验证：

```text
C001 -> --no-save-best-checkpoint --save-last-checkpoint
C002 -> --save-best-checkpoint --save-last-checkpoint
```

因此未来 Stage 1 即使用户不额外重复输入 save-best flag，运行时 metadata 也会记录 Study protocol 的正确值。

## 7. Failed attempt-0 preservation

六个失败的 C002/C003 `attempt_0` directories 均未删除、未覆盖、未修改。它们保留为失败 provenance。修复后的正式 rerun 应使用 `run_attempt=1`，不会复用 attempt-0 目录。

## 8. Stage-2 attempt-1 dry-run

执行命令：

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
OGBENCH_DATASET_DIR=/data/qijunrong/06-RL/offline-rl/data/raw_ogbench \
PYTHONPATH=. \
/home/eai/Tools/miniforge3/envs/brain_nav/bin/python tools/sweep.py \
  --study experiments/M12A_frozen_critic_policy_extraction/study.yaml \
  --configs M12A-C002,M12A-C003 \
  --run-attempt 1 \
  --gpus 0 \
  --run-root /data/qijunrong/06-RL/offline-rl/exp/RLC/runs \
  --dataset-root /data/qijunrong/06-RL/offline-rl/data/raw_ogbench \
  --dry-run \
  --train_steps=1000000 \
  --batch_size=1024 \
  --log_interval=5000 \
  --eval_interval=100000 \
  --eval_tasks=all \
  --eval_episodes=20 \
  --save_interval=100000 \
  --eval_temperature=0.0
```

结果：

```text
total=6 planned=6 completed=0 failed=0 running=0 retained=0 remaining=6
```

计划路径全部为 `seed_000__attempt_001`、`seed_001__attempt_001`、`seed_002__attempt_001`，C002/C003 各 3 个，共 6 个。

## 9. Same-seed checkpoint pairing verification

从现有 C002/C003 attempt-0 runtime metadata 读取 dependency records 后，三组 pair 均严格相同：

| seed | checkpoint role/step | source SHA equal | critic fingerprint equal |
|---:|---|---|---|
| 0 | `last@1000000` | yes | yes |
| 1 | `last@1000000` | yes | yes |
| 2 | `last@1000000` | yes | yes |

具体 SHA/fingerprint 均对应各自同 seed 的 C001 source，未发生跨 seed 或 fallback。

## 10. GO / NO-GO recommendation

实现与 preflight 层面：GO。

- PyTree restore bug 已修复；
- FF/SS-K4 first-update regression 通过；
- critic frozen invariant 通过；
- optimizer state isolation 通过；
- Stage-1 protocol propagation 已修复；
- C001 不需要重跑；
- attempt-1 dry-run 规划 6 个 runs；
- same-seed source SHA/fingerprint pairing 通过。

正式 rerun 层面：本轮只建议在用户 review 代码、手动完成 Git 操作并确认运行环境后 GO；本 agent 不会自动启动 attempt-1。若只依据当前 dry-run 状态，formal execution 仍应视为待用户批准，而不是已经开始。

## 11. Explicit final statement

**No formal M12A Stage-2 rerun was started.**

本轮没有执行 `--execute`、没有修改六个 failed attempt-0 directories、没有重跑 C001、没有执行 Git commit/push/branch/worktree 操作。
