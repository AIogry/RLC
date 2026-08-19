# M10A-R001 checkpoint re-evaluation 方案与验证记录

## 1. 固定实验协议

M10A-R001 是 M10A 500k final checkpoint 的 post-hoc reevaluation，不是重新训练，
也不是对原始训练 `eval.csv` 的改写。

```text
study_id             = M10A
reevaluation_id      = M10A-R001
environment          = antmaze-large-navigate-v0
checkpoint_step      = 500000
configs              = M10A-C001 ... M10A-C011
training seeds       = 0, 1, 2
tasks/checkpoint     = 5
episodes/task        = 100
episodes/checkpoint  = 500
checkpoints          = 33
total episodes       = 16,500
evaluation_seed      = 20260819
eval_temperature     = 0.0
eval_gaussian        = null
video_episodes       = 0
seed scheme          = common_task_episode_v1
```

原始训练期历史保持不变：M10A training-time evaluation 是 20 episodes/task；
M10A-R001 是同一批 500k checkpoints 的 100 episodes/task 高精度复评。

## 2. spec

声明文件为：

```text
experiments/M10A_fixed_budget_placement/reevaluations/M10A-R001.yaml
```

正式输出默认放置于：

```text
/data/qijunrong/06-RL/offline-rl/exp/RLC/reevaluations/M10A/M10A-R001/
```

实现和通用 schema 见 [`docs/checkpoint_reevaluation.md`](../checkpoint_reevaluation.md)。

## 3. 实施验证

### CPU / 单元测试

新增 `tests/experiment/test_reevaluation.py`，覆盖：

- legacy `evaluate` 返回 API；
- streaming episode records 与无 trajectory retention；
- deterministic seed；
- common-random-number pairing；
- success 与 task/overall accounting；
- duplicate episode rejection；
- source resolved-config fingerprint 与 checkpoint SHA；
- campaign seed-level aggregation；
- M10A inventory 的 33 个 checkpoint。

本地执行结果：

```text
Ran 9 tests ... OK
```

### M10A dry-run

使用：

```bash
python tools/reevaluate_study.py \
  --spec experiments/M10A_fixed_budget_placement/reevaluations/M10A-R001.yaml \
  --dry-run --gpus 0,1
```

得到：

```text
source runs       = 33
checkpoints       = 33
tasks/checkpoint  = 5
episodes/task     = 100
episodes/run      = 500
total episodes    = 16500
statuses: planned=33 completed=0 running=0 failed=0 aborted=0 invalid=0
```

dry-run 没有启动任何 rollout。

### 小规模真实 GPU smoke

实现阶段没有运行正式的 100 episodes/task。仅在真实 M10A checkpoint 上运行：

- C005 SingleState，seed 0，5 tasks × 2 episodes = 10 episodes；
- C001 vanilla，seed 0，5 tasks × 2 episodes = 10 episodes。

两次 smoke 的 metadata 均记录：

```text
jax_backend    = gpu
jax_devices    = ["cuda:0"]
task_count     = 5
total_episodes = 10
```

C005 smoke 的 overall success 为 `0.8`，C001 smoke 的 overall success 为 `0.9`。
这两个数仅用于验证 restore、五 task rollout 和 artifact schema，不能当作正式
100-episode 结果。

## 4. 正式执行前检查

正式执行前应确认：

1. reevaluation worktree clean；
2. dry-run 仍显示 33/33 planned 或符合预期的 resumable 状态；
3. 输出根目录没有需要人工覆盖的既有 canonical artifacts；
4. GPU 分配与空闲情况已确认；
5. 执行命令使用 `--execute`，不要在实现阶段用 smoke override 代替正式协议。

正式执行示例：

```bash
python tools/reevaluate_study.py \
  --spec experiments/M10A_fixed_budget_placement/reevaluations/M10A-R001.yaml \
  --source-run-root /data/qijunrong/06-RL/offline-rl/exp/RLC/runs \
  --reeval-root /data/qijunrong/06-RL/offline-rl/exp/RLC/reevaluations \
  --gpus 0,1 --execute
```

实现阶段未执行此正式命令。
