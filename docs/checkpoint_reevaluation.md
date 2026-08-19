# RLC checkpoint re-evaluation 基础设施

本文档说明 RLC 中通用的 checkpoint 后评估机制。它把 checkpoint
re-evaluation 作为独立的 experiment object，不向训练 run 的 `eval.csv`、
`summary.json` 或 checkpoint 写入任何内容。

## 1. 为什么后评估必须独立

训练期评估服务于训练监控，使用训练命令声明的协议；checkpoint 后评估服务于
结果确认，通常需要更多 episodes、更严格的 common-random-number pairing 和
独立的统计汇总。两者不能混为一个文件：

| 类型 | 作用 | M10A 设置 |
|---|---|---|
| training-time evaluation | 训练过程监控，保留原始历史 | 20 episodes/task |
| post-hoc reevaluation | 固定 checkpoint 的高精度结果确认 | 100 episodes/task |

后评估的输出根目录默认为：

```text
/data/qijunrong/06-RL/offline-rl/exp/RLC/reevaluations
```

原始训练目录仍然位于：

```text
/data/qijunrong/06-RL/offline-rl/exp/RLC/runs
```

## 2. 训练 seed 与 evaluation seed

`training_seed` 选择被评估的模型 checkpoint；`evaluation_seed` 选择环境 episode
集合。这两个字段在 metadata、目录和统计中始终分开，不能把 evaluation seed
简称为 `seed`。

当前版本的确定性 seed hierarchy 为 `common_task_episode_v1`：

```text
task_seed    = derive_seed(evaluation_seed, task_id)
episode_seed = derive_seed(task_seed, episode_index, 0)
actor_seed   = derive_seed(task_seed, episode_index, 1)
noise_seed   = derive_seed(task_seed, episode_index, 2)
```

seed 中不包含 config ID、training seed、checkpoint hash 或 GPU ID。因此不同策略
在相同 task/episode 上使用完全相同的环境随机实现，可以通过
`paired_episode_id=taskXX_epYYY` 做配对比较。

seed 实现位于 `impls/utils/evaluation.py::common_episode_seeds`，使用已有的
`impls/utils/reproducibility.py::derive_seed`，并由测试覆盖。

## 3. 评估 API

现有训练调用仍然使用：

```python
stats, trajectories, renders = evaluate(...)
```

该 API 的返回结构和 trajectory 语义保持不变。新的后评估 API 是：

```python
records = evaluate_episodes(
    agent,
    env,
    task_id=task_id,
    task_name=task_name,
    evaluation_seed=evaluation_seed,
    episode_indices=missing_indices,
    ...,
)
```

两个 API 共用同一个内部 `_rollout_episode`，所以不会形成两套策略执行语义。
`evaluate_episodes` 不保存完整 observation/action trajectory，只返回每个 episode
一条紧凑记录，内存不会随 episode horizon 累积。success 也使用共享的
`extract_episode_success`，遇到多个互相冲突的 success 信号会直接失败。

## 4. source-run reconstruction 与 provenance

后评估绝不使用当前 Study YAML 重新推导 agent architecture。它使用 source run
中的 `resolved_config.json`，具体流程是：

```text
source runtime_metadata.json
        +
source resolved_config.json
        ↓
algorithm / resolved agent config / dataset class
        ↓
make_env_and_datasets + one dataset example batch
        ↓
agents registry -> agent skeleton
        ↓
restore_agent(source_run_dir, checkpoint_step)
        ↓
evaluate_episodes(restored_agent, env, ...)
```

启动评估前会验证：

- source `runtime_metadata.json`、`resolved_config.json` 和 checkpoint 存在；
- source status 为 `completed`，formal source git worktree 为 clean；
- source 路径与 study/config/environment/training seed 一致；
- resolved config fingerprint 在 runtime metadata、resolved config 和重新计算值
  之间一致；
- checkpoint 文件存在、SHA256 已记录；
- checkpoint metadata 与 source 的 environment、study、config、slug、commit 和
  training seed 一致；
- restored 参数以及 action probe 是有限值。

## 5. 输出 schema

单个 run 的 canonical layout 为：

```text
<reeval_root>/<study_id>/<reevaluation_id>/
  campaign_metadata.json
  manifest.csv
  config_summary.csv
  task_config_summary.csv
  <config_id>__<config_slug>/
    <environment>/
      seed_<training_seed>/
        reevaluation_metadata.json
        episode_results.csv
        task_summary.csv
        summary.json
```

`episode_results.csv` 每个 episode 一行，至少包含：

- study/config/environment/training seed/checkpoint step；
- task ID/name、episode index；
- evaluation/task/episode/actor/noise seeds；
- success、return、length、terminated、truncated；
- `paired_episode_id`；
- scalar-only 的 `final_info_json`。

`task_summary.csv` 每个 task 一行，包含 episode count、success count/rate、return
均值和总体标准差、episode length 均值和总体标准差、success standard error，以及
95% Wilson interval。

`summary.json` 明确区分：

- `evaluation/overall_success`：五个 task success rate 的宏平均；
- `overall_episode_sampling_se`：按 task 分层计算的 episode sampling SE；
- training-seed variability：写入 campaign-level `config_summary.csv`，不能被
  episode SE 替代。

## 6. resume 与生命周期

单 run 在 rollout 前写入 `status=running` 的 metadata。每完成一个 episode，
`episode_results.csv` 都会 flush。状态可能是：

```text
running / completed / failed / aborted / invalid
```

`--resume` 只允许在 checkpoint SHA256 和 reevaluation protocol fingerprint 都
一致时继续；已有 `(task_id, episode_index)` 必须唯一且有效，只补缺失 episode，
不会重复。没有 `--resume` 时，如果 canonical output 已存在则直接失败，不允许
静默覆盖。

## 7. launcher 与 GPU 调度

单 checkpoint：

```bash
python tools/reevaluate_checkpoint.py \
  --spec experiments/M10A_fixed_budget_placement/reevaluations/M10A-R001.yaml \
  --source-run-dir /path/to/source/seed_000 \
  --reeval-root /data/qijunrong/06-RL/offline-rl/exp/RLC/reevaluations
```

整个 study：

```bash
python tools/reevaluate_study.py \
  --spec experiments/M10A_fixed_budget_placement/reevaluations/M10A-R001.yaml \
  --gpus 0,1 \
  --dry-run
```

正式执行需要显式使用 `--execute`，且 reevaluation worktree 必须 clean。每张
physical GPU 只有一个 persistent worker，worker 依次处理 checkpoint；实际
`CUDA_VISIBLE_DEVICES` 和分配 GPU 会写入 run metadata。

实现没有引入新的分布式框架，调度模式沿用了现有 `tools/sweep.py` 的动态 GPU
队列思想。

## 8. 将来复用到 M10B/M11

未来 study 只需要新增 reevaluation YAML，声明：

- `reevaluation_id`、`source_study_id`、`source_run_root`；
- checkpoint step；
- environments、configs、training seeds；
- task selection、episodes/task、evaluation seed、seed scheme；
- temperature、Gaussian noise 和 video episodes。

通用 Python 代码不包含 M10A-specific config ID 或路径。source architecture 从
每个 run 自带的 resolved config 恢复，所以后续 Study YAML 发生变化也不会改变
历史 checkpoint 的重建方式。
