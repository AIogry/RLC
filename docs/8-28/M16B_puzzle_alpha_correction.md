# M16B：Puzzle-play GCIQL `alpha=1.0` 校正实验

## 1. 目的与结论边界

M16A 的 16 个正式 run 实际全部使用 `alpha=0.3`。根据当前采用的
OGBench Puzzle-play GCIQL benchmark 设定，Puzzle 应使用 `alpha=1.0`。
因此 M16B 是对 M16A 的 alpha 校正与部分复现实验，不覆盖 M16A 的全部结构条件。

本轮只运行：

| 环境 | B000 Flat | S002 Mixer L2 |
|---|---:|---:|
| Puzzle-3x3 | 1 | 1 |
| Puzzle-4x4 | 1 | 1 |
| Puzzle-4x5 | 1 | 1 |
| Puzzle-4x6 | 1 | 1 |

共 8 个 run，全部 `seed=0`。S001 和 S004 明确延期，不得从本轮结果推断它们在
`alpha=1.0` 下的表现。

M16B 的首要受控比较是同一环境内的 `B000` 与 `S002`。M16B 与 M16A 的差异则是
`alpha=1.0` 对比 `alpha=0.3` 的敏感性/校正比较，不是纯粹的结构因果比较。

## 2. 配置保证

M16B 不修改 GCIQL 或 MLP-Mixer 实现。运行时真正生效的 alpha 来源是每个配置中的：

```yaml
agent_overrides:
  alpha: 1.0
```

`fixed_design.alpha` 和 `alpha_policy.value` 是声明层记录；`agent_overrides.alpha`
是 runtime authority。每个配置还在 `factors.alpha` 中重复记录 `1.0`，以避免只改
Study 声明但实际运行仍回退到 GCIQL 默认值 `0.3`。

非执行预检 `tools/m16b_doctor.py` 会检查：

- 4 个 Puzzle 环境、B000/S002 两个条件、seed 0 的完整 8-cell 矩阵；
- train/validation 数据存在；
- Study、factor、agent override 和解析后的 runtime config 均为 `alpha=1.0`；
- B000 的 computation slots 关闭，S002 的 actor/value/critic 均为 Puzzle-token
  MLP-Mixer L2；
- GCIQL 网络可以实例化；
- 规范 run path 不存在，因而不会覆盖既有结果。

预检不会执行 Git 操作、创建 run 目录或启动训练。

## 3. 固定训练与评估协议

除 alpha 外，保持 M16A 的协议：

- GCIQL，`actor_loss=ddpgbc`；
- actor/value hidden dims `(512,512,512)`，`layer_norm=true`，`lr=3e-4`；
- `batch_size=1024`，`train_steps=1,000,000`；
- `discount=0.99`，`expectile=0.9`，`tau=0.005`；
- GCDataset、同样的 goal sampling、`gc_negative=true`、`p_aug=0`；
- 每 5k log、每 100k eval/save；每个 eval task 20 episodes；
- `eval_tasks=all`、temperature `0`、Gaussian noise `None`、video episodes `0`；
- 保存 best 和 last checkpoint；主终点为 `final@1M`，同时保留 best、best step、
  last-3 mean 与 100k–1M normalized AUC。

## 4. 用户手动执行流程

以下步骤只应由用户完成。Agent 不执行 Git 操作，也不启动正式训练。

### 4.1 审阅、提交并创建新的冻结 worktree

在主线 `/home/eai/Research/RLC` 审阅本次新增文件后，由用户自行完成 Git 提交，
并从该提交创建新的冻结 worktree，例如：

```bash
cd /home/eai/Research/RLC
# 用户自行完成 status / add / commit / push（如需要）
git worktree add --detach /home/eai/Research/RLC-m16b-exp <M16B_COMMIT_SHA>
```

不要修改或复用 `/home/eai/Research/RLC-m16a-exp`。M16A 冻结 worktree 保持不变，
M16B 应使用独立的 `/home/eai/Research/RLC-m16b-exp`。

### 4.2 在新的冻结 worktree 中做非执行预检

```bash
cd /home/eai/Research/RLC-m16b-exp
export OGBENCH_DATASET_DIR=/data/qijunrong/06-RL/offline-rl/data/raw_ogbench
export PYTHONPATH=/home/eai/Research/RLC-m16b-exp
export XLA_PYTHON_CLIENT_PREALLOCATE=false
RLC_PYTHON=/home/eai/Tools/miniforge3/envs/brain_nav/bin/python

JAX_PLATFORMS=cpu "$RLC_PYTHON" tools/m16b_doctor.py \
  --study experiments/M16B_puzzle_alpha_correction/study.yaml \
  --dataset-root /data/qijunrong/06-RL/offline-rl/data/raw_ogbench \
  --run-root /data/qijunrong/06-RL/offline-rl/exp/RLC/runs \
  --gpus 0,1
```

预期输出为 `M16B PREFLIGHT: PASS`、8 jobs、`alpha=1.0`。若失败，不得执行训练。

### 4.3 用户检查 dry-run，再手动执行正式训练

先将下面命令以 `--dry-run` 执行并检查 8 个 config ID、数据路径、run root、协议和
GPU 计划：

```bash
bash scripts/run_study.sh \
  --study experiments/M16B_puzzle_alpha_correction/study.yaml \
  --configs M16B-3x3-B000,M16B-3x3-S002,M16B-4x4-B000,M16B-4x4-S002,M16B-4x5-B000,M16B-4x5-S002,M16B-4x6-B000,M16B-4x6-S002 \
  --gpus 0,1 \
  --run-root /data/qijunrong/06-RL/offline-rl/exp/RLC/runs \
  --dataset-root /data/qijunrong/06-RL/offline-rl/data/raw_ogbench \
  --train-steps 1000000 --batch-size 1024 --log-interval 5000 \
  --eval-interval 100000 --eval-tasks all --eval-episodes 20 \
  --eval-temperature 0 --save-interval 100000 \
  --save-best-checkpoint --save-last-checkpoint --dry-run
```

确认无误后，用户将末尾的 `--dry-run` 改为 `--execute`，由用户手动启动正式训练。
两张 GPU 时 sweep 会同时运行两个 job；一个 job 结束后才会从队列取下一个，这是
预期行为，不是只计划了一个任务。

## 5. 后续 alpha 策略

今后所有 Puzzle Study 必须显式选择 alpha，不得依赖
`impls/agents/gciql.py` 的默认值：

1. Study 的 `fixed_design.alpha` 记录研究选择；
2. Study 的 `alpha_policy.value` 重复记录选择及来源；
3. 每个 executable config 的 `factors.alpha` 与
   `agent_overrides.alpha` 必须显式存在且一致；
4. 启动前必须运行对应 doctor，确认解析后的 runtime config 与声明一致；
5. `alpha=0.3` 和 `alpha=1.0` 使用不同 Study ID、config ID、run root 子目录，
   不覆盖历史结果。

当前建议：

- M16A：永久保留为 `alpha=0.3` 的探索性结果；
- M16B：作为 `alpha=1.0` 的 B000/S002 校正实验；
- 等 M16B 完成后，再决定 Puzzle 主线固定使用 `alpha=1.0` 还是 `alpha=0.3`；
- 如果要声称某个 alpha 在 Puzzle 上更好，应在同一配置矩阵、相同 seed 和相同协议
  下比较，而不是只比较 M16A 的结构结果与 M16B 的部分结果。

## 6. 结果解释门槛

本轮只有一个训练 seed，且每个评估点 20 episodes。结果只能作为 alpha 校正后的
探索性证据。必须等 8 个 run 完成后，分别报告每个环境的：

- `last@1M`、`best`、`best_step`、`last-3 mean`、normalized AUC；
- task-level success，而不只报告 overall；
- M16B 内部 S002-B000 差值；
- M16B 与 M16A 在相同环境/条件下的 alpha 敏感性差值。

不得把 M16B 的 B000/S002 结果外推为 S001/S004 的 alpha=1.0 结果。
