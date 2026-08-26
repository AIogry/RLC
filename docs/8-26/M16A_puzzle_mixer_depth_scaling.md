# M16A：GCIQL 在 Puzzle 复杂度上的结构化 Mixer 深度实验

## 1. 研究问题与边界

M16A 检验：在 GCIQL 的算法超参数、训练预算和评估协议保持不变时，面向 Puzzle 的结构化 token 表示是否随任务复杂度产生收益，以及 Mixer 深度 `L=1/2/4` 是否改变这一关系。

本实验只把同一环境内的 B000 与 S001/S002/S004 作为受控表示/深度对比。不同 Puzzle 尺寸之间的比较是复杂度相关的描述性比较，不等同于严格的跨任务因果效应。实验不做 MAC/参数量匹配；因此参数量和计算量必须作为结果解释的一部分记录。

正式实验约束：16 个单元（4 环境 × 4 条件 × seed 0），每个单元 1,000,000 个训练 step；在全部 16 个单元完成前不形成 M16A 正式结论。

## 2. 预注册矩阵

| 条件 | 语义 | 结构 | Mixer blocks | 3x3/4x4/4x5/4x6 各 1 个 |
|---|---|---|---:|---:|
| B000 | Flat baseline | canonical vector MLP | 0 | 4 |
| S001 | Structured Mixer L1 | Puzzle tokens + feedforward MLP-Mixer | 1 | 4 |
| S002 | Structured Mixer L2 | Puzzle tokens + feedforward MLP-Mixer | 2 | 4 |
| S004 | Structured Mixer L4 | Puzzle tokens + feedforward MLP-Mixer | 4 | 4 |

环境及 token 数：

| 环境 | `num_buttons` | 原始 observation 维度 | action 维度 |
|---|---:|---:|---:|
| `puzzle-3x3-play-v0` | 9 | 55 | 5 |
| `puzzle-4x4-play-v0` | 16 | 83 | 5 |
| `puzzle-4x5-play-v0` | 20 | 99 | 5 |
| `puzzle-4x6-play-v0` | 24 | 115 | 5 |

配置文件位于 `experiments/M16A_puzzle_mixer_depth_scaling/configs/`，按环境显式展开为 16 个配置，避免 `num_buttons` 被运行时隐式推断。

## 3. 冻结的算法与训练协议

所有条件都使用 canonical GCIQL：

- actor/value hidden dims：`(512, 512, 512)`；`layer_norm=true`；`lr=3e-4`；`batch_size=1024`。
- `discount=0.99`、`expectile=0.9`、`tau=0.005`、`actor_loss=ddpgbc`、`alpha=0.3`、`const_std=true`。
- `discrete=false`、`dataset_class=GCDataset`、`gc_negative=true`、`p_aug=0`、`frame_stack=null`。
- value goal sampling：current/traj/random = `0.2/0.5/0.3`，geometric sampling 开启。
- actor goal sampling：current/traj/random = `0/1/0`，geometric sampling 关闭。
- Polyak 语义冻结为：`target_new = tau * online_post_gradient + (1 - tau) * target_old`。

结构化条件在 actor、value、critic 三个 GCIQL slot 同时启用，且均为：
`structure=puzzle_tokens`、`topology=feedforward`、`block=mlp_mixer`、`primitive=mlp`、`credit=direct`；`robot_dim=19`、`button_feature_dim=4`、`token_dim=128`、`robot_hidden_dim=128`、`token_mlp_hidden_dim=64`、`channel_mlp_hidden_dim=256`、`index_embedding=true`、`readout=mean`、`tm_mode=none`。唯一改变的结构因素是 Mixer block 数 `L=1/2/4`。

B000 的三个 computation slot 全部关闭，因此它保留 canonical Flat GCIQL MLP。M16A 不引入 recurrence、TwoState、HRM、adaptive depth 或新的 loss/optimizer/reward/goal-sampling 逻辑。

训练/评估：`1M steps`、每 `5k` log、每 `100k` eval/save、每个 canonical task 20 episodes、evaluation temperature 0、Gaussian noise null、video episodes 0；保存 best 与 last checkpoint。主指标为 `evaluation/overall_success` 在 `1M` 的最终值；次指标为 best、best step、100k–1M normalized AUC 和 last-3 mean。

## 4. 预检与架构核算

已使用 `tools/m16a_doctor.py` 做非执行预检，检查内容包括：完整矩阵、唯一 `(environment, condition)`、数据 train/val 文件、seed、算法超参数一致性、结构化字段、无 recurrence/HRM 标记、GCIQL 网络实例化、输出路径不覆盖，以及 GPU 轮转计划。实际 run root 的预检结果为：

```text
M16A PREFLIGHT: PASS
16 jobs; environments=4; conditions=4; seeds=[0]
Formal training was not started. Manual launch remains required.
```

预检得到的总参数量 / Dense MAC / actor-value-critic 顺序深度如下。参数量和 MAC 是整个 GCIQL agent 的合计；critic ensemble 的物理 Dense 参数/MAC 已计入；深度是包含 readout 的顺序 Dense 路径深度。它们不是性能结果，也不是 MAC 匹配设计。

| 环境 | 条件 | 参数量 | Dense MAC | 深度 A/V/C |
|---|---|---:|---:|---:|
| 3x3 | B000 | 2,347,016 | 2,331,648 | 4/4/4 |
| 3x3 | S001 | 833,324 | 3,535,104 | 7/7/7 |
| 3x3 | S002 | 1,101,904 | 6,484,224 | 11/11/11 |
| 3x3 | S004 | 1,639,064 | 12,382,464 | 19/19/19 |
| 4x4 | B000 | 2,461,704 | 2,446,336 | 4/4/4 |
| 4x4 | S001 | 840,520 | 5,857,536 | 7/7/7 |
| 4x4 | S002 | 1,112,712 | 11,100,416 | 11/11/11 |
| 4x4 | S004 | 1,657,096 | 21,586,176 | 19/19/19 |
| 4x5 | B000 | 2,527,240 | 2,511,872 | 4/4/4 |
| 4x5 | S001 | 844,632 | 7,184,640 | 7/7/7 |
| 4x5 | S002 | 1,118,888 | 13,738,240 | 11/11/11 |
| 4x5 | S004 | 1,667,400 | 26,845,440 | 19/19/19 |
| 4x6 | B000 | 2,592,776 | 2,577,408 | 4/4/4 |
| 4x6 | S001 | 848,744 | 8,511,744 | 7/7/7 |
| 4x6 | S002 | 1,125,064 | 16,376,064 | 11/11/11 |
| 4x6 | S004 | 1,677,704 | 32,104,704 | 19/19/19 |

从设计上可预期：结构化 Mixer 的参数量低于 Flat MLP，但随着 `num_buttons` 和 `L` 增加，token-aware MAC 增加；因此若结构化条件表现不同，不能仅解释为“更深所以更好”或“参数更少所以更差”。

## 5. 正式启动前的人工步骤

本轮没有替用户启动正式训练，也没有执行任何 Git 操作。用户应先自行完成 Git 审阅/提交/推送和环境确认，然后在主线 RLC 中再次执行预检。预检通过后，可由用户手动执行通用 sweep（下面的命令只负责训练，不执行 Git 操作）：

```bash
cd /home/eai/Research/RLC
export OGBENCH_DATASET_DIR=/data/qijunrong/06-RL/offline-rl/data/raw_ogbench
export PYTHONPATH=/home/eai/Research/RLC
export XLA_PYTHON_CLIENT_PREALLOCATE=false
RLC_PYTHON=/home/eai/Tools/miniforge3/envs/brain_nav/bin/python

JAX_PLATFORMS=cpu "$RLC_PYTHON" tools/m16a_doctor.py \
  --study experiments/M16A_puzzle_mixer_depth_scaling/study.yaml \
  --dataset-root /data/qijunrong/06-RL/offline-rl/data/raw_ogbench \
  --run-root /data/qijunrong/06-RL/offline-rl/exp/RLC/runs \
  --gpus 0,1

"$RLC_PYTHON" tools/sweep.py \
  --study experiments/M16A_puzzle_mixer_depth_scaling/study.yaml \
  --gpus 0,1 \
  --run-root /data/qijunrong/06-RL/offline-rl/exp/RLC/runs \
  --run-attempt 0 \
  --dataset-root /data/qijunrong/06-RL/offline-rl/data/raw_ogbench \
  --execute \
  --train_steps=1000000 --batch_size=1024 --log_interval=5000 \
  --eval_interval=100000 --eval_tasks=all --eval_episodes=20 \
  --eval_temperature=0 --video_episodes=0 \
  --save_interval=100000 --save-best-checkpoint --save-last-checkpoint
```

命令中省略 `--eval_gaussian`，因此使用 parser 的 `None` 默认值，和协议中的 `null` 一致。正式启动前不得把 `--max-runs` 用于截断 16 个单元，也不得因某个条件早期表现较差而提前停止；只允许基础设施错误、NaN 或 OOM 等必要故障处理。

## 6. 实验结束后的汇总

完成后，用统一分析脚本读取 16 个 run 的 `eval.csv` 和 runtime metadata：

```bash
cd /home/eai/Research/RLC
PYTHONPATH=/home/eai/Research/RLC \
/home/eai/Tools/miniforge3/envs/brain_nav/bin/python tools/analyze_m16a.py \
  --study experiments/M16A_puzzle_mixer_depth_scaling/study.yaml \
  --run-root /data/qijunrong/06-RL/offline-rl/exp/RLC/runs \
  --run-attempt 0 \
  --output-dir docs/milestones/M16A_results
```

脚本输出 long-format CSV、JSON 和 Markdown：每个单元的 final/best/best step/last-3/AUC、完整性状态、参数/MAC/depth，以及同环境相对 B000 的 `Δ last@1M`、`Δ best`、`Δ normalized AUC`。缺少任意评估点时 AUC 留空，缺少 1M 时 final 留空，不把 partial run 当作正式结果。

## 7. 当前状态

M16A 的代码、配置、预检和结果汇总工具已准备完毕；M15 结构化计算与真实 Puzzle tiny lifecycle 回归测试通过（13/13），M16A 预检通过（16/16）。截至本文件生成时，正式 1M 训练尚未启动，因而不存在可供科学分析的 M16A performance result。
