# M16C：Puzzle-4x4 S002 MLP-Mixer alpha 扫描

日期：2026-08-29  
状态：已完成配置与非执行预检准备；尚未进行 Git 操作或正式训练。

## 1. 研究目的

M16A/M16B 显示，Puzzle-token MLP-Mixer L2（S002）在 Puzzle-4x4 上的表现对 DDPG+BC
系数 `alpha` 高度敏感：

| 来源 | alpha | S002 final@1M | S002 norm AUC |
|---|---:|---:|---:|
| M16A | 0.3 | 0.71 | 0.470 |
| M16B | 1.0 | 0.47 | 0.313 |

M16C 仅探索 S002 在 4x4 上的中间/更低 alpha 区域，补足以下四个点：

```text
Puzzle-4x4 × S002 × alpha {0.1, 0.2, 0.5, 0.7} × seed 0
= 4 个新正式 run
```

连同已完成的 `alpha=0.3` 和 `1.0` anchors，M16C 最终形成 S002 的六点 seed-0
响应曲线：`{0.1, 0.2, 0.3, 0.5, 0.7, 1.0}`。

## 2. 明确边界

- M16C **不运行 B000 Flat**，这是为节约算力的明确设计选择。
- 因此 M16C 只能估计 4x4-S002 的 alpha 响应，不能建立新的同-alpha 架构对照。
- M16C 不能证明 Flat 的最佳 alpha，也不能宣称 S002 相对 Flat 的纯结构效应。
- M16C 的 seed 0 网格是 exploratory tuning stage；最终候选 alpha 仍需用新训练 seed 确认。

## 3. 配置矩阵

| config ID | alpha | 条件 | 环境 | seed | 预期正式结果目录 |
|---|---:|---|---|---:|---|
| M16C-4x4-S002-A010 | 0.1 | S002 Mixer L2 | puzzle-4x4-play-v0 | 0 | `runs/M16C/...alpha0p1.../seed_000` |
| M16C-4x4-S002-A020 | 0.2 | S002 Mixer L2 | puzzle-4x4-play-v0 | 0 | `runs/M16C/...alpha0p2.../seed_000` |
| M16C-4x4-S002-A050 | 0.5 | S002 Mixer L2 | puzzle-4x4-play-v0 | 0 | `runs/M16C/...alpha0p5.../seed_000` |
| M16C-4x4-S002-A070 | 0.7 | S002 Mixer L2 | puzzle-4x4-play-v0 | 0 | `runs/M16C/...alpha0p7.../seed_000` |

每个配置在 `factors.alpha` 与 `agent_overrides.alpha` 中都明确记录同一 alpha；运行时
后者覆盖 GCIQL 默认值。M16C 不依赖隐式默认 alpha。

除 alpha 外，所有新 run 与 M16A/M16B 的 S002 保持一致：GCIQL DDPG+BC、joint
actor+value+critic Puzzle-token MLP-Mixer L2、1M steps、batch size 1024、每 100k eval、
每 task 20 episodes、temperature 0、保存 best/last checkpoint。

## 4. 可复用 anchors

M16C Study 只声明、不会读取或修改以下既有结果：

| alpha | 来源 | config ID | final@1M |
|---:|---|---|---:|
| 0.3 | M16A | M16A-4x4-S002 | 0.71 |
| 1.0 | M16B | M16B-4x4-S002 | 0.47 |

M16C preflight 会验证上述 anchors 的状态、环境、seed、resolved alpha 与 10 个完整
evaluation checkpoint。新四个结果只有在该 provenance 检查通过后，才可与 anchors
合并为 alpha 响应曲线。

## 5. 非执行预检

`tools/m16c_doctor.py` 会检查：

- Study 只包含 Puzzle-4x4、S002、seed 0；
- alpha 集合恰为 `0.1/0.2/0.5/0.7`，无重复或遗漏；
- 每个配置解析到的 runtime alpha 与声明一致；
- actor/value/critic 三个 slot 均为同一份 Puzzle-token MLP-Mixer L2 定义；
- 除 alpha 外的 resolved agent config 完全一致；
- 数据 train/validation 文件、M16A/M16B anchors、网络实例化与输出路径均有效；
- 不会覆盖已有 M16C 输出路径。

预检不会执行 Git 操作、创建 run 目录或启动训练。

## 6. 用户手动执行流程

以下操作均由用户完成。Agent 不执行 Git 操作或正式启动。

1. 在主线 `/home/eai/Research/RLC` 审阅 M16C 文件，并自行完成 Git 提交与新的冻结 worktree。
2. 从该提交创建新的独立冻结副本，例如 `/home/eai/Research/RLC-m16c-exp`；不修改
   `RLC-m16a-exp` 或 `RLC-m16b-exp`。
3. 在新的冻结副本中运行：

```bash
cd /home/eai/Research/RLC-m16c-exp
export OGBENCH_DATASET_DIR=/data/qijunrong/06-RL/offline-rl/data/raw_ogbench
export PYTHONPATH=/home/eai/Research/RLC-m16c-exp
export XLA_PYTHON_CLIENT_PREALLOCATE=false
RLC_PYTHON=/home/eai/Tools/miniforge3/envs/brain_nav/bin/python

JAX_PLATFORMS=cpu "$RLC_PYTHON" tools/m16c_doctor.py \
  --study experiments/M16C_puzzle_4x4_mixer_alpha_sweep/study.yaml \
  --dataset-root /data/qijunrong/06-RL/offline-rl/data/raw_ogbench \
  --run-root /data/qijunrong/06-RL/offline-rl/exp/RLC/runs \
  --gpus 0,1
```

4. 检查通过后，先由用户运行 dry-run：

```bash
bash scripts/run_study.sh \
  --study experiments/M16C_puzzle_4x4_mixer_alpha_sweep/study.yaml \
  --configs M16C-4x4-S002-A010,M16C-4x4-S002-A020,M16C-4x4-S002-A050,M16C-4x4-S002-A070 \
  --gpus 0,1 \
  --run-root /data/qijunrong/06-RL/offline-rl/exp/RLC/runs \
  --dataset-root /data/qijunrong/06-RL/offline-rl/data/raw_ogbench \
  --train-steps 1000000 --batch-size 1024 --log-interval 5000 \
  --eval-interval 100000 --eval-tasks all --eval-episodes 20 \
  --eval-temperature 0 --save-interval 100000 \
  --save-best-checkpoint --save-last-checkpoint --dry-run
```

5. 确认 plan、GPU、run root 和协议无误后，由用户将末尾 `--dry-run` 改为 `--execute`。

两张 GPU 下，M16C 会同时运行两个 job；一个结束后队列再分配下一个，直至四个 job 完成。

## 7. 结果选择规则

M16C 的 primary endpoint 是 `final@1M`。与 seed-0 最佳值相差不超过 `0.05` 的 alpha
保留为并列候选；AUC、best 和 last-3 用于诊断，不用来在近似持平的单 seed 结果中强行排序。

后续应对每个候选 alpha 以 seed 1–4 在 4x4-S002 上确认。此后若要将结果迁移到其他
Puzzle 尺寸，应把它表述为 transfer hypothesis，而不是各环境的已证明最优 alpha。

## 8. 相关文件

- [M16C Study](/home/eai/Research/RLC/experiments/M16C_puzzle_4x4_mixer_alpha_sweep/study.yaml)
- [M16C preflight](/home/eai/Research/RLC/tools/m16c_doctor.py)
- [M16A/M16B alpha 报告](/home/eai/Research/RLC/docs/8-29/M16A_M16B_alpha_sensitivity_report_2026-08-29.md)
