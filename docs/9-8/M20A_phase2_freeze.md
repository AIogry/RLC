# M20A Phase 2：relation threshold freeze 与正式 screening 协议

## 冻结决定

用户已在 2026-09-08 的 Phase-2 prompt 中正式冻结 Cube relation geometry。该决定适用于 Cube-triple 的 Zero、Correct、Shuffled 以及 Mean/Hybrid 的全部六个 cell；Zero 虽然运行时 relation tensor 为零，也保留同一 metadata 以完整记录 Study provenance。

| 字段 | 冻结值 | 冻结依据 |
| --- | ---: | --- |
| `current_support_epsilon_xy` | 0.02 | OGBench Cube stack xy alignment 的 source-grounded constant |
| `current_support_epsilon_z` | 0.01 | Phase-1 预先定义的几何 sensitivity bracket；以 cube vertical geometry 0.04 为依据的 conservative tolerance |
| `goal_support_epsilon_xy` | 0.02 | 与 current support 共享同一几何语义 |
| `goal_support_epsilon_z` | 0.01 | 与 current support 共享同一几何语义 |
| `goal_conflict_radius` | 0.04 | OGBench Cube official success position radius |

这些值没有根据 RL success、loss、`Correct − Shuffled` gap 或 relation prevalence 的“吸引力”调优。Phase-1 audit 中的 `UNFROZEN` 状态是当时真实的历史记录；本文件与 M20A Study/config 的 `FROZEN_USER_PHASE2` 状态是后续由用户给出的正式 Phase-2 决定。

## 正式矩阵

正式 screening 恰好为下列 18 个 `seed=0` run，不增加 Flat reference、额外 seed、替代 threshold 或新 relation：

| task | MeanContextReadout | HybridContextQueryReadout |
| --- | --- | --- |
| Puzzle-4x4 | Zero, Correct, Shuffled | Zero, Correct, Shuffled |
| Cube-triple | Zero, Correct, Shuffled | Zero, Correct, Shuffled |
| Scene | Zero, Correct, Shuffled | Zero, Correct, Shuffled |

config ID 分别是：

```text
M20A-PUZZLE-Z-M  M20A-PUZZLE-C-M  M20A-PUZZLE-S-M
M20A-PUZZLE-Z-Q  M20A-PUZZLE-C-Q  M20A-PUZZLE-S-Q
M20A-CUBE-Z-M    M20A-CUBE-C-M    M20A-CUBE-S-M
M20A-CUBE-Z-Q    M20A-CUBE-C-Q    M20A-CUBE-S-Q
M20A-SCENE-Z-M   M20A-SCENE-C-M   M20A-SCENE-S-M
M20A-SCENE-Z-Q   M20A-SCENE-C-Q   M20A-SCENE-S-Q
```

## 保持不变的科学协议

- GCIQL / DDPG+BC，显式 `alpha=1.0`，不依赖 agent default；
- 1,000,000 steps，batch 1024，`lr=3e-4`，discount 0.99，expectile 0.9，tau 0.005；
- value goal sampling `(0.2, 0.5, 0.3, geom=true)`，actor goal sampling `(0, 1, 0, geom=false)`；
- 每 100k eval，`eval_tasks=all`，每 task 50 episodes，temperature 0；每 100k save，保存 best/last；主终点 `final@1M`；
- token dim 128，MLP-Mixer L2/64/256，feedforward/direct；
- RelationAugmenter 256-wide、GELU、两层无 bias Dense、无 normalization/dropout；
- HybridContextQuery 的 `query_dim=128`；
- Puzzle shuffle permutation 与 Cube derangement `[1,2,0]` 不变。

本轮没有修改 relation definitions、entity parser、RelationAugmenter、MLPMixerBlock、GCIQL objective、readout implementation、alpha、evaluation episode 数或 seed。

## 对照与解释边界

未来的 primary contrast 仍是同一 task、同一 readout 下的 `Correct − Shuffled`；`Correct − Zero` 次之。`Hybrid − Mean` 是 readout-package contrast，不是 parameter/MAC-matched attention effect。Puzzle、Cube、Scene raw success 不做 pooling；单 seed screening 不做 significance claim。

## 启动前 gate

启动前必须同时满足：

1. M20A relation/representation/RNG/parameter-parity tests 与历史回归通过；
2. `m20a_doctor.py` 输出 `M20A PHASE-2 FREEZE: PASS` 和 `FORMAL TRAINING: READY`；
3. formal run root 中没有任何既存 M20A artifact；
4. formal dry-run 恰好显示 planned=18、completed=0、remaining=18；
5. Git diff 仅包含 M20A Phase-1 implementation 与本次 Phase-2 freeze/activation；
6. commit、push、detached frozen worktree 与 frozen-worktree doctor 都成功；
7. 只调度实际空闲 GPU，且 startup validation 无 config/provenance/NaN error。

`experiments/M20A_cross_task_oracle_relation_diagnosis/study.yaml` 是当前冻结配置 authority；Phase-1 的 [implementation handoff](M20A_phase1_oracle_relation_implementation.md) 与 [relation audit](M20A_phase1_relation_audit.md) 保留为实现和审计证据。

## Pre-commit gate record

在本次 freeze/activation 后、Git commit 前，以下 non-formal gate 已实际通过：

| gate | 结果 |
| --- | --- |
| `compileall impls tools tests` | PASS |
| M20A relation + frozen-study test | 10 tests PASS |
| M15–M17、M16 alpha studies、M19 factorization、checkpoint/provenance | 63 tests PASS |
| canonical GCIQL、M14、M18/M18-D、M15/M17/M19 real smoke | 44 tests PASS |
| Phase-2 doctor | `M20A PHASE-2 FREEZE: PASS` / `FORMAL TRAINING: READY` |
| formal-root dry-run | planned=18, completed=0, remaining=18; 18 unique config IDs |

测试过程仅使用 `/tmp` 临时 checkpoint 或既有历史 run 作为 regression input；未创建 M20A formal run artifact。后续 commit SHA、push、frozen worktree 和正式 launch provenance 将由 launcher/runtime metadata 记录。
