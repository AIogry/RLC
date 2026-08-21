# M11A 正式实验结果报告

日期：2026-08-21  
项目：RLC 研究平台  
Study：`M11A — CRL Actor × Critic Computation Interaction`  
环境：`antmaze-large-navigate-v0`  
正式实验状态：**7/7 configurations completed**

## 摘要

M11A 是一个 2×2 的 CRL actor/critic computation factorial study，并分别在 SingleState 与 TwoState H2L1 topology 上复现。正式实验共 7 个 configuration，均为 seed 0、1M training steps、batch size 1024、5 个 evaluation tasks、每个 task 20 episodes。

主要结果如下：

- actor-side computation 是最稳定、最明显的正向因素。SingleState actor 在 FF critic 下相对 C001 的 final@1M 提升为 `+0.17`，TwoState actor 提升为 `+0.13`；两者的 best success 都达到 `0.93`。
- critic-side SingleState 在 FF actor 下有正向收益：C002 相对 C001 的 final@1M 提升 `+0.05`，evaluation checkpoints 平均提升 `+0.119`。
- critic-side TwoState 的收益较弱且不稳定：C005 相对 C001 的 final@1M 为 `-0.02`，checkpoint 平均提升仅 `+0.057`，且从 `0.87@800k` 回落到 `0.72@1M`。
- SingleState actor+critic 的绝对表现最好之一：C004 的 best success 为 `0.94`，final@1M 为 `0.90`；但其 factorial interaction 为负，说明“actor 计算收益 + critic 计算收益”并未以简单加法叠加，表现为 descriptive substitution。
- TwoState actor+critic C007 达到全实验最高 best success `0.95`，final@1M 为 `0.89`；其 final interaction 为 `+0.04`，但跨 checkpoint 的平均 interaction 为 `-0.031`，只能称为弱的、时间依赖的近似 additive/weak complementarity，不能称为稳定互补。
- M11A 的最优 configuration 取决于 checkpoint 选择：按 primary `last@1M`，C003 为最高；按训练期间 best，C007 为最高。因此不能只报告 best checkpoint 而忽略 final@1M。

本报告中的 interaction 都是单 seed 的 descriptive effect，不报告统计显著性，不把单次 seed 结果表述为普适规律。

## 1. 数据、协议与可复现性核验

### 1.1 正式 M11A protocol

| 项目 | 设置 |
|---|---|
| algorithm | CRL |
| actor loss | DDPG+BC |
| critic objective | contrastive bilinear |
| critic ensemble | 2 |
| environment | `antmaze-large-navigate-v0` |
| training seed | 0 |
| training steps | 1,000,000 |
| batch size | 1,024 |
| learning rate | `3e-4` |
| train log interval | 5,000 steps |
| evaluation interval | 100,000 steps |
| evaluation tasks | all 5 tasks |
| episodes per task | 20 |
| evaluation temperature | 0 |
| evaluation Gaussian noise | null |
| primary checkpoint | `last@1M` |
| secondary checkpoint | training期间 success 最大的 `best` |

每个 evaluation point 的 overall success 是 5 个 task success 的平均值；每个 task 有 20 episodes，因此 overall 指标对应 100 个 evaluation episodes，步长通常为 `0.01`。

### 1.2 Run 完整性

7 个 run 均满足：

- `runtime_metadata.json` 状态为 `completed`；
- 最后一条训练记录为 step `1,000,000`；
- 每个 run 有 200 条 train log、10 条 eval log；
- `last` checkpoint 为 step `1,000,000`；
- `best` checkpoint 均已写入；
- 所有 run 使用 `jax_backend=gpu`、`cuda:0`；
- 7 个 run 记录的 commit 均为 `22d7a9071727f25813a5ee341756da46ac05c41e`，且 `git_dirty=false`；
- dataset root 均为 `/data/qijunrong/06-RL/offline-rl/data/raw_ogbench`。

本报告只读取已有 run artifacts，没有执行任何 Git 命令，也没有修改、重启或覆盖实验产物。

### 1.3 重要的统计边界

M11A 只有一个 training seed。表中的 `eval_mean` 和 `eval_sd` 是 10 个 evaluation checkpoints 沿训练时间的描述统计，不是跨 seed 的均值和标准差；它们不能用于构造 training-seed confidence interval。20 episodes/task 只能反映一次训练结果的 evaluation sampling variation，不能替代多 seed replication。

## 2. M11A factorial design

| config | actor computation | critic computation | actor topology | critic topology | 说明 |
|---|---|---|---|---|---|
| C001 | FF | FF | feedforward | feedforward | factorial anchor |
| C002 | FF | SingleState | feedforward | SingleState, 4 iterations | critic-only SingleState |
| C003 | SingleState | FF | SingleState, 4 iterations | feedforward | actor-only SingleState |
| C004 | SingleState | SingleState | SingleState, 4 iterations | SingleState, 4 iterations | actor+critic SingleState |
| C005 | FF | TwoState | feedforward | TwoState H2L1, full-BPTT | critic-only TwoState |
| C006 | TwoState | FF | TwoState H2L1, full-BPTT | feedforward | actor-only TwoState |
| C007 | TwoState | TwoState | TwoState H2L1, full-BPTT | TwoState H2L1, full-BPTT | actor+critic TwoState |

固定设计包括 width 512、MLP/GELU、decision-local state、normal buffer initialization、state init std 1.0、无 outer residual，以及 actor/critic 分支各自独立参数。actor recurrent core 使用 `LayerNorm=False`、final activation 保留；critic recurrent core 使用与 CRL vanilla branch 对齐的 `LayerNorm=True`、final activation=False。M11A 不包含 H2L6、residual sweep、其他 normalization、其他 primitive 或其他 environment。

## 3. 原始 evaluation trajectory

下表是每个 100k checkpoint 的 `evaluation/overall_success`。这张表是主要证据，`best` 与 `final` 都由此直接计算。

| step | C001 | C002 | C003 | C004 | C005 | C006 | C007 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 100k | 0.56 | 0.60 | 0.77 | 0.85 | 0.54 | 0.86 | 0.85 |
| 200k | 0.60 | 0.80 | 0.78 | 0.84 | 0.71 | 0.83 | 0.82 |
| 300k | 0.65 | 0.87 | 0.82 | 0.91 | 0.77 | 0.87 | 0.89 |
| 400k | 0.72 | 0.86 | 0.88 | 0.94 | 0.80 | 0.88 | 0.94 |
| 500k | 0.74 | 0.82 | 0.90 | 0.89 | 0.77 | 0.81 | 0.90 |
| 600k | 0.68 | 0.83 | 0.93 | 0.93 | 0.76 | 0.93 | 0.95 |
| 700k | 0.63 | 0.87 | 0.90 | 0.89 | 0.80 | 0.87 | 0.92 |
| 800k | 0.79 | 0.86 | 0.91 | 0.89 | 0.87 | 0.80 | 0.91 |
| 900k | 0.86 | 0.86 | 0.92 | 0.86 | 0.80 | 0.91 | 0.82 |
| 1M | 0.74 | 0.79 | 0.91 | 0.90 | 0.72 | 0.87 | 0.89 |

## 4. 主结果：best、final 与训练过程稳定性

| config | best success | best step | final@1M | eval_mean | eval_sd* | last-3 mean | best→final |
|---|---:|---:|---:|---:|---:|---:|---:|
| C001 | 0.86 | 900k | 0.74 | 0.697 | 0.086 | 0.797 | -0.12 |
| C002 | 0.87 | 300k | 0.79 | 0.816 | 0.077 | 0.837 | -0.08 |
| C003 | 0.93 | 600k | 0.91 | 0.872 | 0.056 | 0.913 | -0.02 |
| C004 | 0.94 | 400k | 0.90 | 0.890 | 0.031 | 0.883 | -0.04 |
| C005 | 0.87 | 800k | 0.72 | 0.754 | 0.083 | 0.797 | -0.15 |
| C006 | 0.93 | 600k | 0.87 | 0.863 | 0.039 | 0.860 | -0.06 |
| C007 | 0.95 | 600k | 0.89 | 0.889 | 0.043 | 0.873 | -0.06 |

`eval_sd*` 是沿 10 个 checkpoints 的时间波动，不是 seed uncertainty。可以看到：

- C004 的 `eval_sd=0.031` 最低，且 final 仍为 0.90，说明 SingleState actor+critic 的曲线在本次 seed 上相对稳定；
- C003 的 best-to-final 仅下降 0.02，是 final 表现最稳的高性能 configuration；
- C001、C005 的 best-to-final 分别下降 0.12、0.15，说明 FF/FF anchor 与 TwoState-critic-only 结果存在明显 late-stage regression；
- C007 达到全实验最高峰值 0.95，但从 600k 后回落到 0.89，因而不能只用 best 结论替代 last@1M 结论。

### 4.1 Final 与 best checkpoint 的 task-level 结果

下表给出 final@1M 的五 task success；括号内为该 configuration 的 best checkpoint 对应 task success。

| config | task1 | task2 | task3 | task4 | task5 | overall final |
|---|---:|---:|---:|---:|---:|---:|
| C001 | 0.90 (0.95) | 0.10 (0.60) | 0.90 (0.85) | 0.95 (1.00) | 0.85 (0.90) | 0.74 |
| C002 | 0.90 (1.00) | 0.85 (0.85) | 0.80 (0.80) | 0.70 (0.80) | 0.70 (0.90) | 0.79 |
| C003 | 0.95 (1.00) | 0.80 (0.85) | 1.00 (0.95) | 0.90 (0.90) | 0.90 (0.95) | 0.91 |
| C004 | 0.95 (1.00) | 0.80 (0.85) | 0.90 (0.95) | 0.95 (0.95) | 0.90 (0.95) | 0.90 |
| C005 | 0.75 (0.95) | 0.45 (0.65) | 0.90 (1.00) | 0.70 (0.90) | 0.80 (0.85) | 0.72 |
| C006 | 0.90 (1.00) | 0.85 (0.85) | 0.90 (1.00) | 0.85 (0.95) | 0.85 (0.85) | 0.87 |
| C007 | 0.95 (0.95) | 0.90 (0.95) | 0.85 (1.00) | 0.95 (0.95) | 0.80 (0.90) | 0.89 |

最明显的 task-level 现象是 task2：C001 final 只有 0.10，而加入 critic SingleState 的 C002 为 0.85；C003/C004/C006/C007 也都达到 0.80 以上。相反，C005 final 的 task2 只有 0.45，说明 TwoState critic-only 并没有复现 SingleState critic-only 对该 task 的改善。由于每个 task 只有 20 episodes，这些差异仍应视为本次固定 seed 的结果，而不是已经建立的 task-level causal law。

## 5. Factorial effect 与 interaction 分析

### 5.1 定义

以 C001 为 factorial anchor，定义：

```text
A_S = J(C003) - J(C001)       actor SingleState main effect
A_T = J(C006) - J(C001)       actor TwoState main effect
C_S = J(C002) - J(C001)       critic SingleState main effect
C_T = J(C005) - J(C001)       critic TwoState main effect

I_S = J(C004) - J(C002) - J(C003) + J(C001)
I_T = J(C007) - J(C005) - J(C006) + J(C001)
```

`J` 可以取共同 checkpoint 的 overall success。`I<0` 仅称为 descriptive substitution，`I>0` 仅称为 descriptive complementarity；在单 seed 条件下不进行显著性检验。

### 5.2 共同 checkpoint 的 effect 汇总

| effect | 10 个 checkpoint 平均 | last@1M | last-3 平均 | 解释 |
|---|---:|---:|---:|---|
| A_S：actor SingleState | +0.175 | +0.17 | +0.117 | 稳定正向，但 late-stage advantage 收窄 |
| A_T：actor TwoState | +0.166 | +0.13 | +0.063 | 正向，但 800k 后相对 C001 优势变小 |
| C_S：critic SingleState | +0.119 | +0.05 | +0.040 | 正向，早中期尤其明显 |
| C_T：critic TwoState | +0.057 | -0.02 | 0.000 | 弱正向平均效应，final 不再占优 |
| I_S：SingleState interaction | -0.101 | -0.06 | -0.070 | descriptive substitution |
| I_T：TwoState interaction | -0.031 | +0.04 | +0.013 | 接近 additive，final 略偏 complementarity |

### 5.3 Actor-side computation

C003 与 C006 是最清晰的 actor-only 对照：critic 均为 FF，因此差异主要对应 actor computation。

- C003 SingleState actor：best `0.93`、final `0.91`、eval_mean `0.872`；
- C006 TwoState actor：best `0.93`、final `0.87`、eval_mean `0.863`；
- 两者都显著高于 C001 FF/FF 的 best `0.86`、final `0.74`、eval_mean `0.697`；
- 本次 seed 下 SingleState actor 的平均与 final 略高于 TwoState actor，但 best peak 相同，因此不能宣称 SingleState 在所有条件下优于 TwoState；
- C003 的 late-stage regression 最小，说明其优势不仅来自早期峰值，至少在本次 run 中 final stability 更好。

### 5.4 Critic-side computation

C002 与 C005 是最清晰的 critic-only 对照：actor 均为 FF。

- SingleState critic C002：best `0.87`、final `0.79`、eval_mean `0.816`；
- TwoState critic C005：best `0.87`、final `0.72`、eval_mean `0.754`；
- 两者都比 C001 的 mean `0.697` 更高，但 C002 的收益更持续，C005 在 800k 达到 `0.87` 后回落至 `0.72`；
- 因而不能把“更多层级状态计算”直接解释为更好的 critic-side computation。至少在当前 H2L1、critic update_depth=3、单 seed 设置下，TwoState critic-only 不稳定。

### 5.5 SingleState interaction

C004 的绝对结果很强：best `0.94`、final `0.90`、eval_mean `0.890`，也是最稳定的 configuration 之一。但 `I_S` 平均为 `-0.101`、final 为 `-0.06`。

这两个事实并不矛盾。负 interaction 的含义是：

```text
C004 的组合收益 < C002 的 critic-only 收益 + C003 的 actor-only 收益
```

它不意味着 C004 的绝对性能差，也不意味着 actor/critic 计算互相破坏。更准确的结论是：SingleState actor 与 SingleState critic 都各自有效，但组合后存在明显的 diminishing returns/substitution，而不是简单 additive。

### 5.6 TwoState interaction

C007 达到全实验最高 best `0.95`，但 final `0.89` 略低于 C004 的 `0.90`；`I_T` 在不同 checkpoint 上正负交替，10 点平均为 `-0.031`，final 为 `+0.04`。

因此 TwoState 的组合不能给出单一稳定标签：

- early/middle checkpoints 中 interaction 多次为负；
- 500k 为 `+0.06`，600k 为 `-0.06`，700k 为 `-0.12`；
- 800k 为 `+0.03`，1M 为 `+0.04`；
- 最稳妥的表述是“近似 additive、时间依赖，final 点略偏 complementarity”，而不是“已证明 TwoState 存在互补性”。

## 6. 训练日志的辅助观察

正式 train log 显示 7 个 run 均完成且没有出现非有限数值终止。共同趋势是：

- training actor MSE 均下降，说明 actor policy 对 dataset action 的拟合改善；
- training critic contrastive loss 均下降；
- validation critic contrastive loss 并不都单调下降，尤其 C005/C007 在 final 附近仍有波动；
- recurrent critic 的 training categorical accuracy 可上升到约 0.34–0.44，但 validation categorical accuracy 仍约 0.09–0.10，不能仅凭 training metric 宣称 critic generalization 已改善；
- C005/C007 的 gradient norm 在后期高于 C001/C002/C003/C004，结合 evaluation 回落，说明其 late-stage training dynamics 更值得关注，但当前日志不足以证明具体机制。

这些是 mechanism clues，不是额外的 causal evidence。M11A 没有预注册 gradient attribution、state norm 或 H/L update magnitude 作为主要结果。

## 7. 与 M9A、M9B、M9B1M 的关系

历史结果用于 contextual reference，不直接替代 M11A 的 factorial control。原因包括：旧 Study 的训练 horizon、代码 commit、actor/critic placement、算法以及 computation semantics 不完全相同。

### 7.1 关键 baseline 对照

| Study/config | 算法与 computation | horizon | best | final | 适合回答的问题 |
|---|---|---:|---:|---:|---|
| M9A-C001 | HIQL vanilla FF | 1M | 0.87 | 0.87 | 外部 HIQL baseline |
| M9A-C002 | CRL vanilla FF | 1M | 0.77 | 0.62 | 历史 CRL vanilla reference |
| M9A-C003 | CRL actor SS K1 no-res | 500k | 0.85 | 0.84 | 早期 SingleState actor |
| M9A-C005 | CRL actor SS K2 no-res | 500k | 0.89 | 0.87 | 早期迭代深度对照 |
| M9A-C007 | CRL actor SS K4 no-res | 500k | 0.92 | 0.91 | 与 M11A C003 最接近的历史 actor reference |
| M9B-C001 | CRL actor TS H2L1 full-BPTT | 500k | 0.90 | 0.84 | 历史 TwoState H2L1 actor |
| M9B-C003 | CRL actor TS H2L6 full-BPTT | 500k | 0.91 | 0.89 | 历史 H2L6 actor |
| M9B1M-C001 | CRL actor TS H2L1 full-BPTT | 1M | 0.93 | 0.92 | 长 horizon H2L1 CRL reference |
| M9B1M-C002 | CRL actor TS H2L6 full-BPTT | 1M | 0.92 | 0.85 | 长 horizon H2L6 CRL reference |
| M9B1M-C003 | HIQL high+low TS H2L1 | 1M | 0.93 | 0.89 | 长 horizon hierarchical HIQL |
| M9B1M-C004 | HIQL high+low TS H2L6 | 1M | 0.94 | 0.91 | 长 horizon H2L6 HIQL |
| M11A-C001 | CRL FF actor + FF critic | 1M | 0.86 | 0.74 | M11A factorial anchor |
| M11A-C003 | CRL SS actor + FF critic | 1M | 0.93 | 0.91 | M11A actor SingleState |
| M11A-C006 | CRL TS H2L1 actor + FF critic | 1M | 0.93 | 0.87 | M11A actor TwoState |
| M11A-C007 | CRL TS H2L1 actor + TS H2L1 critic | 1M | 0.95 | 0.89 | M11A full TwoState combination |

### 7.2 对 M9A 的解释

M9A 的 CRL actor-only trajectory 已经显示迭代 actor 可以带来收益：K4 no-res C007 的 final 为 `0.91`，高于 K1 no-res C003 的 `0.84`。M11A 的 C003 final 也为 `0.91`，best `0.93`，与该历史趋势一致；但 M11A 使用 1M horizon 和修订后的统一 primitive semantics，不能把两者差值当作单独由 M11A 代码修改造成。

M9A 的历史 CRL vanilla C002 final 为 `0.62`，而 M11A anchor C001 final 为 `0.74`。这说明两次 CRL vanilla run 的结果不同，但它们来自不同的实验时期/代码 provenance，不能据此声称 M11A 代码改动带来了 `+0.12` 的确定性提升。M11A 内部 factorial comparison 应优先使用 C001 作为同一 campaign anchor。

M9A 的 HIQL control 也表明 residual、actor placement 与 computation depth 之间存在明显 seed/曲线波动。例如 HIQL high actor K4 residual C014 final `0.94`，但 high+low K4 residual C026 final `0.87`；因此不能把单一 placement 的峰值直接外推到组合 placement。

### 7.3 对 M9B 与 M9B1M 的解释

M9B historical 500k 中，CRL H2L1 full-BPTT C001 为 best/final `0.90/0.84`，H2L6 C003 为 `0.91/0.89`。延长到 1M 后，M9B1M-C001 为 `0.93/0.92`，说明 H2L1 CRL actor 在该长 horizon run 中保持良好；M9B1M-C002 H2L6 则为 `0.92/0.85`，final 低于 H2L1。

M11A actor-only C006（TS H2L1 actor + FF critic）为 `0.93/0.87`。它的 best 与 M9B1M-C001 相同，但 final 低 `0.05`。这提示 M11A 的 critic/actor semantics、campaign context 或跨进程训练顺序可能影响 late-stage stability；不过由于不是完全同一 Study 的 paired rerun，不能将差异归因于某一个单独因素。

M11A C007 的 best `0.95` 高于 M9B1M-C001 的 `0.93`，但 final `0.89` 低于其 `0.92`。因此“加入 TwoState critic 后提高峰值”可以作为描述性现象，但不能称为稳定的最终性能提升。

## 8. M9A 全部 Large 对照组摘要

下表保留 M9A 其他 comparison groups 的 best/final 结果。M9A-C001/C002 是 1M；C003–C026 是历史 500k runs，因此表内横向比较必须区分 horizon。

| group | config | best | final |
|---|---|---:|---:|
| CRL actor SS no-res | C003 K1 | 0.85 | 0.84 |
| CRL actor SS res | C004 K1 | 0.86 | 0.86 |
| CRL actor SS no-res | C005 K2 | 0.89 | 0.87 |
| CRL actor SS res | C006 K2 | 0.90 | 0.87 |
| CRL actor SS no-res | C007 K4 | 0.92 | 0.91 |
| CRL actor SS res | C008 K4 | 0.93 | 0.91 |
| HIQL high SS no-res | C009 K1 | 0.91 | 0.83 |
| HIQL high SS res | C010 K1 | 0.91 | 0.91 |
| HIQL high SS no-res | C011 K2 | 0.92 | 0.92 |
| HIQL high SS res | C012 K2 | 0.85 | 0.85 |
| HIQL high SS no-res | C013 K4 | 0.93 | 0.85 |
| HIQL high SS res | C014 K4 | 0.94 | 0.94 |
| HIQL low SS no-res | C015 K1 | 0.84 | 0.84 |
| HIQL low SS res | C016 K1 | 0.92 | 0.85 |
| HIQL low SS no-res | C017 K2 | 0.89 | 0.89 |
| HIQL low SS res | C018 K2 | 0.90 | 0.86 |
| HIQL low SS no-res | C019 K4 | 0.93 | 0.93 |
| HIQL low SS res | C020 K4 | 0.93 | 0.89 |
| HIQL high+low SS no-res | C021 K1 | 0.92 | 0.92 |
| HIQL high+low SS res | C022 K1 | 0.93 | 0.82 |
| HIQL high+low SS no-res | C023 K2 | 0.83 | 0.83 |
| HIQL high+low SS res | C024 K2 | 0.94 | 0.88 |
| HIQL high+low SS no-res | C025 K4 | 0.92 | 0.79 |
| HIQL high+low SS res | C026 K4 | 0.93 | 0.87 |

另有 M9A-C001 HIQL vanilla `0.87/0.87`、M9A-C002 CRL vanilla `0.77/0.62`，见上一节关键 baseline 表。

## 9. M9B 全部 Large 对照组摘要

M9B historical configuration 均为 500k；M9B1M 是其 1M extension。以下先列 M9B historical full/one-step 与 placement controls，再列 M9B1M extension。

| group | config | best | final |
|---|---|---:|---:|
| CRL actor H2L1 full-BPTT | M9B-C001 | 0.90 | 0.84 |
| CRL actor H2L1 one-step | M9B-C002 | 0.86 | 0.86 |
| CRL actor H2L6 full-BPTT | M9B-C003 | 0.91 | 0.89 |
| CRL actor H2L6 one-step | M9B-C004 | 0.70 | 0.70 |
| HIQL high H2L1 full-BPTT | M9B-C005 | 0.91 | 0.89 |
| HIQL high H2L1 one-step | M9B-C006 | 0.93 | 0.93 |
| HIQL high H2L6 full-BPTT | M9B-C007 | 0.88 | 0.88 |
| HIQL high H2L6 one-step | M9B-C008 | 0.85 | 0.85 |
| HIQL low H2L1 full-BPTT | M9B-C009 | 0.77 | 0.73 |
| HIQL low H2L1 one-step | M9B-C010 | 0.92 | 0.81 |
| HIQL low H2L6 full-BPTT | M9B-C011 | 0.83 | 0.80 |
| HIQL low H2L6 one-step | M9B-C012 | 0.80 | 0.80 |
| HIQL high+low H2L1 full-BPTT | M9B-C013 | 0.95 | 0.91 |
| HIQL high+low H2L1 one-step | M9B-C014 | 0.85 | 0.83 |
| HIQL high+low H2L6 full-BPTT | M9B-C015 | 0.92 | 0.85 |
| HIQL high+low H2L6 one-step | M9B-C016 | 0.89 | 0.80 |

M9B historical 500k 结果的主要信息是：H2L6 对 CRL actor full-BPTT 有利，但 one-step 在 H2L6 上明显失败；HIQL 的高/低 actor placement 与 credit rule 之间存在大幅 variation。这些结果支持 M11A 固定 H2L1、固定 full-BPTT、固定 placement 的设计选择，但不直接决定 M11A critic computation 的结论。

## 10. 结果的科学解释与不应做出的结论

### 可以支持的结论

1. 在本次 seed 0、AntMaze-Large、1M protocol 下，actor-side computation 的收益比 critic-side computation 更稳定、更大。
2. SingleState critic-only 比 TwoState critic-only 更稳定：C002 的 final、均值和 late-stage stability 均优于 C005。
3. SingleState actor+critic 的绝对性能强且曲线相对稳定，但 interaction 为负，表示 diminishing returns/substitution，而不是 additive accumulation。
4. TwoState actor+critic 可以产生最高 peak，但 final 与 peak 有明显差距；其 interaction 不能被概括为稳定 complementarity。
5. M9A/M9B/M9B1M 的历史结果与 M11A 一起表明，computation effect 对 placement、topology、credit、residual 和 training horizon 很敏感。

### 不能支持的结论

- 不能声称 SingleState 在所有 seed、环境或 topology 上优于 TwoState；
- 不能声称 critic computation 没有价值；C002 的正向结果明确反驳这一点；
- 不能声称 C007 的 `0.95` peak 代表稳定最终性能优于 C004 或 C003；
- 不能把 M11A 与 M9A/M9B 的 raw difference 解释为单个代码改动的因果效果；
- 不能从 training critic accuracy 或 contrastive loss 直接推出 policy success 的机制；
- 不能把一次 20-episode/task evaluation 当作 statistically significant test。

## 11. 诊断 artifact 的边界

截至本报告生成时，正式 M11A run artifacts 已存在，但 `/data/qijunrong/06-RL/offline-rl/exp/RLC/diagnostics` 下没有发现 M11A diagnostic bank、candidate pool、critic identity audit、E_eval/E_ext scoring 或 aggregate interaction artifact。因此本报告只报告：

- formal training evaluation；
- train/eval curves；
- factorial performance contrasts；
- 与已有 M9A/M9B/M9B1M raw run 的 contextual comparison。

本报告没有把尚未生成的 critic-value diagnostic 或 temporal/extrapolation metric 写成实验结果。若后续生成这些 source-dependent diagnostics，应另写 analysis report，并保持 last@1M 为 primary checkpoint，不得用 best checkpoint 替换 primary interaction estimate。

## 12. Artifact 索引

M11A 正式 run 根目录：

```text
/data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M11A/
```

每个 configuration 的关键文件：

```text
<config>__*/antmaze-large-navigate-v0/seed_000/eval.csv
<config>__*/antmaze-large-navigate-v0/seed_000/train.csv
<config>__*/antmaze-large-navigate-v0/seed_000/summary.json
<config>__*/antmaze-large-navigate-v0/seed_000/runtime_metadata.json
<config>__*/antmaze-large-navigate-v0/seed_000/checkpoints/index.json
<config>__*/antmaze-large-navigate-v0/seed_000/checkpoints/last/params_1000000.pkl
<config>__*/antmaze-large-navigate-v0/seed_000/checkpoints/best/params_<best_step>.pkl
```

M9A、M9B、M9B1M raw run 根目录分别为：

```text
/data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M9A/
/data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M9B/
/data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M9B-1M/
```

## 13. 最终结论

对 M11A 的 primary `last@1M` 结果，最可靠的总结是：

> CRL 中 actor-side recurrent computation 在本次 AntMaze-Large 1M 单 seed 实验中表现出最强、最稳定的正向作用；SingleState critic computation 有中等正向作用，而 H2L1 TwoState critic computation 单独使用时收益较弱并出现明显 late-stage regression。actor 与 critic computation 的组合不是简单 additive：SingleState 组合表现出 diminishing returns/substitution；TwoState 组合在 peak 上有优势，但 interaction 随训练时间变化，最终只能描述为近似 additive、而非稳定互补。

因此，M11A 的科学结论不是“recurrent computation 越多越好”，而是：**computation placement 和计算角色比单纯增加 recurrent depth 更重要；actor-side computation 的收益在当前设计中比 critic-side computation 更可靠。** 这一结论仍需要多 seed 与跨环境实验确认。
