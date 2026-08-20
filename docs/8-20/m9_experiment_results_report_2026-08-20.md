# M9 实验结果综合报告：M9A、M9B 与 M9B-1M

日期：2026-08-20  
报告范围：M9A SingleState、M9B TwoState、M9B-1M convergence extension  
目标环境：`antmaze-medium-navigate-v0`、`antmaze-large-navigate-v0`  
训练 seed：均为 `seed=0`；因此本文结论是单 seed 的描述性结论，不是统计显著性结论。

## 结论摘要

1. **M9B-1M 的正式运行是有效的、完整的，但没有显示出“训练到 1M 后所有配置持续提升”的统一趋势。** 四个 Large 运行的 1M final overall success 为 `0.85–0.92`，best 为 `0.92–0.94`；所有训练和评估记录均为 finite，运行状态为 completed，GPU backend，工作树 clean。

2. **相对于外部 vanilla baseline，M9B-1M 的 CRL 结果提升最明显。** CRL H2L1 full-BPTT 的 final 为 `0.92`，相对 CRL vanilla 1M final `0.62` 提升 `+0.30`；CRL H2L6 为 `0.85`，提升 `+0.23`。但这是单次运行、不同 source commit 的外部参考比较，应该称为“描述性优势”，不能直接称为已证实的因果提升。

3. **M9B-1M 对 HIQL high+low 的提升较小但方向为正。** H2L1 final `0.89`、H2L6 final `0.91`，相对 HIQL vanilla 1M final `0.87` 分别为 `+0.02`、`+0.04`；best 分别为 `0.93`、`0.94`。这说明 TwoState 对 HIQL 的收益目前更像是稳定性/峰值质量的可能改善，而不是已经明确的 final-score 大幅提升。

4. **M9B-1M 的主要现象是非单调、晚期波动，而非简单饱和。** C001 在 200k 达到 `0.93`，之后回落再回升至 1M `0.92`；C002 在 700–800k 达到 `0.92` 后回落至 `0.85`；C003 在 700k 达到 `0.93` 后回落至 `0.89`；C004 在 400k 达到 `0.94`，1M 为 `0.91`。因此报告最终结果时应同时保留 best checkpoint 与 last@1M，不能只报其中一个。

5. **M9A/M9B 的大环境结果支持“计算结构可能有帮助”，但支持的是特定 placement/schedule 组合，不是所有结构都有效。** M9A 在 Large 上相对同训练步数 baseline 的平均提升最明显的是 CRL actor（约 `+0.127`）；M9B 在 Large 上 HIQL high 组较稳定（约 `+0.078`），low-only 组反而平均下降（约 `-0.025`）。

6. **当前最需要补充的是多 seed 和更可靠的 checkpoint reevaluation，而不是继续扩大 M9 的结构网格。** 在做更多机制解释、论文表述或显著性判断前，优先补 `M9B-1M` 四个配置的 seed 1/2，并对 best/last checkpoint 做 100-episode reevaluation。

## 1. 数据、协议与可比性边界

### 1.1 研究对象

| Study | 配置数 | 环境 | 训练步数 | 研究轴 |
|---|---:|---|---:|---|
| M9A | 26 configs × 2 env = 52 runs | Medium/Large | 计算组 500k；vanilla baseline 1M | SingleState、K、residual、placement |
| M9B | 16 configs × 2 env = 32 runs | Medium/Large | 500k | TwoState、H2L1/H2L6、full-BPTT/one-step、placement |
| M9B-1M | 4 configs × 1 env = 4 runs | Large | 1M | M9B selected configs 的 long-horizon extension |

M9A 与 M9B 的配置定义和实现说明分别见 [M9A study](../../experiments/M9A_single_state_iteration/study.yaml)、[M9B study](../../experiments/M9B_two_state/study.yaml) 以及 [M9B-1M protocol](m9b_two_state_1m.md)。

### 1.2 运行来源

当前报告读取的正式产物位于：

```text
/data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M9A
/data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M9B
/data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M9B-1M
```

M9B-1M 四个正式运行的共同 provenance 为：

| 项目 | 值 |
|---|---|
| source commit | `c2f6d7ef4231e710cd5e6292e77aad3db9a8a8fb` |
| git dirty | `false` |
| backend/device | `gpu` / `cuda:0` |
| dataset | `antmaze-large-navigate-v0`，all five tasks |
| train seed | `0` |
| batch size | `1024` |
| train steps | `1,000,000` |
| evaluation | 每 100k；20 episodes/task；temperature 0；Gaussian noise null |
| numeric checkpoints | 每 100k；另有 best 与 last |
| 记录完整性 | 每个 run 的 `train.csv` 200 rows，`eval.csv` 10 rows，均 finite，status completed |

历史 M9A/M9B 运行的 source commit 为 `f30b64bf81e1738235eef4f213d3019820ee918a`，同样是 clean GPU 运行。两批运行不是同一 commit，因此 M9B-1M 与历史结果之间的差异不能被严格归因于训练时长一个因素。另一个重要边界是：M9A 计算组和 M9B 都是从头训练的新 run，不是从历史 checkpoint 继续训练；M9B-1M 也应理解为从 0 step 重新训练的 1M run，而非把 M9B 的 500k checkpoint 延长到 1M。

### 1.3 指标定义

- `final`：该 run 最后一个评估点的 `evaluation/overall_success`。
- `best`：训练期间所有评估点中最高的 `overall_success`，括号中给出 step。
- M9A/M9B 的 `final` 一般是 500k；vanilla baseline 的 `final` 是 1M。因此在比较 M9A/M9B 计算组与 baseline 时，优先使用 **同一 500k 时刻的 baseline 值**。
- Large 每个 overall success 是五个任务、每任务 20 episodes 的汇总；单次评估总共 100 episodes。这个样本量足以做趋势判断，但不足以消除明显的评估噪声。

## 2. M9B-1M 正式结果

### 2.1 四个配置的完整 Large 曲线

| 配置 | 方法/placement | schedule | credit | 100k | 200k | 300k | 400k | 500k | 600k | 700k | 800k | 900k | 1M |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| M9B1M-C001 | CRL actor | H2L1 | full-BPTT | .70 | .93 | .89 | .86 | .85 | .86 | .90 | .90 | .90 | .92 |
| M9B1M-C002 | CRL actor | H2L6 | full-BPTT | .80 | .86 | .81 | .82 | .77 | .86 | .92 | .92 | .89 | .85 |
| M9B1M-C003 | HIQL high+low | H2L1 | full-BPTT | .77 | .78 | .91 | .84 | .87 | .87 | .93 | .86 | .87 | .89 |
| M9B1M-C004 | HIQL high+low | H2L6 | full-BPTT | .78 | .85 | .83 | .94 | .93 | .91 | .89 | .81 | .89 | .91 |

### 2.2 final、best 与历史 M9B 500k counterpart

| M9B-1M | 500k | 1M final | 1M best | best step | 500k→1M | 历史 M9B counterpart 500k | 新旧 500k 差异 |
|---|---:|---:|---:|---:|---:|---:|---:|
| C001 CRL H2L1 full | .85 | .92 | .93 | 200k | +.07 | M9B-C001 .84 | +.01 |
| C002 CRL H2L6 full | .77 | .85 | .92 | 700k | +.08 | M9B-C003 .89 | -.12 |
| C003 HIQL both H2L1 full | .87 | .89 | .93 | 700k | +.02 | M9B-C013 .91 | -.04 |
| C004 HIQL both H2L6 full | .93 | .91 | .94 | 400k | -.02 | M9B-C015 .85 | +.08 |

这里的“新旧 500k 差异”不是同一训练轨迹的 continuation 对比，而是同一个 training seed、不同正式 run/source commit 的结果对比。四个配置的差异方向不一致，幅度最高达到 0.12，这本身说明单 seed 和评估噪声已经足以影响 500k 的数值判断。

### 2.3 与 vanilla baseline 的比较

vanilla baseline 来自 M9A 的外部 immutable reference：CRL 为 M9A-C002，HIQL 为 M9A-C001。两者都在同一 Large 环境、seed 0、batch 1024、learning rate 0.0003、20 episodes/task、1M steps 下运行，但 source commit 与 M9B-1M 不同。

| M9B-1M | 对应 baseline | baseline 500k | baseline 1M final | baseline best | M9B-1M final | 相对 baseline 1M final | M9B-1M best | 相对 baseline best |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| C001 CRL H2L1 full | CRL vanilla | .75 | .62 | .77 @900k | .92 | +.30 | .93 | +.16 |
| C002 CRL H2L6 full | CRL vanilla | .75 | .62 | .77 @900k | .85 | +.23 | .92 | +.15 |
| C003 HIQL both H2L1 full | HIQL vanilla | .81 | .87 | .87 @1M | .89 | +.02 | .93 | +.06 |
| C004 HIQL both H2L6 full | HIQL vanilla | .81 | .87 | .87 @1M | .91 | +.04 | .94 | +.07 |

若只做同一 500k 时刻的描述性比较，M9B-1M 的 500k 值相对 baseline 500k 分别为：C001 `+0.10`、C002 `+0.02`、C003 `+0.06`、C004 `+0.12`。因此 CRL H2L1 和 HIQL H2L6 的 1M final 优势很清楚；CRL H2L6 的优势主要体现在 700–800k 的峰值而非最后一个 checkpoint；HIQL 两个配置的 final 优势较小，但 best checkpoint 优势更明显。

### 2.4 任务级结果

下表是 1M final 与 best checkpoint 的五个 task success。它用于检查 overall 平均值是否掩盖单任务退化。

| 配置 | final task1/task2/task3/task4/task5 | final overall | best step | best task1/task2/task3/task4/task5 | best overall |
|---|---|---:|---:|---|---:|
| C001 | .95/.85/1.00/.95/.85 | .92 | 200k | .95/.95/.90/.95/.90 | .93 |
| C002 | .85/.70/.95/.95/.80 | .85 | 700k | .95/.90/.95/.90/.90 | .92 |
| C003 | .85/.80/.95/.95/.90 | .89 | 700k | .85/.85/1.00/1.00/.95 | .93 |
| C004 | 1.00/.85/.95/.95/.80 | .91 | 400k | 1.00/.85/.95/.90/1.00 | .94 |

C002 在 1M 的主要短板是 task2=`.70`、task5=`.80`；C004 的 1M task5 也从 best checkpoint 的 `1.00` 回落到 `.80`。这与“overall 下降主要来自少数任务，而不是五个任务整体同步退化”的判断一致。

### 2.5 训练过程与收敛判断

- 四个 run 的 `train.csv` 均有 200 个 5k 间隔记录，未发现 NaN/Inf 或中途终止。
- CRL 两个 run 的 actor/critic loss 与梯度范数在 500k–1M 间保持 finite，但没有呈现足以单独证明收敛的单调趋势。
- HIQL 两个 run 的高层/低层 actor loss、value loss、梯度范数同样保持 finite，但 value loss 与梯度范数的变化不能直接映射为 evaluation success 的改善。
- 成功率曲线的峰值在 200k、400k、700k 分散出现，说明“1M 已完全收敛”目前证据不足；更稳妥的描述是：**模型在 1M 内达到较高性能，但存在明显的 checkpoint-dependent fluctuation**。
- M9B-1M 中只保留了 full-BPTT，因此它回答的是 selected full-BPTT 配置的 long-horizon 行为，不能单独回答 full-BPTT 是否优于 one-step。

## 3. M9A：SingleState 全部对照组

### 3.1 配置映射

| 配置区间 | placement | 变量 |
|---|---|---|
| C001 | HIQL vanilla | baseline |
| C002 | CRL vanilla | baseline |
| C003–C008 | CRL actor | K=1/2/4 × no-residual/residual |
| C009–C014 | HIQL high actor | K=1/2/4 × no-residual/residual |
| C015–C020 | HIQL low actor | K=1/2/4 × no-residual/residual |
| C021–C026 | HIQL high+low actor | K=1/2/4 × no-residual/residual |

### 3.2 M9A baseline 与训练时长问题

| 环境 | HIQL baseline 500k | HIQL baseline 1M final | CRL baseline 500k | CRL baseline 1M final |
|---|---:|---:|---:|---:|
| Large | .81 | .87 | .75 | .62 |
| Medium | .97 | .96 | .96 | .93 |

M9A 的 24 个计算配置只有 500k 评估点，而两个 vanilla baseline 训练到 1M。因此不能把 M9A 计算组的 500k final 直接与 baseline 的 1M final 当成严格公平比较。下表使用 baseline 的 500k 数值作为同训练时刻的参考；baseline 的 1M 结果只作为长训练参考保留。

### 3.3 M9A 分组统计

“组均值”是该组配置在 500k 的 final success 平均值；“best 均值”是每个配置的训练期间 best 再取均值；delta 是每个配置相对同算法 baseline@500k 的差异。

#### Large

| 组 | n | final@500k 均值 | best 均值 | 相对 baseline@500k 的 delta 均值 | delta 范围 |
|---|---:|---:|---:|---:|---:|
| CRL actor | 6 | .877 | .892 | +.127 | +.09 ～ +.16 |
| HIQL high | 6 | .883 | .910 | +.073 | +.02 ～ +.13 |
| HIQL low | 6 | .877 | .902 | +.067 | +.03 ～ +.12 |
| HIQL high+low | 6 | .852 | .912 | +.042 | -.02 ～ +.11 |

#### Medium

| 组 | n | final@500k 均值 | best 均值 | 相对 baseline@500k 的 delta 均值 | delta 范围 |
|---|---:|---:|---:|---:|---:|
| CRL actor | 6 | .937 | .957 | -.023 | -.07 ～ +.01 |
| HIQL high | 6 | .955 | .982 | -.015 | -.05 ～ +.02 |
| HIQL low | 6 | .953 | .967 | -.017 | -.03 ～ 0.00 |
| HIQL high+low | 6 | .945 | .973 | -.025 | -.07 ～ +.01 |

### 3.4 M9A Large 全部计算组

| 配置 | placement/variant | final@500k | best | best step |
|---|---|---:|---:|---:|
| C003 | CRL actor K1 nores | .84 | .85 | 400k |
| C004 | CRL actor K1 res | .86 | .86 | 500k |
| C005 | CRL actor K2 nores | .87 | .89 | 400k |
| C006 | CRL actor K2 res | .87 | .90 | 400k |
| C007 | CRL actor K4 nores | .91 | .92 | 400k |
| C008 | CRL actor K4 res | .91 | .93 | 300k |
| C009 | HIQL high K1 nores | .83 | .91 | 400k |
| C010 | HIQL high K1 res | .91 | .91 | 500k |
| C011 | HIQL high K2 nores | .92 | .92 | 500k |
| C012 | HIQL high K2 res | .85 | .85 | 300k |
| C013 | HIQL high K4 nores | .85 | .93 | 300k |
| C014 | HIQL high K4 res | .94 | .94 | 400k |
| C015 | HIQL low K1 nores | .84 | .84 | 500k |
| C016 | HIQL low K1 res | .85 | .92 | 400k |
| C017 | HIQL low K2 nores | .89 | .89 | 500k |
| C018 | HIQL low K2 res | .86 | .90 | 400k |
| C019 | HIQL low K4 nores | .93 | .93 | 500k |
| C020 | HIQL low K4 res | .89 | .93 | 300k |
| C021 | HIQL high+low K1 nores | .92 | .92 | 500k |
| C022 | HIQL high+low K1 res | .82 | .93 | 400k |
| C023 | HIQL high+low K2 nores | .83 | .83 | 300k |
| C024 | HIQL high+low K2 res | .88 | .94 | 300k |
| C025 | HIQL high+low K4 nores | .79 | .92 | 400k |
| C026 | HIQL high+low K4 res | .87 | .93 | 200k |

M9A Large 的关键信息是：CRL actor 的 K=4 组最稳定地超过 CRL vanilla@500k；HIQL high 组中 C014 最强，但 C012/C013 说明增加 K 并非自动获益；HIQL high+low 组的 best 可能很高，但 final 方差更大，C025 的 `.79` 是明显反例。因此不能将“更多 state computation”作为无条件正向结论。

### 3.5 M9A Medium 全部计算组

| 配置 | placement/variant | final@500k | best | best step |
|---|---|---:|---:|---:|
| C003 | CRL actor K1 nores | .94 | .94 | 500k |
| C004 | CRL actor K1 res | .89 | .91 | 400k |
| C005 | CRL actor K2 nores | .96 | .98 | 300k |
| C006 | CRL actor K2 res | .91 | .97 | 400k |
| C007 | CRL actor K4 nores | .97 | .98 | 300k |
| C008 | CRL actor K4 res | .95 | .96 | 300k |
| C009 | HIQL high K1 nores | .98 | .98 | 500k |
| C010 | HIQL high K1 res | .99 | .99 | 500k |
| C011 | HIQL high K2 nores | .95 | .96 | 300k |
| C012 | HIQL high K2 res | .94 | .99 | 100k |
| C013 | HIQL high K4 nores | .92 | .98 | 300k |
| C014 | HIQL high K4 res | .95 | .99 | 400k |
| C015 | HIQL low K1 nores | .97 | .97 | 500k |
| C016 | HIQL low K1 res | .96 | .96 | 300k |
| C017 | HIQL low K2 nores | .95 | .97 | 200k |
| C018 | HIQL low K2 res | .96 | .97 | 300k |
| C019 | HIQL low K4 nores | .94 | .97 | 200k |
| C020 | HIQL low K4 res | .94 | .96 | 300k |
| C021 | HIQL high+low K1 nores | .97 | .97 | 400k |
| C022 | HIQL high+low K1 res | .93 | .97 | 100k |
| C023 | HIQL high+low K2 nores | .92 | .96 | 200k |
| C024 | HIQL high+low K2 res | .90 | .99 | 100k |
| C025 | HIQL high+low K4 nores | .98 | .98 | 400k |
| C026 | HIQL high+low K4 res | .97 | .97 | 300k |

Medium baseline 已经处于 `.96/.97` 的高位，单次评估噪声和 ceiling effect 会使结构差异难以解释。M9A Medium 不支持“SingleState 普遍提升”的结论；它更适合作为 sanity check，而不是后续结构筛选的主要依据。

## 4. M9B：TwoState 全部对照组

### 4.1 M9B 配置映射

| 配置区间 | placement | schedule/credit |
|---|---|---|
| C001–C004 | CRL actor | H2L1/H2L6 × full-BPTT/one-step |
| C005–C008 | HIQL high actor | H2L1/H2L6 × full-BPTT/one-step |
| C009–C012 | HIQL low actor | H2L1/H2L6 × full-BPTT/one-step |
| C013–C016 | HIQL high+low actor | H2L1/H2L6 × full-BPTT/one-step |

### 4.2 M9B 分组统计

#### Large

| 组 | n | final@500k 均值 | best 均值 | 相对同算法 baseline@500k 的 delta 均值 | delta 范围 |
|---|---:|---:|---:|---:|---:|
| CRL actor | 4 | .823 | .843 | +.073 | -.05 ～ +.14 |
| HIQL high | 4 | .888 | .893 | +.078 | +.04 ～ +.12 |
| HIQL low | 4 | .785 | .830 | -.025 | -.08 ～ 0.00 |
| HIQL high+low | 4 | .847 | .902 | +.038 | -.01 ～ +.10 |

#### Medium

| 组 | n | final@500k 均值 | best 均值 | 相对同算法 baseline@500k 的 delta 均值 | delta 范围 |
|---|---:|---:|---:|---:|---:|
| CRL actor | 4 | .922 | .935 | -.038 | -.12 ～ 0.00 |
| HIQL high | 4 | .980 | .988 | +.010 | 0.00 ～ +.02 |
| HIQL low | 4 | .945 | .975 | -.025 | -.08 ～ +.01 |
| HIQL high+low | 4 | .927 | .960 | -.043 | -.08 ～ -.02 |

### 4.3 M9B Large 全部配置

baseline 为 CRL `.75`、HIQL `.81`，delta 均为相对同训练步数 baseline@500k。

| 配置 | placement | schedule | credit | final | best | best step | delta |
|---|---|---|---|---:|---:|---:|---:|
| C001 | CRL actor | H2L1 | full | .84 | .90 | 300k | +.09 |
| C002 | CRL actor | H2L1 | one-step | .86 | .86 | 500k | +.11 |
| C003 | CRL actor | H2L6 | full | .89 | .91 | 400k | +.14 |
| C004 | CRL actor | H2L6 | one-step | .70 | .70 | 500k | -.05 |
| C005 | HIQL high | H2L1 | full | .89 | .91 | 400k | +.08 |
| C006 | HIQL high | H2L1 | one-step | .93 | .93 | 400k | +.12 |
| C007 | HIQL high | H2L6 | full | .88 | .88 | 500k | +.07 |
| C008 | HIQL high | H2L6 | one-step | .85 | .85 | 500k | +.04 |
| C009 | HIQL low | H2L1 | full | .73 | .77 | 400k | -.08 |
| C010 | HIQL low | H2L1 | one-step | .81 | .92 | 400k | 0.00 |
| C011 | HIQL low | H2L6 | full | .80 | .83 | 300k | -.01 |
| C012 | HIQL low | H2L6 | one-step | .80 | .80 | 500k | -.01 |
| C013 | HIQL high+low | H2L1 | full | .91 | .95 | 400k | +.10 |
| C014 | HIQL high+low | H2L1 | one-step | .83 | .85 | 300k | +.02 |
| C015 | HIQL high+low | H2L6 | full | .85 | .92 | 200k | +.04 |
| C016 | HIQL high+low | H2L6 | one-step | .80 | .89 | 200k | -.01 |

### 4.4 M9B Medium 全部配置

baseline 为 CRL `.96`、HIQL `.97`。

| 配置 | placement | schedule | credit | final | best | best step | delta |
|---|---|---|---|---:|---:|---:|---:|
| C001 | CRL actor | H2L1 | full | .94 | .98 | 400k | -.02 |
| C002 | CRL actor | H2L1 | one-step | .96 | .96 | 500k | 0.00 |
| C003 | CRL actor | H2L6 | full | .95 | .96 | 400k | -.01 |
| C004 | CRL actor | H2L6 | one-step | .84 | .84 | 500k | -.12 |
| C005 | HIQL high | H2L1 | full | .97 | 1.00 | 100k | 0.00 |
| C006 | HIQL high | H2L1 | one-step | .99 | .99 | 200k | +.02 |
| C007 | HIQL high | H2L6 | full | .99 | .99 | 200k | +.02 |
| C008 | HIQL high | H2L6 | one-step | .97 | .97 | 500k | 0.00 |
| C009 | HIQL low | H2L1 | full | .95 | .97 | 400k | -.02 |
| C010 | HIQL low | H2L1 | one-step | .89 | .95 | 300k | -.08 |
| C011 | HIQL low | H2L6 | full | .96 | 1.00 | 200k | -.01 |
| C012 | HIQL low | H2L6 | one-step | .98 | .98 | 300k | +.01 |
| C013 | HIQL high+low | H2L1 | full | .89 | .96 | 100k | -.08 |
| C014 | HIQL high+low | H2L1 | one-step | .93 | .96 | 300k | -.04 |
| C015 | HIQL high+low | H2L6 | full | .95 | .96 | 300k | -.02 |
| C016 | HIQL high+low | H2L6 | one-step | .94 | .96 | 300k | -.03 |

### 4.5 M9B 的机制解读

- **HIQL high 是 M9B Large 最一致的受益 placement。** 四个配置全部超过 HIQL baseline@500k，delta 为 `+0.04–+0.12`。
- **HIQL low-only 不支持“低层 state computation 普遍有效”。** Large 的 full-BPTT H2L1 只有 `.73`，而 one-step H2L1 的 best 虽达到 `.92`，final 却回落到 `.81`；这更像 checkpoint/训练动力学问题，而不是稳定提升。
- **high+low 需要 full-BPTT 的证据较强。** Large 上 full-BPTT 的 C013/C015 分别为 `.91/.85`，one-step 的 C014/C016 为 `.83/.80`；但 H2L6 full 的 final 仍低于其 best，说明结构收益与后期稳定性是两个问题。
- **one-step 与 full-BPTT 不是全局可排序关系。** CRL H2L1 中 one-step 高于 full；CRL H2L6 中 full 明显高于 one-step；HIQL high 也随 schedule 改变方向。因此不应根据 M9B 单 seed 结果宣称某一 credit rule 普遍优于另一种。
- **Medium 上的差异不适合承担主要机制结论。** 多数组已接近 1.0，M9B 的额外结构反而带来若干下降，说明 Large 的困难程度是揭示差异的关键。

## 5. 跨 M9A/M9B/M9B-1M 的综合判断

### 5.1 目前最可信的结论

1. M9 的计算结构在 Large 上有能力超过 vanilla baseline，尤其是 CRL actor 的 SingleState K=4、M9B 的 HIQL high、以及 M9B high+low 的 full-BPTT 配置。
2. 这种优势高度依赖 placement、schedule、credit rule 与训练时刻；“加 state”本身不是充分条件。
3. M9B-1M 证明 selected TwoState 配置可以在 1M horizon 保持高性能，没有出现普遍崩溃；但它没有证明继续训练一定带来收益。
4. best checkpoint 往往早于 final checkpoint，且同一个配置在 500k 与 1M 之间可能上升、下降或先升后降。因此后续实验必须预先规定 best/last 的报告和选择规则。

### 5.2 目前不能声称的结论

- 不能声称 M9B-1M 相对于 vanilla 的提升已经具有统计显著性：每个配置只有一个训练 seed，baseline 也是单 seed 外部参考。
- 不能把 M9B-1M 与历史 M9B 的差异解释为“1M 训练步数带来的纯收益”：它们是不同的 fresh run，且 source commit 不同。
- 不能声称 full-BPTT 普遍优于 one-step，或 H2L6 普遍优于 H2L1。
- 不能从 actor/value loss 或 gradient norm 单独推断 state dynamics 学到了预期的层级计算机制；当前训练日志缺少 `z_H/z_L` 的范数、更新量和跨步耦合统计。
- 不能用 Medium 的接近饱和结果替代 Large 上的结构验证。

### 5.3 对 M9B-1M 当前结果的建议表述

建议在论文/阶段报告中使用如下强度的表述：

> 在 `antmaze-large-navigate-v0`、seed 0 的 1M-step run 中，四个 selected TwoState full-BPTT 配置均达到较高 success；相较已有 vanilla reference，CRL 配置显示出较大的描述性优势，HIQL high+low 配置显示出较小但正向的优势。训练曲线存在显著的 checkpoint-dependent fluctuation，因此结果应同时报告 best checkpoint 与 last@1M。由于尚无多 seed 结果，以上结论仍属于 preliminary single-seed evidence。

## 6. M9 是否需要补充实验

### P0：必须优先补充

#### P0-1：M9B-1M 四配置的多 seed

对以下四个配置保持完全相同的环境、协议、代码 commit 与数据版本，补 `seed=1`、`seed=2`；如果 GPU 预算允许，再加 `seed=3`。

| 配置 | 原因 |
|---|---|
| CRL H2L1 full | 1M final/best 都高，且相对 CRL baseline 优势最大之一 |
| CRL H2L6 full | 700–800k 高峰后回落，最需要判断是不是 seed-specific fluctuation |
| HIQL high+low H2L1 full | 700k best 高但 1M final 回落，验证 high+low 的稳定性 |
| HIQL high+low H2L6 full | 400–500k 很强，后期波动明显，验证 H2L6 是否有稳定优势 |

正式结果至少应报告 mean、standard deviation 或 confidence interval，并分别报告 final@1M、best-over-training 和 best-step 分布。用户需手动启动这些正式实验；本报告不启动它们。

#### P0-2：best/last checkpoint 的高样本 reevaluation

对每个 M9B-1M run 的 best checkpoint 和 last@1M checkpoint，用固定 deterministic policy、每任务至少 100 episodes 重新评估。优先对象是：

- C001：200k best vs 1M last；
- C002：700k best vs 1M last；
- C003：700k best vs 1M last；
- C004：400k best vs 1M last。

这一步不改变训练结果，只是降低 20-episode evaluation 的测量噪声。若重新评估后 best/last 排序改变，应以高样本 reevaluation 作为最终 checkpoint 选择依据，并在报告中保留原始 20-episode 曲线。

### P1：为消除现有比较中的重要混杂因素

#### P1-1：同 commit 的 vanilla reference

当前 vanilla baseline 来自旧 commit `f30b64b...`，M9B-1M 来自 `c2f6d7e...`。如果最终需要做强因果表述，建议在 M9B-1M 使用的 clean commit 上补一对 Large vanilla reference：CRL vanilla 1M、HIQL vanilla 1M，协议保持一致，仍然只用 seed 0 作为 source-matched reference，之后再用多 seed 作为统计验证。

这不是解释当前结果所必需的紧急实验，因为现有 baseline 的数据、环境和主要训练协议已经对齐；但它会显著降低“source commit 差异”对最终结论的威胁。

#### P1-2：M9A 的 selected 1M duration extension

M9A 的计算组只到 500k，而 baseline 到 1M。如果 M9A 的 K/residual/placement 结论要进入最终论文，建议不要把 24 个配置全部延长，而是选择 Large 上有代表性的少量配置，例如：

- CRL actor：M9A-C007（K4 nores）、M9A-C008（K4 res）；
- HIQL high：M9A-C011（K2 nores）、M9A-C014（K4 res）；
- HIQL low：M9A-C019（K4 nores）；
- HIQL high+low：M9A-C021（K1 nores）和一个高方差配置 M9A-C025（K4 nores）。

目标不是继续寻找最高分，而是判断 M9A 的 500k 排名在 1M 是否保持，以及 residual/K 的收益是否只是早期 checkpoint 现象。Medium 不建议优先延长，因为已接近 ceiling。

### P2：机制诊断与实验设计补全

#### P2-1：增加 TwoState 内部状态 telemetry

在后续正式 run 中记录但不参与梯度的诊断量：

- `||z_H||`、`||z_L||` 及其 batch 分布；
- 每次 H/L update 的 `||Δz_H||`、`||Δz_L||`；
- H/L state cosine similarity 或相关性；
- final actor representation 与 warm-up representation 的变化；
- full-BPTT 与 one-step 的梯度范数分解。

这些数据才能判断 TwoState 是否形成了可解释的快/慢状态动力学。当前 `train.csv` 中的 loss 与 gradient norm 不足以完成该机制验证。

#### P2-2：固定报告规则

后续 Study 应预先固定：

- primary metric 是 last@1M 还是高样本 reevaluated best；
- 是否允许使用训练期间 best checkpoint；
- best tie rule；
- 是否按所有任务平均，还是同时报告每任务结果；
- 多 seed 的聚合和置信区间方法。

### 不建议现在补充的内容

- 不建议继续扩大 M9B 的 placement × schedule × credit 全网格；已有 M9B 16 配置足以显示强交互，当前瓶颈是重复性而不是候选结构不足。
- 不建议现在引入 HRM Transformer、Attention、SwiGLU、RMSNorm 或新的 hidden width；那会把 M9 的结构验证扩展成另一个研究问题。
- 不建议把 Medium 作为主要补实验环境；它适合做回归检查，不适合作为选择结构的主要依据。
- 不建议在没有多 seed 和高样本 reevaluation 前，继续针对单个峰值配置做更多长训搜索。

## 7. 建议的后续执行顺序

1. 用户手动确认并启动 M9B-1M seed 1/2 的四配置复现实验。
2. 对已经完成的四个 seed-0 run 做 best/last 100-episode reevaluation。
3. 汇总多 seed 的 final、best、best step、每任务 success 和训练曲线波动。
4. 如果 M9B-1M 的优势在多 seed 后仍保持，再决定是否补同 commit vanilla reference。
5. 只有在需要保留 M9A 的 long-horizon claim 时，才补 M9A selected 1M extension。
6. 在下一轮代码变更中加入 TwoState telemetry，再进行机制层面的解释。

## 8. 最终判定

截至 2026-08-20，M9B-1M 可以作为一个**完成且有效的 preliminary convergence result** 纳入 M9 实验记录：它证明 selected TwoState full-BPTT 配置在 Large、1M steps、seed 0 下能够达到 `.85–.92` 的 final success，并且相对已有 vanilla reference 具有明显的 CRL 描述性优势和较小的 HIQL 描述性优势。

但 M9 系列尚不能作为多 seed、统计稳健、机制已验证的最终实验结论。最小而充分的补实验集合是：**M9B-1M 四配置 × seed 1/2 + best/last 高样本 reevaluation**。在这两项完成前，不需要再增加新的 M9 结构变量。
