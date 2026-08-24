# M12B 最终实验结果与初步分析

日期：2026-08-24  
项目：RLC 研究平台  
实验：M12B Computation Architecture + Structure Isolation  
环境：`antmaze-large-navigate-v0`  
算法：CRL frozen-critic policy extraction  
状态：21/21 active runs completed；仅做结果分析，未启动新实验，未执行任何 Git 操作

## 1. 执行摘要

M12B 的最终结果没有显示整个架构实现失效。最显著的现象是：

> `K=4 + shared parameter + zero state initialization` 明显弱于 `K=4 + shared parameter + normal state initialization`，但仍高于 FF baseline。

这说明异常更接近一个由

```text
state initialization × iteration count K × parameter sharing
```

形成的结构交互，而不是单纯的代码错误或训练崩溃。

核心数字如下：

- M12A K4 shared normal（B004）：last@1M `0.863 ± 0.070`，AUC `0.793 ± 0.042`；
- M12B K4 shared zero（B005）：last@1M `0.737 ± 0.057`，AUC `0.674 ± 0.080`；
- B005 相比 B004 的差值：final `-0.127`，AUC `-0.119`，三条 seed 方向一致；
- M12A FF baseline（B001）：last@1M `0.670 ± 0.125`，AUC `0.586 ± 0.107`；
- B005 仍比 baseline 高：final `+0.067`，AUC `+0.088`；
- K4 untied normal/zero（B006/B007）：final `0.867/0.900`，AUC `0.855/0.859`；
- 9-layer plain FF（B008）：final `0.890 ± 0.060`，AUC `0.850 ± 0.023`；
- residual FF（B009）：final `0.867 ± 0.035`，AUC `0.849 ± 0.010`。

因此，当前不应下结论说“SS K4 在 M12B 中整体失败”。更准确的结论是：**shared K4 对 state initialization 高度敏感；untied K4 和深层 FF 可以恢复性能。**

## 2. 数据范围与统计口径

### 2.1 实验完成状态

M12B active matrix 为 7 个条件 × 3 个 seed，共 21 个 run。当前逐目录核对结果为：

- 21/21 `runtime_metadata.status = completed`；
- 21/21 有 10 个 eval 点，覆盖 `100k, ..., 1M`；
- 21/21 有 200 个 train 记录，覆盖到 `1M`；
- 21/21 有 `last/params_1000000.pkl`；
- 21/21 有 `best` checkpoint 和 `summary.json`；
- 21/21 `accounting_consistency.status = pass`；
- train/eval 数值字段没有检测到 NaN、Inf 或解析失败。

### 2.2 指标定义

- `last@1M`：1M 训练步的 `evaluation/overall_success`；
- `best`：训练过程中按 `evaluation/overall_success` 选择的最佳 checkpoint 值；
- `AUC`：100k 到 1M eval 曲线的梯形积分，再除以该区间长度；
- 表中的 `±`：三个 seed 的样本标准差；
- 每次 evaluation 使用 20 episodes、all tasks、temperature 0、无 Gaussian noise；
- 所有正式运行 train steps 为 1M，eval interval 为 100k，batch size 为 1024。

### 2.3 M12A 对照与 M12B 条件映射

| 条件 | 实际来源 | topology/block | K 或深度 | state init | sharing | 角色 |
|---|---|---|---:|---|---|---|
| B001 | M12A-C002 attempt 1 | plain FF | 3 hidden Dense | — | — | FF baseline anchor |
| B002 | M12B-C001 | SingleState + plain | K=1 | normal | shared | active |
| B003 | M12B-C002 | SingleState + plain | K=1 | zero | shared | active |
| B004 | M12A-C003 attempt 1 | SingleState + plain | K=4 | normal | shared | M12A K4 anchor |
| B005 | M12B-C003 | SingleState + plain | K=4 | zero | shared | active |
| B006 | M12B-C004 | SingleState + plain | K=4 | normal | untied | active |
| B007 | M12B-C005 | SingleState + plain | K=4 | zero | untied | active |
| B008 | M12B-C006 | plain FF | 9 hidden Dense | — | — | active |
| B009 | M12B-C007 | residual FF | 4 blocks × 2 Dense | — | — | active |

特别注意：M12A K4 是 B004，即 `K4 + shared + normal`；M12B active 的 shared K4 是 B005，即 `K4 + shared + zero`。所以 B005 与 B004 的比较首先是 state initialization contrast，而不是同条件版本复现。

## 3. 完整总体结果矩阵

### 3.1 每个 seed 的结果

| 条件 | last@1M s0 / s1 / s2 | best s0 / s1 / s2 | best step s0 / s1 / s2 | AUC s0 / s1 / s2 |
|---|---|---|---|---|
| B001 M12A FF | 0.63 / 0.57 / 0.81 | 0.70 / 0.68 / 0.84 | 700k / 800k / 900k | 0.5033 / 0.5478 / 0.7072 |
| B002 K1 shared normal | 0.68 / 0.50 / 0.81 | 0.71 / 0.50 / 0.81 | 500k / 600k / 1M | 0.5772 / 0.4128 / 0.6256 |
| B003 K1 shared zero | 0.68 / 0.81 / 0.87 | 0.69 / 0.81 / 0.89 | 700k / 1M / 800k | 0.5344 / 0.5894 / 0.7383 |
| B004 M12A K4 shared normal | 0.93 / 0.79 / 0.87 | 0.93 / 0.85 / 0.87 | 1M / 700k / 900k | 0.8322 / 0.7489 / 0.7983 |
| B005 K4 shared zero | 0.72 / 0.69 / 0.80 | 0.80 / 0.71 / 0.81 | 800k / 900k / 600k | 0.7278 / 0.5822 / 0.7128 |
| B006 K4 untied normal | 0.88 / 0.86 / 0.86 | 0.88 / 0.89 / 0.92 | 1M / 400k / 400k | 0.8361 / 0.8506 / 0.8794 |
| B007 K4 untied zero | 0.87 / 0.93 / 0.90 | 0.91 / 0.93 / 0.95 | 900k / 1M / 900k | 0.8633 / 0.8267 / 0.8872 |
| B008 9-layer plain FF | 0.95 / 0.83 / 0.89 | 0.95 / 0.92 / 0.94 | 1M / 600k / 500k | 0.8533 / 0.8256 / 0.8711 |
| B009 residual FF | 0.83 / 0.87 / 0.90 | 0.88 / 0.91 / 0.90 | 800k / 500k / 1M | 0.8567 / 0.8522 / 0.8378 |

### 3.2 三 seed aggregate

| 条件 | last@1M mean ± sd | best mean ± sd | AUC mean ± sd | final min–max |
|---|---:|---:|---:|---:|
| B001 M12A FF | 0.670 ± 0.125 | 0.740 ± 0.087 | 0.586 ± 0.107 | 0.57–0.81 |
| B002 K1 shared normal | 0.663 ± 0.156 | 0.673 ± 0.158 | 0.539 ± 0.112 | 0.50–0.81 |
| B003 K1 shared zero | 0.787 ± 0.097 | 0.797 ± 0.101 | 0.621 ± 0.106 | 0.68–0.87 |
| B004 M12A K4 shared normal | 0.863 ± 0.070 | 0.883 ± 0.042 | 0.793 ± 0.042 | 0.79–0.93 |
| B005 K4 shared zero | 0.737 ± 0.057 | 0.773 ± 0.055 | 0.674 ± 0.080 | 0.69–0.80 |
| B006 K4 untied normal | 0.867 ± 0.012 | 0.897 ± 0.021 | 0.855 ± 0.022 | 0.86–0.88 |
| B007 K4 untied zero | 0.900 ± 0.030 | 0.930 ± 0.020 | 0.859 ± 0.031 | 0.87–0.93 |
| B008 9-layer plain FF | 0.890 ± 0.060 | 0.937 ± 0.015 | 0.850 ± 0.023 | 0.83–0.95 |
| B009 residual FF | 0.867 ± 0.035 | 0.897 ± 0.015 | 0.849 ± 0.010 | 0.83–0.90 |

## 4. 每个 task 的完整结果

下表为 1M 时各 task success 的 seed 均值 ± 样本标准差。

| 条件 | task1 | task2 | task3 | task4 | task5 |
|---|---:|---:|---:|---:|---:|
| B001 M12A FF | 0.817 ± 0.104 | 0.667 ± 0.161 | 0.850 ± 0.050 | 0.583 ± 0.275 | 0.433 ± 0.202 |
| B002 K1 shared normal | 0.850 ± 0.087 | 0.617 ± 0.275 | 0.900 ± 0.087 | 0.533 ± 0.351 | 0.417 ± 0.351 |
| B003 K1 shared zero | 0.950 ± 0.050 | 0.617 ± 0.318 | 0.867 ± 0.058 | 0.833 ± 0.104 | 0.667 ± 0.126 |
| B004 M12A K4 shared normal | 0.867 ± 0.104 | 0.850 ± 0.173 | 0.867 ± 0.058 | 0.917 ± 0.058 | 0.817 ± 0.076 |
| B005 K4 shared zero | 0.883 ± 0.058 | 0.717 ± 0.076 | 0.867 ± 0.076 | 0.700 ± 0.100 | 0.517 ± 0.144 |
| B006 K4 untied normal | 0.883 ± 0.115 | 0.850 ± 0.087 | 0.867 ± 0.076 | 0.850 ± 0.087 | 0.883 ± 0.058 |
| B007 K4 untied zero | 0.900 ± 0.050 | 0.850 ± 0.132 | 0.967 ± 0.029 | 0.917 ± 0.029 | 0.867 ± 0.058 |
| B008 9-layer plain FF | 0.883 ± 0.076 | 0.900 ± 0.087 | 0.883 ± 0.076 | 0.933 ± 0.076 | 0.850 ± 0.100 |
| B009 residual FF | 0.883 ± 0.076 | 0.817 ± 0.104 | 0.883 ± 0.126 | 0.900 ± 0.100 | 0.850 ± 0.100 |

### 4.1 B005 相比 B004 的 task-level 变化

| task | B005 − B004 final mean |
|---|---:|
| task1 | +0.017 |
| task2 | −0.133 |
| task3 | 0.000 |
| task4 | −0.217 |
| task5 | −0.300 |

退化主要集中在 task4 和 task5，而不是所有 task 一致下降。B005 相比 B001 baseline 仍然在五个 task 上均为正增益，分别为约 `+0.067、+0.050、+0.017、+0.117、+0.083`。

## 5. 完整学习曲线

以下为各 condition 的 overall success 均值 ± seed 标准差，时间单位为训练步数。

```text
condition   100k          200k          300k          400k          500k          600k          700k          800k          900k          1M
B001 FF     .367±.174     .437±.153     .513±.133     .590±.095     .563±.100     .617±.075     .663±.139     .693±.131     .680±.140     .670±.125
B004 K4SN   .567±.067     .740±.082     .793±.067     .780±.122     .823±.057     .770±.052     .833±.029     .830±.020     .853±.029     .863±.070
B002 K1SN   .363±.031     .440±.101     .500±.066     .523±.115     .593±.146     .567±.070     .563±.204     .597±.196     .550±.121     .663±.156
B003 K1SZ   .347±.050     .460±.115     .580±.123     .520±.069     .607±.144     .613±.156     .730±.106     .757±.119     .753±.100     .787±.097
B005 K4SZ   .447±.093     .607±.121     .637±.090     .663±.067     .673±.060     .757±.076     .677±.137     .720±.114     .743±.031     .737±.057
B006 K4UN   .850±.020     .833±.006     .857±.042     .880±.046     .860±.026     .837±.042     .887±.025     .857±.023     .830±.062     .867±.012
B007 K4UZ   .810±.092     .810±.010     .850±.046     .837±.072     .853±.035     .877±.061     .847±.051     .883±.035     .920±.026     .900±.030
B008 D9     .750±.080     .837±.006     .843±.021     .853±.055     .860±.072     .860±.060     .873±.032     .880±.072     .823±.046     .890±.060
B009 R      .813±.035     .837±.068     .853±.021     .850±.010     .857±.055     .847±.042     .840±.026     .873±.021     .843±.021     .867±.035
```

曲线显示：

- B005 从 100k 起就低于 B004，并非 1M 附近突然崩溃；
- B006/B007/B008/B009 从早期开始就处于较高区间；
- B005 的下降更像学习到的策略质量差异，而不是训练后期 checkpoint 损坏；
- B006 的 final seed 方差最低，说明 untied K4 的结果不仅高，而且稳定。

## 6. 关键 paired contrasts

以下差值按相同整数 seed 对齐。n=3 适合进行方向和效应量判断，不足以支持强统计显著性结论。

| 对比 | final 差值 mean ± sd | AUC 差值 mean ± sd | 初步解释 |
|---|---:|---:|---|
| B004 K4 shared normal − B001 FF | +0.193 ± 0.122 | +0.207 ± 0.119 | M12A K4 相比 FF 有明显增益 |
| B002 K1 shared normal − B001 FF | −0.007 ± 0.060 | −0.048 ± 0.109 | K1 shared normal 基本没有超过 FF |
| B003 K1 shared zero − B002 K1 shared normal | +0.123 ± 0.164 | +0.082 ± 0.113 | zero 在 K1 中有利，但 seed 间差异较大 |
| B005 K4 shared zero − B004 K4 shared normal | −0.127 ± 0.074 | −0.119 ± 0.042 | shared K4 对 initialization 高敏感 |
| B005 K4 shared zero − B001 FF | +0.067 ± 0.057 | +0.088 ± 0.080 | B005 仍优于 baseline，但增益有限 |
| B006 K4 untied normal − B004 K4 shared normal | +0.003 ± 0.061 | +0.062 ± 0.052 | final 相当，AUC 略高 |
| B007 K4 untied zero − B006 K4 untied normal | +0.033 ± 0.040 | +0.004 ± 0.026 | untied 下 init 差异很小 |
| B007 K4 untied zero − B005 K4 shared zero | +0.163 ± 0.071 | +0.185 ± 0.055 | untied zero 显著恢复 |
| B008 9-layer FF − B007 K4 untied zero | −0.010 ± 0.090 | −0.009 ± 0.008 | 两者基本同档 |
| B009 residual FF − B008 9-layer FF | −0.023 ± 0.085 | −0.001 ± 0.030 | residual 与 plain deep FF 的 AUC 几乎相同 |
| B009 residual FF − B006 K4 untied normal | 0.000 ± 0.046 | −0.007 ± 0.032 | 三者整体处于同一性能层级 |

## 7. 参数量、执行深度与计算量

| 条件 | actor total params | state buffer | unique Dense | executed Dense | full actor dense MACs |
|---|---:|---:|---:|---:|---:|
| B001 FF | 559,624 | 0 | 3 | 3 | 558,080 |
| B002/B003 K1 shared | 559,624 | 512 | 3 | 3 | 558,080 |
| B004/B005 K4 shared | 559,624 | 512 | 3 | 9 | 2,130,944 |
| B006/B007 K4 untied | 2,135,560 | 512 | 9 | 9 | 2,130,944 |
| B008 9-layer FF | 2,135,560 | 0 | 9 | 9 | 2,130,944 |
| B009 residual FF | 2,135,560 | 0 | 9 | 9 | 2,130,944 |

这里存在一个必须保留的混杂因素：B005→B007 的性能恢复同时伴随参数 sharing 从 shared 变为 untied，以及 unique parameters 从约 0.56M 增加到约 2.13M。二者执行 Dense 数和 MAC 相同，但参数量不同。因此当前结果支持“shared K4 zero 受限、untied 可以恢复”，但不能把全部增益严格归因于 parameter sharing 本身。

## 8. 训练诊断指标

下表为训练结束时三 seed 均值 ± 标准差；`grad max` 是所有训练记录中的 gradient norm 最大值。

| 条件 | actor loss@1M | actor MSE@1M | actor Q loss@1M | frozen q_delta@1M | grad norm@1M | grad max |
|---|---:|---:|---:|---:|---:|---:|
| B002 K1SN | 1.7390 ± 0.0018 | 0.0807 ± 0.0024 | 0.9716 ± 0.0022 | 0.1417 ± 0.0209 | 42.01 ± 1.13 | 63.25 |
| B003 K1SZ | 1.7384 ± 0.0024 | 0.0807 ± 0.0028 | 0.9710 ± 0.0027 | 0.1533 ± 0.0149 | 35.54 ± 2.30 | 48.30 |
| B005 K4SZ | 1.7375 ± 0.0022 | 0.0771 ± 0.0032 | 0.9715 ± 0.0032 | 0.1492 ± 0.0137 | 57.71 ± 1.41 | 76.41 |
| B006 K4UN | 1.7353 ± 0.0018 | 0.0767 ± 0.0043 | 0.9695 ± 0.0017 | 0.2335 ± 0.0278 | 80.16 ± 1.50 | 100.46 |
| B007 K4UZ | 1.7344 ± 0.0016 | 0.0759 ± 0.0044 | 0.9689 ± 0.0020 | 0.2454 ± 0.0214 | 81.96 ± 3.31 | 97.03 |
| B008 D9 | 1.7347 ± 0.0023 | 0.0760 ± 0.0041 | 0.9692 ± 0.0026 | 0.2411 ± 0.0157 | 87.69 ± 3.68 | 99.56 |
| B009 R | 1.7345 ± 0.0012 | 0.0752 ± 0.0037 | 0.9693 ± 0.0018 | 0.2388 ± 0.0183 | 80.74 ± 4.21 | 90.68 |

所有 condition 的 frozen critic validation contrastive loss 在相同 seed 下保持一致，三 seed aggregate 约为 `0.006358 ± 0.000204`。这与 frozen critic 依赖没有被 condition-specific 污染的判断一致。

## 9. provenance 与结果可信度

21 个 M12B run 的 metadata 一致显示：

- `git_commit = bb2644ccb23ee77a0c08e8b9cded85a57716df67`；
- `git_dirty = false`；
- runtime code worktree 为 `/home/eai/Research/RLC-M12B-final`；
- backend 为 GPU；
- environment 与 dataset identity 一致；
- 每个 seed 的 7 个 active condition 复用同一 frozen critic checkpoint SHA/module fingerprint；
- 三个 seed 的 critic source 分别为同 seed 的 M12A-C001 last@1M checkpoint；
- accounting consistency 全部 pass；
- 训练和 evaluation artifact 完整。

因此，从 artifact 完整性、运行一致性和 provenance 看，M12B 最终结果可信。需要区分两类可信度：

1. **数据可信度：高。** 当前没有发现缺失、NaN、错误状态或 checkpoint lifecycle 问题。
2. **科学因果结论：中等偏高。** 现有三 seed 结果足以支持结构交互的初步判断，但 B004 是 M12A external anchor，且 untied 对比存在参数量混杂，仍不应写成最终普遍定律。

## 10. 初步科学分析

### 10.1 shared K4 的退化不是普通训练崩溃

B005 的 actor loss、Q loss、q_delta 和 gradient norm 均为正常有限值，没有 NaN/Inf 或梯度爆炸。它从 100k 开始就低于 B004，而不是在 1M 末期突然坍塌。

所以目前更像是：不同 computation structure 学到了不同质量的 policy，而不是训练进程失控或 checkpoint 损坏。

### 10.2 state initialization 的作用依赖 K

在 K1 shared 中：

```text
zero > normal
```

在 K4 shared 中：

```text
zero < normal
```

这说明不能对 zero initialization 给出脱离 topology 和 iteration count 的单向评价。K 增大后，同一个 shared update module 被反复使用，初始 state 对最终迭代轨迹的影响可能被放大。

### 10.3 untied K4 恢复性能，但参数量是混杂因素

B007 相比 B005 的 final 提升约 `+0.163`，AUC 提升约 `+0.185`，而且 task4/task5 恢复明显。这表明 untied K4 或更高参数容量能够摆脱 shared K4 zero 的限制。

但是 B006/B007 的 unique Dense 数从 3 增加为 9，actor total params 从约 0.56M 增加到约 2.14M。因此还不能严格说是“untied parameter sharing 单独带来提升”。

### 10.4 深层 FF 与 recurrent/iterative structure 处于同一高性能层级

B006、B007、B008、B009 都有 9 个 executed Dense、相近的 full actor MAC；其中：

- B006 final `0.867`，AUC `0.855`；
- B007 final `0.900`，AUC `0.859`；
- B008 final `0.890`，AUC `0.850`；
- B009 final `0.867`，AUC `0.849`。

这说明在当前任务上，增加有效执行深度/容量本身已经能带来高性能；目前没有证据证明 iterative state computation 明显优于同等执行深度的 plain FF 或 residual FF。

### 10.5 高执行深度条件的 seed 稳定性更好

final seed 标准差为：

- B002 K1 shared normal：`0.156`；
- B003 K1 shared zero：`0.097`；
- B005 K4 shared zero：`0.057`；
- B006 K4 untied normal：`0.012`；
- B007 K4 untied zero：`0.030`；
- B008 D9：`0.060`；
- B009 residual：`0.035`。

这提示较高执行深度和较高容量可能不仅提高平均性能，也提高训练稳定性。但 seed 数只有 3，当前应将其视为趋势，而非确定性统计结论。

## 11. 当前可以成立与不能成立的结论

### 可以成立

- M12B 21 个 active run 均已完成，结果矩阵完整。
- M12B 没有出现全局性 implementation failure 或数值训练崩溃。
- B005 的退化是三 seed 一致的、可重复的 condition-level phenomenon。
- shared K4 对 state initialization 敏感；zero 在 K1 和 K4 中作用方向相反。
- B005 仍优于 FF baseline，但显著低于 M12A K4 shared normal anchor。
- untied K4、9-layer plain FF、residual FF 均能达到约 0.85–0.90 的 final success。
- task4/task5 是 shared K4 zero 退化最明显的 task。

### 不能成立

- 不能说“SS K4 在 M12B 中整体失效”；B006/B007 和 B004 不支持该说法。
- 不能说 zero initialization 普遍有害；B003 反而优于 B002。
- 不能将 B005→B007 的全部提升归因于 parameter sharing，因为参数量同时变化。
- 不能据此证明 iterative computation 优于普通深层 FF；B008/B009 与 B006/B007 同档。
- 不能基于三 seed 得出强显著性或跨环境普遍化结论。

## 12. 建议的后续实验与分析

### 优先级 P1：确认当前 M12B commit 下的 B004

直接运行 `K4 + shared + normal_buffer`，与 B005 使用完全相同的当前 M12B commit、seed 和 protocol。目的不是重复 sweep，而是确认 M12A external anchor 与当前 runtime 的兼容性和 B004/B005 初始化差异。

### 优先级 P1：提高 evaluation 精度

对 B004、B005、B006、B007 的 last/best checkpoint 使用 100–200 episodes 重新评估，并保留 task-level 结果。现有 20 episodes 已足以支持大方向，但更高 episode 数可减少 0.05 success resolution 带来的噪声。

### 优先级 P2：增加 state trajectory diagnostics

针对 shared K4 normal/zero、untied K4 normal/zero，记录每个 iteration 的：

- state norm；
- update norm；
- input injection norm；
- activation norm；
- action distribution；
- 每个 task 的 Q-policy 与 Q-data gap。

这可以直接检验 zero shared K4 是否进入了不利的 state trajectory，而不是只根据最终 success 进行推断。

### 优先级 P2：参数量匹配的 sharing ablation

若要严格回答 parameter sharing 的作用，需要构造参数量接近的对照，例如缩小 untied K4 宽度或增大 shared K4 宽度，同时保持 K、state init、input injection 和 training protocol 不变。

## 13. 原始数据与复核入口

M12B 原始运行目录：

```text
/data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M12B/
```

每个 seed 目录包含：

```text
eval.csv
train.csv
summary.json
runtime_metadata.json
resolved_config.json
checkpoints/last/
checkpoints/best/
```

M12A 对照目录：

```text
/data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M12A/
```

本报告所有统计均由上述当前 artifacts 重新读取计算；没有修改任何实验结果、运行状态或 Git 状态。

