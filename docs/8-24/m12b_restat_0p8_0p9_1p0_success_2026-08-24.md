# M12B 全部条件重新统计

## 基于 0.8M / 0.9M / 1.0M success_rate

日期：2026-08-24

## 1. 统计口径

本次只使用 evaluation/overall_success 在以下三个精确 checkpoint 的数值：

- 0.8M = step 800,000
- 0.9M = step 900,000
- 1.0M = step 1,000,000

对每个 seed 先计算三点算术平均：

mean_3(seed) = [success(0.8M) + success(0.9M) + success(1.0M)] / 3

再对三个 seed 的 mean_3 做宏平均，作为 condition 的总体指标。没有使用 best、AUC 或其它 checkpoint。

M12B-R 的 K4 shared normal 位于 M12A-C003 的 attempt 2 artifact 路径，但其 resolved semantics 和 M12B protocol 已按前置 preflight 验证；本报告将其作为 M12B 缺失 condition 纳入。

## 2. 每个 seed 的原始三点值与 seed 内平均

百分数格式为 0.8M / 0.9M / 1.0M → 三点平均。

| condition | seed 0 | seed 1 | seed 2 | 跨 seed平均 |
|---|---:|---:|---:|---:|
| K1 shared normal | 58 / 57 / 68 → 61.0 | 41 / 42 / 50 → 44.3 | 80 / 66 / 81 → 75.7 | 60.3 |
| K1 shared zero | 66 / 64 / 68 → 66.0 | 72 / 79 / 81 → 77.3 | 89 / 83 / 87 → 86.3 | 76.6 |
| K4 shared normal | 88 / 91 / 70 → 83.0 | 87 / 85 / 93 → 88.3 | 93 / 89 / 86 → 89.3 | 86.9 |
| K4 shared zero | 80 / 77 / 72 → 76.3 | 59 / 71 / 69 → 66.3 | 77 / 75 / 80 → 77.3 | 73.3 |
| K4 untied normal | 87 / 78 / 88 → 84.3 | 87 / 81 / 86 → 84.7 | 83 / 90 / 86 → 86.3 | 85.1 |
| K4 untied zero | 88 / 91 / 87 → 88.7 | 85 / 90 / 93 → 89.3 | 92 / 95 / 90 → 92.3 | 90.1 |
| Deep FF | 94 / 85 / 95 → 91.3 | 80 / 77 / 83 → 80.0 | 90 / 85 / 89 → 88.0 | 86.4 |
| Residual FF | 88 / 86 / 83 → 85.7 | 89 / 85 / 87 → 87.0 | 85 / 82 / 90 → 85.7 | 86.1 |

总体均值按 seed-level 三点均值计算；不是把 9 个 observation 先按时间混合后再报告。

## 3. 按 checkpoint 的跨 seed 平均

| condition | 0.8M | 0.9M | 1.0M | 三 checkpoint 总体均值 |
|---|---:|---:|---:|---:|
| K1 shared normal | 59.7 | 55.0 | 66.3 | 60.3 |
| K1 shared zero | 75.7 | 75.3 | 78.7 | 76.6 |
| K4 shared normal | 89.3 | 88.3 | 83.0 | 86.9 |
| K4 shared zero | 72.0 | 74.3 | 73.7 | 73.3 |
| K4 untied normal | 85.7 | 83.0 | 86.7 | 85.1 |
| K4 untied zero | 88.3 | 92.0 | 90.0 | 90.1 |
| Deep FF | 88.0 | 82.3 | 89.0 | 86.4 |
| Residual FF | 87.3 | 84.3 | 86.7 | 86.1 |

## 4. 按三点总体均值排序

| rank | condition | mean success |
|---:|---|---:|
| 1 | K4 untied zero | 90.1% |
| 2 | K4 shared normal | 86.9% |
| 3 | Deep FF | 86.4% |
| 4 | Residual FF | 86.1% |
| 5 | K4 untied normal | 85.1% |
| 6 | K1 shared zero | 76.6% |
| 7 | K4 shared zero | 73.3% |
| 8 | K1 shared normal | 60.3% |

## 5. 简单对照分析

### 5.1 K4 shared：normal 明显优于 zero

K4 shared normal 相对 K4 shared zero：

- 0.8M：+17.3 个百分点
- 0.9M：+14.0 个百分点
- 1.0M：+9.3 个百分点
- 三点总体均值：+13.6 个百分点

三个 seed 的三点均值差分别为：

- seed 0：+6.7 个百分点
- seed 1：+22.0 个百分点
- seed 2：+12.0 个百分点

这与此前的正确理解一致：M12B 中表现较弱的是 K4 shared zero，而不是 K4 shared normal。补齐后，K4 shared normal 是 M12B 中表现较强的 condition。

### 5.2 K1 的初始化效应方向相反

K1 shared zero 相对 K1 shared normal 的三点总体均值为：

76.6% - 60.3% = +16.2 个百分点

因此：

- K1 更适合 zero initialization；
- K4 更适合 normal initialization；
- initialization effect 明显依赖 K。

这不是一个“normal 始终更好”或“zero 始终更好”的简单结论。

### 5.3 K4 相对 K1 的优势取决于 initialization

在 normal 条件下：

- K4 shared normal：86.9%
- K1 shared normal：60.3%
- K4 − K1：+26.6 个百分点

在 zero 条件下：

- K4 shared zero：73.3%
- K1 shared zero：76.6%
- K4 − K1：−3.2 个百分点

对应的 K × initialization interaction：

[(K4 normal − K1 normal) − (K4 zero − K1 zero)]

= +29.8 个百分点。

初步看，K4 的优势不是单独由迭代深度决定，而是强烈依赖于 shared K4 与 initialization 的组合。

### 5.4 Shared 与 untied 的影响也依赖 initialization

K4 normal：

- shared：86.9%
- untied：85.1%
- shared − untied：+1.8 个百分点

两者基本接近，shared normal 没有显示出明显劣势。

K4 zero：

- shared：73.3%
- untied：90.1%
- shared − untied：−16.8 个百分点

untied zero 是本次三点均值最高的 condition。说明 parameter sharing 与 initialization 存在明显 interaction：zero initialization 下，untied 参数结构可能显著缓解 shared K4 的弱表现。

### 5.5 与 FF 对照的关系

三点总体均值：

- K4 shared normal：86.9%
- Deep FF：86.4%
- Residual FF：86.1%
- K4 untied normal：85.1%

因此 K4 shared normal 相比两个 FF 对照只高约：

- 相比 Deep FF：+0.4 个百分点
- 相比 Residual FF：+0.8 个百分点

这表示 K4 shared normal 的绝对表现较强，但在这三个 checkpoint 的均值上，并没有形成对 FF 对照的巨大优势。尤其在 1.0M：

- K4 shared normal：83.0%
- Deep FF：89.0%
- Residual FF：86.7%

K4 shared normal 在最后 checkpoint 的优势并不稳定，主要受到 seed 0 从 0.9M 的 0.91 降到 1.0M 的 0.70 的影响。

### 5.6 稳定性与 seed variation

基于每个 seed 的三点均值，粗略观察：

- K4 shared normal：83.0%、88.3%、89.3%，整体较稳定，但 seed 0 的 1.0M 有明显回落。
- K4 untied normal：84.3%、84.7%、86.3%，非常稳定。
- K4 untied zero：88.7%、89.3%、92.3%，不仅均值最高，seed 间也较稳定。
- K1 shared normal：61.0%、44.3%、75.7%，seed variation 最大，整体表现最弱。
- Residual FF：85.7%、87.0%、85.7%，稳定性很好。

由于每个 condition 只有 3 个 seed，以上稳定性判断只能作为初步描述，不应视为正式显著性检验。

## 6. 初步结论

在只使用 0.8M、0.9M、1.0M success_rate 的统计口径下：

1. M12B 的缺失 K4 shared normal 补跑结果良好，三点均值为 86.9%。
2. K4 shared normal 明显优于 K4 shared zero，支持“zero 是 K4 shared 弱点”的判断。
3. K1 与 K4 的 initialization effect 方向相反，存在显著的 K × initialization interaction。
4. K4 untied zero 是当前八个 condition 中最高的，三点均值为 90.1%。
5. K4 shared normal 与 Deep FF、Residual FF 的总体均值接近，当前不能仅凭这三个 checkpoint 宣称 K4 shared normal 显著优于 FF。
6. M12B 的主要规律更像是“结构 × 参数共享 × 初始化”的交互，而不是单一的 K 或单一的 initialization 主效应。

这些结论是基于三个 seed、每个 seed 三个 checkpoint 的描述性统计；还没有进行置信区间、配对检验或多重比较校正。

## 7. 数据来源

现有 M12B condition：

- M12B-C001：K1 shared normal
- M12B-C002：K1 shared zero
- M12B-C003：K4 shared zero
- M12B-C004：K4 untied normal
- M12B-C005：K4 untied zero
- M12B-C006：Deep FF
- M12B-C007：Residual FF

补跑 condition：

- M12B-R / M12A-C003 attempt 2：K4 shared normal

原始指标均来自各 run 的 eval.csv 中 evaluation/overall_success 列。
