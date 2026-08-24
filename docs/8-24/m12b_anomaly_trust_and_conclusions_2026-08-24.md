# M12B 异常现象、结果可信度与阶段性结论

日期：2026-08-24  
项目：RLC 研究平台  
范围：M12A/M12B，`antmaze-large-navigate-v0`，CRL frozen-critic policy extraction  
分析对象：M12B 已产生的正式 run artifacts；不启动新训练，不修改 run 状态，不执行 Git 操作

## 1. 结论先行

M12B 不是“整体实验失效”，当前最明显的异常可以被一个相当清晰的结构交互解释：

> **K=4、shared 参数、zero state initialization 的组合表现明显弱于 K=4、shared 参数、normal initialization；但它仍然优于 FF baseline。**

因此，“M12B 的 SS K4 相比 M12A 严重下降”这个现象本身是真实的，但不能直接解释为“SS K4 在 M12B 中失效”。原因是 M12A 的 K4 结果和 M12B 当前 active 的 K4 结果并不是同一个 condition：

- M12A K4：`shared + normal_buffer`，对应 M12B 的 external anchor B004；
- M12B 当前 active K4：`shared + zero_buffer`，对应 B005。

在正确的条件对应下，当前最合理的表述是：

1. **shared K4 对 state initialization 很敏感；**
2. **zero initialization 的负面作用只在 K=4、shared 组合中明显，并不是 zero initialization 的普遍作用；**
3. **untied K4、9-layer plain FF 和 4×2 residual FF 都恢复到较高水平，说明 M12B 架构实现没有出现全局性失效；**
4. **M12B 的已完成结果可以用于阶段性科学判断，但不能把整个 21-run active matrix 宣布为完整最终结果，因为 residual seed2 尚未完成。**

## 2. 实验条件与比较口径

### 2.1 M12B 的 9 个 conceptual conditions

| 条件 | 实际来源 | topology/block | K | state init | parameter sharing | 说明 |
|---|---|---|---:|---|---|---|
| B001 | M12A-C002 attempt 1 | plain FF | 3 hidden Dense | — | — | FF baseline external anchor |
| B002 | M12B-C001 | SingleState + plain | 1 | normal | shared | active |
| B003 | M12B-C002 | SingleState + plain | 1 | zero | shared | active |
| B004 | M12A-C003 attempt 1 | SingleState + plain | 4 | normal | shared | M12A K4 external anchor |
| B005 | M12B-C003 | SingleState + plain | 4 | zero | shared | 用户所说的 M12B SS K4 |
| B006 | M12B-C004 | SingleState + plain | 4 | normal | untied | active |
| B007 | M12B-C005 | SingleState + plain | 4 | zero | untied | active |
| B008 | M12B-C006 | plain FF | 9 hidden Dense | — | — | active |
| B009 | M12B-C007 | residual FF | 4 blocks × 2 Dense | — | — | active，seed2 未完成 |

### 2.2 为什么 B004 与 B005 的比较不能叫作“同配置复现”

M12A 的 K4 结果是 B004，即 `K4 + shared + normal`。M12B 的正式 active 配置中没有重新跑一个 `K4 + shared + normal`，而是将 M12A 的 B004 作为 external anchor，并新增 `K4 + shared + zero` 的 B005。

所以：

- `B005 - B004` 是一个 state initialization contrast；
- 它不是 M12A K4 与 M12B K4 的无条件版本复现差异；
- 如果希望排除跨 study anchor 和 commit 差异，仍应在当前 M12B worktree 下直接复跑 B004。

这一点是解释当前异常的核心。

## 3. 结果矩阵

以下均为 `evaluation/overall_success`。AUC 为 100k 到 1M 的梯形积分后归一化结果；`±` 为 seed 间样本标准差。除 B009 外，active condition 均为 3 个完整 seed。

| 条件 | last@1M | best | AUC | n | 状态 |
|---|---:|---:|---:|---:|---|
| B001 / M12A FF | 0.670 ± 0.125 | 0.740 ± 0.087 | 0.586 ± 0.107 | 3 | external anchor |
| B002 / K1 shared normal | 0.663 ± 0.156 | 0.673 ± 0.158 | 0.539 ± 0.112 | 3 | completed |
| B003 / K1 shared zero | 0.787 ± 0.097 | 0.797 ± 0.101 | 0.621 ± 0.106 | 3 | completed |
| B004 / M12A K4 shared normal | 0.863 ± 0.070 | 0.883 ± 0.042 | 0.793 ± 0.042 | 3 | external anchor |
| B005 / K4 shared zero | 0.737 ± 0.057 | 0.773 ± 0.055 | 0.674 ± 0.080 | 3 | completed |
| B006 / K4 untied normal | 0.867 ± 0.012 | 0.897 ± 0.021 | 0.855 ± 0.022 | 3 | completed |
| B007 / K4 untied zero | 0.900 ± 0.030 | 0.930 ± 0.020 | 0.859 ± 0.031 | 3 | completed |
| B008 / 9-layer plain FF | 0.890 ± 0.060 | 0.937 ± 0.015 | 0.850 ± 0.023 | 3 | completed |
| B009 / residual FF | 0.850 ± 0.028 | 0.895 ± 0.021 | 0.854 ± 0.003 | 2 | seed2 incomplete，不能作为 n=3 最终均值 |

### 3.1 关键 paired contrasts

这些差值按相同整数 seed 对齐，适合做方向判断；但 n=3 很小，不能据此宣称统计显著性。

| 对比 | last@1M 差值 | AUC 差值 | 解读 |
|---|---:|---:|---|
| B005 − B004：K4 shared zero vs normal | −0.127 | −0.119 | 这就是“SS K4 下降”的主要来源 |
| B005 − B001：K4 shared zero vs FF baseline | +0.067 | +0.088 | B005 没有低于 baseline，而是只获得了有限增益 |
| B003 − B002：K1 shared zero vs normal | +0.123 | +0.082 | zero 在 K1 反而有利 |
| B006 − B004：K4 untied normal vs M12A K4 anchor | +0.003 | +0.062 | untied normal 与 M12A K4 整体相当，AUC 更高 |
| B007 − B006：K4 untied zero vs normal | +0.033 | +0.004 | untied 下 zero/normal 差异很小，zero 略高 |
| B007 − B005：K4 untied zero vs shared zero | +0.163 | +0.185 | 去掉 parameter tying 后性能大幅恢复 |
| B008 − B007：9-layer plain FF vs K4 untied zero | −0.010 | −0.009 | 两者基本同档 |
| B009 − B008：residual FF vs 9-layer plain FF | −0.040 | +0.004 | 目前只能说接近；B009 只有 2 个完整 seed |

## 4. 逐 task 结果：下降集中在哪里

下表是 last@1M 各 task 的 seed 均值。B009 仅使用两个完整 seed，故只作参考。

| 条件 | task1 | task2 | task3 | task4 | task5 |
|---|---:|---:|---:|---:|---:|
| B001 / M12A FF | 0.817 | 0.667 | 0.850 | 0.583 | 0.433 |
| B002 / K1 shared normal | 0.850 | 0.617 | 0.900 | 0.533 | 0.417 |
| B003 / K1 shared zero | 0.950 | 0.617 | 0.867 | 0.833 | 0.667 |
| B004 / M12A K4 shared normal | 0.867 | 0.850 | 0.867 | 0.917 | 0.817 |
| B005 / K4 shared zero | 0.883 | 0.717 | 0.867 | 0.700 | 0.517 |
| B006 / K4 untied normal | 0.883 | 0.850 | 0.867 | 0.850 | 0.883 |
| B007 / K4 untied zero | 0.900 | 0.850 | 0.967 | 0.917 | 0.867 |
| B008 / 9-layer plain FF | 0.883 | 0.900 | 0.883 | 0.933 | 0.850 |
| B009 / residual FF | 0.925 | 0.800 | 0.825 | 0.850 | 0.850 |

B005 相对 B004 的损失主要集中在：

- task4：−0.217；
- task5：−0.300；
- task2：−0.133；
- task1、task3：基本没有稳定下降。

这不是五个 task 等比例整体缩放，也不像所有输出都被同一个数值错误污染，更像是某些较难或对策略结构更敏感的目标被 shared-zero 配置选择性影响。由于当前没有 task 语义映射，不能把 task4/5 进一步解释为具体几何因素。

另一个很有信息量的现象是，B007 相对 B005 几乎完整恢复了 task4/5：

- task4：+0.217；
- task5：+0.350。

这进一步支持“shared K4 zero 的结构性限制”这一判断，而不是单纯的随机 seed 波动。

## 5. 运行与 provenance 审计

### 5.1 已完成运行的完整性

M12B active 运行共计划 21 个：7 个 active condition × 3 seeds。当前状态是：

- 20/21 个 active run 有 `status: completed`；
- 20/21 个完整 run 都有 10 个 eval 点，覆盖 100k…1M；
- 20/21 个完整 run 都有 `last/params_1000000.pkl`；
- 20/21 个完整 run 都有 `summary.json`；
- 所有运行的 `accounting_consistency.status` 都是 `pass`；
- 完整 train CSV 未发现 NaN 或非有限值。

### 5.2 residual seed2 的明确限制

`M12B-C007 / B009 / seed_002` 的 artifact 仍然显示：

- `runtime_metadata.status = running`；
- eval 只到 800k；
- 没有 `last@1M`；
- 没有 `summary.json`；
- 当前没有对应活动进程。

它已有的 800k overall success 是 0.85，但这不是 last@1M，不能和其他条件的最终值混用。B009 的当前均值只能报告为 n=2 的 provisional summary，不能报告成三 seed 最终结果，也不能对 B008/B009 做正式显著性比较。

### 5.3 代码、critic 与数据 provenance

完整 M12B run 的 metadata 一致显示：

- `git_commit = bb2644ccb23ee77a0c08e8b9cded85a57716df67`；
- `git_dirty = false`；
- 运行代码 worktree 为 `/home/eai/Research/RLC-M12B-final`；
- environment 为 `antmaze-large-navigate-v0`；
- frozen critic 为同 seed 的 M12A-C001 `last@1M` checkpoint；
- B002…B009 使用的 frozen critic checkpoint SHA 与 module fingerprint 按 seed 一致；
- batch size、eval protocol、eval interval、train steps 均为统一的 `1024 / 20 episodes / 100k / 1M`。

M12A B001/B004 anchor 来自前一轮 clean commit `b3fde3f91d89169c02c7604ace80d65bdf8ced25`，M12B active run 来自上述 clean commit。此前 preflight 已验证旧 M12A shared K4 checkpoint 在新架构下 restore 成功，且 shared 参数树仍使用 `update_module`，没有发生 checkpoint key migration。

因此，数据层面的可信度是高的；但 B004 仍属于跨 study、跨 commit 的 external anchor，而不是当前 M12B commit 下重新训练的同条件结果。这是科学解释上的限制，不是 artifact 损坏证据。

## 6. 是否存在训练崩溃或实现级异常

目前没有看到支持“实现崩溃”的证据。

### 6.1 没有数值崩溃

B005 的 train CSV 没有 NaN/Inf；actor loss、BC loss、Q loss、gradient norm 都有正常有限值。它的 best checkpoint 在 600k–900k 之间，说明训练和保存生命周期正常工作。

### 6.2 critic 没有发生 condition-specific 污染

M12B 使用 frozen critic，所有 active condition 在相同 seed 下的 validation critic 指标一致。这是预期现象，也说明 B005 的下降不是某一个 condition 意外加载了不同 critic。

### 6.3 训练诊断没有呈现爆炸

以三 seed 均值为例：

- B005 的训练 gradient norm 最后约 57.7，global max 约 76.4；
- B006/B007 的最后 gradient norm 约 80.2/82.0；
- B005 的 `training/frozen/q_delta` 最后约 0.149；
- M12A B004 的同类 q_delta 均值约 0.139；
- B005 与 M12A B004 的 actor Q loss、frozen Q delta 量级接近。

换言之，B005 的性能下降并没有伴随明显的 loss divergence 或 gradient explosion。训练损失相近而行为成功率不同，反而说明问题更可能发生在 learned policy 的表示/决策几何或 state initialization interaction，而不是优化器直接崩溃。

### 6.4 曲线显示从早期开始偏弱，不是 1M 末期突然坍塌

overall success 的三 seed 均值大致为：

- B004：100k 0.567，200k 0.740，1M 0.863；
- B005：100k 0.447，200k 0.607，1M 0.737；
- B006：100k 0.850，1M 0.867；
- B007：100k 0.810，1M 0.900。

B005 从早期就低于 B004，之后基本维持差距，没有出现只在最后阶段发生的灾难性坍塌。因此把它称为“训练完成后的反常低性能”可以，但称为“训练后期崩溃”并不准确。

## 7. 最可能的机制解释

当前证据支持的不是单一因素，而是一个三因素交互：

```text
state initialization × repeated iterations (K=4) × parameter sharing
```

逐项看：

1. K1 shared：zero 比 normal 高约 0.123 final，说明 zero 本身不是坏初始化；
2. K4 shared：zero 比 normal 低约 0.127 final，说明增加迭代后，zero 的影响方向反转；
3. K4 untied：zero 比 normal 高约 0.033 final，说明 untied 后 zero 的负面作用基本消失；
4. K4 shared zero → K4 untied zero：final 提升约 0.163，AUC 提升约 0.185，说明 shared repeated update 是关键嫌疑对象之一。

一种合理但尚未被单独证明的解释是：zero state 下，shared update module 在 K 次执行中反复使用同一变换；随着 K 增加，网络需要在一个由相同参数产生的迭代轨迹上同时完成状态推进和策略输出。该轨迹可能落入不利的表示区域，尤其影响 task4/5。untied K4 允许每一个 execution 使用不同的 update 参数，因而显著放宽了这条轨迹的表达能力。

但必须保留一个重要混杂因素：

- shared K4 的 actor body/core trainable params 约 555,520；
- untied K4 的 core trainable params 约 2,131,456；
- 二者执行 Dense 层数量和 MAC 相同，但 unique parameters 不同。

所以现有结果支持“shared K4 zero 受限、untied 可以恢复”，但不能严格把全部增益归因于 parameter sharing 本身；容量差异和参数 tying 是同时变化的。

## 8. 目前可以成立的科学结论

### 可以成立

- M12B 的 completed artifacts 没有显示全局实现错误或数值训练崩溃。
- B005 的退化是可重复的三 seed 现象，不是单个 seed outlier。
- B005 相比 M12A K4 anchor 的下降主要来自 `zero_buffer`，因为比较对象不是同一个初始化条件。
- zero initialization 对 shared topology 的作用依赖 K：K1 有利，K4 不利。
- K4 shared zero 仍高于 FF baseline，但没有复制 M12A K4 shared normal 的性能增益。
- untied K4 和 9-layer plain FF 达到相近的高性能，说明较高的执行深度/容量能够恢复性能；但这不能单独证明 recurrent computation 优于普通深层 FF。
- B009 residual 的两个完整 seed 已显示高性能潜力，但尚不足以形成三 seed 最终结论。

### 目前不能成立

- 不能说“SS K4 在 M12B 中整体失效”；B006/B007 与 B004 都反驳这一说法。
- 不能说“zero initialization 普遍有害”；B003 明显优于 B002。
- 不能说 parameter sharing 单独导致退化；shared/untied 同时改变了 unique parameter count。
- 不能把 B009 报告为三 seed final mean。
- 不能基于 n=3 小样本宣称显著性或普遍化到其他 task/environment。
- 不能把 B004 external anchor 当作当前 M12B commit 下的完全独立复现结果。

## 9. 建议的补充工作（按优先级）

### P0：先补齐结果矩阵

1. 由用户决定如何处理 B009 seed2：继续完成到 1M，或按规范重新运行；
2. 在它完成前，所有聚合表将 B009 标记为 `n=2 provisional`，不纳入完整三 seed ranking。

### P1：确认当前 commit 下的同条件 anchor

在当前 M12B worktree 下直接运行 B004，即 `K4 + shared + normal_buffer`，优先使用与其他条件完全相同的 protocol 和 seed。目的不是扩大 sweep，而是确认：

- M12A B004 external anchor 在当前代码下是否保持；
- B005 的下降是否仍然由 normal→zero 初始化对比产生；
- 跨 commit anchor 差异是否可忽略。

### P1：对已有 checkpoint 做更低噪声 evaluation

对 B004/B005/B006/B007 的 `last` 与 `best` checkpoint 做 100 或 200 episodes 的重新评估，至少保留 task-level success。现有 20 episodes 的结果足以支持目前的大方向，因为 B005 与 B004 的 task4/5 差距很大，但更高 episode 数可以减少 0.05 分辨率带来的噪声。

### P2：增加 state-trajectory diagnostics

针对 K4 shared normal/zero 和 K4 untied normal/zero，记录每个 execution 的：

- state norm；
- update norm；
- input injection norm；
- activation norm；
- final action distribution；
- 每个 task 的 Q-policy 与 Q-data gap。

这可以把“zero shared K4 表示轨迹进入不利区域”的机制假设变成可检验的诊断，而不是只依据最终成功率推断。

### P2：做 parameter-matched sharing control

如果研究目标是严格回答“parameter sharing 是否导致退化”，需要控制参数量：

- 缩小 untied K4 的每层宽度，使总 trainable params 接近 shared K4；或
- 增大 shared K4 的表示宽度，使其与 untied K4 参数量相当；
- 同时保持 K、state init、input injection 和 training protocol 不变。

当前 B005→B007 的恢复很强，但不能把它直接写成“只由 untied 参数共享策略造成”。

## 10. 最终判断

对用户提出的三个问题，当前答案是：

1. **SS K4 相比 M12A 是否严重下降？**  
   是，但准确说是 `K4 shared zero` 相比 `K4 shared normal` 下降；不是所有 SS K4 都下降，也不是 M12B 代码整体失效。

2. **这样的情况下 M12B 结果还可信吗？**  
   已完成条件的 artifact 和方向性结论可信度高；B005 的异常可信且具有三 seed 一致性。完整矩阵的最终可信度暂为“有一个明确缺口”：B009 seed2 未完成，且 B004 是 external anchor。

3. **能否找到规律和总结性结论？**  
   可以：当前最强规律是 `state initialization × K × parameter sharing` 的交互；shared K4 对 zero initialization 敏感，untied K4 和普通深层 FF 能恢复性能，退化主要集中在 task4/5。这个规律足以指导下一步诊断，但在补齐 B009、确认当前 commit 下 B004、以及做参数量控制前，不应写成最终普遍性定律。

