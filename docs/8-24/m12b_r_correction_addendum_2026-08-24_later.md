# M12B-R 实验解释勘误与重新定义

日期：2026-08-24  
依据：用户对 M12A/M12B 实验实际运行内容的事实更正

## 1. 更正后的事实

此前的解释混淆了“配置文件/历史结果存在”和“该条件已经在对应实验阶段正式运行”这两个概念。应以以下事实为准：

1. M12B 实验中没有运行 SS K4 normal。
2. M12A 实验中没有运行 SS K4 zero。
3. M12B 中看到的 SS K4 低结果对应的是 SS K4 zero，而不是 SS K4 normal。
4. M12A 的 SS K4 normal checkpoint 没有被用于 M12B。

因此，当前实际数据矩阵不是：

| 实验 | SS K4 normal | SS K4 zero |
|---|---:|---:|
| M12A | 有 | 有 |
| M12B | 有 | 有 |

而是：

| 实验 | SS K4 normal | SS K4 zero |
|---|---:|---:|
| M12A | 有 | 无 |
| M12B | 无 | 有 |

## 2. 对此前结论的影响

此前“SS K4 normal 在 M12B 中相比 M12A 效果严重下降”的说法应撤回。M12B 没有 normal 结果，因此不存在这个下降量，也不能据此讨论 M12B normal 是否退化。

同样，当前数据不能支持以下结论：

- M12B 中 normal 与 zero 的直接性能差异；
- zero initialization 相对 normal initialization 在 M12B protocol 下的因果效应；
- M12A normal checkpoint 迁移到 M12B 后是否会改善或恶化结果；
- M12B 阶段本身是否造成了 normal 分支性能下降。

目前唯一可以保留的谨慎陈述是：M12B 已有 SS K4 zero 结果在其实际运行 protocol 下表现较弱。这个现象本身可以作为待解释现象，但不能被表述为“zero 相比 M12B normal 更差”，也不能被表述为“normal 在 M12B 中退化”。

## 3. 对原 preflight 报告的修正

原报告中使用 M12A-C003 作为所谓 M12B normal reference，并规划：

- M12A-C003 normal attempt 2；
- M12B-C003 zero attempt 1；

这不能构成严格的 M12B 内部 paired comparison。尤其是，M12A-C003 是 M12A 阶段的运行/配置语境，不应直接被命名为“已完成的 M12B normal 结果”。

原报告中关于 paired initialization、critic fingerprint 和 dry-run 的实现级验证仍然可以作为“代码路径可实现该 paired design”的证据；但它们不能被误读为 M12B 已经获得了 normal-vs-zero 的正式实验结果。原报告中的相关设计部分由本勘误覆盖。

## 4. 正确的后续实验问题

需要先区分两个不同问题。

### 问题 A：M12B 内部的 initialization effect

如果要判断 M12B protocol 下 normal buffer 与 zero buffer 的差异，必须在同一个 M12B study/protocol 下补齐 normal 条件，并与 zero 条件进行 paired comparison：

- 同一 source commit；
- 同一 M12B critic dependency；
- 同一 task、算法、网络宽度、训练步数、batch、学习率和评估协议；
- 同一 seed 集合；
- actor 初始参数、数据流和前 10 个 batch 对齐；
- 唯一改变 state_init：normal_buffer vs zero_buffer；
- normal 和 zero 各自使用新的、未占用的 attempt 目录；
- 正式结果至少报告 last@1M、best、AUC 以及 seed-level 数据。

这个实验回答的是“在 M12B protocol 下，初始化方式是否影响训练”，不要求把 M12A normal actor checkpoint 当作 M12B 的起点。

### 问题 B：M12A normal checkpoint 的跨阶段迁移效应

如果科研问题是“把 M12A 的 SS K4 normal checkpoint 迁移到 M12B 是否有效”，那是另一个 transfer experiment，不能用问题 A 的 normal-vs-zero 对照替代。它需要明确：

- 迁移的是 actor checkpoint、critic checkpoint，还是二者；
- M12B 哪些模块继续训练、哪些模块冻结；
- checkpoint 的同 seed、step、SHA 和模块 fingerprint；
- 迁移前后参数、optimizer state、RNG 和数据流如何定义；
- 是否设置不迁移 checkpoint 的 matched control。

当前事实只能说明这个 transfer experiment 没有发生，不能从已有 M12B zero 结果推断其结果。

## 5. 当前科学结论

截至目前，关于 M12B 的结论应改写为：

> M12B 的 SS K4 zero 条件已经完成并表现较弱，但 M12B 没有 SS K4 normal 条件，且没有使用 M12A SS K4 normal checkpoint。因此，现有 M12B 数据不足以估计 M12B 内部的 normal-vs-zero 初始化效应，也不足以判断跨阶段 checkpoint transfer 是否有效。后续必须先补齐同 protocol 的 M12B normal 对照，或明确开展独立的 checkpoint transfer 实验。

在补齐对照之前，不应再使用“SS K4 normal 在 M12B 中显著下降”作为实验结论。

## 6. 当前执行状态

- 本勘误不改变“没有执行 Git 操作”的约束。
- 本勘误不启动正式训练。
- 本勘误不把 M12A normal 结果改写成 M12B normal 结果。
- 原 preflight 报告的最终 NO-GO 状态仍然有效；但 NO-GO 之外，实验解释本身已由本文件纠正。
