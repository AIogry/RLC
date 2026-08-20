# RLC Research Questions

本文件区分已完成的实验、当前可支持的观察和仍然开放的科学问题。另一个模型不得把“实现完成”直接改写成“科学问题已经解决”。

## 总体问题

RLC 研究的是：在保持 RL agent 的任务语义和 baseline 目标函数不变时，计算模块的表示变换、拓扑组织、credit assignment 和 placement 如何影响 goal-conditioned RL 的学习与最终性能。

核心因果链应保持为：

```text
scientific factor
    -> declared configuration
    -> training protocol
    -> checkpoint
    -> reevaluation protocol
    -> provenance-aware analysis
```

## M9A：SingleState iterative computation

主要问题：在固定实验语义下，SingleState 的迭代次数和 computation budget 是否改变性能？

当前状态：M9A study、配置矩阵、训练和结果分析已经形成。后续分析应关注：

- 计算次数与性能之间是否存在稳定趋势；
- 不同 budget 下收益是否递减；
- gain 是否来自优化更容易，还是来自最终表达能力变化；
- seed variability 是否足以改变 allocation 排序。

## M9B：TwoState/HRM-style topology

主要问题：TwoState hierarchical computation 是否提供独立于 SingleState 的有效归纳偏置？

当前最重要的限制：TwoState 的 success-rate 表现不明显，不能直接解释为方法无效。至少存在以下混淆：

- 500k training steps 可能不足以完成 TwoState 的优化；
- TwoState 可能需要不同的 warm-up 或 state initialization；
- 两个状态的 credit assignment 可能增加 optimization difficulty；
- 单一最终 checkpoint 无法显示是否只是收敛速度更慢。

因此当前严谨表述应为：

> M9B 当前结果不足以区分 TwoState 的方法效应与训练未充分效应。

建议 follow-up：

- 保存并分析训练过程曲线，而非只分析最终 checkpoint；
- 增加 1M/2M 或等价的 training-duration 条件；
- 保持相同 seeds、dataset、evaluation protocol 和 checkpoint accounting；
- 单独审计 state norm、high/low state update 和 loss trajectory；
- 不在没有 duration control 的情况下做 TwoState superiority/inferiority claim。

## M10A：Fixed-budget HIQL placement

正式问题：

> 在固定总 computation budget 下，将计算分配给 HIQL high actor 或 low actor 是否改变 GCRL performance？随着总预算增加，偏好的 allocation 是否变化？

正式设计：

- environment：`antmaze-large-navigate-v0`；
- training steps：500k；
- training seeds：0、1、2；
- configs：11 个；
- primary metric：overall success；
- reevaluation：M10A-R001，每 task 100 episodes，共同 evaluation seed；
- reference：M10A-C002，within-parameterization reference；
- C001：external vanilla baseline，allocation factors 为空，不应当被误写为某个 `K_H/K_L` allocation。

当前分析可以支持：

- allocation response 的描述性比较；
- task-level response；
- focal task 与 remaining tasks 的分组观察；
- paired episode outcome 统计；
- seed-level mean、population SD、sample SD 的审计。

当前分析不能单独支持：

- 普适的最优 allocation 结论；
- 统计显著性结论；
- 对所有环境或所有训练时长的外推；
- 将某一 task 的难度标签作为已证实的因果解释。

## 长期研究方向

- 计算 budget 与 wall-clock/FLOP/MAC 的关系是否一致；
- topology、primitive、credit assignment 与 placement 是否存在交互；
- high/low actor 的独立 network 设计是否影响优化稳定性；
- computation benefit 是来自更强表达能力、迭代深度、还是优化路径；
- training duration 是否是 TwoState 与 SingleState 比较中的关键 moderator；
- 如何在不破坏 baseline fidelity 的前提下研究 adaptive computation。
