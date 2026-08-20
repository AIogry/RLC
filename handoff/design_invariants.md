# RLC Design Invariants

这些是接手后默认必须保持的设计不变量。改变任何一项都应当先形成 architecture decision，并说明它是否构成新的 scientific factor。

## 模块边界

```text
Agent       = learning algorithm, losses, targets, update semantics
Network     = input/output semantics, encoder, readout, goal semantics
Computation = representation transformation
Compute Slot = replaceable computation location
```

Computation 不应隐式改变 agent loss、readout、target update、dataset sampling 或 evaluation semantics。

## HIQL

- 官方 HIQL 是两个独立的 GCActor：high actor 和 low actor；
- high/low actor 不默认共享 core；
- 如果研究 shared network，必须作为明确的 architecture factor；
- high/low 的 distribution、goal encoding、action readout、loss 和 target semantics 应与 baseline 对齐；
- `value` 分支是否使用 computation 必须由配置显式声明；
- 不要把 readout 共享误认为 computation core 共享。

## CRL

- `compute.critic_state` 和 `compute.critic_goal` 是独立 representation branches；
- bilinear dot product 和 contrastive objective 仍属于 CRL-specific semantics；
- AWR value branches 的迁移范围必须与 milestone 和配置一致；
- 不要因为 critic 已 computationized 就自动改写 actor 或 value 的实验语义。

## Vanilla CoGHP

- 官方 Vanilla CoGHP 的 physical `actor_mixer` core 复用于 subgoal 和 final action；
- `high_actor_head` 和 `low_actor_head` 是独立 readouts；
- Vanilla CoGHP 默认不使用 RLC computation pipeline；
- 不能把 CoGHP 的 shared mixer 语义迁移成 HIQL shared-core 设计。

## 实验与 provenance

- Study、Configuration、Run、Reevaluation、Analysis 是不同层级；
- 任何正式 config 必须可由 YAML 重建；
- checkpoint 必须记录 step、hash、source run、Git commit 和 protocol；
- reevaluation source 只读；
- analysis 输出必须能从 source fingerprint 和 spec fingerprint 重建；
- full/zoom/split view 不能改变 scientific rows 或 statistics；
- training seed variability 与 episode sampling uncertainty 必须分离。

## 代码扩展规则

新增 primitive、topology 或 credit assignment 时，应优先通过：

- `impls/computation/interfaces.py`；
- `impls/computation/factory.py`；
- `impls/computation/accounting.py`；
- 对应 topology/primitive/credit 模块；
- configuration YAML 和针对性 tests。

不要在 agent 文件中复制一套 computation 实现，也不要在 plotting 代码中加入实验专用的数据推断。
