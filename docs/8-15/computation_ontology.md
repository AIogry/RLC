# RLC Computation Ontology

日期：2026-08-15

## Building hierarchy

RLC computation 使用以下层级：

```text
Operator
    ↓
Primitive
    ↓
Block
```

### Operator

Operator 是底层算子，例如 Dense、GELU、normalization、transpose 和
residual/add。Operator 目前只是实现层概念，不作为主要 scientific
configuration axis。M8 不为每个 operator 创建独立 Python abstraction。

### Primitive

Primitive 完成一个局部 computation function 的基本单元，例如 `MLP` 和未来
的 `SwiGLU-FFN`。Primitive 可以通过 topology 被调用，但不负责 agent loss、
环境交互或 algorithm-specific readout。

### Block

Block 是由一个或多个 primitives 以及 residual/wiring 组成的复合计算模块，
例如 `MLPMixerBlock`：token mixing、token weighting、token residual、channel
mixing、channel residual。当前 vanilla CoGHP 的
`networks.coghp.MixerBlock` 是 frozen reference。

M8 新增 computation-side `MLPMixerBlock` 作为 standalone parity candidate；
CoGHP production implementation 暂不切换到该 block。

## Computation description dimensions

computation system 的完整描述至少包含以下五个维度：

```text
State Structure
Topology
Execution Schedule
Parameter Reuse
Credit Structure
```

### State Structure

描述一次 computation 内维护哪些 computational states，以及 state 的 shape、
初始化和更新含义。

当前 RLC 的 internal computational state 只存在于单次 environment decision 内，
不跨越：

```text
environment step t -> t+1
```

当前 FeedForward path 的 state 必须为 `None`。M8 不引入 recurrent-policy
memory、POMDP memory 或跨环境 step state。

### Topology

描述 states 与 computation units 之间允许怎样的 dependency。当前 executable
topology 是 `FeedForward`：它调用一个 primitive 一次并返回 representation。
未来的 stateful/iterative topology 可以由 topology 自己定义 state semantics；
generic `ComputationCore` 不再预先拒绝 state。

### Execution Schedule

描述何时更新哪个 state/unit、执行多少次、以及执行顺序。M8 不实现 schedule，
也不向 `ComputationSpec` 添加 `cycles`、`schedule` 或其他尚无 executable
implementation 的字段。CoGHP 的 subgoal autoregression 属于
algorithm/network semantics，不被重新编码为 generic computation schedule。

### Parameter Reuse

描述不同 execution events 是否共享同一组 trainable parameters。参数统计使用：

- unique trainable parameter count；
- per-slot parameter count；
- per-core parameter count。

M8 不为尚未存在的 iterative execution 构造虚假的 reuse 数值。

### Credit Structure

描述 gradient 可以跨哪些 update、state 或 execution event 传播。当前唯一
executable credit policy 是 `direct`，表示普通 reverse-mode differentiation。
`full_bptt`、truncated credit 和 one-step credit 仍是保留的 scientific axis，
但在没有 stateful execution 时没有真实可区分的 computation graph，因此不在
M8 实现。

## Executable M8 mapping

```text
ComputationSpec(primitive, topology, credit)
    -> resolve_slot_spec
    -> make_computation_core
    -> ComputationCore
        -> FeedForward
            -> MLP
        -> ComputationOutput
```

`ComputationCore` 接受 `x` 和可选 `state`，将 state semantics 委托给 topology，
并负责把 plain output 归一化为 `ComputationOutput`。`FeedForward` 明确拒绝
non-`None` state；未来 topology 可以在自己的接口中实现 state 初始化、使用和
更新。

## Algorithm/computation boundary

```text
Agent
    = learning objective / loss / update / target update / policy orchestration

Network
    = task- and algorithm-specific input/output semantics and readout

Computation
    = internal representation transformation

Compute Slot
    = algorithm/network 中允许替换 computation 的位置
```

因此 HIQL/CRL 的 computation slot 只替换 representation body；CRL bilinear
interaction、contrastive objective、AWR weighting 和 readout 仍属于
algorithm/network。CoGHP 的 subgoal autoregression、teacher forcing、
subgoal-chain construction、high/low heads 和 sharing 仍属于 CoGHP semantics。
CoGHP autoregressive loop 不是 generic computation topology。

## M8 boundary and M9 deferral

M8 只冻结 ontology、放宽 generic interface、建立 Block-level parity、清理
vanilla defaults 和记录 runtime provenance。

明确 deferred 到 M9：single-state iterative computation、stateful topology、
execution schedule/cycles、parameter reuse across iterative events、non-direct
credit policy、recurrent policy memory、HRM、SwiGLU/RMS 或其他新 scientific
model。
