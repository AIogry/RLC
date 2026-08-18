# M8 Computation Foundation Audit

日期：2026-08-15

## 审计范围

本地工作树是本 milestone 的 source of truth。当前已验证的 baseline 为 HIQL、
CRL 和 vanilla CoGHP；本审计不从远端删除本地 docs/tests，也不引入新的
scientific model。

检查范围：

- `impls/computation/`
- `impls/networks/`
- `impls/agents/`
- `impls/main.py`
- `configs/`
- `tests/`
- `docs/`

## 当前 computation call graph

```text
Agent loss/update
  -> algorithm-specific Network module
      -> optional Compute Slot
          -> ComputationCore
              -> FeedForward topology
                  -> MLP primitive
              -> ComputationOutput(representation, state, auxiliary)
      -> task-specific readout / bilinear interaction / distribution
```

### ComputationCore

`ComputationCore` 位于 `impls/computation/interfaces.py`，负责持有 topology、
调用 topology，并将 plain representation 统一包装成 `ComputationOutput`。
M8 前它在 generic core 中直接拒绝非 `None` state；M8 将这一职责下放给
topology，使 core 能够接受并转发可选 state。

### ComputationOutput

`ComputationOutput` 是 `(representation, state, auxiliary)` 的公共结果接口。
当前 `FeedForward` 返回 `state=None`，未来 stateful topology 可以返回更新后
state，而不需要改变 generic core 的结果归一化接口。

### ComputationSpec 与 factory

`ComputationSpec(primitive, topology, credit)` 只描述当前可执行的 computation
轴。`resolve_slot_spec` 负责解析 `compute.<slot>`；`make_computation_core`
负责构造 `ComputationCore(topology=FeedForward(primitive=MLP(...)))`。
M8 不向 `ComputationSpec` 添加尚无 executable implementation 的 state、schedule、
reuse 或 cycles 字段。

### Primitive / topology / credit

- `MLP`：`impls/computation/primitives/mlp.py`，保留 OGBench MLP 的 Dense、
  activation、LayerNorm 顺序和初始化；
- `FeedForward`：`impls/computation/topologies/feedforward.py`，调用 primitive
  一次并明确拒绝非 `None` recurrent state；
- `Direct`：当前唯一 executable credit policy，只表示普通 reverse-mode
  differentiation；
- `accounting.py`：统计 unique trainable parameter、per-slot parameter 和
  独立 core parameter。

### Baseline slots

```text
HIQL:
  low_actor, high_actor, value

CRL:
  actor, critic_state, critic_goal
  value_state, value_goal  (仅 actor_loss='awr')

CoGHP:
  vanilla official actor_mixer，不使用 computation slot
```

HIQL/CRL 的 computation body 只替换内部 representation transformation；
agent loss、network readout、bilinear interaction、target update 和 policy
orchestration 仍属于 baseline algorithm/network semantics。

### CoGHP MixerBlock

`impls/networks/coghp.py::MixerBlock` 是 vanilla CoGHP 的 frozen reference。
它不是当前 generic computation topology；其 autoregressive subgoal chain、
teacher forcing、high/low heads 和 shared Mixer parameters 继续属于 CoGHP
algorithm/network semantics。M8 只建立 standalone computation-side block
parity，不切换 CoGHP production import。

## 当前 package audit

`impls/networks/__init__.py` 存在重复 import 和重复 `__all__` 覆盖；M8 将其
合并为一个明确的 task-network export 列表。`actors.py`、`values.py`、
`encoders.py`、`implicit_hrm.py` 等文件没有被当前 baseline import；本轮不做
大规模目录移动，placeholder 保留并记录为后续清理范围。`implicit_hrm.py`
也不会在 M8 被实现或接入。

## M8 audit conclusion

当前 ontology 与 executable code 的最小对应关系为：

```text
Operator  -> Dense/GELU/normalization/transpose/add 等底层算子
Primitive -> MLP，以及 M8 新增的 computation-side MLPMixerBlock candidate
Block     -> 由 token/channel mixing 与 residual wiring 组成的复合模块
Topology  -> 当前 FeedForward；未来才实现 stateful/iterative topology
Credit    -> 当前 Direct
```

M8 的安全边界是：扩展 generic interface、冻结 ontology、建立 Mixer block
parity、修复默认配置和 runtime provenance；不实现 single-state iteration、
RNN、HRM、SwiGLU/RMS、新 credit policy 或长训练。
