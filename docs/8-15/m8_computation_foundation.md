# M8 Computation Foundation

日期：2026-08-15

## 目标与结论

M8 不实现新的 scientific model。目标是在不改变 HIQL、CRL 和 vanilla CoGHP
已验证 semantics 的前提下，使 `impls/computation/` 不再隐含
feed-forward-only 的 generic core 假设，并形成可安全承载 M9 single-state
iterative computation 的 Primitive/Block/Topology 边界。

M8 foundation gate 已通过：

- `ComputationCore` 接受并转发 optional state；
- `FeedForward` topology 明确拒绝 non-`None` state；
- MLP sequence-shaped interface `[B,T,D]` 保持可用；
- computation-side `MLPMixerBlock` 与 vanilla `MixerBlock` 参数、forward、
  gradient、parameter count exact parity；
- CRL vanilla default slots 全部 disabled；
- runtime metadata 保存 resolved computation slot snapshot；
- networks package 重复 export cleanup 完成；
- 新增 foundation tests `5/5 PASS`。

## 代码变更

### Computation interface

`impls/computation/interfaces.py` 中 `ComputationCore` 不再自行判断 state
是否允许，而是调用 `topology(x, state=state)` 并统一包装 output。

`impls/computation/topologies/feedforward.py` 明确实现：

```text
state is None -> primitive(x) once
state is not None -> ValueError from FeedForward topology
```

这避免将 FeedForward-only 语义写死在 generic core，同时保持当前
`state=None` path 的 Flax parameter scope 不变。

### Block layer

新增：

```text
impls/computation/blocks/__init__.py
impls/computation/blocks/mlp_mixer.py
```

`MLPMixerBlock` reference-faithfully 表达当前 vanilla CoGHP MixerBlock 的：

```text
token mixing
-> causal tm_weights mixing
-> token residual
-> channel mixing
-> channel residual
```

它目前只用于 standalone parity。`impls/networks/coghp.py::MixerBlock` 仍然是
vanilla CoGHP frozen reference，CoGHP production import 没有切换。

### Baseline defaults

CRL `get_config()` 中以下 slots 默认统一为 `enabled=False`：

```text
actor
critic_state
critic_goal
value_state
value_goal
```

`--computation` 仍然是显式 migration/smoke shortcut；AWR value slots 仍只在
`actor_loss='awr'` 下实际实例化。

### Runtime provenance

保留旧的兼容字段 `{"computation": true}`，并新增稳定 JSON-serializable 的：

```json
{
  "compute_slots": {
    "actor": {
      "enabled": true,
      "primitive": "mlp",
      "topology": "feedforward",
      "credit": "direct"
    }
  }
}
```

该 snapshot 写入 runtime metadata 和 checkpoint metadata，使结果仅通过
metadata 即可判断每个 algorithm slot 的 resolved computation。

## Validation

M8 新增 foundation tests：

```text
tests/computation/test_foundation.py
tests/integration/test_computation_provenance.py
```

结果：

```text
5/5 PASS
```

测试覆盖：

- `ComputationCore(x, state=None)`；
- FeedForward non-`None` state contract；
- `[B,T,D]` representation path；
- 多组 batch/token/embed/hidden 配置的 Mixer parameter parity；
- Mixer forward parity；
- Mixer gradient parity；
- parameter count parity；
- CRL vanilla default slot cleanup；
- resolved runtime compute snapshot。

真实 shared runtime provenance smoke 也通过：`runtime_metadata.json` 和
checkpoint metadata 均成功写入 JSON-serializable `compute_slots` snapshot。

随后运行已有完整 HIQL/CRL/CoGHP regression；M8 不要求重新运行 GPU
1000-step experiment，因为没有改变 production baseline forward 或
parameter tree。完整 CPU regression 最终为 `46/46 PASS`。

## Deferred

M8 不实现：single-state iteration、RNN/multi-state/HRM、SwiGLU/RMS
experiments、new credit rules、generic CoGHP MixerBlock production switch、
formal long training 或 scalar compute score `C=aP+bF+cD+dR`。
