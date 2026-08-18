现在我们希望能够设计一个研究计算模块在GCRL各类算法上效果的项目代码框架。尤其需要研究HRM的计算与MLP、MLP-mixer的计算方式在actor、critic等位置上的效果。

目前我们已经在RLC文件夹下大致搭建了一个初步的项目框架。我们已经将原来CoGHP下经过修复seed不可复现的ogbench复制到RLC下了。

下一步我希望你在仔细阅读/offline_rl_baselines/ogbench项目之后把其中实现的一些GCRL方法首先搬运到RLC下，但这只是一个初步的要求。由于我们需要研究各类计算模块在actor、critic上的效果，所以我们必须把它们在actor中使用的MLP与算法方法解耦开，我大致观察了一下其中crl.py、hiql.py等大多使用的是utils/networks.py下的MLP，因此，如果我们后续需要做研究，必须要让它们这些baseline算法能够调用MLP-mixer、HRM等。这在工程上如何实现。
# RLC architecture

RLC separates four concerns:

- **Agent** = learning algorithm: losses, targets, advantage weighting, goal sampling semantics.
- **Network** = task-specific input/output semantics: encoders, actor/value interfaces, distribution heads and readouts.
- **Computation** = representation transformation: primitives and topologies.
- **Compute Slot** = a named location where a computation body can be replaced.

The canonical OGBench environment/runtime is under `RLC/ogbench`. It is kept separate from the algorithm reference under `offline_rl_baselines/ogbench/impls` and from computation experiments.

The first migrated slot is `HIQL.low_actor`. Its baseline path remains the original OGBench MLP; an explicit slot configuration can route only the actor body through `MLP + FeedForward + Direct` while preserving the encoder, action distribution, and HIQL loss.

The shared slot configuration resolver is `impls.computation.factory.resolve_slot_spec`; it converts `compute.<slot_name>` configuration into an optional `ComputationSpec`. Agent modules use it but do not own its configuration semantics.

See [architecture_audit.md](architecture_audit.md), [baseline_compute_slots.md](baseline_compute_slots.md), and [computation_migration_plan.md](computation_migration_plan.md).
