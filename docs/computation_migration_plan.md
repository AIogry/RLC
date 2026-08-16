# Computation migration plan

## 已执行的第一阶段

1. 保留 `RLC/ogbench` canonical runtime，不覆盖 seed/reproducibility 修复。
2. 增加轻量 `ComputationSpec -> ComputationCore` 工厂，第一阶段只接受：
   - `primitive=mlp`
   - `topology=feedforward`
   - `credit=direct`
   同时由 `computation.factory.resolve_slot_spec` 统一解析 `compute.<slot_name>`，避免各个 Agent 重复实现 slot 配置逻辑。
3. 用 OGBench 原始 MLP 语义实现 primitive：GELU、`activate_final`、LayerNorm 位置、bias、dtype 和 `variance_scaling(fan_avg, uniform)` 均不改变。
4. 在 `GCActor` 中加入可选 `computation_spec`，只在 slot 明确启用时替换 actor body；默认 legacy path 仍直接构造原始 `MLP`。
5. 首个 slot 是 HIQL `low_actor`。encoder、action head、std、distribution、HIQL loss 和 goal representation 均保持原实现。
6. 增加 primitive parity、policy output parity、low-actor loss/gradient parity 和 parameter accounting 测试。

## 后续顺序

### Step 1: HIQL high actor

复用同一 `GCActor` slot 注入，只替换 `actor_net` body；保持 rep-dimension action head 和 target `goal_rep` 不变。先增加 policy output、high actor loss、gradient parity。

### Step 2: HIQL value — completed locally

把 `GCValue` 的 hidden MLP 作为 `value` slot；target value 自动镜像 online
value architecture，并继续由原有 Polyak update 维护。已检查 ensemble axis、
final scalar readout、expectile loss、gradient、update 和 N-step parity。

### Step 3: CRL actor

迁移 `GCActor` body，保持 CRL actor loss、DDPG+BC/AWR 分支和 action distribution 不变。

### Step 4: CRL critic/value

为 `GCBilinearValue` 的 phi/psi 分支分别增加 `critic_state`、`critic_goal`、`value_state`、`value_goal` slots。先迁移 FF MLP，再检查 bilinear logits、contrastive loss 和 gradient parity。

### Step 5: CoGHP slots

先把 CoGHP planner、executor、value 的原始 GELU MLP 或 MLP-Mixer 标定为已有 topology/primitive 的 reference implementation；不把 CoGHP agent 与 computation 变体组合成新 Agent 类。

### Step 6: MLP-Mixer primitive

在 reference parity 完成后加入 `coghp_gelu` 或 mixer primitive；先保持 feedforward topology，比较参数量、输出和训练梯度。

### Step 7: SwiGLU/RMS primitive

加入 SwiGLU + RMSNorm primitive，并与 feedforward topology 组合。primitive 和 topology 仍保持独立。

### Step 8: Single-State topology

在 `[B, D]` 与 `[B, T, D]` 输入约定下加入 recurrent state；增加 trace/state shape 和 direct/full credit 测试。

### Step 9: HRM two-state topology

最后加入 H/L two-state、fixed schedule 和可配置 credit propagation。此步骤之前不开始正式 HRM 实验，也不改变 HIQL/CRL 的 algorithm semantics。

## 每个 slot 的验收门槛

固定参数和输入，依次验证：primitive output、network/policy output、loss、gradient，条件允许时再验证 one optimizer step。误差使用明确的浮点 tolerance，并报告参数计数和 slot/core 参数计数。
