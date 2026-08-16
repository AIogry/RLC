# Vanilla CoGHP 迁移记录

日期：2026-08-13  
官方来源：[`wlsdn9350/CoGHP`](https://github.com/wlsdn9350/CoGHP)，`main`，commit `8f362e9f86bf97fdbc9ce36d1b7b73b024e18b36`

## 迁移内容

Vanilla CoGHP 已接入 RLC 的 shared runtime：

- `impls/agents/coghp.py`：官方 `goal_rep`、双 value/target-value、actor mixer、high/low AWR loss、teacher forcing、autoregressive sampling、Polyak target update；
- `impls/networks/coghp.py`：原始官方 `impls/utils/coghp_network.py` 在 RLC
  中更合理的 network 归类，包含官方 `MixerBlock` 和
  `HierarchicalPolicyNetwork`；
- `impls/utils/datasets.py`：官方 `MultiHGCDataset` 字段和 goal/boundary/num-subgoals 采样；
- `impls/utils/encoders.py`：增加官方 actor mixer 所需的 `listwise=True` encoder 返回路径；
- `impls/main.py` 与 `configs/agents/coghp.py`：agent registry、dataset registry、loss aggregation、evaluation/checkpoint/runtime 接入。

没有修改 `impls/computation/`，没有新增 computation primitive/topology，也没有引入 HRM、recurrence、SwiGLU/RMS 或 matched feed-forward 变体。`--computation` 对 CoGHP 明确报错，因为 vanilla CoGHP 的官方 Mixer core 不是 computation slot。

`MixerBlock` 当前暂时保留在 `impls/networks/coghp.py`，因为它承担
official vanilla CoGHP baseline 的 reference-faithful implementation。未来
当 MLP-Mixer 正式纳入统一 computation framework 时，它很可能移动到
`impls/computation/` 下更通用的 primitive/block 层；本轮不提前移动。

## 官方共享关系保留

`actor_mixer` 中只有一份 `mixer_blocks` 列表。每个 autoregressive token step 重复使用相同的 MixerBlock 参数；subgoal 和最终 action 使用同一个物理 Mixer core。`high_actor_head` 与 `low_actor_head` 是独立的 readout 网络，且 `prev_tokens`、`feature_embed` 各只有一份。迁移没有拆分 high/low core，也没有改变官方共享关系。

当前 effective Mixer token count 为 `num_subgoals + 3`。此外，
`joint_embed_dim == state_dim` 是 token replacement 的隐含 invariant：
high-level subgoal representation 必须能直接替换 previous-token embedding。

小型初始化审计（`feature_dim=4`、`num_subgoals=2`、`num_mixer_blocks=2`、actor/value hidden `(6,)`、encoder hidden `(5,)`）得到：

| 模块 | 参数标量 |
|---|---:|
| `modules_goal_rep` | 79 |
| `modules_value` | 146 |
| `modules_target_value` | 146 |
| `modules_actor_mixer` | 361 |
| 总计 | 732 |

actor mixer 的 361 个标量只计数一次，未将 autoregressive 循环中的重复调用错误地计为多套参数。

官方 forward 的两个接口细节也保持明确：`subgoal_reps` 实际参与
teacher forcing；`action_seq` 只是接口兼容参数，当前 forward 未使用。
`init_scale` 与 `decay_alpha` 是保留的 inactive/compatibility fields，当前
实现不读取它们。每个 high-level step 复用同一个 `high_seed`，保持 official
behavior；本轮不修改为 per-step RNG。

## Parity 验证

### Synthetic official/RLC parity

同一 seed、同一小型配置、同一 batch 下：

- 初始化 parameter tree：51/51 leaves，最大绝对误差 `0.0`；
- `goal_rep`、`value`、`target_value`：最大误差 `0.0`；
- high/low distributions、mode、sample：最大误差 `0.0`；
- value loss、high actor loss、low actor loss、全部 diagnostics：最大误差 `0.0`；
- 单步更新后的 online/target 参数、optimizer state、agent RNG：语义结果一致。

RLC 既有 `TrainState.apply_loss_fn` 的 `grad/norm` 是对 raw pytree leaves 做 L1 聚合，官方实现先对每个 leaf 求 norm 再聚合；因此该单一诊断字段不作为算法 parity 断言，参数更新和其他梯度统计保持一致。

### Real OGBench N=20

环境：`antmaze-medium-navigate-v0`；数据：实际 OGBench `MultiHGCDataset`；配置、seed、batch size、network initialization、target initialization、Adam 和 RNG 对齐。

- dataset required fields：20/20 batch bitwise parity；
- first divergence：step 0 的 optimizer reduction 数值差异（非语义）；
- loss 最大误差：`7.62939453125e-06`；
- online 参数最大误差：`1.4901161193847656e-08`；
- target value 参数最大误差：`2.9103830456733704e-11`；
- optimizer state 最大误差：`1.1920928955078125e-07`；
- agent RNG 最大误差：`0.0`；
- 最终参数保持 finite。

这些差异属于官方与 RLC 两套 `TrainState` 梯度诊断/float32 reduction 路径的数值级差异，不是网络、损失、采样、target update 或 RNG 语义差异。

### Shared runtime smoke

已运行 shared `impls.main` 的真实数据 smoke，包含 update、evaluation、CSV logging、checkpoint save/restore。loss finite，evaluation 和 checkpoint action/value probe 已接入；checkpoint probe 对 CoGHP 的 batch-to-unbatched policy API 做了专门形状处理。

GPU 1000-step vanilla CoGHP smoke 已通过：CUDA backend、真实
`antmaze-medium-navigate-v0`、batch size 256、默认 512x3 MLP 配置。10 个
logging rows 的全部 training/validation/gradient diagnostics 都是 finite；
step 1000 的 value/high-actor/low-actor loss 分别为
`0.3899533`、`24.7590027`、`75.8899841`。evaluation 完成并记录
`task1_success=0.0`，checkpoint save/restore action/value probe PASS，未发现
NaN/Inf。

## 当前状态

Vanilla CoGHP 的官方语义迁移、共享关系、真实数据 N=20 parity、shared
runtime、完整 CPU 回归和 GPU 1000-step smoke 均已完成。本 milestone 的
integration gate **通过**；这不等同于长时间 baseline reproduction。
