# Vanilla CoGHP 官方迁移审计

日期：2026-08-13  
官方来源：[`wlsdn9350/CoGHP`](https://github.com/wlsdn9350/CoGHP)，`main`，审计 commit `8f362e9f86bf97fdbc9ce36d1b7b73b024e18b36`

## 审计范围与结论

本审计在迁移生产代码前完成，读取官方仓库中的：

- `impls/agents/coghp.py`
- `impls/utils/coghp_network.py`
- `impls/utils/datasets.py`
- `impls/utils/networks.py`
- `impls/utils/encoders.py`
- `impls/main.py`

在 RLC 中，原始官方文件 `impls/utils/coghp_network.py` 被归类为
`impls/networks/coghp.py`。这是更符合当前项目分层的归类：CoGHP 的
`HierarchicalPolicyNetwork` 是算法专用 network，而不是通用 computation
模块。本轮不移动或重构其中的实现。

官方 CoGHP 的四个网络模块是 `goal_rep`、`value`、`target_value` 和 `actor_mixer`。本项目迁移时保持这些官方名称和职责，不引入新的 high/low core 命名，也不把 actor mixer 拆成两个网络。

## 官方 agent 语义

`value_loss` 使用 target value 的双头最小值构造 TD target，并分别以 expectile loss 更新两个 value heads。`actor_loss` 同时计算：

- high actor：对每个 autoregressive subgoal 使用 `goal_rep` 产生的目标表示、value advantage、AWR 权重、`high_discount` 和 `subgoal_steps` 归一化；
- low actor：以最后一个 high-level target（无 subgoal 时使用 high actor goal）作为目标，使用 value advantage 和 AWR 权重计算动作 log-prob loss。

总损失为 value loss、high actor loss 和 low actor loss 的和。target value 通过 Polyak update 更新；采样时 actor mixer 以 autoregressive 方式先生成 subgoals，再生成最终 action。官方 `goal_rep` 的 gradient stop 位置、teacher-forcing 的 `subgoal_reps`/`action_seq` 输入、`value`/`target_value`/`actor_mixer` 调用方式均作为迁移的 parity 约束。

其中 `subgoal_reps` 会实际参与 teacher forcing：当它不为 `None` 时，
autoregressive loop 使用给定的前缀替换已经生成的 subgoal token。`action_seq`
目前只是保持官方调用接口的兼容参数，当前 forward 不读取它；它不会改变
当前 action token 的计算。

## 官方 MixerBlock 精确结构

`MixerBlock` 的输入为 `(B, num_tokens, embed_dim)`。在当前 CoGHP
autoregressive layout 中，有 observation token、goal token 和
`num_subgoals + 1` 个 previous/prediction token，因此 effective
`num_tokens = num_subgoals + 3`。每个 block 是独立的物理参数集合，包含：

- `token_dense1`: `num_tokens -> hidden_dim_tokens`
- GELU
- `token_dense2`: `hidden_dim_tokens -> num_tokens`
- token mixing 后乘以可训练 `tm_weights`，并应用 `jnp.tril(tm_weights)` 下三角 mask
- token residual：`x = x + y`
- `channel_dense1`: `embed_dim -> hidden_dim_channels`
- GELU
- `channel_dense2`: `hidden_dim_channels -> embed_dim`
- channel residual：`output = x + z`

官方实现没有 LayerNorm，没有额外的 pre/post normalization；`tm_weights` 以 `normal(stddev=0.02)` 初始化。`init_scale` 与 `decay_alpha` 是 inactive/compatibility fields：它们保留在模块签名中，但官方 `__call__` 不使用它们。token mixing 的实际维度变换是先将输入转为 `(B, embed_dim, num_tokens)`，经过两层 Dense 后转回，再执行 `einsum('btd,ts->bsd', y, tm_weights)`。

当前 token replacement 还依赖一个隐含 invariant：
`joint_embed_dim == state_dim`。high actor 产生的 subgoal representation
必须能直接写入 Mixer 的 previous-token embedding；当前官方 vanilla
配置与 RLC 迁移保持这一相等关系。该 invariant 本轮只记录，不通过改变
forward 或新增校验来重构实现。

## 共享关系与参数树

官方 `HierarchicalPolicyNetwork.setup()` 只创建一个 `self.mixer_blocks` 列表；autoregressive token loop 中每个 token step 都重复调用同一列表。因此：

1. subgoal prediction 和 final action 使用同一个物理 Mixer block 参数集合；
2. high actor head 与 low actor head 是两个独立的 head 参数集合；
3. `feature_embed` 和 `prev_tokens` 各自只有一份；
4. 不存在 high/low 两套 Mixer core，也不存在因为循环而复制的 Mixer 参数；
5. `target_value` 在 official agent 初始化时复制 `value` 的初始数值，但之后是独立的 target 参数树；
6. high-level autoregressive distributions 在每个 high step 采样时复用同一个 `high_seed`，这是 official behavior，不在 RLC 中拆分为 per-step seeds。

参数树顶层为：

```text
modules_goal_rep
modules_value
modules_target_value
modules_actor_mixer
  prev_tokens
  feature_embed
  mixer_blocks_0 ... mixer_blocks_{N-1}
  high_actor_head
  low_actor_head
```

在官方代码上用小型初始化（`feature_dim=4`、`num_subgoals=2`、`num_mixer_blocks=2`、`actor/value hidden=(6,)`、`enc_hidden=(5,)`）实测：

| 模块 | 标量参数数 |
|---|---:|
| `modules_goal_rep` | 79 |
| `modules_value` | 146 |
| `modules_target_value` | 146 |
| `modules_actor_mixer` | 361 |
| 总计 | 732 |

其中 actor mixer 的 361 个参数包含一份 `prev_tokens`、一份 `feature_embed`、两套独立 MixerBlock 参数和一套 high/low head；autoregressive 循环没有额外参数。该结果也作为 RLC 迁移后的 parameter-accounting 回归基线。

## 数据集语义

官方 `dataset_class='MultiHGCDataset'`。它在保留 `value_goals`、`rewards`、`masks` 的同时，生成：

- `low_actor_goals`：`subgoal_steps` 后的状态；
- `high_actor_goals`：一个 high-level 终点；
- `high_actor_targets`：形状 `(B, num_subgoals, state_dim)` 的逐级目标。

trajectory sampling、random-goal 分支、episode boundary 截断、`num_subgoals` 的反向/正向 target offsets、frame stack 与 augmentation 均按官方实现迁移。RLC 保留已有的显式 `numpy.random.Generator` 参数，避免回退到官方的 process-global `numpy.random`；这只改变 RNG 注入方式，不改变采样公式或输出字段。

## 文件归类与后续边界

当前 `impls/networks/coghp.py` 是原始 `impls/utils/coghp_network.py` 在
RLC 中更合理的归类。`MixerBlock` 暂时继续保留在
`impls/networks/coghp.py`，因为它承担 official vanilla CoGHP baseline 的
reference-faithful implementation。后续当 MLP-Mixer 正式纳入统一
computation framework 时，`MixerBlock` 很可能从这里移出，进入
`impls/computation/` 下更通用的 primitive/block 层；本轮不提前移动。

## 迁移边界

本 milestone 不修改 `impls/computation/`，不新增 Mixer primitive/topology，不引入 HRM、recurrence、SwiGLU/RMS 或 matched-FF 变体。Vanilla CoGHP 以 RLC 的 agent/network/dataset/registry/shared trainer 接口接入，并保持官方模块名、loss 语义、共享关系和 autoregressive 行为。
