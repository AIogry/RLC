# M10A：HIQL 固定预算下的计算放置实验

状态：实现与短验证完成，正式 500k 训练尚未启动。

## 1. 科学问题

M10A 研究在总计算预算匹配时，把 SingleState 计算深度分配到 HIQL 的
`high_actor` 与 `low_actor` 是否会改变 GCRL 性能；同时观察当总预算从中等
水平扩大到较大水平时，优选的分配比例是否发生变化。

这是一项 computation placement/allocation study，而不是无约束的网络架构
搜索。所有非 vanilla 条件保持同一 primitive、topology、状态语义、输入注入、
hidden width、readout、HIQL loss 与 value network；同一个预算组内只改变
`high_actor` 与 `low_actor` 的计算深度分配。

### 假设

- 零假设：在相同 `B_body` 下，不同的 `(K_H, K_L)` 分配对
  `evaluation/overall_success` 没有系统性影响。
- 备择假设：在相同 `B_body` 下，至少有一种 `(K_H, K_L)` 分配产生不同的
  GCRL 性能。
- 次级问题：预算从 `B_body=5` 增加到 `B_body=16` 时，分配—性能关系是否
  改变。

实验前不预设 `high_actor` 应当优于 `low_actor`，也不预设更大的预算一定更好。

## 2. 记号与计算预算

```text
K_H     = high_actor 的 SingleState computation depth
K_L     = low_actor 的 SingleState computation depth
B_body  = K_H + K_L
```

当前 M10A 的 homogeneous hidden width 是 `h=512`。一个 SingleState update
module 执行包含两个 `Dense(512,512)`，因此重复 update body 的 Dense MAC 为：

```text
C_update = 2 * h^2 = 524,288 MACs
C_body   = (K_H + K_L) * C_update
```

`C_body` 只表示重复 update body，不等于完整 actor/policy 的全部 MAC。报告和
runtime audit 另外记录：

- input mapping Dense MACs；
- 每次 update 的 Dense MACs；
- update 执行次数与总 update MACs；
- computation-core Dense MACs；
- readout Dense MACs；
- 一次完整 actor forward 的 Dense MACs；
- trainable parameter 数量；
- non-trainable buffer 元素数量。

MAC 只统计实际 Flax 参数树中 Dense kernel 的
`input_features * output_features`，不把 bias、GELU、环境交互、评估或优化器
开销伪装进 MAC 数字。输入维度和 readout 维度从真实初始化参数树中读取，未在
通用 accounting library 中写死 AntMaze 维度。

在当前 AntMaze-large 形状中，审计得到：

| actor | 输入映射维度 | input mapping MACs | readout 输出 | readout MACs |
|---|---:|---:|---:|---:|
| high_actor | 58 → 512 | 29,696 | 512 → 10 | 5,120 |
| low_actor | 39 → 512 | 19,968 | 512 → 8 | 4,096 |

因此非 vanilla 条件的 combined computation-core MAC 是：

```text
C_core,combined = 29,696 + 19,968 + B_body * 524,288
```

例如：

| B_body | C_body | combined computation-core MACs |
|---:|---:|---:|
| 2 | 1,048,576 | 1,098,240 |
| 5 | 2,621,440 | 2,671,104 |
| 16 | 8,388,608 | 8,438,272 |

输入映射成本在预算组内恒定，所以不影响同组 placement 对比；它被单独列出，
避免把完整 core 成本错误地等同于 `B_body * C_update`。

## 3. 固定架构与语义

所有非 vanilla M10A 条件均为：

```yaml
algorithm: hiql
high_actor:
  primitive: mlp
  topology: single_state
  credit: direct
  residual: false
  input_injection: z_plus_x
  state_dim: 512
  state_init: zero_buffer
low_actor:
  primitive: mlp
  topology: single_state
  credit: direct
  residual: false
  input_injection: z_plus_x
  state_dim: 512
  state_init: zero_buffer
```

同时固定：

- GELU MLP，不使用 LayerNorm、RMSNorm、SwiGLU、gating 或新 activation；
- 一个物理共享的 update module 在本次 call 内重复执行 K 次；
- 参数在 K 次 execution 间共享；
- state 只在一次 decision 内存在，每次 call 从 buffer 重新 broadcast；
- 不跨 environment step carry state；
- `direct` 是完整 unrolled graph 的普通 reverse-mode differentiation；
- HIQL 的 high/low distribution、readout、goal encoding、loss、target update
  以及 vanilla value network 不变；
- 不启用 auxiliary loss，不改变 critic/value computation，也不让 critic 在
  inference 中参与 action。

`zero_buffer` 的语义为：

```text
z_init = zeros(state_dim)
```

它保存在 `buffers` collection 中，不属于 `params`，不会被 optimizer 更新，
每次 forward 只读不写，也不需要随机数。已有的 `normal_buffer` 仍然是默认值，
并保持原有 `state_init_std > 0` 校验和 checkpoint 语义。

vanilla HIQL 与 `(K_H,K_L)=(1,1)` zero-buffer SingleState 同时保留：前者是
外部算法 baseline，后者是同一 computation parameterization 下的 reference。
两者不能被宣称为 native initialization tree 完全相同；K=1 parity 只在明确的
semantic parameter mapping 后验证。

## 4. 11 个配置与 planned runs

研究目录为
[`experiments/M10A_fixed_budget_placement/`](../experiments/M10A_fixed_budget_placement/)。

| config | slug | K_H | K_L | B_body | high fraction | low fraction | 角色 |
|---|---|---:|---:|---:|---:|---:|---|
| M10A-C001 | `hiql_vanilla` | — | — | — | — | — | vanilla external baseline |
| M10A-C002 | `hiql_alloc_h1_l1_b2` | 1 | 1 | 2 | 0.5 | 0.5 | within-parameterization reference |
| M10A-C003 | `hiql_alloc_h4_l1_b5` | 4 | 1 | 5 | 0.8 | 0.2 | matched-budget placement |
| M10A-C004 | `hiql_alloc_h3_l2_b5` | 3 | 2 | 5 | 0.6 | 0.4 | matched-budget placement |
| M10A-C005 | `hiql_alloc_h2_l3_b5` | 2 | 3 | 5 | 0.4 | 0.6 | matched-budget placement |
| M10A-C006 | `hiql_alloc_h1_l4_b5` | 1 | 4 | 5 | 0.2 | 0.8 | matched-budget placement |
| M10A-C007 | `hiql_alloc_h15_l1_b16` | 15 | 1 | 16 | 0.9375 | 0.0625 | matched-budget placement |
| M10A-C008 | `hiql_alloc_h11_l5_b16` | 11 | 5 | 16 | 0.6875 | 0.3125 | matched-budget placement |
| M10A-C009 | `hiql_alloc_h8_l8_b16` | 8 | 8 | 16 | 0.5 | 0.5 | matched-budget placement |
| M10A-C010 | `hiql_alloc_h5_l11_b16` | 5 | 11 | 16 | 0.3125 | 0.6875 | matched-budget placement |
| M10A-C011 | `hiql_alloc_h1_l15_b16` | 1 | 15 | 16 | 0.0625 | 0.9375 | matched-budget placement |

研究协议为：

- environment：`antmaze-large-navigate-v0`；
- seeds：`0, 1, 2`；
- planned runs：`11 × 1 × 3 = 33`；
- primary metric：`evaluation/overall_success`；
- primary checkpoint metric：500k step 的 final success；
- 不使用 best checkpoint 作为 primary scientific metric；
- 正式实验的训练步数计划为 500k，但本次实现任务没有启动正式 sweep。

主要比较必须是同一 `B_body` 内的 allocation：

```text
primary comparison: allocations within the same B_body
secondary comparison: allocation response from B_body=5 to B_body=16
```

## 5. 参数与 MAC 审计

代码位置：

- [`impls/computation/accounting.py`](../impls/computation/accounting.py)：从
  实际参数/缓冲区树统计 Dense MAC、参数和 buffer；
- [`impls/main.py`](../impls/main.py)：将 actor/policy accounting 写入运行时
  metadata；
- [`tools/audit_m10a_budget.py`](../tools/audit_m10a_budget.py)：构造真实
  AntMaze-large shape 的 HIQL，并打印 11 配置 compact budget table。

对所有 M10A allocation 条件，两个 actor 的 unique trainable parameter 数量
不随 K 改变：

| slot | trainable params | computation-core params | buffer elements |
|---|---:|---:|---:|
| high_actor | 560,650 | 555,520 | 512 |
| low_actor | 549,896 | 545,792 | 512 |
| high + low | 1,110,546 | 1,101,312 | 1,024 |

每个 update module 只有一份物理参数，K 只影响 execution 次数，不复制参数。
`C002–C011` 的 combined core MAC 按预算组分别恒定为上表中的 `B_body=2/5/16`
数字；但完整 actor forward 还包括输入映射和 readout，因此不能只报告
`C_body`。

由真实 AntMaze-large 参数树得到的完整 compact audit 如下。`M_actor,total` 是
一次完整 high+low actor forward 的 Dense MAC，不是参数树中 kernel 的去重总和；
因此它会随 `(K_H,K_L)` 变化，而 `P_H+L` 与 `Bf_H+L` 不会变化。

| config | K_H | K_L | B_body | C_update | M_core,total | M_actor,total | P_H+L | Bf_H+L |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C001 | — | — | — | 0 | 0 | 1,107,456 | 1,110,546 | 0 |
| C002 | 1 | 1 | 2 | 524,288 | 1,098,240 | 1,107,456 | 1,110,546 | 1,024 |
| C003 | 4 | 1 | 5 | 524,288 | 2,671,104 | 2,680,320 | 1,110,546 | 1,024 |
| C004 | 3 | 2 | 5 | 524,288 | 2,671,104 | 2,680,320 | 1,110,546 | 1,024 |
| C005 | 2 | 3 | 5 | 524,288 | 2,671,104 | 2,680,320 | 1,110,546 | 1,024 |
| C006 | 1 | 4 | 5 | 524,288 | 2,671,104 | 2,680,320 | 1,110,546 | 1,024 |
| C007 | 15 | 1 | 16 | 524,288 | 8,438,272 | 8,447,488 | 1,110,546 | 1,024 |
| C008 | 11 | 5 | 16 | 524,288 | 8,438,272 | 8,447,488 | 1,110,546 | 1,024 |
| C009 | 8 | 8 | 16 | 524,288 | 8,438,272 | 8,447,488 | 1,110,546 | 1,024 |
| C010 | 5 | 11 | 16 | 524,288 | 8,438,272 | 8,447,488 | 1,110,546 | 1,024 |
| C011 | 1 | 15 | 16 | 524,288 | 8,438,272 | 8,447,488 | 1,110,546 | 1,024 |

建议使用以下命令执行审计（需要 OGBench 数据目录）：

```bash
cd /home/eai/Research/RLC
OGBENCH_DATASET_DIR=/path/to/ogbench/data \
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl JAX_PLATFORMS=cpu PYTHONPATH=. \
/home/eai/Tools/miniforge3/envs/brain_nav/bin/python \
tools/audit_m10a_budget.py
```

## 6. 测试覆盖

新增/更新测试覆盖：

- K=`1,2,3,4,5,8,11,15` 的构造和 forward；
- K≤0、非整数和 bool 拒绝；
- 一个共享 update module 与跨 K 的 trainable count invariant；
- zero buffer 的全零、非 trainable、确定性、不可变和不依赖随机 draw；
- normal buffer 的已有行为回归；
- zero-buffer non-residual K=1 的 semantic output parity；
- semantic mapping 后的参数梯度 parity；
- K=5/K=15 手工 unroll parity 与 K=15 finite gradient；
- HIQL `(15,1)`、`(8,8)`、`(1,15)` 的 high/low output/action shape；
- value 与 target value 保持 vanilla path；
- 11 配置和 33 planned runs；
- B=5、B=16 的 combined core MAC invariant；
- C002–C011 的 trainable parameter invariant；
- M8/M9/M9B 既有测试仍需在完整 CPU regression 中通过。

## 7. 解释边界与限制

`K` 是当前 homogeneous SingleState update body 的 execution depth，不是任意
异构 module 的普适计算单位。跨 primitive、hidden width 或 topology 的实验必须
重新定义成本，不能直接比较 K。

三 seed、单一 AntMaze-large 环境只能作为 pilot，不能单独构成最终论文证据；
正式结论还需要预注册/冻结协议下的完整训练、最终 checkpoint 的更高 episode
数评估，以及后续环境验证。后续 final-checkpoint evaluation 应在不改变训练
checkpoint 的前提下提高 episode 数，并同时保留每个 task 的原始结果、均值和
不确定性统计。

本任务明确不实施 M10B、TwoState/HRM topology 改动、ACT、critic-at-inference、
adaptive computation 或其它新的 scientific factor，也不改变 M9 配置和冻结
实验 worktree。
