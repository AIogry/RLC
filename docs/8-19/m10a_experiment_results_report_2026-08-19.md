# M10A 正式实验结果报告

日期：2026-08-19  
实验：M10A — HIQL fixed-budget computation placement  
状态：33 个正式 run 全部完成，均训练至 500,000 steps。

本报告严格区分两类内容：

1. “原始数据”部分直接保留正式 run 目录中的 `eval.csv`、`summary.json` 原文；
2. “初步分析”部分只做明确标记的均值、标准差、差值和学习曲线整理，不覆盖、不修改、不平滑原始数据。

`eval.csv` 和 `summary.json` 的浮点字符串，包括例如
`0.8699999999999999`、`0.8400000000000001` 等，均按原始文件保留。

---

## 1. 实验目的与研究问题

M10A 研究在固定 recurrent-body computation budget 下，将 SingleState
computation depth 分配到 HIQL `high_actor` 与 `low_actor` 的不同方式，是否会
改变 GCRL performance；同时观察总预算从 `B_body=5` 增加到 `B_body=16` 时，
allocation-performance response 是否变化。

记号：

```text
K_H    = high_actor 的 SingleState computation depth
K_L    = low_actor 的 SingleState computation depth
B_body = K_H + K_L
```

主要科学比较是同一个 `B_body` 内的 allocation：

```text
B_body=5:  (4,1), (3,2), (2,3), (1,4)
B_body=16: (15,1), (11,5), (8,8), (5,11), (1,15)
```

`M10A-C001` 是 vanilla HIQL external baseline；`M10A-C002` 是
`(K_H,K_L)=(1,1)` 的 zero-buffer SingleState reference。C001 与 C002 不属于
完全相同的 native initialization tree，因此报告中不把二者差异解释为纯粹的
allocation effect。

本报告不把结果解释为普适的 architecture search 结论，也不把 K 宣称为所有
异构 computation module 的通用计算单位。

## 2. 正式实验数据位置与 provenance

原始实验根目录：

```text
/data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M10A
```

每个 run 的标准路径为：

```text
M10A-Cxxx__slug/antmaze-large-navigate-v0/seed_NNN/
```

所有 33 个 run 的 metadata 一致记录：

| 字段 | 原始记录 |
|---|---|
| `status` | `completed`（33/33） |
| `jax_backend` | `gpu`（33/33） |
| JAX device | `cuda:0` |
| hostname | `eai-pro` |
| git commit | `f1d2a4a95ce510f4d476f9d8c74f092c92e0558c` |
| `git_dirty` | `false`（33/33） |
| environment | `antmaze-large-navigate-v0` |
| dataset directory | `/data/qijunrong/06-RL/offline-rl/data/raw_ogbench` |
| OGBench module | `/home/eai/Research/RLC-m10a-exp/ogbench/__init__.py` |

本报告只读取了上述原始目录，没有修改其中的训练日志、评估日志、metadata 或
checkpoint。

## 3. 正式训练 protocol

以下内容直接来自每个 run 的 `runtime_metadata.json` / `resolved_config.json`：

| protocol 字段 | 数值 |
|---|---:|
| `train_steps` | 500,000 |
| `batch_size` | 1,024 |
| `log_interval` | 5,000 |
| `eval_interval` | 100,000 |
| `eval_episodes` | 20 |
| `eval_tasks` | `null`，运行时覆盖该环境的全部 5 个 task |
| `eval_temperature` | 0.0 |
| `eval_gaussian` | `null` |
| `save_interval` | 500,000 |
| `video_episodes` | 0 |
| optimizer learning rate | 0.0003 |
| seeds | 0, 1, 2 |

因此每份 `eval.csv` 有 5 个评估时间点：100k、200k、300k、400k、500k；每个
时间点包含 5 个 task success 和一个 `overall_success`。每个 task 使用 20
episodes，`overall_success` 是 5 个 task success 的平均值。

每份 `train.csv` 有 100 行训练记录，对应 5,000 到 500,000，每 5,000 steps
一行。

## 4. 配置矩阵

| config | slug | K_H | K_L | B_body | high fraction | low fraction | 角色 |
|---|---|---:|---:|---:|---:|---:|---|
| M10A-C001 | `hiql_vanilla` | — | — | — | — | — | vanilla baseline |
| M10A-C002 | `hiql_alloc_h1_l1_b2` | 1 | 1 | 2 | 0.5 | 0.5 | within-parameterization reference |
| M10A-C003 | `hiql_alloc_h4_l1_b5` | 4 | 1 | 5 | 0.8 | 0.2 | matched-budget allocation |
| M10A-C004 | `hiql_alloc_h3_l2_b5` | 3 | 2 | 5 | 0.6 | 0.4 | matched-budget allocation |
| M10A-C005 | `hiql_alloc_h2_l3_b5` | 2 | 3 | 5 | 0.4 | 0.6 | matched-budget allocation |
| M10A-C006 | `hiql_alloc_h1_l4_b5` | 1 | 4 | 5 | 0.2 | 0.8 | matched-budget allocation |
| M10A-C007 | `hiql_alloc_h15_l1_b16` | 15 | 1 | 16 | 0.9375 | 0.0625 | matched-budget allocation |
| M10A-C008 | `hiql_alloc_h11_l5_b16` | 11 | 5 | 16 | 0.6875 | 0.3125 | matched-budget allocation |
| M10A-C009 | `hiql_alloc_h8_l8_b16` | 8 | 8 | 16 | 0.5 | 0.5 | matched-budget allocation |
| M10A-C010 | `hiql_alloc_h5_l11_b16` | 5 | 11 | 16 | 0.3125 | 0.6875 | matched-budget allocation |
| M10A-C011 | `hiql_alloc_h1_l15_b16` | 1 | 15 | 16 | 0.0625 | 0.9375 | matched-budget allocation |

总计划数：`11 × 1 environment × 3 seeds = 33`。  
实际完成数：`33/33`。

固定设计：

- HIQL algorithm；
- high/low actor 独立 GCActor；
- actor hidden dims `(512,512,512)`；
- SingleState + GELU MLP；
- non-residual；
- `input_injection=z_plus_x`；
- `state_dim=512`；
- `state_init=zero_buffer`；
- decision-local state，不跨 environment step carry；
- `credit=direct`；
- value network、HIQL loss、readout/distribution semantics 不变；
- 不使用 auxiliary loss、ACT、adaptive computation 或 critic-at-inference。

## 5. 实验完整性审计

| 原始 artifact | 数量 | 总大小 | 总行数/说明 |
|---|---:|---:|---:|
| `eval.csv` | 33 | 12,145 bytes | 198 行（含各文件 header） |
| `summary.json` | 33 | 5,517 bytes | 231 行 |
| `train.csv` | 33 | 1,918,442 bytes | 3,333 行 |
| `runtime_metadata.json` | 33 | 177,840 bytes | 5,799 行 |
| `resolved_config.json` | 33 | 196,461 bytes | 7,332 行 |
| `checkpoints/params_500000.pkl` | 33 | 1,533,723,279 bytes | 每个 run 一个 |

完整性判断：

- 33/33 有 `runtime_metadata.json`；
- 33/33 有 `resolved_config.json`；
- 33/33 有 `train.csv`；
- 33/33 有 `eval.csv`；
- 33/33 有 `summary.json`；
- 33/33 有 `params_500000.pkl`；
- 33/33 的 metadata `status=completed`；
- 33/33 的最后训练记录为 step 500,000；
- 33/33 的最后评估记录为 step 500,000；
- 33/33 的 summary `status=completed`。

## 6. 参数与 Dense-MAC accounting

正式 run 的 `actor_parameter_accounting` 记录了实际初始化参数树的审计结果。
非 vanilla 条件中：

```text
h = 512
C_update = 2 * h² = 524,288 Dense MACs
C_body = (K_H + K_L) * C_update
```

实际 slot accounting：

| slot | input mapping MACs | update MACs/execution | readout MACs | trainable params | buffer elements |
|---|---:|---:|---:|---:|---:|
| high_actor | 29,696 | 524,288 | 5,120 | 560,650 | 512 |
| low_actor | 19,968 | 524,288 | 4,096 | 549,896 | 512 |
| high+low | 49,664 | — | 9,216 | 1,110,546 | 1,024 |

allocation 只改变 execution count，不复制 update module 参数。因此 C003–C006
的 repeated update body 为 `5 × 524,288 = 2,621,440 MACs`，C007–C011 为
`16 × 524,288 = 8,388,608 MACs`。完整 forward 还包含 input mapping 和
readout，不能只使用 `C_body` 代表完整 policy 成本。

## 7. 原始结果对应的最终/最佳 summary

下表中的 `seed0/seed1/seed2` 是从各自原始 `summary.json` 直接读取的
`final_success`；`best` 是原始文件中的 `best_success`。括号中的 step 是
原始 `best_step`。

| config | K_H | K_L | B_body | final seed0 | final seed1 | final seed2 | best seed0 | best seed1 | best seed2 |
|---|---:|---:|---:|---:|---:|---:|---|---|---|
| C001 | — | — | — | 0.75 | 0.8699999999999999 | 0.8400000000000001 | 0.82 (400k) | 0.8699999999999999 (500k) | 0.8800000000000001 (400k) |
| C002 | 1 | 1 | 2 | 0.8699999999999999 | 0.7999999999999999 | 0.95 | 0.8699999999999999 (500k) | 0.8099999999999999 (400k) | 0.95 (500k) |
| C003 | 4 | 1 | 5 | 0.8400000000000001 | 0.9199999999999999 | 0.72 | 0.89 (400k) | 0.9199999999999999 (500k) | 0.8200000000000001 (400k) |
| C004 | 3 | 2 | 5 | 0.82 | 0.89 | 0.82 | 0.86 (400k) | 0.89 (500k) | 0.8400000000000001 (300k) |
| C005 | 2 | 3 | 5 | 0.8699999999999999 | 0.9 | 0.9 | 0.8699999999999999 (500k) | 0.9099999999999999 (400k) | 0.9199999999999999 (300k) |
| C006 | 1 | 4 | 5 | 0.8300000000000001 | 0.7299999999999999 | 0.8899999999999999 | 0.89 (200k) | 0.76 (400k) | 0.8899999999999999 (500k) |
| C007 | 15 | 1 | 16 | 0.8699999999999999 | 0.8099999999999999 | 0.8099999999999999 | 0.97 (400k) | 0.8099999999999999 (400k) | 0.8700000000000001 (400k) |
| C008 | 11 | 5 | 16 | 0.8699999999999999 | 0.9 | 0.78 | 0.8699999999999999 (500k) | 0.9 (500k) | 0.8699999999999999 (400k) |
| C009 | 8 | 8 | 16 | 0.85 | 0.8300000000000001 | 0.8399999999999999 | 0.9 (300k) | 0.89 (400k) | 0.8799999999999999 (400k) |
| C010 | 5 | 11 | 16 | 0.8299999999999998 | 0.9099999999999999 | 0.9100000000000001 | 0.9 (300k) | 0.9199999999999999 (400k) | 0.9100000000000001 (500k) |
| C011 | 1 | 15 | 16 | 0.8100000000000002 | 0.9200000000000002 | 0.8800000000000001 | 0.85 (400k) | 0.9200000000000002 (500k) | 0.9 (400k) |

---

## 8. 初步派生分析：final success

本节统计均由第 7 节的三个原始 `final_success` 派生。`mean` 是三 seed
算术平均，`SD` 是 population standard deviation（`ddof=0`），不是新增实验
数据，也不替换原始 seed 值。

| config | K_H | K_L | B_body | final seed values（原始） | mean | SD |
|---|---:|---:|---:|---|---:|---:|
| C001 | — | — | — | 0.75, 0.8699999999999999, 0.8400000000000001 | 0.8200 | 0.0510 |
| C002 | 1 | 1 | 2 | 0.8699999999999999, 0.7999999999999999, 0.95 | 0.8733 | 0.0613 |
| C003 | 4 | 1 | 5 | 0.8400000000000001, 0.9199999999999999, 0.72 | 0.8267 | 0.0822 |
| C004 | 3 | 2 | 5 | 0.82, 0.89, 0.82 | 0.8433 | 0.0330 |
| C005 | 2 | 3 | 5 | 0.8699999999999999, 0.9, 0.9 | 0.8900 | 0.0141 |
| C006 | 1 | 4 | 5 | 0.8300000000000001, 0.7299999999999999, 0.8899999999999999 | 0.8167 | 0.0660 |
| C007 | 15 | 1 | 16 | 0.8699999999999999, 0.8099999999999999, 0.8099999999999999 | 0.8300 | 0.0283 |
| C008 | 11 | 5 | 16 | 0.8699999999999999, 0.9, 0.78 | 0.8500 | 0.0510 |
| C009 | 8 | 8 | 16 | 0.85, 0.8300000000000001, 0.8399999999999999 | 0.8400 | 0.0082 |
| C010 | 5 | 11 | 16 | 0.8299999999999998, 0.9099999999999999, 0.9100000000000001 | 0.8833 | 0.0377 |
| C011 | 1 | 15 | 16 | 0.8100000000000002, 0.9200000000000002, 0.8800000000000001 | 0.8700 | 0.0455 |

### 8.1 B_body=5 的初步观察

四个同预算条件的 final mean 为：

```text
C003 (4,1): 0.8267
C004 (3,2): 0.8433
C005 (2,3): 0.8900
C006 (1,4): 0.8167
```

在这组数据中，`(K_H,K_L)=(2,3)` 的 C005 final mean 最高，且 SD=0.0141
是 B=5 组中最低；但是三 seed 数量很小，不能据此宣称存在稳定统计显著的
最优 allocation。C003 和 C006 的 seed 间波动更大，说明 allocation effect
与 seed sensitivity 可能同时存在。

相对于 C001 vanilla mean=0.8200：

- C003：+0.0067；
- C004：+0.0233；
- C005：+0.0700；
- C006：−0.0033。

这些是三 seed mean 的描述性差值，不是显著性检验结果。

### 8.2 B_body=16 的初步观察

五个同预算条件的 final mean 为：

```text
C007 (15,1): 0.8300
C008 (11,5): 0.8500
C009 (8,8): 0.8400
C010 (5,11): 0.8833
C011 (1,15): 0.8700
```

在 B=16 组中，C010 的 `(5,11)` final mean 最高，C011 的 `(1,15)` 次之；
C007 的 `(15,1)` 与 vanilla final mean 接近。C009 balanced placement 的
seed SD 最低（0.0082），但 mean 不是最高。

相对于 C001 vanilla mean=0.8200，B=16 allocation 的描述性差值为：

```text
C007: +0.0100
C008: +0.0300
C009: +0.0200
C010: +0.0633
C011: +0.0500
```

### 8.3 从 B=5 到 B=16 的 allocation response

从当前 3-seed AntMaze-large 结果看，较大预算下表现较好的方向从 B=5 的
`(2,3)` 偏向 low_actor，转为 B=16 的 `(5,11)` 和 `(1,15)`，仍然是
low_actor 分配较高的条件。这与“优选 allocation 可能随总预算改变”的次级
问题相容，但不能排除：

- 三 seed noise；
- 不同 K 的优化难度差异；
- 单一环境的 task composition；
- 500k 截止时点造成的 checkpoint timing effect。

因此这里应记录为初步趋势，而不是已确认的机制性结论。

## 9. 初步派生分析：学习曲线

下表为三个 seed 在每个 eval step 的 `overall_success` 的 mean；括号内为
population SD。原始逐 seed 曲线见第 12 节的 33 个 `eval.csv`。

| config | 100k | 200k | 300k | 400k | 500k |
|---|---:|---:|---:|---:|---:|
| C001 | 0.6167 (0.0772) | 0.7367 (0.0249) | 0.7833 (0.0368) | 0.8467 (0.0249) | 0.8200 (0.0510) |
| C002 | 0.7167 (0.0236) | 0.7600 (0.0245) | 0.8000 (0.0356) | 0.8200 (0.0141) | 0.8733 (0.0613) |
| C003 | 0.7600 (0.0082) | 0.7500 (0.0497) | 0.7933 (0.0411) | 0.8567 (0.0287) | 0.8267 (0.0822) |
| C004 | 0.7267 (0.0386) | 0.7267 (0.0249) | 0.8067 (0.0403) | 0.8067 (0.0411) | 0.8433 (0.0330) |
| C005 | 0.7267 (0.0236) | 0.7933 (0.0125) | 0.8700 (0.0374) | 0.8600 (0.0374) | 0.8900 (0.0141) |
| C006 | 0.7333 (0.0205) | 0.7800 (0.0779) | 0.7767 (0.0330) | 0.8133 (0.0386) | 0.8167 (0.0660) |
| C007 | 0.7233 (0.0544) | 0.8233 (0.0411) | 0.8367 (0.0822) | 0.8833 (0.0660) | 0.8300 (0.0283) |
| C008 | 0.7433 (0.0386) | 0.7900 (0.0356) | 0.8233 (0.0309) | 0.8400 (0.0356) | 0.8500 (0.0510) |
| C009 | 0.7633 (0.0822) | 0.8267 (0.0309) | 0.8533 (0.0340) | 0.8600 (0.0356) | 0.8400 (0.0082) |
| C010 | 0.7433 (0.0047) | 0.8533 (0.0330) | 0.8733 (0.0205) | 0.8467 (0.0772) | 0.8833 (0.0377) |
| C011 | 0.7233 (0.0236) | 0.7400 (0.0408) | 0.8100 (0.0216) | 0.8533 (0.0368) | 0.8700 (0.0455) |

初步学习动态观察：

- C005 在 300k 已达到 0.8700 mean，之后 500k 为 0.8900，曲线较平稳；
- C010 在 300k 达到 0.8733，400k 有回落后在 500k 回升到 0.8833；
- C007 在 400k 达到 0.8833，但 500k 回落到 0.8300，说明 best checkpoint
  与 final checkpoint 的差异不能忽略；
- C002 在 500k 明显高于 400k，说明 `(1,1)` reference 仍在持续变化；
- C006 的 500k mean=0.8167，且 seed variation 较大；
- 不能仅依据早期 100k/200k 排名推断最终排名。

## 10. 初步派生分析：task-level final success

以下为每个 config 在 500k 时五个 task success 的跨 seed mean；括号为
population SD。task 顺序严格对应原始列 `task1`–`task5`。

| config | task1 | task2 | task3 | task4 | task5 |
|---|---:|---:|---:|---:|---:|
| C001 | 0.9000 (0.0707) | 0.4000 (0.1080) | 0.9333 (0.0624) | 0.9167 (0.0624) | 0.9500 (0.0408) |
| C002 | 1.0000 (0.0000) | 0.5500 (0.2944) | 0.9667 (0.0236) | 0.9167 (0.0236) | 0.9333 (0.0624) |
| C003 | 0.8667 (0.1027) | 0.5333 (0.2357) | 0.8500 (0.0707) | 0.9667 (0.0236) | 0.9167 (0.0624) |
| C004 | 0.9167 (0.0236) | 0.4167 (0.1650) | 0.9667 (0.0471) | 0.9500 (0.0408) | 0.9667 (0.0236) |
| C005 | 0.9333 (0.0236) | 0.7500 (0.1225) | 0.9500 (0.0000) | 0.9000 (0.0408) | 0.9167 (0.0624) |
| C006 | 0.8667 (0.0471) | 0.5000 (0.3082) | 0.9167 (0.0236) | 0.9500 (0.0408) | 0.8500 (0.0707) |
| C007 | 0.9000 (0.0408) | 0.5500 (0.2483) | 0.9333 (0.0236) | 0.9000 (0.0707) | 0.8667 (0.1027) |
| C008 | 0.8667 (0.0236) | 0.7667 (0.1312) | 0.9167 (0.0471) | 0.8167 (0.1434) | 0.8833 (0.0471) |
| C009 | 0.9333 (0.0236) | 0.6000 (0.1472) | 0.9500 (0.0408) | 0.8667 (0.0850) | 0.8500 (0.0408) |
| C010 | 0.9667 (0.0236) | 0.6833 (0.1700) | 0.9667 (0.0236) | 0.9167 (0.0236) | 0.8833 (0.0236) |
| C011 | 0.8667 (0.0850) | 0.6667 (0.1929) | 0.9667 (0.0471) | 0.9000 (0.0408) | 0.9500 (0.0408) |

最明显的 task-level 特征是 task2 在所有条件下均明显低于其他 task，并且
seed SD 普遍较高。比如 C005 的 task2 mean=0.7500，而 C001 为 0.4000；但
task2 的三 seed 波动仍然较大，因此需要更高 episode 数和更多环境验证。

## 11. 运行时间与训练日志的初步观察

`runtime_metadata.json` 的 `start_time` 与 `end_time` 给出的总 wall-clock：

- 33 个 run 合计约 21.695 小时；
- 单 run 平均约 2,366.7 秒；
- 最短 1,842 秒；
- 最长 3,158 秒。

按 config 的平均 wall-clock（分钟，三 seed population SD）：

| config | mean min | SD min | min–max min |
|---|---:|---:|---:|
| C001 | 31.85 | 0.39 | 31.40–32.35 |
| C002 | 31.68 | 0.77 | 30.70–32.57 |
| C003 | 33.76 | 0.70 | 32.88–34.58 |
| C004 | 33.76 | 0.56 | 32.97–34.20 |
| C005 | 33.41 | 0.66 | 32.93–34.33 |
| C006 | 33.89 | 0.75 | 32.83–34.50 |
| C007 | 45.14 | 1.32 | 43.28–46.10 |
| C008 | 49.35 | 2.77 | 45.87–52.63 |
| C009 | 51.23 | 1.48 | 49.17–52.57 |
| C010 | 47.48 | 1.60 | 45.90–49.67 |
| C011 | 42.33 | 1.12 | 40.95–43.68 |

这些 wall-clock 数字是 provenance 中的 start/end 差值，不是严格的硬件 benchmark。
正式运行存在 GPU 调度、进程并行和首次编译等因素，因此只能作为资源使用的
初步记录。总体上 B=16 比 B=5 更慢，且具体方向之间存在差异，但不能仅根据
该表建立“某一 placement 的单步计算效率”结论。

所有 `train.csv` 的最后记录均为 500k，且当前已检查所有原始数值为有限值；
完整训练日志没有在本报告中重排为新表，以避免破坏其原始列顺序和浮点文本。
完整原始文件索引见第 13 节。

## 12. 结果解释边界

当前结果支持的最谨慎描述是：

1. M10A 的 33 个正式 run 全部完成，实验基础设施和 checkpoint 产物完整；
2. 在 B=5 的本次三 seed 结果中，C005 `(2,3)` 的 final mean 最高；
3. 在 B=16 的本次三 seed 结果中，C010 `(5,11)` 的 final mean 最高；
4. 两个预算组的较高均值都出现在 low_actor 获得相对较多 computation depth 的
   条件中；
5. 这一趋势尚不足以作为统计显著性或普适机制结论；
6. C007 的 best 与 final 差距、C006 的 seed variation、task2 的低均值与高
   波动，说明 checkpoint selection、seed noise 和 task composition 都需要
   在后续分析中单独处理；
7. 500k final checkpoint 是预先指定的 primary metric，不能因为 best checkpoint
   更高就把 primary 结论替换成 best 结论。

建议的后续工作：

- 对 final checkpoint 做更高 episode 数的独立 evaluation；
- 保留每个 task 的 episode-level 原始结果，而不只保留 task mean；
- 使用更多 seeds 或至少对当前 seed 进行预注册的统计检验；
- 在其他 OGBench 环境复核 C005/C010 的 low-actor-heavy 趋势；
- 单独分析 task2 的成功率和失败轨迹；
- 将 wall-clock 作为资源记录，不把它直接当成科学性能结论。

---

## 13. 原始文件索引

以下路径均为原始正式实验文件的绝对路径。`bytes`、`lines` 和 SHA-256 是对
原文件的审计索引；没有对原文件进行重写。由于 33 份 `train.csv` 合计约
1.92MB，本报告不复制粘贴其全部内容，而保留完整原始路径和哈希，原文件是
训练过程的唯一权威文本来源。

通用路径前缀：

```text
/data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M10A/
```

每个 run 目录都包含：

```text
eval.csv
summary.json
train.csv
runtime_metadata.json
resolved_config.json
checkpoints/params_500000.pkl
```

### 13.1 每个 run 的原始文件大小和行数

| config | seed | eval bytes/lines | summary bytes/lines | train bytes/lines | metadata bytes/lines | resolved bytes/lines | checkpoint bytes |
|---|---:|---:|---:|---:|---:|---:|---:|
| C001 | 0 | 334/6 | 150/7 | 58211/101 | 4456/143 | 5015/194 | 46472174 |
| C001 | 1 | 392/6 | 178/7 | 58201/101 | 4456/143 | 5015/194 | 46472174 |
| C001 | 2 | 376/6 | 178/7 | 58229/101 | 4456/143 | 5015/194 | 46472174 |
| C002 | 0 | 364/6 | 178/7 | 58181/101 | 5473/179 | 6044/225 | 46476891 |
| C002 | 1 | 361/6 | 178/7 | 58100/101 | 5473/179 | 6044/225 | 46476891 |
| C002 | 2 | 361/6 | 150/7 | 58147/101 | 5473/179 | 6044/225 | 46476891 |
| C003 | 0 | 368/6 | 164/7 | 58215/101 | 5477/179 | 6040/225 | 46476891 |
| C003 | 1 | 363/6 | 178/7 | 58071/101 | 5477/179 | 6040/225 | 46476891 |
| C003 | 2 | 345/6 | 164/7 | 58162/101 | 5477/179 | 6040/225 | 46476891 |
| C004 | 0 | 347/6 | 150/7 | 58155/101 | 5481/179 | 6040/225 | 46476891 |
| C004 | 1 | 372/6 | 150/7 | 58096/101 | 5481/179 | 6040/225 | 46476891 |
| C004 | 2 | 364/6 | 164/7 | 58210/101 | 5481/179 | 6040/225 | 46476891 |
| C005 | 0 | 377/6 | 178/7 | 58178/101 | 5481/179 | 6040/225 | 46476891 |
| C005 | 1 | 373/6 | 163/7 | 58219/101 | 5481/179 | 6040/225 | 46476891 |
| C005 | 2 | 361/6 | 163/7 | 58158/101 | 5481/179 | 6040/225 | 46476891 |
| C006 | 0 | 362/6 | 164/7 | 58172/101 | 5477/179 | 6040/225 | 46476891 |
| C006 | 1 | 345/6 | 164/7 | 58129/101 | 5477/179 | 6040/225 | 46476891 |
| C006 | 2 | 394/6 | 178/7 | 58166/101 | 5477/179 | 6040/225 | 46476891 |
| C007 | 0 | 362/6 | 164/7 | 58090/101 | 5486/179 | 6054/225 | 46476893 |
| C007 | 1 | 365/6 | 178/7 | 58053/101 | 5486/179 | 6054/225 | 46476893 |
| C007 | 2 | 394/6 | 178/7 | 58115/101 | 5486/179 | 6054/225 | 46476893 |
| C008 | 0 | 367/6 | 178/7 | 58145/101 | 5490/179 | 6054/225 | 46476893 |
| C008 | 1 | 365/6 | 148/7 | 58088/101 | 5490/179 | 6054/225 | 46476893 |
| C008 | 2 | 380/6 | 164/7 | 58157/101 | 5490/179 | 6054/225 | 46476893 |
| C009 | 0 | 374/6 | 149/7 | 58084/101 | 5483/179 | 6052/225 | 46476892 |
| C009 | 1 | 377/6 | 164/7 | 58099/101 | 5483/179 | 6052/225 | 46476892 |
| C009 | 2 | 375/6 | 178/7 | 58126/101 | 5483/179 | 6052/225 | 46476892 |
| C010 | 0 | 361/6 | 163/7 | 58070/101 | 5490/179 | 6054/225 | 46476893 |
| C010 | 1 | 377/6 | 178/7 | 58069/101 | 5490/179 | 6054/225 | 46476893 |
| C010 | 2 | 368/6 | 178/7 | 58035/101 | 5490/179 | 6054/225 | 46476893 |
| C011 | 0 | 370/6 | 164/7 | 58073/101 | 5486/179 | 6054/225 | 46476893 |
| C011 | 1 | 379/6 | 178/7 | 58081/101 | 5486/179 | 6054/225 | 46476893 |
| C011 | 2 | 372/6 | 163/7 | 58157/101 | 5486/179 | 6054/225 | 46476893 |

---

## 14. 原始 `eval.csv` 与 `summary.json`

以下代码块逐字保留正式 run 的两个主要结果文件。代码块中的内容不包含均值、
排序、平滑或四舍五入。

### M10A-C001 — vanilla

#### seed 0 — `eval.csv`

```csv
evaluation/task1_success,evaluation/task2_success,evaluation/task3_success,evaluation/task4_success,evaluation/task5_success,evaluation/overall_success,step
0.85,0.0,0.9,0.8,0.7,0.65,100000
0.85,0.0,0.95,0.95,0.8,0.71,200000
0.9,0.0,0.9,0.95,0.95,0.74,300000
0.95,0.15,1.0,1.0,1.0,0.82,400000
0.8,0.25,0.85,0.85,1.0,0.75,500000
```

```json
{
  "best_step": 400000,
  "best_success": 0.82,
  "final_success": 0.75,
  "status": "completed",
  "success_column": "evaluation/overall_success"
}
```

#### seed 1 — `eval.csv`

```csv
evaluation/task1_success,evaluation/task2_success,evaluation/task3_success,evaluation/task4_success,evaluation/task5_success,evaluation/overall_success,step
0.7,0.0,0.9,0.75,0.2,0.51,100000
0.95,0.05,0.95,0.9,0.8,0.7300000000000001,200000
0.95,0.2,0.95,0.9,0.9,0.7799999999999999,300000
0.95,0.45,0.95,0.9,0.95,0.8399999999999999,400000
0.95,0.45,1.0,1.0,0.95,0.8699999999999999,500000
```

```json
{
  "best_step": 500000,
  "best_success": 0.8699999999999999,
  "final_success": 0.8699999999999999,
  "status": "completed",
  "success_column": "evaluation/overall_success"
}
```

#### seed 2 — `eval.csv`

```csv
evaluation/task1_success,evaluation/task2_success,evaluation/task3_success,evaluation/task4_success,evaluation/task5_success,evaluation/overall_success,step
0.75,0.0,0.95,0.85,0.9,0.69,100000
0.8,0.25,1.0,0.95,0.85,0.77,200000
0.95,0.4,0.95,0.95,0.9,0.8300000000000001,300000
1.0,0.6,1.0,0.9,0.9,0.8800000000000001,400000
0.95,0.5,0.95,0.9,0.9,0.8400000000000001,500000
```

```json
{
  "best_step": 400000,
  "best_success": 0.8800000000000001,
  "final_success": 0.8400000000000001,
  "status": "completed",
  "success_column": "evaluation/overall_success"
}
```

### M10A-C002 — `(K_H,K_L)=(1,1)`

#### seed 0 — `eval.csv`

```csv
evaluation/task1_success,evaluation/task2_success,evaluation/task3_success,evaluation/task4_success,evaluation/task5_success,evaluation/overall_success,step
0.8,0.15,1.0,0.9,0.9,0.75,100000
1.0,0.1,0.95,0.95,0.95,0.79,200000
1.0,0.15,1.0,0.95,1.0,0.82,300000
0.95,0.5,0.95,0.85,0.95,0.8400000000000001,400000
1.0,0.65,0.95,0.9,0.85,0.8699999999999999,500000
```

```json
{
  "best_step": 500000,
  "best_success": 0.8699999999999999,
  "final_success": 0.8699999999999999,
  "status": "completed",
  "success_column": "evaluation/overall_success"
}
```

#### seed 1 — `eval.csv`

```csv
evaluation/task1_success,evaluation/task2_success,evaluation/task3_success,evaluation/task4_success,evaluation/task5_success,evaluation/overall_success,step
0.95,0.0,0.9,0.7,0.95,0.7,100000
0.8,0.2,0.95,0.8,0.9,0.73,200000
1.0,0.1,0.9,0.95,0.8,0.75,300000
0.95,0.35,0.85,0.95,0.95,0.8099999999999999,400000
1.0,0.15,0.95,0.9,1.0,0.7999999999999999,500000
```

```json
{
  "best_step": 400000,
  "best_success": 0.8099999999999999,
  "final_success": 0.7999999999999999,
  "status": "completed",
  "success_column": "evaluation/overall_success"
}
```

#### seed 2 — `eval.csv`

```csv
evaluation/task1_success,evaluation/task2_success,evaluation/task3_success,evaluation/task4_success,evaluation/task5_success,evaluation/overall_success,step
0.85,0.0,0.95,0.8,0.9,0.7,100000
1.0,0.0,1.0,0.95,0.85,0.76,200000
0.8,0.5,1.0,0.95,0.9,0.8300000000000001,300000
0.85,0.75,0.9,0.8,0.75,0.8099999999999999,400000
1.0,0.85,1.0,0.95,0.95,0.95,500000
```

```json
{
  "best_step": 500000,
  "best_success": 0.95,
  "final_success": 0.95,
  "status": "completed",
  "success_column": "evaluation/overall_success"
}
```

### M10A-C003 — `(K_H,K_L)=(4,1)`

#### seed 0 — `eval.csv`

```csv
evaluation/task1_success,evaluation/task2_success,evaluation/task3_success,evaluation/task4_success,evaluation/task5_success,evaluation/overall_success,step
0.85,0.15,1.0,0.85,0.95,0.76,100000
0.95,0.0,0.85,0.85,0.95,0.72,200000
0.95,0.55,0.9,0.85,0.95,0.8400000000000001,300000
0.8,0.75,0.95,0.95,1.0,0.89,400000
0.85,0.7,0.8,0.95,0.9,0.8400000000000001,500000
```

```json
{
  "best_step": 400000,
  "best_success": 0.89,
  "final_success": 0.8400000000000001,
  "status": "completed",
  "success_column": "evaluation/overall_success"
}
```

#### seed 1 — `eval.csv`

```csv
evaluation/task1_success,evaluation/task2_success,evaluation/task3_success,evaluation/task4_success,evaluation/task5_success,evaluation/overall_success,step
0.65,0.45,0.9,0.9,0.95,0.7699999999999999,100000
0.8,0.0,0.95,0.85,0.95,0.71,200000
0.95,0.0,0.9,0.95,0.9,0.74,300000
0.95,0.4,0.95,1.0,1.0,0.86,400000
1.0,0.7,0.95,0.95,1.0,0.9199999999999999,500000
```

```json
{
  "best_step": 500000,
  "best_success": 0.9199999999999999,
  "final_success": 0.9199999999999999,
  "status": "completed",
  "success_column": "evaluation/overall_success"
}
```

#### seed 2 — `eval.csv`

```csv
evaluation/task1_success,evaluation/task2_success,evaluation/task3_success,evaluation/task4_success,evaluation/task5_success,evaluation/overall_success,step
0.8,0.1,0.95,0.9,1.0,0.75,100000
0.95,0.15,1.0,1.0,1.0,0.82,200000
0.95,0.2,0.9,0.95,1.0,0.8,300000
1.0,0.3,0.95,0.95,0.9,0.8200000000000001,400000
0.75,0.2,0.8,1.0,0.85,0.72,500000
```

```json
{
  "best_step": 400000,
  "best_success": 0.8200000000000001,
  "final_success": 0.72,
  "status": "completed",
  "success_column": "evaluation/overall_success"
}
```

### M10A-C004 — `(K_H,K_L)=(3,2)`

#### seed 0 — `eval.csv`

```csv
evaluation/task1_success,evaluation/task2_success,evaluation/task3_success,evaluation/task4_success,evaluation/task5_success,evaluation/overall_success,step
0.9,0.15,1.0,0.95,0.9,0.78,100000
0.9,0.1,0.95,0.85,0.7,0.7,200000
0.95,0.4,0.9,0.95,0.95,0.8300000000000001,300000
0.85,0.5,1.0,0.95,1.0,0.86,400000
0.9,0.45,0.9,0.9,0.95,0.82,500000
```

```json
{
  "best_step": 400000,
  "best_success": 0.86,
  "final_success": 0.82,
  "status": "completed",
  "success_column": "evaluation/overall_success"
}
```

#### seed 1 — `eval.csv`

```csv
evaluation/task1_success,evaluation/task2_success,evaluation/task3_success,evaluation/task4_success,evaluation/task5_success,evaluation/overall_success,step
0.95,0.0,0.9,0.8,0.9,0.7100000000000001,100000
0.95,0.2,0.95,0.9,0.8,0.76,200000
0.9,0.3,0.9,0.8,0.85,0.7500000000000001,300000
0.85,0.3,0.95,1.0,0.9,0.7999999999999999,400000
0.9,0.6,1.0,0.95,1.0,0.89,500000
```

```json
{
  "best_step": 500000,
  "best_success": 0.89,
  "final_success": 0.89,
  "status": "completed",
  "success_column": "evaluation/overall_success"
}
```

#### seed 2 — `eval.csv`

```csv
evaluation/task1_success,evaluation/task2_success,evaluation/task3_success,evaluation/task4_success,evaluation/task5_success,evaluation/overall_success,step
0.75,0.05,0.9,0.8,0.95,0.6900000000000001,100000
0.9,0.15,0.9,0.8,0.85,0.72,200000
0.9,0.4,0.95,0.95,1.0,0.8400000000000001,300000
0.85,0.2,0.95,0.85,0.95,0.76,400000
0.95,0.2,1.0,1.0,0.95,0.82,500000
```

```json
{
  "best_step": 300000,
  "best_success": 0.8400000000000001,
  "final_success": 0.82,
  "status": "completed",
  "success_column": "evaluation/overall_success"
}
```

### M10A-C005 — `(K_H,K_L)=(2,3)`

#### seed 0 — `eval.csv`

```csv
evaluation/task1_success,evaluation/task2_success,evaluation/task3_success,evaluation/task4_success,evaluation/task5_success,evaluation/overall_success,step
0.6,0.7,0.9,0.7,0.65,0.7099999999999999,100000
0.85,0.35,1.0,0.95,0.75,0.78,200000
0.9,0.55,0.95,0.9,1.0,0.8600000000000001,300000
0.9,0.45,1.0,0.95,0.8,0.82,400000
0.95,0.6,0.95,0.85,1.0,0.8699999999999999,500000
```

```json
{
  "best_step": 500000,
  "best_success": 0.8699999999999999,
  "final_success": 0.8699999999999999,
  "status": "completed",
  "success_column": "evaluation/overall_success"
}
```

#### seed 1 — `eval.csv`

```csv
evaluation/task1_success,evaluation/task2_success,evaluation/task3_success,evaluation/task4_success,evaluation/task5_success,evaluation/overall_success,step
1.0,0.3,1.0,0.8,0.7,0.76,100000
0.95,0.35,0.95,0.9,0.9,0.8099999999999999,200000
0.75,0.8,0.85,0.85,0.9,0.8300000000000001,300000
1.0,0.75,1.0,1.0,0.8,0.9099999999999999,400000
0.9,0.9,0.95,0.9,0.85,0.9,500000
```

```json
{
  "best_step": 400000,
  "best_success": 0.9099999999999999,
  "final_success": 0.9,
  "status": "completed",
  "success_column": "evaluation/overall_success"
}
```

#### seed 2 — `eval.csv`

```csv
evaluation/task1_success,evaluation/task2_success,evaluation/task3_success,evaluation/task4_success,evaluation/task5_success,evaluation/overall_success,step
0.8,0.1,0.95,0.9,0.8,0.71,100000
1.0,0.15,1.0,0.95,0.85,0.7899999999999999,200000
0.95,0.7,1.0,0.95,1.0,0.9199999999999999,300000
1.0,0.55,1.0,0.8,0.9,0.85,400000
0.95,0.75,0.95,0.95,0.9,0.9,500000
```

```json
{
  "best_step": 300000,
  "best_success": 0.9199999999999999,
  "final_success": 0.9,
  "status": "completed",
  "success_column": "evaluation/overall_success"
}
```

### M10A-C006 — `(K_H,K_L)=(1,4)`

#### seed 0 — `eval.csv`

```csv
evaluation/task1_success,evaluation/task2_success,evaluation/task3_success,evaluation/task4_success,evaluation/task5_success,evaluation/overall_success,step
0.95,0.2,0.9,0.95,0.8,0.76,100000
0.9,0.8,0.95,0.95,0.85,0.89,200000
0.95,0.3,0.95,0.95,0.95,0.8200000000000001,300000
1.0,0.3,1.0,1.0,0.95,0.85,400000
0.9,0.55,0.9,1.0,0.8,0.8300000000000001,500000
```

```json
{
  "best_step": 200000,
  "best_success": 0.89,
  "final_success": 0.8300000000000001,
  "status": "completed",
  "success_column": "evaluation/overall_success"
}
```

#### seed 1 — `eval.csv`

```csv
evaluation/task1_success,evaluation/task2_success,evaluation/task3_success,evaluation/task4_success,evaluation/task5_success,evaluation/overall_success,step
0.9,0.1,0.95,0.7,0.9,0.71,100000
0.9,0.0,0.95,0.9,0.85,0.72,200000
0.8,0.15,1.0,0.85,0.9,0.74,300000
0.8,0.15,1.0,0.95,0.9,0.76,400000
0.9,0.1,0.9,0.95,0.8,0.7299999999999999,500000
```

```json
{
  "best_step": 400000,
  "best_success": 0.76,
  "final_success": 0.7299999999999999,
  "status": "completed",
  "success_column": "evaluation/overall_success"
}
```

#### seed 2 — `eval.csv`

```csv
evaluation/task1_success,evaluation/task2_success,evaluation/task3_success,evaluation/task4_success,evaluation/task5_success,evaluation/overall_success,step
0.85,0.15,0.95,0.9,0.8,0.7300000000000001,100000
0.9,0.05,0.95,0.85,0.9,0.73,200000
0.7,0.25,0.95,1.0,0.95,0.7699999999999999,300000
0.95,0.6,0.95,0.75,0.9,0.8300000000000001,400000
0.8,0.85,0.95,0.9,0.95,0.8899999999999999,500000
```

```json
{
  "best_step": 500000,
  "best_success": 0.8899999999999999,
  "final_success": 0.8899999999999999,
  "status": "completed",
  "success_column": "evaluation/overall_success"
}
```

### M10A-C007 — `(K_H,K_L)=(15,1)`

#### seed 0 — `eval.csv`

```csv
evaluation/task1_success,evaluation/task2_success,evaluation/task3_success,evaluation/task4_success,evaluation/task5_success,evaluation/overall_success,step
0.95,0.4,0.9,0.9,0.85,0.8,100000
0.85,0.85,1.0,0.95,0.7,0.8700000000000001,200000
0.95,0.8,0.95,1.0,0.95,0.93,300000
0.9,1.0,0.95,1.0,1.0,0.97,400000
0.9,0.75,0.9,0.95,0.85,0.8699999999999999,500000
```

```json
{
  "best_step": 400000,
  "best_success": 0.97,
  "final_success": 0.8699999999999999,
  "status": "completed",
  "success_column": "evaluation/overall_success"
}
```

#### seed 1 — `eval.csv`

```csv
evaluation/task1_success,evaluation/task2_success,evaluation/task3_success,evaluation/task4_success,evaluation/task5_success,evaluation/overall_success,step
0.8,0.0,0.9,0.85,0.85,0.68,100000
0.85,0.2,0.95,1.0,0.85,0.77,200000
0.9,0.1,0.95,0.85,0.85,0.73,300000
0.9,0.35,1.0,0.85,0.95,0.8099999999999999,400000
0.95,0.2,0.95,0.95,1.0,0.8099999999999999,500000
```

```json
{
  "best_step": 400000,
  "best_success": 0.8099999999999999,
  "final_success": 0.8099999999999999,
  "status": "completed",
  "success_column": "evaluation/overall_success"
}
```

#### seed 2 — `eval.csv`

```csv
evaluation/task1_success,evaluation/task2_success,evaluation/task3_success,evaluation/task4_success,evaluation/task5_success,evaluation/overall_success,step
0.85,0.0,0.95,0.85,0.8,0.6900000000000001,100000
0.85,0.45,0.95,0.95,0.95,0.8300000000000001,200000
1.0,0.55,0.9,1.0,0.8,0.85,300000
0.9,0.6,0.95,0.95,0.95,0.8700000000000001,400000
0.85,0.7,0.95,0.8,0.75,0.8099999999999999,500000
```

```json
{
  "best_step": 400000,
  "best_success": 0.8700000000000001,
  "final_success": 0.8099999999999999,
  "status": "completed",
  "success_column": "evaluation/overall_success"
}
```

### M10A-C008 — `(K_H,K_L)=(11,5)`

#### seed 0 — `eval.csv`

```csv
evaluation/task1_success,evaluation/task2_success,evaluation/task3_success,evaluation/task4_success,evaluation/task5_success,evaluation/overall_success,step
0.85,0.0,1.0,1.0,0.95,0.76,100000
0.95,0.1,0.95,0.95,0.85,0.76,200000
0.95,0.65,0.95,0.8,0.85,0.8399999999999999,300000
0.85,0.35,1.0,0.85,0.9,0.79,400000
0.9,0.65,0.95,1.0,0.85,0.8699999999999999,500000
```

```json
{
  "best_step": 500000,
  "best_success": 0.8699999999999999,
  "final_success": 0.8699999999999999,
  "status": "completed",
  "success_column": "evaluation/overall_success"
}
```

#### seed 1 — `eval.csv`

```csv
evaluation/task1_success,evaluation/task2_success,evaluation/task3_success,evaluation/task4_success,evaluation/task5_success,evaluation/overall_success,step
0.9,0.2,0.85,0.75,0.75,0.6900000000000001,100000
0.95,0.4,1.0,1.0,0.85,0.8400000000000001,200000
0.85,0.3,0.9,0.95,0.9,0.78,300000
0.85,0.7,0.95,0.85,0.95,0.86,400000
0.85,0.95,0.95,0.8,0.95,0.9,500000
```

```json
{
  "best_step": 500000,
  "best_success": 0.9,
  "final_success": 0.9,
  "status": "completed",
  "success_column": "evaluation/overall_success"
}
```

#### seed 2 — `eval.csv`

```csv
evaluation/task1_success,evaluation/task2_success,evaluation/task3_success,evaluation/task4_success,evaluation/task5_success,evaluation/overall_success,step
0.8,0.35,0.95,0.85,0.95,0.7799999999999999,100000
0.9,0.45,0.85,0.85,0.8,0.7700000000000001,200000
0.9,0.6,0.9,0.9,0.95,0.85,300000
0.9,0.75,1.0,0.85,0.85,0.8699999999999999,400000
0.85,0.7,0.85,0.65,0.85,0.78,500000
```

```json
{
  "best_step": 400000,
  "best_success": 0.8699999999999999,
  "final_success": 0.78,
  "status": "completed",
  "success_column": "evaluation/overall_success"
}
```

### M10A-C009 — `(K_H,K_L)=(8,8)`

#### seed 0 — `eval.csv`

```csv
evaluation/task1_success,evaluation/task2_success,evaluation/task3_success,evaluation/task4_success,evaluation/task5_success,evaluation/overall_success,step
1.0,0.6,1.0,0.95,0.8,0.8699999999999999,100000
0.95,0.6,0.95,0.9,0.95,0.8699999999999999,200000
0.95,0.75,1.0,0.9,0.9,0.9,300000
0.9,0.6,0.95,0.8,0.8,0.8099999999999999,400000
0.95,0.8,0.95,0.75,0.8,0.85,500000
```

```json
{
  "best_step": 300000,
  "best_success": 0.9,
  "final_success": 0.85,
  "status": "completed",
  "success_column": "evaluation/overall_success"
}
```

#### seed 1 — `eval.csv`

```csv
evaluation/task1_success,evaluation/task2_success,evaluation/task3_success,evaluation/task4_success,evaluation/task5_success,evaluation/overall_success,step
0.9,0.0,0.95,0.95,0.95,0.75,100000
0.95,0.25,1.0,0.9,0.95,0.8099999999999999,200000
0.8,0.65,0.95,1.0,0.8,0.8400000000000001,300000
0.9,0.75,0.95,0.95,0.9,0.89,400000
0.9,0.45,1.0,0.9,0.9,0.8300000000000001,500000
```

```json
{
  "best_step": 400000,
  "best_success": 0.89,
  "final_success": 0.8300000000000001,
  "status": "completed",
  "success_column": "evaluation/overall_success"
}
```

#### seed 2 — `eval.csv`

```csv
evaluation/task1_success,evaluation/task2_success,evaluation/task3_success,evaluation/task4_success,evaluation/task5_success,evaluation/overall_success,step
0.85,0.1,1.0,0.7,0.7,0.6699999999999999,100000
0.9,0.35,0.9,0.9,0.95,0.8,200000
0.8,0.5,0.95,0.9,0.95,0.82,300000
1.0,0.65,1.0,0.95,0.8,0.8799999999999999,400000
0.95,0.55,0.9,0.95,0.85,0.8399999999999999,500000
```

```json
{
  "best_step": 400000,
  "best_success": 0.8799999999999999,
  "final_success": 0.8399999999999999,
  "status": "completed",
  "success_column": "evaluation/overall_success"
}
```

### M10A-C010 — `(K_H,K_L)=(5,11)`

#### seed 0 — `eval.csv`

```csv
evaluation/task1_success,evaluation/task2_success,evaluation/task3_success,evaluation/task4_success,evaluation/task5_success,evaluation/overall_success,step
0.9,0.25,0.9,0.9,0.75,0.74,100000
0.9,0.6,0.9,0.9,1.0,0.86,200000
0.9,0.65,1.0,0.95,1.0,0.9,300000
0.95,0.65,1.0,0.95,0.85,0.8799999999999999,400000
0.95,0.45,0.95,0.9,0.9,0.8299999999999998,500000
```

```json
{
  "best_step": 300000,
  "best_success": 0.9,
  "final_success": 0.8299999999999998,
  "status": "completed",
  "success_column": "evaluation/overall_success"
}
```

#### seed 1 — `eval.csv`

```csv
evaluation/task1_success,evaluation/task2_success,evaluation/task3_success,evaluation/task4_success,evaluation/task5_success,evaluation/overall_success,step
0.95,0.0,0.95,0.9,0.95,0.75,100000
0.95,0.5,1.0,1.0,1.0,0.89,200000
0.95,0.65,0.85,1.0,0.9,0.8700000000000001,300000
0.85,0.9,1.0,1.0,0.85,0.9199999999999999,400000
1.0,0.85,0.95,0.9,0.85,0.9099999999999999,500000
```

```json
{
  "best_step": 400000,
  "best_success": 0.9199999999999999,
  "final_success": 0.9099999999999999,
  "status": "completed",
  "success_column": "evaluation/overall_success"
}
```

#### seed 2 — `eval.csv`

```csv
evaluation/task1_success,evaluation/task2_success,evaluation/task3_success,evaluation/task4_success,evaluation/task5_success,evaluation/overall_success,step
0.8,0.15,0.85,0.95,0.95,0.74,100000
0.95,0.2,0.95,0.95,1.0,0.8099999999999999,200000
0.95,0.6,0.95,0.9,0.85,0.85,300000
0.85,0.25,0.85,0.9,0.85,0.74,400000
0.95,0.75,1.0,0.95,0.9,0.9100000000000001,500000
```

```json
{
  "best_step": 500000,
  "best_success": 0.9100000000000001,
  "final_success": 0.9100000000000001,
  "status": "completed",
  "success_column": "evaluation/overall_success"
}
```

### M10A-C011 — `(K_H,K_L)=(1,15)`

#### seed 0 — `eval.csv`

```csv
evaluation/task1_success,evaluation/task2_success,evaluation/task3_success,evaluation/task4_success,evaluation/task5_success,evaluation/overall_success,step
0.9,0.2,1.0,0.8,0.8,0.74,100000
0.9,0.4,0.95,0.8,0.9,0.7899999999999999,200000
0.95,0.3,0.95,1.0,0.9,0.8200000000000001,300000
0.95,0.4,1.0,0.9,1.0,0.85,400000
0.9,0.4,0.9,0.85,1.0,0.8100000000000002,500000
```

```json
{
  "best_step": 400000,
  "best_success": 0.85,
  "final_success": 0.8100000000000002,
  "status": "completed",
  "success_column": "evaluation/overall_success"
}
```

#### seed 1 — `eval.csv`

```csv
evaluation/task1_success,evaluation/task2_success,evaluation/task3_success,evaluation/task4_success,evaluation/task5_success,evaluation/overall_success,step
0.85,0.1,1.0,0.75,0.75,0.6900000000000001,100000
1.0,0.0,0.95,0.95,0.8,0.74,200000
0.85,0.3,1.0,0.9,0.85,0.78,300000
1.0,0.2,0.95,0.95,0.95,0.8099999999999999,400000
0.95,0.75,1.0,0.95,0.95,0.9200000000000002,500000
```

```json
{
  "best_step": 500000,
  "best_success": 0.9200000000000002,
  "final_success": 0.9200000000000002,
  "status": "completed",
  "success_column": "evaluation/overall_success"
}
```

#### seed 2 — `eval.csv`

```csv
evaluation/task1_success,evaluation/task2_success,evaluation/task3_success,evaluation/task4_success,evaluation/task5_success,evaluation/overall_success,step
0.85,0.35,1.0,0.9,0.6,0.74,100000
0.95,0.0,0.8,0.9,0.8,0.6900000000000001,200000
0.9,0.5,0.95,0.9,0.9,0.8299999999999998,300000
0.95,0.65,1.0,0.9,1.0,0.9,400000
0.75,0.85,1.0,0.9,0.9,0.8800000000000001,500000
```

```json
{
  "best_step": 400000,
  "best_success": 0.9,
  "final_success": 0.8800000000000001,
  "status": "completed",
  "success_column": "evaluation/overall_success"
}
```

## 15. 原始训练日志、metadata 与 checkpoint 的读取方式

本报告没有把 `train.csv` 的 1,918,442 个原始 bytes 重新排版到 Markdown 中，
原因是重新排版会改变原始文件的列顺序、换行和浮点文本。它们仍保存在第 13 节
给出的正式路径中。需要完整查看所有训练原始数据时，可以使用：

```bash
ROOT=/data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M10A
find "$ROOT" -path '*/seed_*/train.csv' -print | sort
find "$ROOT" -path '*/seed_*/train.csv' -print0 \
  | sort -z \
  | xargs -0 -n1 sh -c 'echo "===== $0 ====="; sed -n "1,999999p" "$0"'
```

原始 `runtime_metadata.json`、`resolved_config.json` 和 checkpoint 也没有被复制
或修改；它们的数量、大小、行数和路径均已在第 5、13 节审计。`runtime_metadata`
中最重要的运行时事实是：33/33 completed、GPU backend、相同 commit、clean
worktree、相同 dataset root、相同训练 protocol。

报告生成时没有修改正式实验目录，也没有重新运行训练、评估或 checkpoint。

## 16. 基于 `train.csv` 的训练过程补充分析

前面的 success-rate 分析回答的是“当前 checkpoint 在环境中表现如何”，而
`train.csv` 回答的是“优化过程发生了什么”。本节只对正式的 33 个
`train.csv` 做过程审计和低层次汇总；所有均值、差值和相关系数都属于派生统计，
原始浮点文本仍以第 13 节列出的文件为准。

### 16.1 `train.csv` 的记录语义与完整性

本次 M10A 的训练日志具有以下结构：

- 33 个文件、每个文件 100 条记录，共 3,300 条记录；
- 每个文件的 step 为 `5000, 10000, ..., 500000`，间隔恒为 5,000；
- 每条记录 34 个字段：1 个 step、2 个时间字段、3 个梯度字段、5 个
  training/high-actor 字段、5 个 training/low-actor 字段、4 个
  training/value 字段，以及对应的 14 个 validation 字段；
- 3,300 条记录中没有 NaN、正负无穷或无法解析的数值；
- `training/*` 是当前训练 batch 上的 update 信息，`validation/*` 是独立
  validation batch 上通过 `total_loss(..., grad_params=None)` 计算出的信息，
  不是环境 success-rate，也不是第 7 节的 `eval.csv`；
- 日志是每 5,000 step 的单个采样 batch 快照，不是 epoch 平均值。因此下面的
  “早期”与“后期”比较是日志窗口统计，不应解释成完整数据集上的 epoch 曲线。

这些语义可以由训练入口和 HIQL loss 定义直接确认：训练与 validation 的日志
分别在 `impls/main.py` 的训练循环中写入；`value_loss` 是两个 value head 的
expectile loss 之和；actor 的 `mse` 是 distribution mode 与目标之间的平方误差；
`grad/max`、`grad/min` 和 `grad/norm` 在 `TrainState.apply_loss_fn` 中由原始梯度
计算，并且 `grad/norm` 使用的是 `jnp.linalg.norm(flat, ord=1)`，即展平梯度的
L1 norm。它不能直接与 L2 gradient norm 混用。

### 16.2 全部 33 个 run 的训练过程概览

下表把每个 run 的前 10 条记录（5k–50k）和后 10 条记录（455k–500k）分别取均值，
再在 33 个 run 上汇总。`Δ` 为“后期均值 − 早期均值”。这些数值是为了看趋势，
不替代原始 `train.csv`。

| 指标 | 早期均值 | 后期均值 | Δ | 直接观察 |
|---|---:|---:|---:|---|
| training/high_actor/mse | 0.137210 | 0.071680 | -0.065530 | high actor 的行为拟合误差明显下降 |
| validation/high_actor/mse | 0.136973 | 0.093583 | -0.043390 | validation 也下降，但幅度小于 training |
| training/low_actor/mse | 0.068441 | 0.049110 | -0.019331 | low actor training 误差下降 |
| validation/low_actor/mse | 0.077344 | 0.076693 | -0.000651 | low actor validation 基本没有净改善 |
| training/high_actor/adv | 3.454190 | 4.599710 | +1.145520 | high-level advantage 逐步增大 |
| training/low_actor/adv | 0.341004 | 0.437950 | +0.096946 | low-level advantage 逐步增大 |
| training/high_actor/bc_log_prob | -9.875440 | -9.547790 | +0.327650 | log probability 变得不那么负 |
| training/low_actor/bc_log_prob | -7.625270 | -7.547950 | +0.077325 | 同方向，但变化较小 |
| training/value/value_loss | 1.032894 | 1.487177 | +0.454283 | 没有呈现传统意义上的 loss 下降 |
| validation/value/value_loss | 1.028178 | 1.719781 | +0.691603 | validation value loss 上升更明显 |
| training/value/v_mean | -29.055900 | -39.973300 | -10.917500 | value 输出均值持续向负值移动 |
| training/value/v_min | -69.776800 | -100.958000 | -31.181700 | value 输出下界移动幅度更大 |
| training/grad/norm | 22166.49 | 18804.03 | -3362.46 | 均值受尖峰影响；稳健中位数基本不变 |

关于最后一行，早期与后期的 `grad/norm` 中位数分别为 17,893.27 和
17,911.73，95 分位数分别为 25,188.64 和 24,993.23。因此“均值下降”不能简单
解释为梯度整体衰减；更准确的说法是，典型梯度规模大体稳定，但个别时间点存在
极端尖峰。

### 16.3 actor 的训练与 validation 动力学

#### high actor

high actor 的 training MSE 从早期均值 0.137210 降到后期 0.071680，降幅约为
47.8%；validation MSE 从 0.136973 降到 0.093583，降幅约为 31.7%。后期
validation-training gap 约为 `0.093583 - 0.071680 = 0.021903`。

这说明 high actor 的参数确实在学习，且 validation 仍然有一定改善；不过
training MSE 的改善并没有完全传递到 validation。由于每个 validation 点也只是
一个 validation batch 快照，不能仅凭该 gap 断言严重过拟合，但至少说明“训练误差
下降”不等价于“策略行为在所有 validation batch 上等比例改善”。

#### low actor

low actor 的 training MSE 从 0.068441 降到 0.049110，降幅约为 28.2%；但
validation MSE 只从 0.077344 变为 0.076693，净变化约为 -0.000651，基本处于
平台。后期 validation-training gap 约为 `0.076693 - 0.049110 = 0.027583`。

这是一条比 final success 更直接的训练过程信号：low actor 仍在把训练 batch 上的
行为拟合得更好，但 validation batch 上的对应误差没有同步改善。可能原因包括
行为数据覆盖限制、目标构造与训练/validation 分布差异、MSE 指标与最终动作质量
之间不完全一致，或 500k step 后主要进入局部平台。仅凭现有字段无法区分这些原因。

四个 actor 的 `std` 字段在 3,300 条记录中均恒等于 `1.0`。因此本实验的日志中
没有观测到 policy scale 的学习或收缩；这与 M10A 配置中的 constant-standard-
deviation 设定一致，不能把它当成额外的收敛证据。

### 16.4 value 学习过程：输出尺度在变化，loss 没有收敛到更低水平

value 相关字段呈现出与 actor MSE 不同的过程：

- training value loss：`1.032894 → 1.487177`；
- validation value loss：`1.028178 → 1.719781`；
- training `v_mean`：`-29.0559 → -39.9733`；
- training `v_min`：`-69.7768 → -100.9580`；
- training `v_max`：`0.137265 → 0.161508`，上界总体相对稳定。

因此，500k 内 value head 并未表现出“loss 单调下降并稳定”的传统收敛图景，
而更像是 value 输出分布持续向更负的范围移动，同时 expectile regression loss
在较高水平波动。validation value loss 后期比 training value loss 高约 0.233，
说明 value 的 validation gap 比 actor MSE 的 gap 更值得关注。

这里不能直接把 value loss 上升等同于算法失败。HIQL 的 value target 依赖 reward、
discount、target value 与 expectile 权重；`v_mean`/`v_min` 的负向移动可能同时反映
目标尺度变化。因此当前 `train.csv` 能严谨支持的结论是“value 学习仍在发生尺度和
误差变化，未显示出清晰的 loss 收敛”，但不能仅据此判断 value 估计已经发散。若要
进一步判断，需要额外记录 q target、q-v residual、expectile 正负样本比例以及
target network 的统计量。

### 16.5 按配置汇总的训练过程对比

下表仍使用“早期 10 点 → 后期 10 点”的每配置三 seed 均值。`H-tr/H-val` 分别是
high actor 的 training/validation MSE，`L-tr/L-val` 分别是 low actor 的
training/validation MSE，`V-tr/V-val` 是 value loss；最后一列是后期窗口的
`training/grad/norm` 均值。第 7 列的最终 success mean 仅作为对照，不是本表的
优化目标。

| 配置 | final success mean | H-tr | H-val | L-tr | L-val | V-tr | V-val | 后期 grad/norm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C001 vanilla | 0.8200 | 0.1412→0.0732 | 0.1436→0.0996 | 0.0710→0.0513 | 0.0784→0.0784 | 1.0256→1.3976 | 1.0640→1.6781 | 15222 |
| C002 (1,1) | 0.8733 | 0.1427→0.0771 | 0.1446→0.1009 | 0.0706→0.0511 | 0.0781→0.0791 | 1.0276→1.5824 | 0.9770→1.6919 | 15725 |
| C003 (4,1) | 0.8267 | 0.1331→0.0707 | 0.1355→0.0919 | 0.0709→0.0511 | 0.0781→0.0786 | 0.9828→1.5819 | 1.0279→1.6584 | 18308 |
| C004 (3,2) | 0.8433 | 0.1395→0.0755 | 0.1439→0.0951 | 0.0674→0.0487 | 0.0759→0.0777 | 0.9967→1.3935 | 1.0151→1.7423 | 19366 |
| C005 (2,3) | 0.8900 | 0.1277→0.0662 | 0.1332→0.0899 | 0.0661→0.0475 | 0.0758→0.0776 | 1.1676→1.3672 | 1.1539→1.7496 | 17108 |
| C006 (1,4) | 0.8167 | 0.1331→0.0727 | 0.1371→0.0975 | 0.0667→0.0468 | 0.0764→0.0772 | 0.9951→1.4075 | 0.9754→1.7446 | 16487 |
| C007 (15,1) | 0.8300 | 0.1190→0.0723 | 0.1206→0.0922 | 0.0704→0.0513 | 0.0779→0.0799 | 0.9842→1.4982 | 0.9875→1.6173 | 21921 |
| C008 (11,5) | 0.8500 | 0.1388→0.0731 | 0.1424→0.0932 | 0.0663→0.0464 | 0.0761→0.0747 | 1.0397→1.5459 | 1.0116→1.7035 | 22536 |
| C009 (8,8) | 0.8400 | 0.1303→0.0672 | 0.1345→0.0864 | 0.0673→0.0470 | 0.0778→0.0746 | 1.0053→1.5642 | 1.0219→1.7652 | 22128 |
| C010 (5,11) | 0.8833 | 0.1671→0.0640 | 0.1311→0.0859 | 0.0680→0.0483 | 0.0782→0.0727 | 1.1045→1.5151 | 1.0375→1.8523 | 20574 |
| C011 (1,15) | 0.8700 | 0.1367→0.0764 | 0.1402→0.0969 | 0.0681→0.0507 | 0.0781→0.0729 | 1.0327→1.5055 | 1.0383→1.7145 | 17469 |

从过程角度可以看到：

1. 各配置的 actor MSE 变化方向高度一致，配置间差异主要体现在幅度和
   validation gap，而不是完全不同的训练形态；
2. C005 的后期 high/low training MSE 较低，且最终 success mean 也是本实验最高，
   但它的 validation value loss 仍从 1.1539 上升到 1.7496，不能把 success 的
   提升简单归因于 value loss 更低；
3. C010 的后期 actor MSE 也较低，但 validation value loss 后期均值最高，为
   1.8523，并且存在一次极端 gradient spike。因此 C010 的高 success mean 与
   训练稳定性之间并非简单的单调关系；
4. B=16 配置的后期典型梯度规模略高于 B=5 配置，但这个差异容易被异常点影响，
   应优先比较中位数和分位数，而不是只比较均值。

### 16.6 梯度尖峰与训练稳定性

由于 `grad/norm` 是原始展平梯度的 L1 norm，本次 3,300 条记录的统计为：中位数
约 19,666，90 分位数约 25,120，95 分位数约 27,391，最大值为 1,126,891。
按阈值统计：

- `grad/norm > 30,000`：91 条记录，涉及 24 个 run；
- `grad/norm > 50,000`：12 条记录，涉及 7 个 run；
- `grad/norm > 100,000`：4 条记录，涉及 4 个 run；
- `grad/norm > 1,000,000`：1 条记录。

超过 100,000 的四个原始位置如下：

| 配置 | seed | step | grad/norm | grad/max | grad/min |
|---|---:|---:|---:|---:|---:|
| C010 (5,11) | 2 | 35,000 | 1,126,891.0 | 2251.1123 | -2014.8721 |
| C007 (15,1) | 2 | 330,000 | 193,113.25 | 240.7110 | -203.6128 |
| C007 (15,1) | 1 | 340,000 | 104,098.13 | 110.4440 | -129.1461 |
| C005 (2,3) | 1 | 5,000 | 103,008.24 | 155.4076 | -117.3309 |

第一行的 C010/seed 2 尖峰是最明确的局部异常：30k、35k、40k 的
`training/grad/norm` 分别为 19,647.98、1,126,891.0、17,053.21；同一时刻
high actor training MSE 为 1.335611，而相邻两点分别为 0.109157 和 0.085633。
这说明它是一个非常短暂的异常点，后续日志没有持续发散。另一方面，C007 在
330k–355k 附近出现多次高梯度记录，属于比 C010 单点尖峰更值得继续检查的局部
不稳定区间。

这些尖峰不应被静默删除或用均值“修正”。当前报告只将它们标出；如果后续需要
重新评估训练稳定性，建议同时保存梯度裁剪前后统计、update norm、optimizer
step/scale，以及发生尖峰时的 batch/target 分布。

### 16.7 训练过程指标与最终 success 的探索性关系

为避免把 33 个 seed 当成 33 个独立算法结论，下面只报告探索性 Pearson 相关，
不做因果解释。以每个 run 的最终 success 为响应变量时，部分后期窗口指标的
相关系数如下：

| 指标 | run-level n=33 | config-mean n=11 | 说明 |
|---|---:|---:|---|
| training/high_actor/mse | 0.1780 | -0.2727 | 关系弱且方向不稳定 |
| validation/high_actor/mse | 0.1493 | -0.2688 | 不能作为 success 的单独预测量 |
| training/low_actor/mse | -0.1084 | -0.1288 | 几乎没有稳定线性关系 |
| validation/low_actor/mse | -0.2981 | -0.4295 | config 均值下有一定负相关，但样本仅 11 个配置 |
| training/value/value_loss | 0.0404 | 0.0703 | 基本没有线性关系 |
| validation/value/value_loss | -0.1973 | 0.5042 | 两种聚合层级方向相反，不能过度解释 |
| training/grad/norm | 0.0503 | -0.0151 | 典型后期梯度规模几乎不能解释最终 success |

这组结果说明，最终 success 不是任意单个训练字段的简单函数。尤其是 value
validation loss 的 run-level 与 config-level 相关方向不同，正是小样本和聚合方式
影响的例子。当前更可靠的结论是：train.csv 能揭示优化是否仍在变化、哪些指标
出现 train/validation 分离和异常点，但不能用一个低 loss 或低 gradient norm 直接
替代环境评估。

### 16.8 对“是否训练充分”的初步判断

基于训练过程本身，当前 M10A 可以作出比仅看最终 success 更细的判断：

1. actor 的行为拟合仍然有清晰的学习轨迹，尤其是 high actor training MSE；
2. low actor 的 training MSE 还在改善，但 validation MSE 已基本平台化，说明
   继续训练不一定会带来同等比例的泛化收益；
3. value loss 与 value 输出尺度仍在明显变化，不能把 500k 视为 value 侧已经
   充分收敛；
4. 梯度的典型规模没有随训练明显衰减，且存在少数极端尖峰，所以“训练未完成”、
   “训练目标仍在漂移”和“局部数值不稳定”需要区分，不能仅用 success 曲线判断；
5. 因此，若 M10A 的研究问题是比较 computation placement，现阶段较稳妥的结论
   是：配置间 actor 训练趋势总体相似，最终 success 的差异不能简单归因于某个
   actor MSE；value 学习状态和梯度尖峰是后续解释配置差异时必须保留的上下文。

### 16.9 建议的后续过程诊断

如果要把“训练是否充分”从初步判断推进到更强证据，最有价值的补充不是再次只看
500k 的 success，而是：

- 延长表现较好的代表配置和表现较差配置，比较 500k 之后 actor MSE、value loss
  与 success 是否继续变化；
- 为每个 log point 保存 value target、q1/q2、q-v residual 和 expectile 权重的
  分布，而不只保存其聚合 loss；
- 单独记录 raw gradient、optimizer update、gradient clipping/normalization 后
  的统计，定位 C007 和 C010 的尖峰是否来自特定模块或特定 batch；
- 对 `training/*` 与 `validation/*` 使用同一批固定 probe batch，降低当前每个点
  随机采样 batch 带来的噪声；
- 对关键配置增加 seed 或采用 checkpoint-level repeated evaluation，以区分
  训练动力学差异与三 seed 随机波动。
