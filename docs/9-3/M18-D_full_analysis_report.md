# M18-D：完整分析实验说明、结果与初步评估

日期：2026-09-03  
研究对象：M18，Puzzle-4x4，GCIQL  
本文用途：用于组会汇报、与其他研究者讨论 M18-D 的实验设计、结果和当前可以/不可以得出的结论。

## 1. 摘要

M18-D 是针对 M18 异常结果设计的一组后验机制诊断实验。M18 的主实验只改变共享 recurrent computation 的执行次数 K，结果显示 K=4 表现很好，而 K=8 长期表现很差。M18-D 进一步回答以下问题：

1. K 是否只是测试时可以自由调整的计算预算？
2. K=8 的隐藏状态是否发生了数值爆炸、过度平滑或更新停滞？
3. 深层状态经过 readout 后，是否真的改变了动作？
4. critic 是否把深层 actor 的动作评价得过高？
5. K=8 在真实环境中究竟是完全不会交互，还是能够局部进步但无法维持闭环？
6. K=8 的 actor 和 critic 是否形成了内部自洽、但与真实环境效果不一致的 action preference？

目前所有 M18-D 产物均已完成。综合结果可以概括为：

> K=8 的主要问题不是简单的 actor 数值爆炸或动作饱和，而是形成了一个相对稳定、但无法在真实 Puzzle 闭环中持续产生正确行为的深层 actor–critic 系统。D1 和 D5 对这一行为结论提供了较强证据；D2+ 和 D6 分别指出 mean pooling 信息瓶颈和 actor–critic co-adaptation 是值得进一步干预验证的候选机制，但二者目前都还不是因果结论。

## 2. M18 的基础实验设置

### 2.1 M18 研究的问题

M18 的问题是：

> 在结构化 MLP-Mixer block 容量和可训练参数量固定的情况下，增加共享 recurrent computation 的执行次数 K，是否能够改善 Puzzle-4x4 上的 GCIQL 性能？

M18 的主要配置来自：

- [study.yaml](../../experiments/M18_puzzle_recurrent_compute_scaling/study.yaml)
- [M18 配置目录](../../experiments/M18_puzzle_recurrent_compute_scaling/configs/)

主要固定设置如下：

| 项目 | 设置 |
|---|---|
| 算法 | GCIQL |
| 环境 | puzzle-4x4-play-v0 |
| training seed | 0 |
| train steps | 1,000,000 |
| batch size | 1024 |
| actor/value/critic | 三个 slot 都使用 structured computation |
| structure | puzzle_tokens |
| token 数 | 16 |
| token dim | 128 |
| computation block | MLP-Mixer |
| block 内部深度 L | 2 |
| topology | SingleState |
| state 初始化 | zero_buffer，即 Z0=0 |
| input mapping | identity |
| input injection | z+x |
| topology residual | false |
| recurrent 参数共享 | shared |
| readout | mean_context |
| alpha | 0.4 |
| 总可训练参数量 | 1,112,712 |

只有 recurrent execution budget K 改变：

| K | unique Mixer layers | 每次 forward 执行的 Mixer layers |
|---:|---:|---:|
| 1 | 2 | 2 |
| 2 | 2 | 4 |
| 4 | 2 | 8 |
| 8 | 2 | 16 |

因此 M18 比较的是“同一个共享计算块重复执行不同次数”，而不是比较不同参数量的网络。

单个 computation slot 可以抽象为：

~~~text
X = adapter(observation, goal)
Z^0 = 0
Z^(k+1) = B_theta(Z^k + X)
Y^K = readout(Z^K, context)
output = task head(Y^K)
~~~

其中 B_theta 是共享参数的两层 MLP-Mixer block。由于每一次更新都使用 Z^k + X，输入 X 会被重复注入每一个 recurrent iteration。

### 2.2 M18 主实验结果

M18 主实验的结果是 M18-D 的背景：

| K | best success | best step | final@1M | 解释 |
|---:|---:|---:|---:|---|
| 1 | 0.95 | 900k | 0.90 | 最终性能稳定 |
| 2 | 0.75 | 800k | 0.74 | 非单调地低于 K=1 和 K=4 |
| 4 | 0.98 | 900k | 0.89 | 学习速度最好，最终接近 K=1 |
| 8 | 0.12 | 200k | 0.03 | 从早期开始就没有形成有效策略 |

K=8 并不是在训练后期突然崩溃。它从 100k 到 1M 一直处于低成功率区间，因此需要进一步判断它是优化困难、深度不匹配，还是结构导致的表示/critic 问题。

## 3. M18-D 的整体流程

M18-D 的分析链路如下：

~~~text
M18 K4/K8 best checkpoint
        |
        +--> D1: actor-only K_train x K_actor_test rollout
        |
        +--> D2/D3/D4: same fixed offline batch
        |       |
        |       +--> D2: hidden-state geometry
        |       +--> D3: intermediate action refinement
        |       +--> D4: source critic action ranking
        |                |
        |                +--> D2+: mean-pooling retained-energy analysis
        |                +--> D6: cross actor x cross critic preference
        |
        +--> D5: native K4/K8 paired closed-loop rollout
        |
        +--> analyze_m18_d.py: final report and hypothesis table
~~~

所有 M18-D 诊断都满足以下性质：

- evaluation-only；
- 不执行 optimizer update；
- 不进行 finetuning；
- 不覆盖 source checkpoint；
- 对 checkpoint 做 hash 检查；
- D2/D3/D4/D6 使用固定 offline batch；
- D5 使用锁定的原始 checkpoint 和配对 episode seed。

目前完整产物统计如下：

| 产物 | 完成情况 |
|---|---:|
| D1 | 16 个 K_train × K_actor_test 单元 |
| D2/D3/D4 | 810 条 aggregate rows |
| D2+ | 144 条 retained-energy rows |
| D5 | 100 对 paired episodes，合计 200 条 model episodes |
| D6 | 固定 batch 1024 个样本 |

本次锁定的 source checkpoint 为：

| 模型 | checkpoint role | step | SHA256 |
|---|---|---:|---|
| K4 | best | 900,000 | 54fb1ce920e08cf9593acd36eb67f6421f0df35e089cbcb81d95190ca387d628 |
| K8 | best | 200,000 | 1cd4a2eb7f2f428aa4c51f6860a73ebd20a3bcbcc2c0b84b8a694f42e02b3743 |

## 3.1 诊断之间的数据依赖

D2/D3/D4 使用 diagnostic seed=18018、batch size=1024 的同一个固定 offline batch。D3 保存的 K4/K8 native clipped action 又被 D6 直接复用；D6 不重新 forward actor，因此 D6 的 a4/a8 与 D3 结果保持一致。

D5 是独立的真实环境配对 rollout，每个 task 20 个 episode，共 5 个 task、100 对 episode。它不使用 D2/D3/D4 的 fixed batch。

## 4. D1：actor inference-depth probe

### 4.1 D1 要回答什么

D1 检查训练时的 K 和测试时 actor 执行的 K 是否可以解耦。

例如，对于一个 K_train=4 的模型，D1 分别用：

- K_actor_test=1；
- K_actor_test=2；
- K_actor_test=4；
- K_actor_test=8；

进行测试。

每个 K_train × K_actor_test 单元包含 5 个 Puzzle task，每个 task 20 个 episode。

D1 只修改 actor 的 inference depth：

~~~python
config['compute']['actor']['topology_kwargs']['iterations'] = k_actor_test
~~~

value 和 critic 始终保持 K_train。由于 rollout 时动作由 actor 直接生成，所以 D1 主要测量 actor 对测试深度改变的敏感性。

实现位置：

- [m18_cross_k_eval.py](../../tools/m18_cross_k_eval.py)
- 修改 actor K 的位置约在 [m18_cross_k_eval.py:155](/home/eai/Research/RLC/tools/m18_cross_k_eval.py:155)

### 4.2 D1 结果

| K_train \ K_actor_test | 1 | 2 | 4 | 8 |
|---:|---:|---:|---:|---:|
| 1 | 0.87 | 0.00 | 0.00 | 0.00 |
| 2 | 0.00 | 0.75 | 0.00 | 0.00 |
| 4 | 0.00 | 0.00 | 0.93 | 0.00 |
| 8 | 0.00 | 0.00 | 0.00 | 0.15 |

所有非对角线结果都是 0。

### 4.3 D1 的含义

D1 说明当前网络不是一个可以在测试时任意改变计算深度的 depth-agnostic policy。模型学习到的是：

~~~text
特定训练 K
    -> 特定的 Z^K 表示分布
    -> 与该分布匹配的 readout 和 actor head
    -> 特定动作策略
~~~

它没有学习到：

~~~text
任意中间状态 Z^k
    -> 都可以被同一个 readout 正确解释
~~~

因此：

- K=4 模型不能通过测试时改成 K=1、2 或 8 来保持性能；
- K=8 模型也不能通过测试时改成 K=1、2 或 4 来恢复性能；
- K 不是当前模型中独立于训练的“额外测试计算预算”；
- 深度改变会造成表示分布和 readout 输入分布的变化。

这不能证明计算深度本身一定具有因果危害，因为 D1 仍然可能受到训练过程、checkpoint 质量和 actor head 共同影响。但它强烈证明当前模型的计算深度与训练结果是绑定的。

### 4.4 关于 checkpoint 口径的说明

实际 M18-D 产物位于 checkpoint_best 目录，D1 使用的是 best checkpoint。study.yaml 的 cross_k_diagnostic 描述中仍有 checkpoint_selector=last 的旧文本，这是元数据没有同步更新，不应覆盖实际产物中的 checkpoint_best 证据。

D1 原始结果：

- [D1 success matrix](/data/qijunrong/06-RL/offline-rl/exp/RLC/diagnostics/M18D/summary/checkpoint_best/m18d_cross_k_success.csv)
- [D1 delta summary](/data/qijunrong/06-RL/offline-rl/exp/RLC/diagnostics/M18D/summary/checkpoint_best/m18d_cross_k_delta.csv)

## 5. D2：hidden-state geometry

### 5.1 D2 要回答什么

D2 在同一个固定 offline batch 上追踪每个 computation slot 的中间状态：

~~~text
Z^0, Z^1, Z^2, ..., Z^8
~~~

分析的 slot 包括：

- actor；
- value；
- critic 的两个 ensemble member。

实现位于：

- [m18_trace_diagnostics.py](../../tools/m18_trace_diagnostics.py)
- 状态指标构造约在 [m18_trace_diagnostics.py:278](/home/eai/Research/RLC/tools/m18_trace_diagnostics.py:278)

主要指标：

| 指标 | 含义 |
|---|---|
| state_rms | 当前 hidden state 的 RMS 尺度 |
| relative_update_from_previous | 相邻状态更新量相对于上一状态的比例 |
| state_cosine_from_previous | 相邻状态方向相似度 |
| token_variance | token 相对样本内 token 均值的方差 |
| pairwise_token_cosine | token 两两之间的平均 cosine |
| readout_rms | 当前中间状态经过正常 readout 后的输出 RMS |

由于 Z0=0，k=0 和 k=1 的 relative update/cosine 不应被当作正常的收敛统计。

### 5.2 Actor state 结果

#### K=4 checkpoint

| 指标 | k=1 | k=4，训练深度 | k=8，深度外推 |
|---|---:|---:|---:|
| state RMS | 0.3531 | 0.5444 | 0.8377 |
| relative update | — | 0.9516 | 0.9648 |
| state cosine | — | 0.5463 | 0.6516 |
| token variance | 0.0878 | 0.3024 | 0.6612 |
| pairwise token cosine | 0.3282 | 0.1045 | 0.1008 |
| readout RMS | 0.2798 | 0.2691 | 0.3153 |

K=4 在 k=4 之后继续执行到 k=8，状态尺度仍明显变化。说明超过训练深度以后，K=4 的 representation distribution 也不能保持不变。

#### K=8 checkpoint

| 指标 | k=1 | k=4 | k=8，训练深度 |
|---|---:|---:|---:|
| state RMS | 0.2649 | 0.5904 | 0.6739 |
| relative update | — | 0.6072 | 0.4907 |
| state cosine | — | 0.8788 | 0.8735 |
| token variance | 0.0507 | 0.3132 | 0.4273 |
| pairwise token cosine | 0.4467 | 0.0714 | 0.0413 |
| readout RMS | 0.2692 | 0.2597 | 0.2576 |

K=8 的 actor 结果有三个特点：

1. state RMS 从 0.2649 增加到 0.6739，但没有呈现无限发散；
2. relative update 逐渐下降到约 0.49；
3. 相邻状态 cosine 达到约 0.87，说明状态方向逐渐稳定。

因此不能把 K=8 的 actor 失败解释成简单的 hidden-state explosion。

同时，token variance 从 0.0507 增加到 0.4273，pairwise token cosine 从 0.4467 降到 0.0413。这说明 token 不是被压成完全相同的向量，而是越来越分化。

所以 K=8 更像是：

> token representation 逐渐分化，整体状态逐渐稳定，并可能收敛到一个任务上错误的 attractor。

这里的“错误 attractor”是对 D2 和 D5 的联合解释，不是 D2 单独能够证明的因果结论。

### 5.3 value 和 critic 的补充观察

K=8 的 value slot 在 k=8 时有较大的内部尺度：

- value state RMS 约为 3.38；
- value token variance 约为 8.46。

K=8 critic 的 state RMS 约为 1.64–1.75，token variance 约为 1.98–2.06。

这提示 K=8 的 value/critic 表示存在明显的 scale growth，可能影响训练期间的 value target、advantage 和 actor gradient。

但是不能直接把这些数值称为数值发散，因为：

- relative update 仍然下降；
- cosine 仍然较高；
- value、critic、actor 的状态尺度不能跨 slot 直接比较；
- K4 value 在未训练的 k=8 外推处同样会出现非常大的状态尺度。

更稳妥的说法是：K=8 的 value/critic 路径存在值得进一步干预检查的尺度问题，但当前没有足够证据把它确定为失败根因。

## 6. D2+：mean-pooling retained energy

### 6.1 D2+ 要回答什么

D2+ 是在 D2 trace 上进行的后处理，用来检查 token mean 在整个 hidden state 中保留了多少能量。

定义为：

~~~text
rho(i,k)
  = mean_token_rms(i,k)^2
    / (state_rms(i,k)^2 + 1e-8)

discarded_energy = 1 - rho
~~~

rho 表示平均 token 成分所占的 state energy fraction，不是：

- task information；
- mutual information；
- policy information；
- 因果影响大小。

### 6.2 结果

| source K | depth | retained energy rho | discarded energy |
|---:|---:|---:|---:|
| K4 | k=1 | 0.3344 | 0.6656 |
| K4 | k=4 | 0.1217 | 0.8783 |
| K4 | k=8 | 0.1202 | 0.8798 |
| K8 | k=1 | 0.4203 | 0.5797 |
| K8 | k=4 | 0.1152 | 0.8848 |
| K8 | k=8 | 0.0981 | 0.9019 |

K8 在 native k=8 时，mean token 只保留约 9.8% 的 state energy。与此同时，token variance 从 0.0507 增长到 0.4273。

这支持以下候选解释：

~~~text
computation 越深
    -> token-specific difference 越强
    -> mean pooling 仍然只保留 token average
    -> 一部分深层 token 差异无法进入 readout
~~~

但 K4 在 native k=4 时 retained energy 也只有约 12.2%，却能够达到 0.93 的 D5 success。因此不能说“rho 低就一定失败”，也不能确定 mean pooling 是唯一根因。

当前准确结论是：

> mean_context readout 与深层 token differentiation 之间存在结构性不匹配的可能性；它是一个中等强度的候选机制，需要通过替换 readout 的干预实验验证。

## 7. D3：intermediate action refinement

### 7.1 D3 要回答什么

D3 把 actor 在每个中间 computation depth 产生的确定性动作保存下来，比较：

- 动作是否随着深度大幅漂移；
- 动作是否逐渐不再变化；
- 动作是否接近边界或饱和；
- 动作是否远离 dataset action。

主要指标：

| 指标 | 含义 |
|---|---|
| action_delta_from_previous | a^k 与 a^(k-1) 的 RMS 差异 |
| action_drift_from_k1 | a^k 与 a^1 的 RMS 差异 |
| dataset_action_mse | 当前动作与 dataset action 的 MSE |
| action_mean_saturation_fraction | 动作均值接近 ±1 的比例 |
| action_near_boundary_fraction | 动作接近动作边界的比例 |

实现位置：

- [m18_trace_diagnostics.py](../../tools/m18_trace_diagnostics.py)
- 动作指标构造约在 [m18_trace_diagnostics.py:324](/home/eai/Research/RLC/tools/m18_trace_diagnostics.py:324)

### 7.2 关键结果

| source K | depth | action delta | drift from k=1 | dataset action MSE | saturation | near boundary |
|---:|---:|---:|---:|---:|---:|---:|
| K4 | k=4 | 0.0618 | 0.1250 | 0.0227 | 0.0176 | 0.0398 |
| K4 | k=8 | 0.0950 | 0.1560 | 0.0442 | 0.0361 | 0.0506 |
| K8 | k=4 | 0.0354 | 0.0788 | 0.0309 | 0.0184 | 0.0410 |
| K8 | k=8 | 0.0263 | 0.1193 | 0.0262 | 0.0174 | 0.0396 |

K8 在 native k=8 时的 action delta 只有约 0.0263，反而小于 K4 native k=4 的约 0.0618。它没有出现明显的动作爆炸，也没有比 K4 更严重的 saturation。

这与 D5 组合后说明：

> K8 更像是产生了稳定但语义方向错误的动作，而不是产生了幅值异常的动作。

同时，dataset action MSE 也不能直接作为 task success 的替代指标。K8 动作接近 dataset action，并不意味着它能够在当前闭环中解决任务；K4 的动作变化更大，也不妨碍它获得较高成功率。

## 8. D4：source-critic action ranking

### 8.1 D4 要回答什么

D4 使用 source-K critic 对 actor 在不同中间深度产生的动作进行评价，并与 dataset action 比较：

~~~text
qgap_vs_dataset_action
  = Qmin(s, g, a^k) - Qmin(s, g, a_dataset)
~~~

这里的 critic 始终使用训练时的 source K。D4 的 Q gap 是 critic 输出之间的差异，不是环境真实 return。

实现位置：

- [m18_trace_diagnostics.py](../../tools/m18_trace_diagnostics.py)
- qgap 定义约在 [m18_trace_diagnostics.py:361](/home/eai/Research/RLC/tools/m18_trace_diagnostics.py:361)

### 8.2 关键结果

| source K | actor depth | qgap vs dataset | critic member disagreement |
|---:|---:|---:|---:|
| K4 | k=4 | 0.7122 | 0.2310 |
| K4 | k=8 | 0.0719 | 0.2515 |
| K8 | k=4 | 0.2364 | 0.4932 |
| K8 | k=8 | 0.8446 | 0.4534 |

K8 critic 在 k=8 时给自身 actor action 相对于 dataset action 的平均 Q gap 更高，同时其两个 critic member 的 disagreement 也较大。

这提示：

1. K8 actor action 的 critic ranking 与 K4 不同；
2. K8 critic 更强地认为自己的 actor action 优于 dataset action；
3. 这一内部评价与 K8 较差的真实 rollout 结果之间存在脱钩迹象。

但不能直接做以下推断：

- 不能把 qgap 当成真实任务回报；
- 不能直接比较 Q4 和 Q8 的绝对 Q 值；
- 不能仅凭 qgap 证明 Q8 critic 过估计；
- 不能证明 critic 是 K8 失败的唯一原因。

## 9. D5：paired closed-loop rollout

### 9.1 D5 要回答什么

D1、D2、D3、D4 大多是离线或局部分析。D5 用 native K4/K8 policy 在真实 Puzzle 环境中进行完整闭环 rollout，回答：

> K8 的局部表示和动作异常，是否真的会转化成长期的 logical progress failure？

实现位置：

- [m18_paired_rollout_diagnostics.py](../../tools/m18_paired_rollout_diagnostics.py)

### 9.2 配对设计

D5 对 K4 和 K8 使用相同的：

- task；
- reset seed；
- goal；
- episode seed；
- actor seed；
- evaluation seed。

由于两个策略一旦采取了不同动作，后续环境状态就会不同，因此 D5 不会错误地把两个 divergent trajectory 进行逐 timestep state distance 对齐。

D5 还使用一个共享 goal manifest，确保同一个 paired episode 的 policy goal 向量由一次真实环境 reset 产生，并 byte-for-byte 复用于 K4 与 K8。

D5 同时记录 Puzzle 的 logical configuration，并计算：

- initial logical distance；
- final logical distance；
- minimum logical distance；
- net logical progress；
- best logical progress；
- progress transitions；
- regressive transitions；
- time to first logical progress；
- episode length。

### 9.3 总体结果

| 指标 | K4 | K8 |
|---|---:|---:|
| success | 0.93 | 0.17 |
| initial d* | 5.80 | 5.80 |
| final d* | 0.33 | 4.71 |
| minimum d* | 0.30 | 3.93 |
| net logical progress | 5.47 | 1.09 |
| best logical progress | 5.50 | 1.87 |
| time to first progress | 25.31 | 46.87 |
| episodes with first progress | 99 | 77 |
| logical transitions | 7.03 | 4.15 |
| progress transitions | 6.25 | 2.62 |
| regressive transitions | 0.78 | 1.53 |
| episode length | 212.49 | 445.78 |

配对 success outcome：

| K4 | K8 | episode pairs |
|---:|---:|---:|
| 0 | 0 | 6 |
| 0 | 1 | 1 |
| 1 | 0 | 77 |
| 1 | 1 | 16 |

K4 相比 K8 多出 76 个百分点的 success，并且 paired outcome 中 K4-only 是 77 对，K8-only 只有 1 对。

### 9.4 Task-level 结果

| task | K4 success | K8 success | K4 net progress | K8 net progress |
|---:|---:|---:|---:|---:|
| 1 | 1.00 | 0.40 | 4.00 | 0.40 |
| 2 | 0.75 | 0.00 | 4.55 | 0.05 |
| 3 | 0.90 | 0.45 | 5.80 | 3.35 |
| 4 | 1.00 | 0.00 | 6.00 | 0.05 |
| 5 | 1.00 | 0.00 | 7.00 | 1.60 |

K8 并非所有任务都完全没有局部能力。它在 task 1 和 task 3 上还能偶尔成功，也能取得部分 logical progress；但在 task 2、4、5 上完全没有成功，并且通常达到最大 episode length。

K8 的平均 regression transition 是 K4 的约两倍，而 progress transition 明显更少。这说明 K8 的问题是：

> 不能稳定地把局部正确动作组织成连续的、朝向目标的操作序列。

### 9.5 D5 的结论边界

D5 强烈支持“深度导致的行为退化是真实闭环退化，而不是单纯 reward logging 问题”。

但是 D5 不能判断失败究竟来自：

- 高层 logical reasoning 选错了目标按钮；
- 低层连续控制没有准确执行；
- action-to-logical-transition 的接口；
- 或上述因素的组合。

当前环境没有暴露 policy intended button，所以 reasoning failure 和 motor-control failure 在 D5 中不可识别。

## 10. D6：cross actor × cross critic preference

### 10.1 D6 要回答什么

D6 研究 K4/K8 actor 产生的动作，是否被各自 critic 以不同方式排序。

D6 使用：

- 同一个 D234 fixed offline batch；
- D3 中保存的 K4 native action，记为 a4；
- D3 中保存的 K8 native action，记为 a8；
- dataset action，记为 a_data；
- K4 source critic，执行深度 4；
- K8 source critic，执行深度 8。

D6 只比较同一个 critic 内部的动作排序：

~~~text
Q4(a4) vs Q4(a8)
Q8(a8) vs Q8(a4)
~~~

不比较 Q4 和 Q8 的绝对数值，因为不同 critic 的 Q scale 不能直接比较。

实现位置：

- [m18_cross_actor_critic.py](../../tools/m18_cross_actor_critic.py)

### 10.2 原始结果

| 指标 | 结果 |
|---|---:|
| Q4 偏好 a4 的比例 | 0.5488 |
| Q8 偏好 a8 的比例 | 0.7744 |
| 两个 critic 同时偏好自身 actor 的比例 | 0.3623 |
| Q4 self preference margin mean | 0.2126 |
| Q8 self preference margin mean | 0.3381 |
| Q4 self preference margin median | 0.0113 |
| Q8 self preference margin median | 0.0955 |
| Q4 own vs data 为正的比例 | 0.7432 |
| Q8 own vs data 为正的比例 | 0.8008 |
| Q4 self ties | 0 |
| Q8 self ties | 0 |

### 10.3 D6 的解释

Q4 对自身 actor action 的偏好只有 54.9%，接近均衡；Q8 对自身 actor action 的偏好达到 77.4%，明显更强。

这说明 K8 critic 与 K8 actor 之间存在更强的 depth-specific action preference geometry。考虑到 K8 的真实 success 只有 0.17，这种偏好可能是：

> critic 与 actor 在 offline objective 上形成了内部自洽，但这种内部自洽没有被环境长期成功验证。

这提供了 actor–critic co-adaptation 的中等强度证据，但不能直接称为 critic overestimation。

D6 目前支持：

- K8 critic 比 K4 critic 更强地偏好 K8 自身 actor action；
- critic 的 action ranking 具有深度依赖；
- K8 的 critic ranking 可能与真实闭环成功率脱钩。

D6 目前不能证明：

- Q8 一定错误；
- Q8 一定过估计；
- critic 是唯一根因；
- Q8 的绝对 Q 值大于 Q4 就代表 Q8 更乐观。

### 10.4 最终报告中的 D6 显示问题

当前自动生成的 [M18D_report.md](/data/qijunrong/06-RL/offline-rl/exp/RLC/diagnostics/M18D/reports/checkpoint_locked_final/M18D_report.md) 中，D6 数值栏为空，并显示 No completed formal D6 artifact。

这不是 D6 实验没有完成，而是汇总器的字段读取问题：

- [m18d_d6_summary.json](/data/qijunrong/06-RL/offline-rl/exp/RLC/diagnostics/M18D/cross_actor_critic/checkpoint_locked/fixed_batch_N1024_seed18018/m18d_d6_summary.json) 的 D6 rows 没有 scope 字段；
- [analyze_m18_d.py:724](/home/eai/Research/RLC/tools/analyze_m18_d.py:724) 调用 _summary_metric 时默认要求 scope=overall；
- 因此 self_preference_4、self_preference_8、joint_self_preference 被错误过滤。

D6 的原始 CSV/JSON 才是本次报告的有效数据源。按照当前预设的 decision rule，D6 更准确的状态应是 mixed，而不是 insufficient evidence：p4 和 p8 都高于 0.5，但 joint self preference 只有 0.3623。

## 11. 各个诊断实验的角色对照

| 诊断 | 核心问题 | 数据类型 | 主要结果 |
|---|---|---|---|
| D1 | 测试时改变 actor K 是否仍然有效 | 真实 rollout | 只有 native K 有效，非对角线全为 0 |
| D2 | hidden state 是否爆炸、停滞或过平滑 | 固定 offline batch | K8 actor 无简单爆炸，token 反而分化 |
| D2+ | mean pooling 保留了多少 token state energy | D2 后处理 | K8 native rho 约 0.098 |
| D3 | 深度如何改变动作 | 固定 offline batch | K8 动作变化小，没有明显饱和，更像稳定但错误 |
| D4 | critic 如何评价中间动作 | 固定 offline batch | K8 native action 的 qgap 较高，存在 ranking 脱钩警告 |
| D5 | 局部问题是否转化为真实闭环失败 | 配对真实 rollout | K8 success 0.17，progress 少且 regression 多 |
| D6 | actor 和 critic 是否形成深度相关的自洽偏好 | 固定 offline batch | Q8 对自身 actor 偏好 0.774 |

## 12. 综合分析：当前能确定什么

### 12.1 已经有较强证据的结论

#### 结论一：K8 的闭环性能退化是真实的

D5 使用 logical distance、progress 和 regression 指标复核了 K8 的行为。K8 不是只在 reward 汇总上表现差，而是在真实 Puzzle 状态转移中表现为：

- 更少的有效 progress；
- 更多的 regression；
- 更长的 episode；
- 更高的 timeout 比例；
- 更高的 final logical distance。

#### 结论二：当前 K 是训练绑定的

D1 的非对角线全部为 0，说明当前 actor/readout 并不支持测试时任意改变 K。

#### 结论三：K8 不是简单的 actor numerical explosion

D2 和 D3 没有显示：

- actor hidden state 无界增长；
- actor action 大面积饱和；
- action delta 随深度爆炸；
- 所有 token 被压成相同向量。

### 12.2 有证据支持但尚未证实的候选机制

#### 候选机制一：mean_context readout 丢失 token-specific 信息

K8 越深，token variance 越大，而 mean retained energy 越低。由于 readout 主要使用 token mean，深层 token 差异可能无法被 actor head 使用。

但 K4 也有较低 rho，所以这不是充分条件。必须通过 readout ablation 才能验证。

#### 候选机制二：K8 actor–critic co-adaptation

Q8 对自身 actor action 的偏好率为 0.774，而 K8 真实 rollout success 只有 0.17。这说明 critic 的内部 action preference 可能没有反映长期环境效果。

但 D6 只在 fixed offline batch 上测量，不能直接证明 rollout 状态分布上的 Q 错误。

#### 候选机制三：value/critic representation scale growth

K8 value/critic 的 hidden state 和 token variance 相对较大，可能影响 advantage 和 actor gradient。

但当前没有 value output calibration、TD target error 或 rollout-state critic calibration，因此只能作为训练侧候选问题。

### 12.3 目前没有被支持的解释

以下解释目前不成立或证据不足：

- “K8 只是动作饱和了”：D3 不支持；
- “K8 是 token oversmoothing”：pairwise cosine 下降、token variance 上升，不支持；
- “K8 一定发生了数值发散”：actor 的 relative update 下降，不支持简单发散；
- “K8 critic 一定过估计”：D6 不能单独证明；
- “失败一定来自 reasoning”：D5 没有 intended button 标签；
- “K8 结构必然不适合”：仍有优化和训练预算混杂因素。

## 13. 当前最合理的工作假设

把所有结果串起来，当前最合理的工作假设是：

~~~text
K=8
  -> shared computation 被重复执行更多次
  -> token-specific state difference 增强
  -> mean_context 只保留 token 平均成分
  -> actor 得到的任务相关信息可能被压缩或偏移
  -> value/critic 路径同时出现较大的 representation scale
  -> Q8 更强地偏好自身 actor action
  -> actor 和 critic 在 offline objective 上形成内部自洽
  -> 真实闭环中表现为 progress 少、regression 多、最终失败
~~~

这个假设解释了：

- 为什么 K8 可以数值稳定但行为错误；
- 为什么 K8 的动作不一定大幅偏离 dataset；
- 为什么 critic loss 正常不能保证 rollout success；
- 为什么测试时降低 K 不能恢复 K8；
- 为什么 D5 中 K8 仍能偶尔局部 progress，但不能完成复杂任务。

不过该假设仍需干预实验验证，不能在论文中表述为已经证明的因果链。

## 14. 结果可信度和限制

### 14.1 可信的部分

- D1 使用了完整的 K_train × K_actor_test 矩阵；
- D5 使用了相同 task/reset/goal/episode seeds 的 paired design；
- D5 有真实 logical distance；
- D6 使用 D3 保存的 actor action，并进行了 parity check；
- source checkpoint hash 前后一致；
- 所有诊断均没有 optimizer update；
- D2/D3/D4 使用相同 fixed offline batch；
- K4/K8 的 D5 结果与 D1 native-depth 结果方向一致。

### 14.2 必须保留的限制

1. M18 只有 training seed=0，不能给出跨 seed 泛化结论。
2. D5 只测试 Puzzle-4x4 的 task 1–5，每个 task 20 个 episode。
3. D2/D3/D4/D6 是 fixed offline batch，不是完整 rollout state distribution。
4. D5 的 paired trajectories 在动作分叉后不能逐 timestep 对齐。
5. K4 best checkpoint 在 900k，K8 best checkpoint 在 200k，checkpoint stage 不同。
6. K8 虽然训练到 1M，但它可能需要更长的优化过程，当前不能完全排除 undertraining。
7. D6 的 self-preference 是 critic 内部排序，不是真实 return。
8. 没有 intended button 或 logical action label，因此不能分离 reasoning 和 motor control。
9. 当前没有多 seed 的显著性检验，也没有将 episode 聚类到 task 后进行统计推断。

## 15. 与他人讨论时建议使用的表述

### 15.1 推荐的准确说法

可以这样总结：

> M18-D 证明了 K=8 在当前 Puzzle-4x4/GCIQL 设置下存在稳定且显著的闭环行为退化。D1 表明 actor 的表示和 readout 与训练深度绑定；D2/D3 排除了简单的 actor 数值爆炸、动作饱和和 token oversmoothing；D2+ 指出 mean pooling 可能无法充分利用深层 token-specific information；D6 发现 K8 critic 更强地偏好自身 actor action，提示 actor–critic co-adaptation。当前最合理的解释是深层表示、readout bottleneck 和 critic action ranking 共同造成了稳定但错误的闭环策略，但还需要干预实验来建立因果关系。

### 15.2 不应使用的过强表述

以下说法目前过强：

- “K=8 必然失败。”
- “mean pooling 已经被证明是根因。”
- “Q8 一定发生了 overestimation。”
- “K8 的失败一定是 reasoning failure。”
- “所有额外 recurrent computation 都没有用。”
- “K4 在任何任务上都优于 K8。”

## 16. 下一步最有价值的干预实验

M18-D 的 post-hoc 诊断已经足够完整，下一步应减少继续添加描述性指标，转向直接干预：

### 16.1 区分结构问题和优化问题

- 对 K=4、K=8 使用至少 3 个 training seeds；
- 延长 K=8 的训练步数；
- 保存并比较完整 checkpoint 曲线；
- 分别比较相同 optimizer steps 和相同 wall-clock budget；
- 记录 best checkpoint 与 final checkpoint 的差异。

### 16.2 验证 mean pooling 假设

在 K8 上保持参数量和训练协议尽量一致，替换：

- mean_context；
- token concat；
- attention pooling；
- last-token 或 token-aware readout；
- mean 与 token variance 的联合 readout。

如果只有 token-aware readout 能恢复 K8，才能把 mean-pooling mismatch 提升为较强因果证据。

### 16.3 验证 depth compatibility

尝试：

- 在训练中随机采样 computation depth；
- 对多个中间深度同时施加 readout/auxiliary loss；
- 使用 depth-conditioned readout；
- 对中间状态加入 normalization；
- 训练时显式要求不同深度的 policy output 保持一致或逐步改善。

### 16.4 验证 actor–critic co-adaptation

可以尝试：

- 固定 critic，仅训练 actor；
- 使用更保守的 critic/action regularization；
- 比较 actor action 在 rollout states 上的 critic ranking；
- 对 Q ranking 做 success-weighted calibration；
- 比较 dataset action、K4 action、K8 action 在真实 logical transition 上的结果。

### 16.5 区分 reasoning 和 motor control

需要让环境或诊断记录：

- policy intended button；
- 实际被按下的 button；
- logical transition 是否符合 intended action；
- 连续动作到 button event 的映射结果。

这样才能回答 K8 是“想错了”，还是“想对但做错了”。

## 17. 主要文件索引

### 实验定义

- [M18 study.yaml](../../experiments/M18_puzzle_recurrent_compute_scaling/study.yaml)
- [M18 configs](../../experiments/M18_puzzle_recurrent_compute_scaling/configs/)

### 分析脚本

- [D1: m18_cross_k_eval.py](../../tools/m18_cross_k_eval.py)
- [D2/D3/D4: m18_trace_diagnostics.py](../../tools/m18_trace_diagnostics.py)
- [D5: m18_paired_rollout_diagnostics.py](../../tools/m18_paired_rollout_diagnostics.py)
- [D6: m18_cross_actor_critic.py](../../tools/m18_cross_actor_critic.py)
- [final analyzer: analyze_m18_d.py](../../tools/analyze_m18_d.py)

### 原始结果

- [最终自动汇总报告](/data/qijunrong/06-RL/offline-rl/exp/RLC/diagnostics/M18D/reports/checkpoint_locked_final/M18D_report.md)
- [D5 summary JSON](/data/qijunrong/06-RL/offline-rl/exp/RLC/diagnostics/M18D/closed_loop/checkpoint_locked/puzzle4x4_tasks1-2-3-4-5_episodes20_evalSeed18018/m18d_d5_summary.json)
- [D5 summary CSV](/data/qijunrong/06-RL/offline-rl/exp/RLC/diagnostics/M18D/closed_loop/checkpoint_locked/puzzle4x4_tasks1-2-3-4-5_episodes20_evalSeed18018/m18d_d5_summary.csv)
- [D6 summary JSON](/data/qijunrong/06-RL/offline-rl/exp/RLC/diagnostics/M18D/cross_actor_critic/checkpoint_locked/fixed_batch_N1024_seed18018/m18d_d6_summary.json)
- [D6 summary CSV](/data/qijunrong/06-RL/offline-rl/exp/RLC/diagnostics/M18D/cross_actor_critic/checkpoint_locked/fixed_batch_N1024_seed18018/m18d_d6_summary.csv)
- [K4 trace summary](/data/qijunrong/06-RL/offline-rl/exp/RLC/diagnostics/M18D/trace/checkpoint_best/fixed_batch_N1024_seed18018/trainK4/maxTraceK8/trace_summary.csv)
- [K8 trace summary](/data/qijunrong/06-RL/offline-rl/exp/RLC/diagnostics/M18D/trace/checkpoint_best/fixed_batch_N1024_seed18018/trainK8/maxTraceK8/trace_summary.csv)

## 18. 最终结论

M18-D 的最终结论应分成三层：

1. **行为层：已确认。** K8 native policy 在当前 Puzzle-4x4 闭环中明显失败，且失败表现为 progress 少、regression 多和 timeout 多。
2. **表示/评价层：有较强线索。** K8 的计算状态趋于稳定但 token 分化明显；mean pooling 可能丢失深层 token-specific information；K8 critic 更强地偏好自身 actor action。
3. **因果层：尚未确认。** 目前还不能断言 mean pooling、critic overestimation、value scale 或训练不足中的哪一个是主要根因。

因此，M18-D 最准确的研究结论不是“找到唯一 bug”，而是：

> 找到了 K8 失败的可重复行为证据，并将问题范围从“计算量增加是否有害”缩小到“深度绑定的表示/readout、训练侧 value/critic scale，以及 actor–critic action ranking 是否形成了错误但自洽的闭环策略”。
