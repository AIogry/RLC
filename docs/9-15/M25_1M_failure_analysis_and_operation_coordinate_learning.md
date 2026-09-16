# M25：1M 步工程长跑失败定位与无 $M^{-1}$ 输入的操作坐标学习方案

**日期：** 2026-09-15
**状态：** 初步分析讨论稿，不是正式实验报告，不改变 M25 的 Study 状态
**目的：** 记录最近一次 1M step 实验的事实结果，并分析在不把 $M^{-1}$ 或任何特权操作标签作为数据输入时，怎样让模型学会可操作的坐标。

## 1. 结论摘要

这次实验没有学到操作坐标，但现有证据不支持“数据中没有足够信息”这一解释。更准确的结论是：

1. **数据是可识别的。** 4×5 数据中观测到 20 个独特非零效果，GF(2) 秩为 20，覆盖完整的 20 维任务空间；每一种效果都跨越多个 episode 出现。
2. **目标本身存在。** 只用事件端点计算出的原始 XOR 效果集合，不使用 Puzzle 的真实 $M$、$M^{-1}$、按钮 ID 或 solver，直接做一次 GF(2) 线性代数求解，就可以把 20/20 种效果变成 one-hot 变化。这是分析用的 oracle-free positive control，不是给 learner 的输入。
3. **当前结构的硬性质是正确的。** 1M 结束时 flow 仍然是精确二进制、GF(2) 线性、可逆、满秩；因此不是逆映射或二进制实现错误。
4. **失败集中在优化/参数化。** 1M 步后有效矩阵仍是 20×20 单位矩阵，`hard_axis_success=0` 从头到尾没有改善；训练损失的均值几乎等于单位映射在该数据上的理论基线，梯度范数从约 $1.2\times10^3$ 衰减到约 $1.0\times10^{-8}$。这与 hard threshold + sigmoid STE 把所有 coupling gate 推回关闭状态的塌缩相符。
5. **不应再简单增加步数。** 下一步应该先做：数据驱动可识别性正控、固定效果集合上的连续 parity relaxation、显式 one-hot 轴匹配、GF(2) 投影/分解，以及当前固定 32 层 schedule 的可达性检查。通过这些 gate 后再设计正式 M25，而不是再次重复同一套 1M 配置。

这里的“失败”只针对当前 `M25-E001` 的训练方法；它不等价于否定“学习控制坐标”这一科学假设。

## 2. Provenance 与科学边界

### 2.1 代码与运行身份

| 项目 | 观测值 |
| --- | --- |
| 开发 worktree | `/home/eai/Research/RLC-M25-dev` |
| 开发分支 | `m25-linear-boolean-flow` |
| 源代码 HEAD | `6e20e56f82ed998520ed7b44b9156bb27d7085d6` |
| Study/config | `M25` / `M25-E001` |
| 运行环境 | `puzzle-4x5-play-v0` |
| 运行归档 | [`/data/.../seed_000`](</data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M25/M25-E001__4x5_linear_boolean_flow_l32_engineering/puzzle-4x5-play-v0/seed_000>) |
| 运行状态 | `completed` |
| 声明阶段 | `engineering_stage1_nonformal` |
| `formal_training` | `false` |
| GPU | 物理 GPU 1，RTX 4090；运行时 JAX backend=`gpu` |
| 训练时间 | 2026-09-14 16:11:29–17:31:53 UTC，约 1 小时 20 分钟 |

运行开始时元数据记录了 `git_dirty=true`，原因是启动日志在 worktree 中产生了未跟踪文件；这也是该运行不能被视为正式结果的 provenance 限制。运行结果目前已按项目约定归档到 `/data`。

本次实际读取的数据文件为：

| 文件 | 字节数 | SHA256 |
| --- | ---: | --- |
| `puzzle-4x5-play-v0.npz` | 816,520,065 | `7f25e3161280aa4ff1234a279ba7a0a4466df6a8e4f5af64fd2b1bfe8df892cc` |
| `puzzle-4x5-play-v0-val.npz` | 81,683,658 | `26bfb8d59290be2838a4c8185fc8221355ff45027d4c4bff4e26f6f4a7c2624c` |

### 2.2 M25 Stage-1 的输入和输出边界

当前研究对象是

\[
H_\theta(b)=A_\theta b,
\qquad A_\theta\in GL(N,2),
\]

其中 $b\in\{0,1\}^N$ 是 Puzzle 的任务 board。对一个相邻、同一 episode 内的状态变化，定义

\[
\delta=b\oplus b'.
\]

如果真实操作效果集合 $\{m_i\}$ 构成一组基，理想目标是

\[
A m_i=e_{\pi(i)},
\]

其中 $\pi$ 是任意坐标置换。坐标编号不需要等于物理按钮编号。

本阶段 learner 的实际输入只有 `start_board` 和 `end_board`。`actions` 不进入 Stage-1 的模型输入；原始 XOR effect signature 只用于数据审计和 post-hoc 诊断。没有使用：

- 真实 Puzzle 操作矩阵 $M$ 或 $M^{-1}$；
- oracle press set、solver、target button、physical button ID；
- privileged `button_states` 或 scripted solution；
- GCIQL、RL success 或物理 executor。

## 3. 这次到底跑了什么

`M25-E001` 是 4×5 的 20-bit、32 层线性可逆 Boolean flow。每层是固定置换下的 GF(2) additive coupling，coupling logits 用 hard threshold 做前向、用 sigmoid straight-through estimator 提供伪梯度。

| 因素 | 实际值 |
| --- | --- |
| `num_bits` | 20 |
| `num_layers` | 32（工程候选，不是科学固定深度） |
| permutation seed | 25001 |
| coupling logit 初始化 | mean=`-2.0`，std=`0.1` |
| binary temperature | 1.0 |
| learning rate | 0.003 |
| batch size | 256 |
| 训练步数 | 命令行覆盖为 1,000,000 |
| event sampling | uniform |
| eval/save interval | 500 / 1,000 steps |

当前代码中的 Stage-1 surrogate 是：

\[
q_\theta(b,b')=
H_\theta(b)\oplus H_\theta(b'),
\qquad
L_{\rm axis}=\mathbb E\left[(\|q_\theta\|_1-1)^2\right].
\]

对精确的线性 Boolean flow，$q_\theta=A_\theta\delta$。因此目标是让每个真实事件的 latent Hamming distance 恰好为 1，而不是让 RL success 变高。

## 4. 数据支撑：目标不是因为 rank 不足而不可学

来源：运行目录中的 [`event_audit.json`](</data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M25/M25-E001__4x5_linear_boolean_flow_l32_engineering/puzzle-4x5-play-v0/seed_000/event_audit.json>)。

| 审计量 | 数值 | 含义 |
| --- | ---: | --- |
| episode 数 | 3,000 | 有效观测边界已按 canonical `valids` 语义恢复 |
| 同 episode 非零事件数 | 88,765 | learner 实际训练的事件集合 |
| 独特原始 XOR 效果 | 20 | 与 $N=20$ 相同 |
| 观测效果 GF(2) rank | 20 | 满秩 |
| 是否张成完整任务空间 | `true` | 没有 rank-deficiency gate 问题 |
| 有跨 episode 覆盖的效果 | 20/20 | 不是单一轨迹偶然性 |
| 单一效果频率范围 | 4,224–4,550 | 频率相当均衡，非严重长尾 |
| 效果频率均值 / 标准差 | 4,438.25 / 74.73 | 频率变异约 1.7% |

原始效果的 Hamming weight 分布为：

| 原始 $\|\delta\|_1$ | 事件数 | 比例 |
| ---: | ---: | ---: |
| 3 | 17,590 | 19.82% |
| 4 | 44,258 | 49.86% |
| 5 | 26,917 | 30.32% |
| 合计 | 88,765 | 100% |

因此，单位映射 $A=I$ 在这批数据上的理论 surrogate 基线是

\[
\mathbb E[(\|\delta\|_1-1)^2]
 =
\frac{17590\cdot4+44258\cdot9+26917\cdot16}{88765}
 =10.13185377.
\]

这给了一个很有用的诊断参照：如果训练结束时损失和平均距离仍然等于这个数，模型实际上没有离开 $A=I$。

## 5. 1M step 的真实结果

### 5.1 评估曲线

运行生成了 2,001 行评估记录（第 0 步以及每 500 步至第 1,000,000 步）。[`eval.csv`](</data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M25/M25-E001__4x5_linear_boolean_flow_l32_engineering/puzzle-4x5-play-v0/seed_000/eval.csv>) 中 `hard_axis_success` 的 2,001 个值全部为 0。

| eval step | hard-axis success | 平均 latent distance | median | distance > 1 | one-hot effects |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.0000 | 4.105075 | 4 | 1.0000 | 0/20 |
| 5,000 | 0.0000 | 4.105075 | 4 | 1.0000 | 0/20 |
| 50,000 | 0.0000 | 4.105075 | 4 | 1.0000 | 0/20 |
| 250,000 | 0.0000 | 4.105075 | 4 | 1.0000 | 0/20 |
| 500,000 | 0.0000 | 4.105075 | 4 | 1.0000 | 0/20 |
| 1,000,000 | 0.0000 | 4.105075 | 4 | 1.0000 | 0/20 |

4.105075 正好等于上面由原始效果权重计算出的单位映射平均 Hamming weight。也就是说，评估曲线不是“有一点学习但没有达到阈值”，而是从统计上一直保持原始坐标。

### 5.2 训练损失与梯度

`train.csv` 每 100 步记录一次 batch 指标，共 10,001 条训练记录（不含 header）。选取的行如下；batch loss 和 batch distance 会有抽样噪声，但趋势足以判断是否离开恒等映射。

| step | axis surrogate loss | batch distance mean | gradient norm |
| ---: | ---: | ---: | ---: |
| 1 | 10.1015625 | 4.1015625 | 1209.0647 |
| 1,000 | 9.9140625 | 4.0703125 | 219.3773 |
| 10,000 | 10.1679688 | 4.1054688 | 0.9493 |
| 50,000 | 10.3867188 | 4.1445312 | $7.37\times10^{-7}$ |
| 500,000 | 9.7109375 | 4.0390625 | $2.37\times10^{-8}$ |
| 1,000,000 | 10.2578125 | 4.1328125 | $9.97\times10^{-9}$ |

所有记录的 loss 均值为 10.132099，与单位映射理论基线 10.131854 几乎相同。最后的 loss 10.2578 只是最后一个随机 batch 的波动，并不代表全数据性能改善。

### 5.3 最终矩阵与结构审计

来源：[`control_coordinate_summary.json`](</data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M25/M25-E001__4x5_linear_boolean_flow_l32_engineering/puzzle-4x5-play-v0/seed_000/control_coordinate_summary.json>)、[`effective_matrix.json`](</data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M25/M25-E001__4x5_linear_boolean_flow_l32_engineering/puzzle-4x5-play-v0/seed_000/effective_matrix.json>) 和最终诊断文件。

| 项目 | 结果 |
| --- | ---: |
| final step | 1,000,000 |
| best step | 0 |
| best / final hard-axis success | 0 / 0 |
| effective matrix rank | 20 |
| effective matrix density | 0.05 |
| 行/列 Hamming weight | 全部为 1 |
| 256 个随机状态的矩阵一致性 | 256/256，exact |
| forward/inverse exact | `true` |
| forward outputs exactly binary | `true` |
| GF(2) linearity exact | `true` |
| zero maps to zero | `true` |
| same-effect consistency | 20/20 |
| cross-episode consistency | `true` |

完整矩阵是 $I_{20}$，而不是一个学到的非平凡 $PM^{-1}$。同效果 consistency 为真不能被解读为成功：单位映射本身当然会对同一个原始 XOR 给出相同 latent delta；它只说明结构确定性，没有说明 axis locality。

## 6. 为什么当前方法学不成功

下面把“已验证事实”和“待验证解释”分开。

### 6.1 已排除：数据不可识别

可以只从运行审计中的 20 个 raw XOR 效果构造矩阵 $E$，按确定顺序把它们作为行向量，并在 GF(2) 上求逆：

\[
E=
\begin{bmatrix}
\delta_1^\top\\\cdots\\\delta_{20}^\top
\end{bmatrix},
\qquad
A_{\rm data}=(E^{-1})^\top.
\]

这样有 $A_{\rm data}\delta_j=e_j$。对本次 4×5 审计数据重新计算得到：

| data-only positive control | 结果 |
| --- | ---: |
| (E) 的 rank | 20 |
| $A_{\rm data}$ 的 rank | 20 |
| 变换后 one-hot 效果 | 20/20 |
| 是否全部 one-hot | `true` |
| $A_{\rm data}$ density | 0.445 |

这一步没有读取真实 Puzzle algebra，也没有把 $A_{\rm data}$ 送进 learner；它只是说明“从标准观测边推导出一个满足 M25 目标的坐标系”在信息上是可能的。正式研究中应把它作为 identifiability/positive-control baseline，而不能冒充神经模型已经学会。

### 6.2 已排除：flow 的硬结构损坏

最终矩阵满秩，所有结构不变量均通过。即使模型不学习，当前 parameterization 也能保持信息、二进制输出和可逆性。因此失败不是因为 forward 把 Boolean 变成了连续 bottleneck，也不是 inverse 实现不一致。

### 6.3 最主要解释：hard threshold + STE 的离散优化塌缩

当前 coupling gate 的前向是

\[
g=\mathbf 1[\ell/T\ge 0],
\]

只有 $\ell$ 的 sigmoid 导数被用作 straight-through 伪梯度。初始化 mean=`-2.0`、std=`0.1` 时，几乎所有 gate 的 hard 值都是 0，所以初始 flow 是恒等映射。

这会产生三个问题：

1. **前向矩阵是离散的。** 大量参数更新不会改变任何 hard gate，因而不会改变真实评估指标。
2. **伪梯度不等于离散目标的真实梯度。** 它可以在当前 hard 矩阵不变时把 logits 推向更负的方向，优化器于是选择“继续关闭 gate”而不是探索一个需要多个 gate 同时翻转的非平凡矩阵。
3. **一旦 logits 更负，sigmoid 导数迅速消失。** 本次运行的 gradient norm 从 1209 降到 $10^{-8}$ 量级，同时有效矩阵保持 $I_{20}$，这与 gate 饱和完全一致。

这不是单凭代码推测：训练 loss 等于恒等映射基线、全程 success=0、最终矩阵为 (I_{20})、梯度消失四项证据相互吻合。

### 6.4 目标函数对“轴分配”的优化信号太弱

当前损失只惩罚 latent delta 的总 Hamming weight：

\[
\left(\sum_k q_{jk}-1\right)^2.
\]

在 hard binary 前向中，$q$ 是 one-hot 当且仅当总和为 1；但在 relaxation/STE 阶段，它没有显式表达“每个效果要占不同的轴”。精确可逆性在最终 hard 矩阵中会保证不同基向量不能映射到同一个 one-hot，但训练过程看不到一个平滑的 permutation assignment 信号。

此外，$N!$ 个坐标置换都是等价答案。这个对科学指标不是问题，但对梯度搜索意味着需要同时解决“变成 one-hot”和“不同效果选择不同轴”两个离散问题。当前标量平方损失没有为第二个问题提供稳定的连续代理。

### 6.5 endpoint 路径与 delta 路径的 STE 梯度可能不同

在精确线性 Boolean 前向中，

\[
H(b)\oplus H(b')=H(b\oplus b')=H(\delta).
\]

但当前实现通过两次 endpoint encode 再 XOR；在 hard gate 的伪微分下，这条路径与直接将 $\delta$ 送入 flow 的梯度并不必然相同。前向代数等价，不代表 STE 的优化几何等价。直接以 $\delta$ 为 learner 内部的差分对象可以减少两次端点传播和中间 XOR 的梯度抵消；这应作为明确的优化因素做小规模 ablation，而不是默认认为两种写法完全一样。

### 6.6 32 层候选的表达能力/优化能力尚未分离

容量预检使用随机满秩 GF(2) 目标矩阵、每个 batch 都包含完整 identifying basis，不使用 Puzzle 的 $M$。结果如下；这些文件已归档到 [`/data/.../diagnostics/M25/capacity_preflight`](</data/qijunrong/06-RL/offline-rl/exp/RLC/diagnostics/M25/capacity_preflight>)。

| N | layers | init mean | lr | steps | heldout bit accuracy | exact state accuracy | target exact | gate |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 4 | 6 | -0.1 | 0.001 | 10,000 | 1.0000 | 1.0000 | true | pass |
| 20 | 32 | -2.0 | 0.003 | 1,000 | 0.5042 | 0.0000 | false | fail |
| 24 | 32 | -2.0 | 0.003 | 1,000 | 0.4998 | 0.0000 | false | fail |

这说明低维、较温和初始化时同一类 flow 可以学会一个随机 GF(2) map；在 N=20/24、当前初始化和优化设置下则接近随机。它不能单独证明 32 层一定表达不了目标，因为失败可能来自 fixed schedule 的 reachability，也可能来自优化；但结合本次 1M 的恒等塌缩，当前配置不能被冻结为 formal depth。

### 6.7 当前没有证据表明事件频率是主因

20 种效果的频率在 4,224–4,550 之间，标准差只有 74.73；因此这次失败不能优先归因于某些操作没有被采样。后续仍可采用每种 unique effect 等量采样来降低方差，但这应被视为优化改进，而不是对本次失败的主要解释。

## 7. 不提供 $M^{-1}$ 数据输入时，怎样学习操作坐标

关键区分是：

- **不允许的做法：** 把真实 $M^{-1}$、真实物理按钮对应关系或 oracle operation identity 作为输入或标签。
- **允许且必要的做法：** 从标准观测边 $(b,b')$ 内部计算 $\delta=b\oplus b'$，利用所有观察到的效果的代数关系或自监督目标来学习 $H$。

操作坐标发现和物理执行也必须分成两个阶段：Stage 1 学 $H$，后续阶段再用普通 `actions` 学“latent axis → physical action”的 grounding。没有 grounding，$H$ 可以告诉我们“哪个 latent bit 被操作”，但不能凭空知道应该按哪个物理按钮。

### 7.1 第一优先级：oracle-free data-only positive control

先把上述 $A_{\rm data}$ 做成一个明确的分析工具，而不是 learner 输入：

1. 从 `observations` 按 canonical Puzzle parser 得到 binary boards。
2. 只保留同一 episode 内的非零相邻差分。
3. 去重得到 unique effects；审计数量、rank、跨 episode 覆盖。
4. 如果 rank=N，在 GF(2) 上求一个逆并构造 $A_{\rm data}$。
5. 在全部事件、随机状态和 inverse round-trip 上验证 hard exact 指标。

它的作用有三个：

- 证明数据是否包含足够信息；
- 给当前 flow 提供一个可达到目标的正控；
- 作为后续模型/优化器的上界和回归测试。

如果研究问题坚持“必须由梯度从随机初始化发现坐标”，$A_{\rm data}$ 不能用于训练，只能作为 post-hoc positive control；如果允许“从数据推导初始化但不使用特权信息”，则可以把它用于初始化/蒸馏，但必须把这声明为新的方法因素，而不是与当前 M25 Stage-1 混在一起。

### 7.2 推荐的可训练路径：连续 expected-parity relaxation

不要在训练第一步就把 coupling gate 硬阈值化。令每条 coupling edge 的概率为

\[
p_k=\sigma(\ell_k/T),
\]

先用软概率训练，最后再 harden。对于固定的 binary source bits $s_k$，如果每条 edge 是独立 Bernoulli gate，则 parity 为 1 的期望概率可以写成

\[
\Pr\left[\bigoplus_k s_k g_k=1\right]
=
\frac{1-\prod_{k:s_k=1}(1-2p_k)}{2}.
\]

这给出一个比 hard STE 更平滑、方差更低的 parity 概率。实现时应：

1. 在 warm-up 阶段使用较高温度，保留连续概率；
2. 逐步退火温度，再将概率投影到 hard (0/1)；
3. 每个周期都用 GF(2) rank、forward/inverse exact 和全事件 hard success 检查投影结果；
4. 如果软路径不保持精确可逆，维护一个独立的 hard GF(2) shadow map，最终只接受通过 exact projection 的 checkpoint。

这不是把 $M^{-1}$ 喂给模型，而是改变 gate 的优化估计器。

### 7.3 加入显式 permutation-invariant one-hot assignment

对 N 个 unique effects（当前 4×5 就是 20 个）在每一个优化周期都计算软 latent delta，并将每一行归一化为 axis distribution：

\[
Q_{jk}\approx\Pr[(A_\theta\delta_j)_k=1].
\]

令 $P$ 是 effect-to-axis 的 permutation assignment，可由 Hungarian 或 Sinkhorn 根据当前 $Q$ 求得。训练目标可以采用：

\[
\mathcal L_{\rm assign}
=-\sum_{j,k}P_{jk}\log Q_{jk}
 +\lambda\sum_j H(Q_j)
 +\mu\left\|\sum_jQ_{j,:}-\mathbf 1\right\|_2^2,
\]

其中 $H(Q_j)$ 是 row entropy，目标是每行低熵且每个 axis 都被恰好占用。assignment 可以 stop-gradient 后作为当前轮的伪目标；正式实现需把 assignment 规则、温度和 tie-breaking 写入 Configuration。

相比当前的单一 $(\|q\|_1-1)^2$，这个目标明确提供：

- 每个效果应该靠近某一个 axis 的信号；
- 不同效果不能全部挤到同一个 axis 的信号；
- 对 $N!$ 个等价坐标置换保持不变的科学目标。

### 7.4 delta-first，但不改变特权边界

保留数据接口 `start_board`、`end_board`，在 learner 内部临时计算

```text
delta = start_board XOR end_board
```

然后用 $H_\theta(\delta)$ 计算 effect loss。这里的 delta 不是 ground-truth operation label，也不是预先提供的 $M^{-1}b$；它完全由两个标准观测端点决定。对当前线性模型，这与 endpoint encode 的 hard 前向严格等价，但优化路径更直接。

该改动应标记为一个明确的 implementation/optimization factor，做以下对照：

- endpoint encode + XOR（当前 baseline）；
- internal delta + flow；
- internal delta + expected parity + assignment。

先在 unique effects 的全量小数据上比较，而不是直接跑 1M。

### 7.5 精确 GF(2) 投影、分解和 schedule 可达性检查

连续 relaxation 得到软矩阵后，不能只看 soft loss。需要：

1. 将候选矩阵 round/project 到 binary；
2. 检查 rank=N；
3. 检查是否能用当前固定 permutation schedule 的 coupling layers 分解；
4. 在全部 unique effects 上要求 100% one-hot；
5. 在全部事件上要求 100% hard axis success。

特别要做一个“可达性而非优化”的 gate：给定数据驱动的 $A_{\rm data}$，用 GF(2) 分解、SAT/整数搜索或确定性 elementary-shear factorization 检查它是否能由当前 32 层固定 schedule 表示。如果不能，继续调学习率没有意义，应增加 depth 或放宽 schedule；如果能而梯度仍失败，问题才可以明确归因于优化器/relaxation。

### 7.6 噪声或非唯一效果时的备用方法

当前 4×5 数据的效果已经稳定且满秩，不需要立即引入复杂方法。但如果以后出现：同一物理操作产生多个观测效果、效果签名受噪声污染、或者 rank 不足，可使用：

- raw delta clustering + consensus effect；
- same-effect cross-context consistency；
- Puzzle transition graph 的 cycle/commutativity 约束；
- 对候选 $A$ 的离散 coordinate descent、beam search 或 simulated annealing。

这些方法属于后续扩展，不能用来掩盖当前基本优化 gate 尚未通过的事实。

## 8. 建议的最小下一轮实验顺序

每一轮都应写入 `/data/qijunrong/06-RL/offline-rl/exp/RLC/`，并使用独立的 Configuration/Run 身份；不要覆盖本次失败 run。

### Gate 0：数据正控（不训练）

- 对 4×5、4×6 各自提取 unique effects。
- 记录数量、rank、频率、跨 episode 覆盖。
- 计算 $A_{\rm data}$，要求所有 unique effects 变为 one-hot。
- 输出完整 hard audit，作为 Stage-1 的 identifiability baseline。

### Gate 1：结构可达性/容量正控

- 对 N=4、20、24 的随机 GL(N,2) 目标做 exact recovery。
- 比较 L=16/32/64（或由分解 gate 决定的最小深度）。
- 至少 seeds 0、1、2；每个 batch 固定包含完整 basis。
- 同时报告 target matrix exact、heldout exact、rank 和 active gate 数。
- 当前 N=20/24、L=32、init=-2 的失败必须保留，不能被新结果覆盖。

### Gate 2：固定 4×5 unique-effects 的优化 ablation

建议先用 10k–50k steps，而不是再次直接 1M：

| cell | 差分路径 | parity estimator | assignment | 目的 |
| --- | --- | --- | --- | --- |
| A | endpoint | hard STE | 无 | 重现当前 baseline |
| B | internal delta | hard STE | 无 | 检查 endpoint/STE 路径 |
| C | internal delta | expected parity | 无 | 检查离散 gate 梯度 |
| D | internal delta | expected parity | Hungarian/Sinkhorn | 检查 one-hot/diversity 信号 |
| E | D + temperature annealing | expected parity | assignment | 候选主方案 |

每个 cell 必须记录：soft assignment cost、row entropy、active gate 数、hard success、unique one-hot 数、rank、inverse exact，以及梯度范数。

建议通过标准：

```text
unique_effect_one_hot = N
full_event_hard_axis_success = 1.0
effective_matrix_rank = N
forward_inverse_exact = true
```

如果 Gate 2 仍然失败，应优先做 schedule factorization/expressivity 检查，不再增加步数。

### Gate 3：4×6 与多 seed

只有 Gate 0–2 通过后，才把同一方法扩展到 N=24 的 4×6，并把 seed variability 写进 Study。当前 Study 只有 seed 0、工程阶段，不能直接据此冻结正式深度或启动正式 campaign。

### Gate 4：操作 grounding 与 RL 集成

在 $H$ 通过 Stage-1 后再做：

1. 冻结 $H$，对每个观测 transition 得到 latent effect $e_k$。
2. 用普通 offline `actions` 或短 physical windows 学 $e_k\rightarrow$ physical action/distribution。
3. 将 latent goal difference $H(b)\oplus H(g)$ 交给后续 policy/critic。
4. 把“坐标发现”“物理执行”“GCIQL 集成”作为不同 Study/Configuration 因素记录。

这样即使 latent axis 的编号被任意 permutation，executor 也可以从数据中学习对应关系；不需要把真实 $M^{-1}$ 作为输入。

## 9. 当前不应采取的修复

- 不应只把 `train_steps` 从 1M 提高到更大；本次已经显示梯度饱和。
- 不应把真实 $M^{-1}$、真实按钮 ID 或 oracle effect label 加入 learner，以换取成功率。
- 不应把 data-only $A_{\rm data}$ 的成功直接报告为 neural learner 的成功。
- 不应在容量/可达性 gate 未通过前冻结 32 层作为科学因素。
- 不应把 same-effect consistency 或 full-rank 误报为 axis locality 成功。
- 不应把 Stage-1 的坐标发现与后续 action grounding、RL success 混成一个指标。

## 10. 讨论时需要明确的开放问题

1. 研究口径是否接受“从标准观测边推导 $A_{\rm data}$”作为 oracle-free positive control？本报告建议接受为分析正控，但不作为当前 learner 输入。
2. 固定 32 层 schedule 是否能表达本次 $A_{\rm data}$？需要先做确定性分解/可达性检查。
3. 正式候选优化器选择 expected parity + assignment，还是采用 data-only 初始化后再学习？两者应作为不同方法因素，不应混报。
4. Stage-1 的 primary metric 是否继续使用 exact hard one-hot，还是同时报告 assignment cost/entropy 作为训练诊断？建议 primary metric 保持 exact hard 指标，soft 指标只作优化诊断。
5. 何时进入 action grounding 和 GCIQL？建议以 Gate 2 的全事件 exact success 为硬门槛，而不是以 surrogate loss 下降为门槛。

## 11. 事实来源

- M25 设计约束：[`docs/9-14/prompt2.md`](../9-14/prompt2.md)
- M25 Study：[`study.yaml`](../../experiments/M25_linear_boolean_control_coordinates/study.yaml)
- M25 learner/objective：[`impls/agents/control_coordinate.py`](../../impls/agents/control_coordinate.py)
- Boolean flow：[`impls/computation/blocks/linear_boolean_flow.py`](../../impls/computation/blocks/linear_boolean_flow.py)
- M25 README：[`experiments/M25_linear_boolean_control_coordinates/README.md`](../../experiments/M25_linear_boolean_control_coordinates/README.md)
- 1M canonical run：[`seed_000`](</data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M25/M25-E001__4x5_linear_boolean_flow_l32_engineering/puzzle-4x5-play-v0/seed_000>)
- 1M summary：[`control_coordinate_summary.json`](</data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M25/M25-E001__4x5_linear_boolean_flow_l32_engineering/puzzle-4x5-play-v0/seed_000/control_coordinate_summary.json>)
- 1M evaluation curve：[`eval.csv`](</data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M25/M25-E001__4x5_linear_boolean_flow_l32_engineering/puzzle-4x5-play-v0/seed_000/eval.csv>)
- 1M training curve：[`train.csv`](</data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M25/M25-E001__4x5_linear_boolean_flow_l32_engineering/puzzle-4x5-play-v0/seed_000/train.csv>)
- 容量预检：[`capacity_preflight`](</data/qijunrong/06-RL/offline-rl/exp/RLC/diagnostics/M25/capacity_preflight>)
