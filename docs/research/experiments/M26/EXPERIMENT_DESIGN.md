# M26 4×5 / seed0 实验设计

2026-09-19，配置阶段。来源是 `/home/eai/Research/docs/9-19/` 中的
`M26_EXPERIMENT_PLAN.md` 和 `M26_CODEX_EXPERIMENT_PROMPT.md`，修订
`m26-4x5-seed0-manual-release-r1`。已有 [DESIGN.md](DESIGN.md) 是能力阶段历史记录；
本文件不改写那时的未提交/未运行状态。

## 科学问题、因素、固定项、解释边界

问题：在完全相同的 v2 输入参数化下，显式操作距离、operation 坐标重编码、
actor/value-side 供给位置分别能解释多少 Puzzle-4x5 最终任务成功率差异？

操纵因素：actor coordinate、value_side coordinate、固定 identity/P/B、zero/精确距离。
配置与完整矩阵的唯一运行定义是
[Study](../../../../experiments/M26_puzzle_goal_coordinate_diagnostics/study.yaml)
及同目录 configs；[README](../../../../experiments/M26_puzzle_goal_coordinate_diagnostics/README.md) 列出10行矩阵。

固定项：GCIQL DDPG+BC、alpha0.4、lr3e-4、batch1024、discount0.99、expectile0.9、
tau0.005、hidden dims512×3、原 post-gradient Polyak；原目标采样和 board-equality
reward/mask；actor+V+Q Mixer-L2，token128/robot128/token-MLP64/channel-MLP256，
物理 index embedding 和 mean/规范化 mean_context readout；无 encoder/augmentation/frame-stack。
完整字段及真实 M24 来源见 [M24_INHERITANCE_AUDIT](M24_INHERITANCE_AUDIT.md)。

解释边界：oracle mechanism diagnostic，不是 learned operation、规划器或执行器。
精确距离是最少有效按压次数/N，不是机器人运动路径或环境时间步。
单一 seed/P/B 的结果不建立显著性或跨环境稳健性。稠密编码同时改变坐标解耦与解码难度，
不能唯一归因于“单坐标动力学”。没有引入新 loss、reward shaping、监督、warm start 或 M25 代码。

## 输入与参照

所有组固定 v2、9D paired token（含始终存在的1D aux）；zero组也保留该通道。
令 d=b(s) XOR b(g)，x=M_inverse d。残差是d；操作坐标是x；
P 用 `y[...,j]=x[...,p[j]]`；B 用 `y=B@x mod2`。距离始终是原始 x 的 `sum(x)/20`。
不重排 current state、物理 token/index embedding。V/Q/target Q 绑定同一 value_side。

M24 的 8D 和 M26 的 9D 不具有同初始化/checkpoint等价性。此次生产前向检查观测到
全部10组参数（含target）都是1,679,354元素，参数、optimizer、RNG初值逐元素相同；
这只是 v2 组内控制验证，不宣称和历史 v1 全因素相同。

首批默认 C03–C10，8 runs。C01/C02 完整定义、保持可执行、未纳入首批；不会用旧M24成绩
补齐。完整10组和仅补2组是人工可选清单，不能同时启动两个scheduler。

| 对比 | 首批8组是否具备全部条件 |
| --- | --- |
| C03−C01、C04−C02（距离增益） | 否，匹配参照pending |
| C04−C03（有距离时的坐标差异） | 是 |
| C02−C05（无距离对齐） | 否 |
| C04−C06（有距离对齐） | 是 |
| C08−C07（dense补距离） | 是 |
| C06−C08（有距离时P与B） | 是 |
| C01/C02/C09/C10 placement 2×2 | 否 |

这里“是”仅指计划覆盖，不是已经取得结果。

## 预固定变换

配置准备层工具 `tools/prepare_goal_coordinate_transform.py` 复用生产 resolver、
GF(2) rank/inverse，采用独立 `numpy.Generator(PCG64(seed))`。
不修改已有无约束生成器，不将筛选放入训练，不使用全局/training RNG。

| 变换 | seed | 候选生成/接受规则 | 接受候选（从1计数） | content SHA-256 |
| --- | --- | --- | --- | --- |
| perm_v1 | 26001 | permutation(20)，首个无固定点置换 | 2 | e68864ecc5f5f3ecbfb76a8302807190730436197bba85e324a6b5ac465f7de1 |
| dense_v1 | 26002 | integers(0,2,(20,20),dtype=uint8)，首个满秩且每行/列权重[5,15] | 7 | 34e92593c51a1c923644473a752acdeaee3dc728a7870fafb3a0215ef37eca17 |

每种最多10000候选；失败报错，不换seed。固定 NumPy生成版本记录为2.4.6，来源版本1。
P/B实际内容、方向、hash、rank、weights、provenance已嵌入使用它们的每个config；
C05/C06共享P，C07/C08共享B，其他组没有未用P/B。
接受序号和 B_inverse 的行列权重仅保存在 Study 的准备审计字段，不输入模型。
B 的行权重范围6–13，列7–15；B_inverse 行4–13、列5–13，两者rank20。

## 训练、终点和执行身份

仅 puzzle-4x5-play-v0、training seed0、1M updates；4×6和多seed deferred。
log/eval/save = 5000/100000/100000；所有5tasks×20episodes，temperature0、gaussianNone、video0。
保存best和last；best依据 overall_success严格大于，平手保留早者；主终点last@1M，
报告每task和HardTaskMean=Tasks2–5均值，不用best替换主终点。

正式父目录 `/data/qijunrong/06-RL/offline-rl/exp/RLC/runs`，Study自动追加M26。
initial attempt0；已有任何选中目标目录都不覆盖。失败保留，重试只选失败配置并显式新attempt。
物理GPU1、同一scheduler两个进程是操作设置，不是科学因素。
Git/冻结/GPU单双smoke/正式训练均由用户执行，见 [MANUAL_OPERATIONS](MANUAL_OPERATIONS.md)。
配置验证完成不等于 formal_preflight_ready。
