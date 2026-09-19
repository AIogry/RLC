# M26 — 目标坐标机制诊断能力

日期：2026-09-19。范围是代码能力与 CPU 工程验证，不是正式 Study。
执行来源与验收记录见 [IMPLEMENTATION_HANDOFF.md](IMPLEMENTATION_HANDOFF.md)。

## 科学问题与边界

可操纵因素：residual / operation 坐标；零 / 精确最少按压比例辅助量；
operation 的 identity / 固定 permutation / 固定可逆 GF(2) 重编码；
actor 与 value_side 的供给位置。value_side 必须统一包含 V、Q、target Q。

固定因素：GCIQL continuous DDPG+BC、board-equality reward/mask、原始目标采样、
目标更新、动作和分布语义、Puzzle Mixer、readout。没有新 loss、reward shaping、
planner、operation executor 或学习型坐标模块。不依赖 M25。

这是 oracle mechanism diagnostic。工程测试通过不证明性能提升、操作学习或跨任务
泛化；本轮没有指定正式 alpha、训练 seeds、训练预算、GPU、并发数或输出 namespace。

## 一个入口、两种输入契约

唯一配置入口仍为顶层 `agent_overrides.goal_conditioning`。

| 契约 | v1 / 缺省 | 显式 v2 |
| --- | --- | --- |
| 条件 | 原 None/canonical/board/residual/oracle_operation | 两个 role 各自声明 coordinate/transform_id/distance_feature |
| 原始 observation | `19+4N` | `19+4N`，不变 |
| body 输入 | 原数组 | `StructuredNetworkInput(flat_inputs, token_aux)` |
| 每个按钮的 projection 输入 | current 4 + goal 4 | current 4 + goal 4 + aux 1 |
| zero 条件 | 不新增通道 | 固定保留一个零通道 |
| checkpoint | 原 legacy 契约 | 必须通过目标输入语义 fingerprint 检查 |

v2 必须声明 `schema_version: 2`、`input_schema: token_aux_v1`、`roles` 和
`transforms`。角色只有 `actor`、`value_side`；各自必须且只能包含：

```text
coordinate: residual | operation
transform_id: identity | 一个已定义 ID
distance_feature: zero | exact_press_fraction
```

布局字段沿用 M24，robot/transient goal policy 都为 zero。`transforms: {}` 表示只用
保留的 identity；完整解析配置会补上 identity 的内容、约定和元数据。
不接受旧 mode 与 v2 混用、独立 Q/target role、compute 内条件覆盖或未知配置字段。
非 identity transform 只允许用于 operation。

`resolve_goal_conditioning` 返回不可变 `GoalConditioningPlan`。
`make_goal_conditioners` 使用这一 resolver，v2 role 暴露显式 `.prepare(s,g)`；
旧 `make_goal_conditioner` 继续返回数组 callable，对 v2 明确报错。
`goal_conditioning_layout` 只提取原始布局/equality 信息，不解析矩阵或构造逆。

首版 v2 只支持 continuous GCIQL DDPG+BC、PuzzleBoardGCDataset、无 encoder / frame stack、
三个 enabled 的 relation-free Puzzle feedforward Mixer slot 和原 mean/mean_context readout。
这些限制不收窄旧 v1 的支持范围。

## 代数、静态内容与逐调用准备

`d=b(s) XOR b(g)`；需要 operation 或距离时才计算 `x=M_inverse d`。
残差加零辅助的路径不构造或使用 Puzzle inverse，故可用于奇异布局；operation / 精确
距离要求满秩 M，奇异布局失败，不提供伪逆或任意解。

| transform kind | inline payload | 方向 |
| --- | --- | --- |
| identity | `{"kind":"identity"}` | y=x |
| permutation | `{"kind":"permutation","permutation":[...]}` | `y[...,j]=x[...,p[j]]`，`P[j,p[j]]=1` |
| gf2_linear | `{"kind":"gf2_linear","matrix":[[...],...]}` | `y=B@x mod 2`，行是编码坐标、列是原操作坐标 |

每个完整 payload 含 num_coordinates、convention、content_sha256、rank、row_weights、
column_weights，以及 provenance(generator/version/transform_seed)。inline 手写来源默认
为 inline/version=1/seed=null；随机生成由独立工具明确指定种类、N 和 transform_seed。
仅 seed 或可变文件路径不是合法的 transform 定义。生成器有 max_attempts 上限，失败报错；
不隐含非恒等、derangement 或特定密度约束，后续科学协议须另行决定这些要求。

内容 hash 基于规范化矩阵和方向，不含生成来源。语义 fingerprint 基于实际使用的各角色
坐标、变换内容、距离、布局、输入版本和所需的 M 内容。改名、不同生成来源或未使用的
合法 transform 不改变实际输入语义；改变被角色使用的 P/B 或 role 则改变 fingerprint。

所有校验、求逆、hash 和生成在 setup；模块中的矩阵/置换是不可变 tuple。
每次调用从本次 s 与外部 raw g 重算，包含 V(s',g)；不写入 Dataset 或 batch。
距离始终为 `sum(x)/N`，而非 `sum(y)/N`，并广播为 `[...,N,1]`。
current observation、物理按钮 token 与 index embedding 不置换。

沿用既有 eager one-hot/binary 校验，v2 还拒绝 eager observation/goal 非有限值。
JIT 路径保持纯确定性 JAX 运算，不引入 host callback；与既有 parser 一样，traced 数据值
遵循已验证的 canonical 数据契约，形状错误仍在 tracing 时失败。

## 网络、梯度与参数

`GCActor/GCValue.prepare_inputs` 是真实 forward 使用的准备入口；它们保留原
`[obs, goal]` / `[obs, goal, action]` 顺序，只给 v2 包装 typed input。
adapter 先用原 `_split_input`，再在按钮 projection 前追加 aux。前导维、N 和辅助宽度
必须精确匹配，不隐式广播 batch；不改变 raw parser、q/qdot、robot 或 action tail。

actor 绑定 actor preparer；V/Q 绑定 value_side preparer；target Q 仍 deepcopy Q。
actor loss 内的 Q 独立接收 raw actor_goals 并使用 value_side。没有 stop-gradient 整个 Q，
也没有修改任何 GCIQL loss、update、sample_actions 或 post-gradient Polyak 函数。

每个独立 projection 新增 `token_dim` 个参数和每 sample `N*token_dim` 个 Dense MAC；
ensemble 按实际 kernel 轴计数，target 不算独立优化模型。现有 accounting 已满足要求，
未修改其实现；GF(2) 等前处理不伪装成 Dense MAC。
v2 条件间同 seed 的参数/optimizer 初始化一致；v1/v2 不声明 checkpoint 或初始化互换。

## 恢复与审计

checkpoint metadata 的 `goal_conditioning_semantics` 保存完整 resolved 条件、整体和角色
fingerprints。保存/恢复会校验 config 与实际绑定模块；恢复前校验保存语义与 fresh agent
重建语义。同形状 P/B、距离、role 或版本变更拒绝；缺少该 metadata 的 checkpoint 不能
进入 v2。legacy 没有新增 metadata。单模块恢复也不得绕过检查。

通用 reevaluation 仅做两个必要修补：注册 PuzzleBoardGCDataset；语义不匹配不能走旧
Mixer shape-compatibility fallback。受控 Puzzle replay 原来已经使用同一 reconstruction
入口，不另建路径。

`impls/diagnostics/goal_conditioning.py` 使用 Flax intermediate capture，读取真实 forward
的 prepare_inputs 输出，不重写 oracle 公式。工具入口：

- `tools/audit_goal_conditioning.py`：默认 CPU 小型 synthetic fixture，无数据加载、更新或 rollout。
- `tools/generate_goal_coordinate_transform.py`：生成单个 inline payload，不创建 Study。
- `tests/integration/test_role_goal_conditioning_real_smoke.py`：只有显式 opt-in 才执行；CPU、
  真实 4x5/4x6、小网络各两次更新与 fresh restore，无评估 rollout。缺数据按错误报告。

真实 smoke 和正式实验本轮均未运行。正式配置与 Git 发布由用户下一阶段决定。
