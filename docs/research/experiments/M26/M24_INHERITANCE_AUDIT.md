# M26 对 M24 的继承审计

2026-09-19，本次实际读取结果，**runtime_verified**，不是仅声明值。
工具：[audit_study_inheritance.py](../../../../tools/audit_study_inheritance.py)。
机器报告：`/tmp/rlc-m26-experiment.qO8B5C/inheritance_audit.json`，status=pass、errors=[]。

## 精确来源

M24 Study和config来自已提交 `38fccf0ff8e034723e85a2016026244f425f0309`；本轮能力入口
`f0484f425cfd4092d1b95c7bf3e1348ad00ea0ff` 中这些历史文件完全未改。
只读取下面两个精确run的 resolved_config.json / runtime_metadata.json，没有遍历/载入历史checkpoint。

共同根：`/data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M24A/`。

| 来源 | 相对run路径 |
| --- | --- |
| residual / G2 | M24A-C003__4x5_residual_mixer_l2_alpha0p4/puzzle-4x5-play-v0/seed_000__attempt_001 |
| oracle_operation / G4 | M24A-C005__4x5_oracle_operation_mixer_l2_alpha0p4/puzzle-4x5-play-v0/seed_000__attempt_001 |

两个runtime均记录training seed0、attempt1、source SHA=`38fccf0ff8e034723e85a2016026244f425f0309`、
git_dirty=false、detached、status=completed。本次未审核其分数或完整M24/M22运行状态。
旧PROJECT_STATE/M24 README里的“0/6未启动”是9-11历史快照，不能代表这两个当前artifact；
本次不重写历史handoff，也不凭这两个run推断整个campaign已完成。

| 文件/指纹 | C003 | C005 |
| --- | --- | --- |
| resolved_config_fingerprint（重算并与两文件核对） | e829d0fc7a9cfd98ba94e354b2ac369ebbd9e69e5c79777765ecc2d01448b80e | d47ca641182d70f6c481747accad401faf87732fb48e5cc8f3b3f11b09a3d5a0 |
| resolved_config.json SHA256 | 8f0e01c54b3e1ca46c7322f795ba93004b2103ba92ddad14b41956a9c3841579 | b0ee9b815d7978128f78ea299fbbbf80c41afaff6b8eae286075c4debe28b31d |
| runtime_metadata.json SHA256 | 58e3030a92ca666c7a24803d9286572f27ddc160f3ba6f7a0863f8baa0e09e62 | 6867106a85cfbe3684cb6cb34ca697553e31626d06c1d6d35e70804bff66d0f1 |

## 字段级继承

两份实际 agent 与当前 M24 声明解析结果完全相等。M26全部10组与两份M24解析agent逐字段比较，
唯一允许差异的agent顶层字段是 `goal_conditioning`；compute没有被排除。没有其他差异。
完整路径差异和字段名称列表在机器报告，避免把庞大resolved config复制进上下文。

| 字段 | 本次核验值 |
| --- | --- |
| agent_name / actor_loss / alpha | gciql / ddpgbc / 0.4 |
| lr / batch / budget | 0.0003 / 1024 / 1000000 |
| discount / expectile / tau | 0.99 / 0.9 / 0.005 |
| actor/value hidden dims | [512,512,512] / [512,512,512] |
| layer_norm / const_std / discrete | true / true / false |
| Dataset / success | PuzzleBoardGCDataset / board_equality |
| value sampling | current0.2 / trajectory0.5 / random0.3 / geometric=true |
| actor sampling | current0 / trajectory1 / random0 / geometric=false |
| gc_negative / p_aug | true / 0.0 |
| encoder / frame_stack | null / null |
| compute placement | actor + value + critic；target Q沿用Q结构 |
| structure / topology / block / credit | puzzle_tokens / feedforward / mlp_mixer / direct |
| widths / blocks | token128、robot128、token-MLP64、channel-MLP256、2blocks |
| index embedding / tm_mode | true / none |
| readout | 配置mean，实际accounting mean_context；未改变 |
| current observation / goal robot/transient | current全部保留 / goal两类均zero |
| target update | 原GCIQL post-gradient Polyak；impls未改，相关回归通过 |
| log / eval / save | 5000 / 100000 / 100000 |
| eval | all五tasks、每task20episodes、temperature0、gaussianNone、video0 |
| checkpoints | best+last；overall_success严格大于/平手保留更早；主终点last@1M |

协议检查使用真实 launcher 的全部12个有效字段，不把缺字段当None来放过；
selection rule另从runtime checkpoint_lifecycle核对。Study协议仅去掉历史的
`formal_training_started`声明字段后完全相等。

有意差异 allowlist：v1→v2、8→9D固定aux参数化、角色/坐标/P/B/距离；环境范围缩为仅4×5、
training seeds仍为[0]；Study/config/slug/output/attempt身份不同；物理GPU1双进程操作设置。
不允许 alpha/loss/采样/readout/训练预算等静默变化，不声明历史M24与M26参数初始化可互换。

## 数据身份及真实CPU覆盖

目录：`/data/qijunrong/06-RL/offline-rl/data/raw_ogbench`。
机器报告：`/tmp/rlc-m26-experiment.qO8B5C/cpu_audit.json`，包含所有NPZ成员shape/dtype/CRC、
文件大小/mtime/SHA256、实际import路径和10组真实forward的fingerprints。

| 文件 | 文件bytes | observations / actions shape | SHA256 |
| --- | --- | --- | --- |
| puzzle-4x5-play-v0.npz | 816520065 | [3003000,99] / [3003000,5] | 7f25e3161280aa4ff1234a279ba7a0a4466df6a8e4f5af64fd2b1bfe8df892cc |
| puzzle-4x5-play-v0-val.npz | 81683658 | [300300,99] / [300300,5] | 26bfb8d59290be2838a4c8185fc8221355ff45027d4c4bff4e26f6f4a7c2624c |

真实train/val分别对10组使用相同派生seed，各连续抽2个batch4，逐元素核对raw batch、
transition/value/actor goal indices、reward/mask完全相同。仅C008做tiny工程模型2次CPU更新
（batch2、hidden8、token/robot/channel8、tokenMLP4、1block）及不同初始化seed的fresh restore，
动作、V/Q/target、参数/optimizer/RNG相同。没有rollout。小模型不是生产GPU显存/速度证据。
生产宽度所有10组另行初始化/前向，输入宽度与参数初值一致。

数据在核验时可读；后续仍须保持immutable并在冻结后的GPU smoke/preflight重核对。
报告只能证明当前继承与工程边界，不能提供新的科学成绩或formal-ready结论。
