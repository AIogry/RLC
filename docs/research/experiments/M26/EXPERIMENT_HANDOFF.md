# M26 实验配置阶段交接

2026-09-19。配置、继承审计、CPU工程验证完成；**不是 formal_preflight_ready**。
人工下一步：[MANUAL_OPERATIONS.md](MANUAL_OPERATIONS.md)，从C的配置审查/提交开始。

## 当前事实与来源

| 项目 | 本轮实查 |
| --- | --- |
| main | /home/eai/Research/RLC，main，38fccf0ff8e034723e85a2016026244f425f0309，clean |
| M26 | /home/eai/Research/RLC-M26，m26-goal-conditioning-diagnostics |
| 入口capability SHA | f0484f425cfd4092d1b95c7bf3e1348ad00ea0ff，已独立commit，入口clean |
| 本轮来源 | capability SHA + 未提交的实验配置/工具/测试/文档diff |
| 结束dirty | 3个tracked上下文修改、22个新增文件；没有staged修改 |
| common Git directory | /home/eai/Research/RLC/.git |
| 远端核验 | ls-remote读到main=38fccf0…、M26分支=f0484f4…；cached refs也一致；本轮未fetch |
| main旧untracked说明 | 不存在；实际任务说明已移至 /home/eai/Research/docs/9-19/ |
| 正式M26 namespace | 检查时 /data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M26 不存在 |

已完整阅读实际AGENTS（无更深层）、项目三份上下文、能力DESIGN/IMPLEMENTATION_HANDOFF、
本轮两份要求和人工模板，再核验M24配置/实际resolved/runtime、resolver/generator、
Dataset/网络audit、checkpoint/reevaluation、Study管理/launcher及相关测试。
能力阶段handoff的“当时未提交/144 tests”保留原历史，不把它冒充本次结果。

输入文档为 `m26-4x5-seed0-manual-release-r1`：

```text
M26_EXPERIMENT_PLAN.md          8e712ce79492ea1e1c9bb831ae029c7913c711fbd31a887497964e7de7697508
M26_CODEX_EXPERIMENT_PROMPT.md   b4e64a8965818f9481ef1923c4c407daf9ba38e0d9e5c2a2420be98ad004bec6
M26_MANUAL_OPERATIONS.md        deff02af502ff4e270fa09b09f6019fc685a2282d0672c89c2472c5514f29272
```

## 交付与科学完成度

- `experiments/M26_puzzle_goal_coordinate_diagnostics/`：Study、10个独立稳定配置、README。
  仅4×5/seed0，全v2/9D；默认初始C003–C010共8组。C001/C002只是未排入首批，不是假completed。
- P(seed26001)接受第2候选、B(seed26002)第7；实际内容/来源/hash/rank/weights嵌入配置；
  筛选边界、接受序号和inverse统计入Study。transform seeds不生成额外training runs。
- 配置继承已对照M24 C003/C005 attempt001真实文件，完整agent只有goal_conditioning允许差异，
  协议无冲突。详见 [M24_INHERITANCE_AUDIT](M24_INHERITANCE_AUDIT.md)。
- 四个工具：受约束静态变换准备、CPU Study输入/真实数据审计、字段级继承审计、默认不执行的
  通用Study GPU smoke。新增两个测试模块；四份本阶段研究文档和三个项目上下文更新。
- 没有改impls、ogbench、既有生成器/算法/损失/采样/readout/launcher或M24/M25配置。
  原M21/M22 smoke不满足本轮默认opt-in、生产配置、全batch、阶段显存/短评估要求，因此新建
  通用工具；没有运行或改写旧benchmark。

P hash `e68864ecc5f5f3ecbfb76a8302807190730436197bba85e324a6b5ac465f7de1`；
B hash `34e92593c51a1c923644473a752acdeaee3dc728a7870fafb3a0215ef37eca17`。
不能用M24的v1/8D成绩/checkpoint补C01/C02；首批8组完成仍缺两个匹配参照，placement 2×2等
依赖它们的结论pending。没有新训练成绩；单seed/P/B只用于方向选择。

准确逐文件allowlist已写入人工手册C节（25文件）；无需把数据/log/checkpoint提交。

## 本次验证

工作目录始终为M26；公共命令前缀：

```bash
cd /home/eai/Research/RLC-M26
export PYTHONPATH=/home/eai/Research/RLC-M26
export JAX_PLATFORMS=cpu
export PYTHONDONTWRITEBYTECODE=1
M26_PY=/home/eai/Tools/miniforge3/envs/brain_nav/bin/python
```

这些export只用于该CPU复查shell，人工正式命令显式去除CPU设置并强制CUDA，不改共享安装。
Python3.11.15，JAX/JAXlib0.10.2，NumPy2.4.6；77个实际加载impls/ogbench模块都位于M26。
没有安装/升级依赖。当前测试使用现有unittest，不依赖缺失pytest。

共 **115个不同的CPU unittest通过，0失败、0skip**，不重复计入重跑，也不计入历史144项。

| 日志（均在下述证据根） | 本次结果 / 边界 |
| --- | --- |
| test_preparation_verified.log | 22项：8个M26矩阵/生产前向测试 + 14个工具/安全测试；GPU设备/worker控制流为mock，不是实际GPU验收 |
| regression_goal_inputs.log | 46项：v2角色/梯度/多步paired-value/初始化/语义checkpoint/fresh恢复与M24 legacy、board dataset、变换/CLI |
| regression_lifecycle.log | 34项：M24 Study、checkpoint生命周期、management、attempt、sweep、controlled goal replay合成测试 |
| regression_reevaluation.log | 13项：ReevaluationTest除外部历史inventory项外的合成测试 |

实际对应命令（以上公共前缀）：

```bash
"$M26_PY" -m unittest tests.experiment.test_m26 tests.experiment.test_study_preparation_tools -v
"$M26_PY" -m unittest tests.integration.test_role_goal_conditioning tests.integration.test_gciql_puzzle_goal_conditioning tests.integration.test_puzzle_board_dataset tests.computation.test_puzzle_conditioning tests.computation.test_goal_coordinate_transforms tests.experiment.test_goal_conditioning_tools -v
"$M26_PY" -m unittest tests.experiment.test_m24a tests.experiment.test_checkpoint_lifecycle tests.experiment.test_management tests.experiment.test_run_attempt tests.experiment.test_sweep tests.diagnostics.test_puzzle_controlled_goal_replay -v
"$M26_PY" - <<'PY'
import unittest
from tests.experiment.test_reevaluation import ReevaluationTest
names=[n for n in unittest.defaultTestLoader.getTestCaseNames(ReevaluationTest)
       if n!='test_m10a_inventory_has_33_valid_checkpoints']
result=unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(ReevaluationTest(n) for n in names))
raise SystemExit(not result.wasSuccessful())
PY
```

未跑全历史suite，也没有跑包含 `git init/add/commit` fixture 的旧launcher测试（本轮所有Git写
均禁止，包括toy fixture）。以现有sweep单测、新的配置子集测试、真实wrapper的3次dry-run和
人工Bash语法检查覆盖本轮直接依赖。没有把未运行项算作skip或pass。

独立CPU审计命令已执行成功（复跑须换新output，工具拒绝覆盖）：

```bash
"$M26_PY" tools/audit_goal_conditioning_study.py \
  --study experiments/M26_puzzle_goal_coordinate_diagnostics/study.yaml \
  --dataset-root /data/qijunrong/06-RL/offline-rl/data/raw_ogbench \
  --real-data-smoke-config M26-C008 \
  --output /tmp/rlc-m26-experiment.qO8B5C/cpu_audit.json

"$M26_PY" tools/audit_study_inheritance.py \
  --study experiments/M26_puzzle_goal_coordinate_diagnostics/study.yaml \
  --reference-study experiments/M24A_puzzle_goal_conditioning_intervention/study.yaml \
  --reference-configs M24A-C003,M24A-C005 --allow-agent-field goal_conditioning \
  --reference-run /data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M24A/M24A-C003__4x5_residual_mixer_l2_alpha0p4/puzzle-4x5-play-v0/seed_000__attempt_001 \
  --reference-run /data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M24A/M24A-C005__4x5_oracle_operation_mixer_l2_alpha0p4/puzzle-4x5-play-v0/seed_000__attempt_001 \
  --output /tmp/rlc-m26-experiment.qO8B5C/inheritance_audit.json
```

生产宽度10组真实forward审计通过，参数/optimizer/RNG逐元素相同。全部参数含target共
1,679,354：actor280109、V279337、Q559954、targetQ559954；target不是独立科学模型。
init状态fingerprint：`1cafcc0cfb859f2d24593fa90126c21af1474889e54907e94d4c713d57642c4f`。
实际captured输入与独立NumPy d/x/P/B/原x距离核对，physical current保留、robot/transient目标清零、
Q action tail保留、actor按钮projection kernel为(9,128)。

真实CPU数据覆盖：只读取4×5 train/val，10组各2个batch4的原始采样及trace/reward/mask配对；
C008派生tiny模型2个batch2更新，finite loss，fresh恢复动作/V/Q/target及参数/optimizer/RNG一致。
这是低成本工程测试，未跑4×6、未评估rollout、不代表生产batch1024 GPU资源验收。

wrapper三次 `--dry-run` 的完整协议见人工手册F2，实跑日志分别为
`dry_run_initial.log` / `dry_run_full.log` / `dry_run_references.log`：
planned/remaining分别8/10/2，completed均0；未创建run目录。GPU smoke单/双CLI也只生成plan，
`execute_requested=false`，没有创建smoke输出路径。工具help、手册Bash/内嵌Python语法均已检查。
最终静态检查 `final_static_checks_pass.log` 核对25文件allowlist、M26链接、全部新文件whitespace、
Python AST、main/runtime/M24未改及正式namespace未创建；生成YAML的额外EOF空行已清除。

开发中发现并修复两个审计/测试问题：通用tree_fingerprint只接受mapping/array，故tuple/optimizer
先走Flax state serialization；ConfigDict与dict比较先规范化。初次失败日志
`cpu_audit.log` / `test_m26.log`保留，没有隐藏失败或放宽科学断言。最终验证按上表计算。
真实CPU加载有非致命X11显示及Box dtype警告；没有渲染、GPU调用或rollout。

## 证据、资源和剩余gate

证据根：`/tmp/rlc-m26-experiment.qO8B5C/`，不是正式run。

```text
cpu_audit.json SHA256         e75eda02a100cb46bc07f7f12ad8b3a368750ce4a09be6389077492234e0b969
inheritance_audit.json SHA256 c1d00f3e5fb6acf0a583297d08e5cb943a819b494d905758cae77a732e960d32
```

长期归档由用户在冻结后复制到外部engineering/release目录，具体命令已给；/tmp可能被清理。
不要将庞大报告、数据或训练checkpoint放入仓库。最终训练SHA尚不存在，不能预填或写回待提交文档。

沙箱内初次 `nvidia-smi`/远端DNS查询失败；只读sandbox外查询成功，非实际driver故障。
GPU1为RTX4090、24564MiB，UUID `GPU-8c1feab3-0fcf-396d-1a2e-f12af187f066`；查询时18MiB、
0% utilization、无计算进程。GPU0有其他进程，未触碰。主存约46GiB available、磁盘约3.3TiB free。
只读资源快照不预留GPU、不代替启动前复查；未建立外部engineering目录。

| Gate | 状态 |
| --- | --- |
| 已提交M24、独立capability、远端publication | verified |
| 10组配置 / 8+10+2筛选 / M24继承 / P/B / CPU测试与真实tiny smoke | passed，限上述覆盖 |
| 本轮配置commit/push/merge | pending，用户操作 |
| 包含全部实验配置的最终clean detached TRAIN_SHA | pending，用户操作 |
| GPU1生产宽度batch1024单/双smoke | pending；入口与CLI已核验，GPU执行未授权 |
| 冻结后数据/源码/资源/选中namespace/preflight manifest | pending；当前快照不能替代 |
| 正式M26训练 | 未运行；用户控制 |

没有科学协议冲突。唯一历史状态冲突是9-11文档的“未启动”与本次两份completed M24 artifact；
已明确时点，不扩大为整个旧campaign的结论。全程无Git写、GPU smoke、正式训练、长评测、
kill其他进程、依赖升级或新worktree；main/M25/冻结树/历史结果保持未修改。
