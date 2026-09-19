# M26 implementation handoff

日期：2026-09-19。能力实现和 CPU 工程验证完成；**不是 formal preflight ready**。
输入/参数/科学契约见 [DESIGN.md](DESIGN.md)。

## 来源、授权和工作树

| 项目 | 执行前 / 执行后事实 |
| --- | --- |
| main 工作树 | `/home/eai/Research/RLC`，源码只读 |
| base_ref | 本地 `refs/heads/main`，不是 origin/main 或 M25 |
| BASE_SHA / main HEAD | `38fccf0ff8e034723e85a2016026244f425f0309`，前后未变 |
| Git common directory | `/home/eai/Research/RLC/.git` |
| cached origin/main | `2485648137db01f456e7c25860c7419e7093409e`；本地 main ahead 1 / behind 0；未 fetch |
| main staged / unstaged | 前后均为空 |
| main untracked | 前后均只有 `docs/9-19/M26_CODE_DESIGN.md` 和 `docs/9-19/M26_CODEX_PROMPT.md` |
| M26 worktree | `/home/eai/Research/RLC-M26`；创建前确认目录、symlink、branch 和注册均无冲突 |
| M26 branch | `m26-goal-conditioning-diagnostics` |
| M26 HEAD | 创建时与现在均为 BASE_SHA；创建时工作树 clean |
| 当前实现来源 | **BASE_SHA + 未提交 tracked diff / 新增文件**，不是已经包含 M26 的 clean commit |
| 当前 dirty state | 14 个 tracked 修改、11 个 untracked 新文件；没有 staged 文件 |

仅执行了获准的 `git worktree add -b ... <BASE_SHA>` 项目 Git 写操作；没有项目
add/commit/push/merge/rebase/cherry-pick/pull、移动 main、stash/reset/clean/restore。
launcher 回归按既有测试在独立临时目录创建 toy Git fixture，不是提交 RLC 源码，也不运行真实训练。
没有修改 main、M25、其它运行/冻结 worktree 的源码，或任何正式数据/结果。
最终 main 快照与初始相同，未观察到需要协调的并发变化。

任务规范是 main 中两份未提交文档的 `main-m24-worktree-r2`，完整阅读后实施；未将它们
复制进 feature 树来冒充已提交基线。其 SHA-256 为：

```text
M26_CODE_DESIGN.md  0faaf7a843d1e95d57fc516e11b5e60aa413d1029e3af9369e1eb3e2522c8c36
M26_CODEX_PROMPT.md 2bf06d3944decbb3fc5abfc1c8b35f10f2ead4a0d63132c7030a70c480b28b28
```

已读 main 的 AGENTS、PROJECT_STATE、DECISIONS、EXPERIMENT_REGISTRY；这些规则与
BASE_SHA 无 tracked 差异。新树重新读取 AGENTS；没有适用的更深层 AGENTS。
历史上下文中的 M24A 进度是 2026-09-11 快照，本轮没有据其判断当前正式运行进度，也未
根据聊天重写任何 M24/M22 成绩或外部状态。

## 已提交 M24 基线门槛

使用 `git show <BASE_SHA>:<path>` / `git ls-tree` 核验，不以主树磁盘文件存在为依据：

- `impls/agents/gciql.py` / `networks/common.py`：conditioner 在原始输入入口接入
  actor/V/Q，target Q deepcopy，每次以该调用的 observation 重算。
- `networks/goal_conditioning.py` / `representation/puzzle_conditioning.py`：
  canonical/board/residual/oracle_operation、goal transient/robot 清零和公共 factory。
- `representation/puzzle_algebra.py` / `puzzle.py`：唯一 GF(2) operator/rank/inverse/map、
  标准 `19+4N` parser、8D paired token 和 physical index embedding。
- `utils/puzzle_datasets.py`：继承 GCDataset 的采样，仅替换 board-equality predicate；
  `main.py` 已注册该 dataset，M24A 配置联通。
- `computation/factory.py` / `structured.py`：既有 Puzzle Mixer/adapter/readout 路径。
- `experiments/M24A_puzzle_goal_conditioning_intervention/` 和现有 algebra/conditioning/
  board-dataset/integration 测试均已提交。路径映射无需改名或跨分支补文件。
- 已核验 target Polyak 使用 post-gradient online Q；无全网络梯度裁剪或共享 actor/value
  参数破坏本轮 paired-value 不变量的前置冲突。

## 实现清单

Tracked 修改（均位于 M26 树）：

```text
impls/agents/gciql.py
impls/computation/factory.py
impls/experiment/reevaluation.py
impls/main.py
impls/networks/common.py
impls/networks/goal_conditioning.py
impls/representation/interfaces.py
impls/representation/puzzle.py
impls/representation/puzzle_conditioning.py
impls/utils/checkpointing.py
impls/utils/flax_utils.py
docs/research/PROJECT_STATE.md
docs/research/EXPERIMENT_REGISTRY.md
docs/research/DECISIONS.md
```

Untracked 新文件（不能只用 `git diff --stat` 审查）：

```text
impls/representation/goal_coordinate_transforms.py
impls/diagnostics/goal_conditioning.py
tools/generate_goal_coordinate_transform.py
tools/audit_goal_conditioning.py
tests/reference/goal_conditioning.py
tests/computation/test_goal_coordinate_transforms.py
tests/integration/test_role_goal_conditioning.py
tests/integration/test_role_goal_conditioning_real_smoke.py
tests/experiment/test_goal_conditioning_tools.py
docs/research/experiments/M26/DESIGN.md
docs/research/experiments/M26/IMPLEMENTATION_HANDOFF.md
```

未修改 Dataset、GF(2) solver、Mixer block、structured body、readout、accounting 实现、
OGBench 环境、历史配置、AGENTS 或其它 agent 算法。
GCIQL 的 value_loss/critic_loss/actor_loss/total_loss/target_update/update/sample_actions
七个函数的 AST 与 BASE_SHA 完全相同，另有实际梯度和更新验证，不只依赖语法比较。

## 验证环境与真实 legacy reference

所有测试从 M26 目录执行，显式 `PYTHONPATH=/home/eai/Research/RLC-M26`、
`JAX_PLATFORMS=cpu`、`PYTHONDONTWRITEBYTECODE=1`。

```text
Python: /home/eai/Tools/miniforge3/envs/brain_nav/bin/python (3.11.15)
jax/jaxlib: 0.10.2    flax: 0.12.7    optax: 0.2.8
numpy: 2.4.6         ml-collections: 1.1.0
impls:   /home/eai/Research/RLC-M26/impls/__init__.py
ogbench: /home/eai/Research/RLC-M26/ogbench/__init__.py
devices: [CpuDevice(id=0)]
```

环境没有 pytest，初始依赖探测报告 PackageNotFoundError；使用现成 unittest，未安装或
升级环境，未修改 editable install。没有把缺依赖、skip 或真实 smoke 未运行算作通过。
核验了 56 个已加载仓库模块的实际路径，并对 20 个变化的 Python 源文件 AST / 246 条
显式 import 做检查；没有新增 M25 专属模块或外部 worktree 路径依赖。

临时证据根目录：`/tmp/rlc-m26-validation.Fgnjyr/`（不是正式 runs/diagnostics；系统清理
临时目录后这些证据可能消失，长期归档由后续人工决定）。

在新树 clean、HEAD=BASE_SHA、源码尚未修改时运行 `capture_legacy.py`，采集 flat 和
structured 各五种 None/canonical/board/residual/oracle_operation 的小型参考：初始参数、
optimizer、动作/分布、V/Q/target、loss/logs、更新、RNG、fresh restore 和下一步更新。
修改后用同一脚本/环境/fixture 与这些独立参考逐元素比较，**10/10 EXACT PARITY**。
参考从未在 main 或 M25 运行；不是两条已修改路径互相比。

```text
reference: base_reference/manifest.json
final:     parity_final/manifest.json, parity_final.log
script SHA256:    34ac806abcca19d85748ac263c3c32b890cb165435d288a04d92d35e33f2b4fa
reference SHA256: 2a81294867a9b60f9898a4fc4a99bd5ec62838bee1555f3144abb3ae78e0e801
final SHA256:     30a7d530978849cbfacbfacc9e574a9b386b41d0b896e5ca9461cb71b0f3653b
```

## 验收结果

共 **144 个 dependency-aware CPU unittest 通过，0 failed、0 skipped**，另有上述 10 条
修改前 reference parity。没有无条件执行全历史 suite。

| 分组 / 日志 | 结果 |
| --- | --- |
| 新代数/schema/roles/CLI + 既有 Puzzle algebra/conditioning/board dataset/integration；`validation_core_final.log` | 52 passed |
| M14、M17、computation provenance、MLP parity；`validation_networks.log` | 42 passed |
| checkpoint lifecycle、management、M24A、attempt、sweep、launcher、controlled-goal replay；`validation_runtime.log` | 37 passed |
| reevaluation 合成测试；`validation_reevaluation.log` | 13 passed |

逐项覆盖：

1. identity/P/B 方向、inverse、非零起始状态单次操作变化；P 的 Hamming 不变性；B 的距离
   仍来自原始 x；奇异 M/B、重复 P、维度、one-hot、NaN、未知/冲突字段失败。
2. unbatched/batched，纯代数额外 leading axes，zero/距离，current/goal/action 分离；
   物理 token 不动；ensemble 共用输入且参数独立；s' 和 actor/value raw goals 不混用。
3. 从真实 forward capture body 输入，与显式 typed 输入和相同参数 body 前向比较；
   aux 确实进入按钮 projection，未进入 robot/action context。
4. 同 v2 schema 不同坐标/变换/距离/placement 的参数树、初值、optimizer、RNG 一致；
   新增 kernel 参数/MAC 按实际 ensemble 数量正确计数，target 非独立优化 slot。
5. actor Q-only 梯度非零，critic/value/target 子树梯度为零，dQ/da 非零；V/Q loss 的
   target 分支不受 grad_params 更新；初始化 target 同步、一步 post-gradient Polyak 正确。
6. 在 residual 和 operation 两种 value_side 下各做 4 步 paired 更新，只改变 actor：
   V/Q/target 参数、各自 Adam mu/nu、Adam count 和 RNG 均逐元素一致。
7. JSON 配置保存后丢弃原 agent，用不同初始化 seed 构建 fresh agent 再 restore；
   动作/V/Q/target、state、role fingerprints、同 batch 下一步更新与 logs 一致。
8. 同形状 P/B、distance、role 和版本变动被拒绝；缺 metadata 的 legacy checkpoint
   不进入 v2；单模块 restore 和 reevaluation fallback 不能绕过语义检查。
9. 构建后 eager/JIT preparation 和网络 forward 不重做 operator/rank/inverse/hash/随机
   transform；residual+zero 无 oracle；Dataset 不解析 role 或构造 P/B/M inverse。
10. 真实数据 sampler 代码未变，合成 dataset 上原始采样/trace/reward/mask/RNG 一致；
    4x5/4x6 的标准生产宽度另以 CPU synthetic 输入验证，没有加载真实数据。

独立工具命令也已执行：生成 N=9、显式 seed 的 GF(2) payload 和默认 production-input
synthetic audit，分别写到该临时根目录的 `generated_transform.json` / `production_audit.json`。
工具生成来源/hash 与拒绝覆盖已有文件由 CLI 测试覆盖。

## 复查指令（不涉及发布或正式实验）

```bash
cd /home/eai/Research/RLC-M26
git status --short --untracked-files=all
git diff --check
git diff --stat
git ls-files --others --exclude-standard

env PYTHONPATH=/home/eai/Research/RLC-M26 JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 \
  /home/eai/Tools/miniforge3/envs/brain_nav/bin/python -m unittest \
  tests.computation.test_goal_coordinate_transforms \
  tests.integration.test_role_goal_conditioning \
  tests.experiment.test_goal_conditioning_tools \
  tests.computation.test_puzzle_algebra tests.computation.test_puzzle_conditioning \
  tests.integration.test_gciql_puzzle_goal_conditioning \
  tests.integration.test_puzzle_board_dataset -v
```

其余已执行命令使用同样的 env/Python 前缀：

```text
-m unittest tests.integration.test_m14_base_algorithm_computation tests.integration.test_m17_modular_structured_computation tests.integration.test_computation_provenance tests.computation.test_mlp_parity -v
-m unittest tests.experiment.test_checkpoint_lifecycle tests.experiment.test_management tests.experiment.test_m24a tests.experiment.test_run_attempt tests.experiment.test_sweep tests.integration.test_run_study_launcher tests.diagnostics.test_puzzle_controlled_goal_replay -v
```

reevaluation 使用 unittest loader 选择 `ReevaluationTest` 中除
`test_m10a_inventory_has_33_valid_checkpoints` 外的全部 13 项；该外部历史 inventory 不属于
本轮合成恢复依赖，未读取其正式 checkpoint 来补测试数。
legacy 最终命令：

```text
python /tmp/rlc-m26-validation.Fgnjyr/capture_legacy.py --output /tmp/rlc-m26-validation.Fgnjyr/parity_final --compare /tmp/rlc-m26-validation.Fgnjyr/base_reference
```

其中 `python` 指上述解释器和显式 CPU/PYTHONPATH 环境。复跑必须选择新的 `--output`，
脚本拒绝覆盖现有证据。

## 未执行、适配与下一步

未执行真实数据 smoke、GPU smoke、正式训练、长 rollout 或 reevaluation campaign；没有
创建 M26 Study/config matrix，没有查看/分配 GPU 或宣称输出 namespace 已完成 formal gate。
新的真实 smoke 入口需要 `RLC_RUN_GOAL_INPUT_REAL_SMOKE=1` 与显式 dataset root，默认不运行；
本轮未检查其真实资源可用性，也没有把 skip 算作 passed。旧的历史真实 smoke 同样未运行。

无科学设计偏离。最小工程适配：两种轻量 NamedTuple 共置于 representation/interfaces
以避免循环依赖；使用实际 forward 的 Flax capture 审计；accounting 已满足要求无需改；
reevaluation 原本缺少 PuzzleBoardGCDataset 注册且 ValueError 会进入 legacy fallback，
因此补上注册与专用语义异常分流。没有遗留的基线或验证阻塞。

| 状态 | 结果 |
| --- | --- |
| main_m24_base_verified | yes |
| m26_worktree_created | yes |
| implementation_ready | yes，未提交 source diff |
| cpu_validation_passed | yes，仅上述覆盖范围 |
| real_data_smoke_not_run | yes，未获本轮执行授权 |
| formal_study_not_defined | yes |
| formal_launch_not_authorized | yes |

下一步只建议用户审阅能力实现，决定是否提交，以及开始制定正式 M26 配置。
main 尚有两份用户 untracked 文档，应继续保留；本交付不提供直接 merge/push 或未知
GPU/输出路径的 launch 命令。将来正式运行仍需要用户决定协议、提交和干净 frozen 来源。
