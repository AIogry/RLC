# M20A Phase 1：跨任务 Oracle Relation 诊断实现与交接

> 历史状态说明：本文记录 Phase-1 完成时全部 18 个 config 尚处于 blocked skeleton 的事实。后续由用户正式冻结 Cube threshold 并激活 Phase-2；当前可执行协议见 [M20A Phase-2 freeze](M20A_phase2_freeze.md)。Phase-1 implementation/audit 结论本身未被改写。

## 状态与范围

本文件记录 M20A 的 Phase 1。已完成的目标是建立一个受控的、可审计的
`Correct / Shuffled / Zero` relation 框架；它不是任何正式 RL 结果报告。

- Phase 1 状态：`PASS`。
- 正式 Phase 2 配置数：18，全部 `executable: false`。
- 正式 M20A run root：`/data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M20A` 在 doctor 检查时不存在。
- 真实数据 smoke 仅写入 `/tmp/m20a_phase1_smoke`，不在正式 run root。
- 未执行 Git 命令。按照用户的更严格长期约束，连只读 Git 命令也没有执行。因此实际起始 HEAD 记录为：`not independently read by Codex; user-only Git policy`；这不是对某个 HEAD 的猜测。

Doctor 的最终标准输出为：

```text
M20A PHASE-1 IMPLEMENTATION/AUDIT: PASS
FORMAL TRAINING: BLOCKED PENDING USER PHASE-2 FREEZE
```

## 科学问题与解释边界

在同一 task 内固定 GCIQL、entity representation、RelationAugmenter、Mixer、readout、初始化和数据采样后，M20A 未来比较 task-grounded 的正确 relation，是否优于 edge-count 匹配但语义错误的 relation。

主要对比是 `Correct − Shuffled`；`Correct − Zero` 为次要对比。`HybridContextQuery − Mean` 只能解释为 readout-package contrast，因为 Hybrid 多了 Q/K/V、attention 和更宽的 fusion 输入，并不 parameter/MAC matched。Puzzle、Cube、Scene 的 raw success 不会合并成跨任务的“universal relation score”。单 seed 的 Phase 2 screening 也不能产生显著性或因果性声明。

本实现不允许将未来结果表述为“证明模型学会 relational reasoning / compositional reasoning”，也不把 oracle relation 称为 learned relation。

## 冻结协议

未来每个 M20A Phase 2 cell 均在 study/config/runtime override 三层显式固定 `alpha=1.0`，不会依赖当前 GCIQL 默认的 `alpha=0.3`。本地 M16B/M19A 记录了 Puzzle 的 `alpha=1.0` 校正协议；同时，Phase 1 没有在本地找到覆盖三个 manipulation-play 环境的独立官方 GCIQL command grid，因此这一局限已显式写入 study metadata，而没有被静默替换为一个推测的来源。

| 项目 | 冻结值 |
| --- | --- |
| 算法 | GCIQL，`actor_loss=ddpgbc`，`const_std=true`，`layer_norm=true`，`discrete=false` |
| 优化 | `lr=3e-4`，batch `1024`，`discount=0.99`，`expectile=0.9`，`tau=0.005` |
| 训练 / 评估 | 1,000,000 steps；每 5k log；每 100k eval/save；`eval_tasks=all`；每 task 50 episode；temperature 0 |
| 数据 | `GCDataset`；value `(0.2,0.5,0.3,geom=true)`；actor `(0,1,0,geom=false)`；`gc_negative=true`；`p_aug=0`；`frame_stack=null` |
| 主终点 | `final@1M overall success`；辅助为 best、best step、last-3 mean、100k–1M normalized AUC、per-task final |
| Backbone | token dim 128；MLP-Mixer L2；token hidden 64；channel hidden 256；`tm_mode=none`；feedforward/direct |
| RelationAugmenter | hidden 256，GELU，两个无 bias Dense，无 normalization/dropout，输出 128 |
| Hybrid readout | `query_dim=128`，scaled dot-product attention |
| 随机性 | training seed 0；Puzzle shuffle design seed 20020；Cube derangement 固定 `[1,2,0]` |

M16B/M19A 的历史 Puzzle checkpoint 若进入未来 M20A 数值表，必须只做 checkpoint-only 的 50-episode 重评估；它们此前的 20-episode 数值只能作历史参考，不能与 M20A 主表混用。

## First-class representation contract

`StructuredRepresentation` 保持前四个 positional 字段不变，并在末尾追加两个带默认值的字段：

```python
StructuredRepresentation(
    tokens,
    context=None,
    mask=None,
    auxiliary=None,
    relations=None,
    relation_mask=None,
)
```

公开未 batch 边界为：`tokens [T,D]`、`context [C] | None`、`mask [T] | None`、`relations [T,T,K] | None`；内部统一为 batch 形式：`[B,T,D]`、`[B,C]`、`[B,T]`、`[B,T,T,K]`。`relation_mask` 可为 `[B,T,T]` 或 `[B,T,T,K]`。若 entity mask 存在，实际使用的边 mask 为 source mask、target mask 与 optional relation mask 的交集。

`relations=None` 与 `relation_mode=legacy_none` 仍是历史 bypass；M20A 的 `zero` 则显式构造全零 relation tensor 并真实执行 RelationAugmenter。因而两者不能混称为 “None”。

## 三个 entity parser

所有 parser 都先从真实 OGBench observation layout 审计而来；不从特权 dataset field 取 production 输入。

| 环境 | 观测 / entity 切分 | token 与 context |
| --- | --- | --- |
| `puzzle-4x4-play-v0` | 83 = robot 19 + 16 buttons × 4 | 每个 raw token 是 `[button_i(state), button_i(goal)]` 的 8 维对，经既有共享 button projection 和绝对 index embedding 后为 `[16,128]`；robot state/goal 进入 context。critic action 只进 context，不 broadcast 到 token。 |
| `cube-triple-play-v0` | 46 = robot 19 + 3 cubes × 9；每 cube 为 xyz(3)、quaternion(4)、cos/sin yaw(2) | 每 cube 保留完整 9 维 canonical feature；state-goal pair 是 18 维，经共享 `CubeEntityEncoder` 成 `[3,128]`；robot state/goal 是 context。 |
| `scene-play-v0` | 40 = robot 19 + cube 9 + button_0 4 + button_1 4 + drawer 2 + window 2 | state-goal raw pair 分别为 cube 18、两个 button 各 8、drawer 4、window 4；使用 type-specific Cube/Button/Drawer/Window encoder，button 共用 encoder 且叠加稳定 role embedding，得到 `[5,128]`。 |

Cube reset 中 task role 可能被 permutation，但同一 physical slot 的 state/goal 在 observation 中仍成对出现；raw per-cube feature 不带稳定语义 ID。故 `slot_identity_embedding: false`，也不对 token features 进行 Cube shuffled permutation。这是 source audit 结论，不是 performance 选择。

## Relation schema、方向与三种处理

全局方向定义为 `R[b,i,j,k]=1`：source entity `i` 经 relation type `k` 指向 target entity `j`。production tensor 是与 token compatible 的 float；其逻辑语义始终是 binary 0/1。

| Task | shape | Correct | Shuffled | Zero |
| --- | --- | --- | --- | --- |
| Puzzle | `[B,16,16,1]` | `i → j` 当 press button `i` 会 toggle `j`：self + in-bounds 上下左右 | `P R P^T`，固定 `P=[2,10,15,12,11,8,7,3,6,0,5,14,9,4,1,13]`，且该 P 非 toggle graph automorphism | 同形状全零 |
| Cube | `[B,3,3,3]` | `current_support`：current lower → upper；`goal_support`：goal lower → upper；`goal_conflict`：current occupant → goal owner | 每 type 用固定 `P=[1,2,0]` 做 `P R P^T`；不动 entity features | 同形状全零 |
| Scene | `[B,5,5,1]` | `button_0 → drawer`，`button_1 → window` | 仅交换两个 target：`button_0 → window`，`button_1 → drawer` | 同形状全零 |

Puzzle relation 是从 `PuzzleEnv.post_step` 实际的 self+cardinal toggle rule 得到的静态图。Scene mapping 来自 `_apply_button_states`，而不是按照 task 名猜测。Cube 仅使用该 forward call 得到的 `(state, goal)`：support 同时要求 xy alignment 和垂直关系，conflict 检查 current cube 是否靠近另一 cube 的 goal-success region，始终排除 self edge。

`relation_mode`、`relation_kwargs`、`relation_augmenter`、`relation_augmenter_kwargs` 是 computation/factory 的一等配置字段；不会藏在通用 `structure_kwargs` 中。所有 18 个未来 config 都显式包含它们。

## RelationAugmenter

对 tokens `H∈R^{B×T×D}`，先将 relation 与有效 mask 相乘为 `R̄`。对每个 type `k`：

\[
o_i^k=\frac{\sum_j \bar R_{i,j,k}H_j}{\max(\sum_j \bar R_{i,j,k},1)},\qquad
u_i^k=\frac{\sum_j \bar R_{j,i,k}H_j}{\max(\sum_j \bar R_{j,i,k},1)}.
\]

将所有 outgoing 与 incoming summary 拼接为 `f_i=[o_i^1,…,o_i^K,u_i^1,…,u_i^K]`，并计算：

\[
\Delta H_i=W_2\,\operatorname{GELU}(W_1 f_i),\qquad H'_i=H_i+\Delta H_i.
\]

`W1` 为 `2KD→256`、`W2` 为 `256→128`，均无 bias；没有 normalization、dropout 或 learned constant token。`H_i` 不会被拼入 relation MLP 输入。故 `R=0` 时 `f=0`、`ΔH=0`、`H'=H`，已经以 strict numerical test 验证。

这保证 Zero/Correct/Shuffled 的 parameterized architecture 和 nominal compute 相同，但 Zero 的 relation MLP 输入永远为零，可能是 dormant path；不能声称三者 effective trainable capacity 相同。

## Readout

Mean 条件直接复用既有 `MeanContextReadout`，没有复制新的“等价”实现。Hybrid 的正式名称为 `HybridContextQueryReadout`，以 context（critic 时也包含 action）产生 query：

\[
m=\operatorname{maskedMean}(H),\quad q=W_Qc,\quad k_i=W_KH_i,\quad v_i=W_VH_i,
\]
\[
a_i=\operatorname{softmax}_i(q^\top k_i/\sqrt{128}),\quad
q_{\rm summary}=\sum_i a_iv_i,
\]

invalid token 在 softmax 前 masked 为 `-∞`，其 attention 权重为 0。最终 `concat[m,q_summary,c]` 进入 fusion Dense，并沿用 MeanContextReadout 的 slot-specific final activation/LayerNorm 语义。Q/K/V 均为 learned linear projection，`query_dim=128`。

## GCIQL wiring：objective 未改，relation 按 call 实时构建

没有修改 GCIQL 的 value objective、critic TD objective、DDPG+BC actor objective、target critic/Polyak、action distribution、reward/mask、goal relabeling、dataset sampling 或 optimizer。Relation 只是 representation/computation input。

doctor 在 GCIQL forward path 做的 call-specific gate 如下：

| 网络调用 | 实际 relation |
| --- | --- |
| actor loss | `R(observations, actor_goals)` |
| actor-loss 中 critic 评价 actor action | `R(observations, actor_goals)` |
| value loss | `R(observations, value_goals)` |
| critic TD | `R(observations, value_goals)` |
| next-state value | `R(next_observations, value_goals)` |
| evaluation | `R(current_observation, evaluation_goal)` |

测试构造 `actor_goals != value_goals` 的 batch，证明 actor 与 value relation 的 tensor 确会不同；hook/diagnostic 同时证明 critic TD 对齐 value relation，actor-critic 对齐 actor relation，evaluation 对齐传入的 evaluation goal。

## 参数、RNG 与 accounting 控制

对于同一 task、readout、training seed=0，Zero/Correct/Shuffled 的 module tree、parameter paths、shapes、count、初始化数值、随机 split 顺序完全相同。仅 relation tensor 内容不同。真实 `GCDataset.sample(..., return_sampling_trace=True)` 审计还验证同 seed 的 transition IDs、actor/value goal IDs、observations、next observations、actor goals、value goals 在 Z/C/S 间逐项相同。

Generic accounting 明确分开 logical active edges 和已执行的 dense relation aggregation cost。实现使用 dense `[T,T,K]` reduction，稀疏 relation 不能被报告为硬件 sparse saving。每个 structured body forward 的 recorded dense aggregation cost 为：Puzzle `65,536`、Cube `6,912`、Scene `6,400`；其含义是 accounting 定义的 dense reduction 工作量，不是由 audit sparsity 推出的加速。

| Task / readout | GCIQL actor+value+critic params | relation augmenter params | actor dense relation aggregation cost |
| --- | ---: | ---: | ---: |
| Puzzle / Mean | 1,505,928 | 393,216 | 65,536 |
| Puzzle / Hybrid | 1,966,216 | 393,216 | 65,536 |
| Cube / Mean | 2,013,728 | 917,504 | 6,912 |
| Cube / Hybrid | 2,474,016 | 917,504 | 6,912 |
| Scene / Mean | 1,502,256 | 393,216 | 6,400 |
| Scene / Hybrid | 1,962,544 | 393,216 | 6,400 |

Hybrid 相比 Mean 多出 query parameters（actor/value 各 49,536，双 critic 合计 99,072）；因此只可视为 readout package 对比。Correct 与 Shuffled 在真实 batch 上每 sample、每 channel 的 active-edge count 都严格相等。

## 代码与测试变更

| 范围 | 文件 / 改动 |
| --- | --- |
| relation core | 新增 `impls/representation/relations.py`、`impls/computation/relation.py`；新增 Cube/Scene parser `impls/representation/manipulation.py` |
| representation / computation 接线 | 修改 `interfaces.py`、`puzzle.py`、representation/computation `__init__.py`、`structured.py`、`readouts.py`、`factory.py`、`slots.py`、`accounting.py` |
| runtime / diagnostic 接线 | 修改 `impls/networks/common.py`、`impls/utils/flax_utils.py`、`impls/utils/datasets.py`、`impls/main.py`、`tools/sweep.py` |
| Phase 1 tooling | 新增 `tools/m20a_relation_audit.py`、`tools/m20a_doctor.py`、`tools/m20a_smoke.py`、`tools/analyze_m20a.py` |
| study | 新增 `experiments/M20A_cross_task_oracle_relation_diagnosis/study.yaml` 与 18 个 config skeleton |
| tests | 新增 `tests/computation/test_m20a_relations.py`、`tests/integration/test_m20a_phase1.py`；同步 M17 的 representation-field assertion，使其检查前四字段仍原样且新增字段默认安全 |

`impls/agents/gciql.py` 的 objective 没有改动；`impls/computation/blocks/mlp_mixer.py` 的 normal forward、parameter tree 和 initializer 没有改动。

已通过的验证包括：

- 新增 M20 relation/Phase-1 tests：10 tests；
- M15 legacy structured：12 tests；M17 modular structured：7 tests；
- M16/M19 + MLP/EntityMLP 回归：35 tests；
- M18/M18-D/canonical computation 回归：29 tests；
- M19A real Puzzle smoke：1 test；
- `compileall`；
- 真实三环境 Correct+Mean 的两步 GCIQL smoke：loss/gradient finite、parameters changed、target update valid、relation shape、checkpoint roundtrip 全部通过。Zero/Shuffled 的 forward/create/parameter parity 也通过。

smoke 未把 evaluation episode 当成科学结果；其 checkpoint 仅在 `/tmp/m20a_phase1_smoke/{puzzle,cube,scene}/params_2.pkl`。

## Phase 2 skeleton、analyzer 与剩余边界

study 位于 `experiments/M20A_cross_task_oracle_relation_diagnosis/`，包含 Puzzle/Cube/Scene × Zero/Correct/Shuffled × Mean/Hybrid 的 18 个 cell。每个均为 `protocol_stage: phase2_skeleton`、`executable: false`、`blocked_by: user_phase2_freeze`。dry-run 不会排队正式 job；launcher 报告 `formal executable runs = 0` 与 `blocked phase2 skeletons = 18`。

Cube 的五个 threshold 字段在每个 skeleton 中均为 `null` 且 `threshold_status: UNFROZEN_PHASE1`。`executable=true` 若仍带 null，schema/doctor 会失败。未来 analyzer 已实现 `final@1M`、best、best step、last-3、normalized AUC、per-task final 与 task 内 contrast 的 schema；其当前输出明确没有任何正式结果，且禁止 cross-task raw-success pooling。

完整真实数据审计见 [relation audit](M20A_phase1_relation_audit.md)，机器可读原始统计见 `M20A_phase1_relation_audit.json`，doctor 记录见 `M20A_phase1_doctor.json`。

## 用户在 Phase 2 前唯一需要冻结的科学决策

1. `current_support_epsilon_xy`
2. `current_support_epsilon_z`
3. `goal_support_epsilon_xy`
4. `goal_support_epsilon_z`
5. `goal_conflict_radius`

除这五项 Cube geometry threshold 外，Phase 2 协议、alpha、architecture、readout 定义、shuffle、seed 与安全 gate 都已冻结。阈值必须由用户在独立 Phase 2 prompt 中确认；不能由未来 RL performance 或本次 audit 的“好看 prevalence”自动选择。

## 用户手动 review / commit 前的建议命令

以下仅供用户自行运行，Codex 未执行其中任何 Git 或正式训练命令：

```bash
cd /home/eai/Research/RLC
git status --short
git diff --check

JAX_PLATFORMS=cpu PYTHONPATH=. /home/eai/Tools/miniforge3/envs/brain_nav/bin/python \
  tools/m20a_doctor.py \
  --study experiments/M20A_cross_task_oracle_relation_diagnosis/study.yaml \
  --audit docs/9-8/M20A_phase1_relation_audit.json \
  --smoke /tmp/m20a_phase1_smoke/m20a_phase1_smoke.json \
  --run-root /data/qijunrong/06-RL/offline-rl/exp/RLC/runs \
  --report docs/9-8/M20A_phase1_doctor.json

PYTHONPATH=. /home/eai/Tools/miniforge3/envs/brain_nav/bin/python \
  tools/sweep.py \
  --study experiments/M20A_cross_task_oracle_relation_diagnosis/study.yaml \
  --run-root /tmp/m20a_phase1_dry_run \
  --gpus 0 \
  --dry-run
```

不要将 `--dry-run` 改为 `--execute`；在用户给出独立 Phase 2 freeze 前，所有 18 个 config 都应保持 blocked。
