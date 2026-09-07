# M20A Phase 1：真实 GCIQL relation prevalence 与语义审计

> 历史状态说明：`CUBE RELATION THRESHOLDS NOT FROZEN` 是本次 Phase-1 audit 当时的正确结论。用户随后已在独立 Phase-2 prompt 中冻结五个 Cube threshold；当前正式值、依据和实验矩阵见 [M20A Phase-2 freeze](M20A_phase2_freeze.md)。

## 结论先行

本审计完成了三个 M20A 环境的真实 `GCDataset.sample()` relation prevalence 检查、Correct/Shuffled/Zero 对照有效性检查和 Cube 几何敏感性检查。它不含正式 RL training，也不以 relation prevalence 推断未来 success。

最重要的审计结论如下：

- 每个环境均从真实 `GCDataset.sample()` 取得 100,000 条训练 pair，`audit_seed=20020`。
- 每个环境内 Z/C/S 使用独立但同 seed 的 dataset sampler；transition ID、actor/value goal ID、以及 sampled observations/goals 均逐项完全一致。relation mode 不改变采样或 goal relabeling。
- Puzzle 与 Scene 的 static Correct/Shuffled 在所有 sampled pair 上 relation tensor 均不同，且每 sample/per-type edge count 完全相等。
- Cube 的固定 derangement `[1,2,0]` 不移动 token feature；在真实 batch 上 actor/value/TD-next 的 C/S tensor 分别有 44.019% / 40.781% / 40.808% 的 sample 不同，同时每 sample/per-type edge count 完全相等。该控制在当前 token-identity semantics 下有效。
- Cube 只得到 source-grounded/geometry-grounded 的候选与敏感性描述，**没有**得到正式阈值。

**CUBE RELATION THRESHOLDS NOT FROZEN**

## 版本、数据和可复现性边界

| 项目 | 记录 |
| --- | --- |
| 本地 OGBench distribution | `1.2.1`，模块为 `RLC/ogbench/__init__.py` |
| 起始 HEAD / worktree status | 未由 Codex 读取：`not independently read by Codex; user-only Git policy` / `git_status_not_read_due_to_user_only_git_policy`。这是遵守用户“所有 Git 操作由用户完成”的更严格约束，不是缺失记录的猜测。 |
| dataset root | `/data/qijunrong/06-RL/offline-rl/data/raw_ogbench` |
| audit method | `GCDataset.sample(batch_size=100000, return_sampling_trace=True)` |
| audit seed | `20020` |
| alpha | M20A skeleton/runtime override 显式 `1.0`；绝不依赖 GCIQL 默认 `0.3`。本地未找到覆盖三个 manipulation-play 环境的独立官方 GCIQL command grid，已作为 provenance limitation 记录，而非伪造替代来源。 |

| 环境 | `.npz` SHA-256 | bytes | observation dim |
| --- | --- | ---: | ---: |
| `puzzle-4x4-play-v0` | `396a029c4c347498eed3e21cd85f99c72c56f786e84c29787715808f27cdfe8a` | 262,414,247 | 83 |
| `cube-triple-play-v0` | `4e09a9f5ba7eaba59350dbef7e85887b9a2746490c69eaa83af34a3f9029202b` | 1,003,544,255 | 46 |
| `scene-play-v0` | `66625e0cd9f2fcf92f5988f4d1bedeb2aa8b1e748316e507e89f4a9afb87c341` | 267,055,508 | 40 |

实际解析到的 M20A GCIQL goal sampling 为：

| distribution | `p_curgoal` | `p_trajgoal` | `p_randomgoal` | `geom_sample` |
| --- | ---: | ---: | ---: | --- |
| actor | 0.0 | 1.0 | 0.0 | false |
| value / critic | 0.2 | 0.5 | 0.3 | true |

其余固定项为 `discount=0.99`、`gc_negative=true`、`p_aug=0`、`frame_stack=null`。因此 actor-goal 与 value-goal prevalence 必须且已经分开报告；不能把它们平均为一个“training prevalence”。建议性的 TD-next `(next_observations, value_goals)` 也单列。

对每个 task，Zero-vs-Correct 与 Zero-vs-Shuffled 的四个 batch field（`observations`、`next_observations`、`actor_goals`、`value_goals`）均 exact equal；三个 sampling trace（transition、actor-goal、value-goal index）也均 exact equal。完整 hash/fingerprint 与逐实体 degree 向量保存在同目录的 [machine-readable JSON](M20A_phase1_relation_audit.json)。

## Puzzle-4x4：transition rule 与 static topology

语义证据来自 `ogbench/manipspace/envs/puzzle_env.py` 的 `PuzzleEnv.post_step`：当 button `i` 被 press 时，环境遍历 `(0,0),(1,0),(-1,0),(0,1),(0,-1)`，只保留 in-bounds 邻居。因此 relation `toggle_effect` 的方向为 `button i → 被 i toggle 的 button j`，并且包含 self edge。

Correct matrix 的列和行均为 button index `0…15`；`1` 表示 `i→j`：

```text
1 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0
1 1 1 0 0 1 0 0 0 0 0 0 0 0 0 0
0 1 1 1 0 0 1 0 0 0 0 0 0 0 0 0
0 0 1 1 0 0 0 1 0 0 0 0 0 0 0 0
1 0 0 0 1 1 0 0 1 0 0 0 0 0 0 0
0 1 0 0 1 1 1 0 0 1 0 0 0 0 0 0
0 0 1 0 0 1 1 1 0 0 1 0 0 0 0 0
0 0 0 1 0 0 1 1 0 0 0 1 0 0 0 0
0 0 0 0 1 0 0 0 1 1 0 0 1 0 0 0
0 0 0 0 0 1 0 0 1 1 1 0 0 1 0 0
0 0 0 0 0 0 1 0 0 1 1 1 0 0 1 0
0 0 0 0 0 0 0 1 0 0 1 1 0 0 0 1
0 0 0 0 0 0 0 0 1 0 0 0 1 1 0 0
0 0 0 0 0 0 0 0 0 1 0 0 1 1 1 0
0 0 0 0 0 0 0 0 0 0 1 0 0 1 1 1
0 0 0 0 0 0 0 0 0 0 0 1 0 0 1 1
```

edge list（source → targets）为：

```text
0→[0,1,4]; 1→[0,1,2,5]; 2→[1,2,3,6]; 3→[2,3,7]
4→[0,4,5,8]; 5→[1,4,5,6,9]; 6→[2,5,6,7,10]; 7→[3,6,7,11]
8→[4,8,9,12]; 9→[5,8,9,10,13]; 10→[6,9,10,11,14]; 11→[7,10,11,15]
12→[8,12,13]; 13→[9,12,13,14]; 14→[10,13,14,15]; 15→[11,14,15]
```

corner degree 为 3、非角 edge degree 为 4、interior degree 为 5；完整 degree vector（亦为 Correct 的 incoming/outgoing vector）是：

```text
[3,4,4,3,4,5,5,4,4,5,5,4,3,4,4,3]
```

| distribution | mean | std | median | p10 | p25 | p75 | p90 | max | samples ≥1 | global density |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| actor `(s,actor_goal)` | 64 | 0 | 64 | 64 | 64 | 64 | 64 | 64 | 1.0 | 0.25 |
| value `(s,value_goal)` | 64 | 0 | 64 | 64 | 64 | 64 | 64 | 64 | 1.0 | 0.25 |
| TD-next `(s',value_goal)` | 64 | 0 | 64 | 64 | 64 | 64 | 64 | 64 | 1.0 | 0.25 |

这里的 table 是 `toggle_effect` 的 active edges/sample。Correct 与 Shuffled 每个 sample 的 edge count 均为 64，Zero 始终为 0。固定 study-level permutation 为：

```text
[2,10,15,12,11,8,7,3,6,0,5,14,9,4,1,13]
```

它使 Correct/Shuffled adjacency 产生 80 个不同 entry，且不是 toggle graph automorphism；真实 sampled batch 的 C/S tensor-difference fraction 为 actor/value/TD-next 均 `1.0`。这证明 semantic corruption 不会退化为同一 graph，而非性能结论。

## Scene：entity order 与 controls topology

`SceneEnv.compute_observation` 的 canonical order 为：`[robot(19), cube(9), button_0(4), button_1(4), drawer(2), window(2)]`。`SceneEnv._apply_button_states` 的真实语义是 button state 0/1 分别 lock/unlock drawer/window，故 M20A Scene V0 只包含：

```text
Correct:  button_0 → drawer, button_1 → window
Shuffled: button_0 → window, button_1 → drawer
```

这不是 near/reachable/prerequisite relation，也没有附加 cube-drawer 或 temporal relation。entity index 为 `[cube,button_0,button_1,drawer,window]` 时，Correct 的 outgoing degree 是 `[0,1,1,0,0]`，incoming degree 是 `[0,0,0,1,1]`；Shuffled 的 source/target type、degree pattern 和 2 条 edge 数相同。Zero 的全部 degree 都为 0。

| distribution | mean | std | median | p10 | p25 | p75 | p90 | max | samples ≥1 | global density |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| actor `(s,actor_goal)` | 2 | 0 | 2 | 2 | 2 | 2 | 2 | 2 | 1.0 | 0.08 |
| value `(s,value_goal)` | 2 | 0 | 2 | 2 | 2 | 2 | 2 | 2 | 1.0 | 0.08 |
| TD-next `(s',value_goal)` | 2 | 0 | 2 | 2 | 2 | 2 | 2 | 2 | 1.0 | 0.08 |

Correct/Shuffled 在每 sample edge count 完全相同，C/S tensor-difference fraction 为 actor/value/TD-next 均 `1.0`，Zero relation 全零。这是一个 static functional topology control，而不是 dataset-dependent relation prevalence 选择。

## Cube-triple：source geometry、identity 与 candidate provenance

Cube observation 为 robot 19 加三个 `[xyz(3), quaternion(4), cos(yaw), sin(yaw)]` block。环境使用 `(raw_xyz / 10) + xyz_center` 表示物理位置，relation builder 在计算几何距离前恢复该 canonical scaling；encoder 仍保留全部 raw 9 维 feature。

source audit 可用的几何依据为：stack xy alignment source constant `0.02`、cube vertical geometry `0.04`、Cube success position radius `0.04`。它们只产生候选和 sensitivity bracket，绝不是正式选择依据：

| 参数 | source-grounded candidate | sensitivity bracket | provenance |
| --- | ---: | --- | --- |
| support `epsilon_xy` | 0.02 | 0.01 / 0.02 / 0.03 | stack alignment constant |
| support `epsilon_z` | 0.01 | 0.005 / 0.010 / 0.020 | cube height 0.04 与 conservative vertical tolerance |
| conflict radius | 0.04 | 0.02 / 0.04 / 0.06 | success radius 0.04 |

三个 channel 的精确定义为：

| channel | direction | condition |
| --- | --- | --- |
| `current_support` | current lower cube `i` → directly supported current upper cube `j` | `i≠j`，xy 对齐且 z separation 匹配 support geometry |
| `goal_support` | goal lower cube `i` → directly supported goal upper cube `j` | 与 current support 同一几何规则，作用于本次 call 的 goal |
| `goal_conflict` | current occupant cube `i` → goal-owner cube `j` | `i≠j`，cube `i` current position 接近 cube `j` 的 goal-success region |

Cube reset 的 role permutation 不构成稳定 slot semantic ID，因此没有新增 slot embedding。Shuffled 用固定 derangement `P=[1,2,0]` 在 relation endpoint 上做 `P R P^T`，token feature 不 permutation。故它维持每 type 的 edge budget，却破坏 relation-to-token alignment。

## Cube 几何分布（真实 sampled pairs）

以下 `Δxy` 与 signed `Δz` 针对所有有序 non-self cube pair；upper-pair fraction 是几何方向可视作上方的 pair 比例。actor 与 value 分开，因为目标采样不同。

| distribution / positions | Δxy mean | Δxy p10 / p50 / p90 | Δz p10 / p50 / p90 | upper-pair fraction |
| --- | ---: | --- | --- | ---: |
| actor / current state | 0.200707 | 0.008271 / 0.181678 / 0.407846 | -0.067747 / 0 / 0.067747 | 0.281882 |
| actor / actor goal | 0.198748 | 0.007387 / 0.179489 / 0.406727 | -0.051201 / 0 / 0.051201 | 0.279377 |
| value / current state | 0.200707 | 0.008271 / 0.181678 / 0.407846 | -0.067747 / 0 / 0.067747 | 0.281882 |
| value / value goal | 0.200361 | 0.007934 / 0.181297 / 0.408344 | -0.058356 / 0 / 0.058356 | 0.283455 |

审计并未把“某个 prevalence 看起来合适”当作 threshold selection criterion。下表和 JSON 仅描述候选下的 relation density。

## Cube reference-candidate prevalence

本节使用审计 reference candidate `(epsilon_xy,epsilon_z,radius)=(0.02,0.01,0.04)`，只为把三个 relation type 的统计放在同一行展示。`mean` 是 active edges/sample，`fraction` 是至少有一条该 type edge 的 sample 比例，`density` 是该 channel 的 global activation density。

| distribution | relation type | mean | std | median | p10 | p25 | p75 | p90 | max | fraction ≥1 | density |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| actor | current_support | 0.29296 | 0.494646 | 0 | 0 | 0 | 1 | 1 | 2 | 0.27419 | 0.032551 |
| actor | goal_support | 0.31400 | 0.510063 | 0 | 0 | 0 | 1 | 1 | 2 | 0.29162 | 0.034889 |
| actor | goal_conflict | 0.07752 | 0.283497 | 0 | 0 | 0 | 0 | 0 | 3 | 0.07312 | 0.008613 |
| value | current_support | 0.29296 | 0.494646 | 0 | 0 | 0 | 1 | 1 | 2 | 0.27419 | 0.032551 |
| value | goal_support | 0.30072 | 0.499467 | 0 | 0 | 0 | 1 | 1 | 2 | 0.28113 | 0.033413 |
| value | goal_conflict | 0.07027 | 0.279843 | 0 | 0 | 0 | 0 | 0 | 4 | 0.06393 | 0.007808 |
| TD-next | current_support | 0.29337 | 0.494938 | 0 | 0 | 0 | 1 | 1 | 2 | 0.27454 | 0.032597 |
| TD-next | goal_support | 0.30072 | 0.499467 | 0 | 0 | 0 | 1 | 1 | 2 | 0.28113 | 0.033413 |
| TD-next | goal_conflict | 0.07189 | 0.281144 | 0 | 0 | 0 | 0 | 0 | 4 | 0.06582 | 0.007988 |

combined global activation density 是 actor `0.025351`、value `0.024591`、TD-next `0.024666`。Correct 的 per-entity degree 采用 entity order `[cube_0,cube_1,cube_2]`，下面列出 mean outgoing / incoming degree，因而可检查 relation 既不是只集中在某一个假定 slot，也没有把 direction 反置：

| distribution | relation | outgoing mean vector | incoming mean vector |
| --- | --- | --- | --- |
| actor | current_support | `[0.09954,0.09340,0.10002]` | `[0.09790,0.09947,0.09559]` |
| actor | goal_support | `[0.10621,0.10116,0.10663]` | `[0.10452,0.10731,0.10217]` |
| actor | goal_conflict | `[0.02498,0.02609,0.02645]` | `[0.02645,0.02585,0.02522]` |
| value | current_support | `[0.09954,0.09340,0.10002]` | `[0.09790,0.09947,0.09559]` |
| value | goal_support | `[0.10191,0.09640,0.10241]` | `[0.10000,0.10134,0.09938]` |
| value | goal_conflict | `[0.02280,0.02316,0.02431]` | `[0.02317,0.02348,0.02362]` |
| TD-next | current_support | `[0.09968,0.09352,0.10017]` | `[0.09807,0.09964,0.09566]` |
| TD-next | goal_support | `[0.10191,0.09640,0.10241]` | `[0.10000,0.10134,0.09938]` |
| TD-next | goal_conflict | `[0.02331,0.02374,0.02484]` | `[0.02356,0.02405,0.02428]` |

Shuffled 的 aggregate density/statistical count 与 Correct 相同，因其只是 endpoint derangement；Zero 的 mean, std, quantiles, degrees, density 与 `fraction≥1` 均为 0。完整 Correct/Shuffled/Zero per-type statistics（包括 all quantiles and degree vectors）保留在 JSON，以便机器重算。

## Cube sensitivity（描述，不做选择）

### Support candidates

| goal distribution | `epsilon_xy` | `epsilon_z` | current support mean / fraction ≥1 | goal support mean / fraction ≥1 |
| --- | ---: | ---: | --- | --- |
| actor | 0.01 | 0.005 | 0.27231 / 0.25560 | 0.29171 / 0.27202 |
| actor | 0.02 | 0.010 | 0.29296 / 0.27419 | 0.31400 / 0.29162 |
| actor | 0.03 | 0.020 | 0.30122 / 0.28150 | 0.32186 / 0.29838 |
| value | 0.01 | 0.005 | 0.27231 / 0.25560 | 0.27859 / 0.26100 |
| value | 0.02 | 0.010 | 0.29296 / 0.27419 | 0.30072 / 0.28113 |
| value | 0.03 | 0.020 | 0.30122 / 0.28150 | 0.30863 / 0.28795 |

### Goal-conflict candidates

| goal distribution | `goal_conflict_radius` | mean conflict edges/sample | fraction samples ≥1 |
| --- | ---: | ---: | ---: |
| actor | 0.02 | 0.00948 | 0.00947 |
| actor | 0.04 | 0.07752 | 0.07312 |
| actor | 0.06 | 0.64547 | 0.43138 |
| value | 0.02 | 0.01037 | 0.01024 |
| value | 0.04 | 0.07027 | 0.06393 |
| value | 0.06 | 0.62481 | 0.37295 |

这张表显示 candidate 改变会显著改变 channel frequency，尤其 conflict radius 从 0.04 到 0.06；它**不**支持“挑一个最优 prevalence”的结论，也未依据 RL loss、success、Correct−Shuffled gap 选择任何值。

## Shuffled / Zero 有效性总表

| task | C/S per-sample per-type edge-count equality | Zero all-zero | C/S tensor-difference fraction (actor / value / TD-next) | validity conclusion |
| --- | --- | --- | --- | --- |
| Puzzle | 1.0 / 1.0 / 1.0 | true | 1.0 / 1.0 / 1.0 | fixed non-automorphism semantic corruption |
| Cube | 1.0 / 1.0 / 1.0 | true | 0.44019 / 0.40781 / 0.40808 | derangement changes relation-to-token alignment; not a pure token relabeling |
| Scene | 1.0 / 1.0 / 1.0 | true | 1.0 / 1.0 / 1.0 | controls target swap preserves source/target types and degree pattern |

`Zero` 的 semantic role 是 relation-input structured control：它保留 RelationAugmenter 的 parameter tree 和 nominal Dense graph，但其 input 为全零。`legacy_none` 是另一条历史 compatibility path，完全不应拿来替代 M20A Zero。

## USER DECISIONS REQUIRED BEFORE PHASE 2

下列五项必须由用户在独立的 Phase 2 freeze 中明确写成非 null 值；当前所有 Cube skeleton 仍为 `null` / `UNFROZEN_PHASE1`：

1. `current_support_epsilon_xy`
2. `current_support_epsilon_z`
3. `goal_support_epsilon_xy`
4. `goal_support_epsilon_z`
5. `goal_conflict_radius`

actor/value support prevalence 在相同 candidate 下相近，但目标分布并不相同；本审计不以此代替用户判断，也不把 `share_support_thresholds` 标为已推荐或已冻结。除了这五个 geometry threshold，M20A 的 alpha、training/evaluation protocol、Mixer、RelationAugmenter、readout、shuffle 与 seed policy 已冻结。

## 解释限制与 artifact 状态

relation prevalence 是数据/语义描述，不是 downstream performance。当前没有正式 M20A Phase 2 curve、success、AUC 或 relation-effect estimate。未来的 primary inference 必须在每个 task 内先比较相同 readout 下的 `Correct − Shuffled`，而不是跨 Puzzle/Cube/Scene pooling raw success。

本审计的完整原始字段、dataset fingerprints、sampling trace fingerprints、per-entity metrics、pair geometry quantiles、全部 C/Z/S distribution 都保存在 [M20A_phase1_relation_audit.json](M20A_phase1_relation_audit.json)。正式 run root 未创建 M20A artifact；唯一真实优化 smoke 仅在 `/tmp/m20a_phase1_smoke`。
