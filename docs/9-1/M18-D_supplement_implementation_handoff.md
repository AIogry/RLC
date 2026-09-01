# M18-D 补充诊断：实现、验证与人工执行交接报告

日期：2026-09-01

## 结论与当前边界

本轮已完成 M18-D 最后一个补充诊断基础设施的实现与验证：D2+（mean-pooling retained energy）、D5（paired closed-loop logical rollout）和 D6（cross-actor × cross-critic preference）。没有启动正式 D5 的 100 个 paired initial conditions / 200 rollouts，也没有启动正式 D6 的 N=1024 计算；正式科学结论仍须由用户手动执行下文命令后产生。

本轮没有执行任何 Git 命令或 Git 写操作，也没有启动训练、修改训练配置、修改 GCIQL loss、修改 recurrent computation、Mixer、normal readout、参数/检查点树或 Puzzle 环境训练语义。

这里必须区分两项 provenance 事实：

- 本任务 prompt 指定的远端 `main` 基线为 `0835cafde0be0d5d9f9a47a2b1612619045e80d7`（`8-31 RLC M18-D: add recurrent computation diagnostics`）。由于本轮严格不运行 Git，本文不声称已用 Git 验证当前工作树 HEAD。
- 实际被锁定的 M18 source-run runtime metadata 记录的 source code commit 为 `d8200ffbacb6bb821a9a025a77c5571815a9406c`。D5/D6 使用的是现有 D1/D234 artifact 所记录的精确 checkpoint 文件和 SHA，而不是依据当前 Git 状态或当前 `best` 指针推断模型身份。

## 已审计的既有 M18-D 工件

既有 D1/D234 工件位于：

`/data/qijunrong/06-RL/offline-rl/exp/RLC/diagnostics/M18D`

锁定契约已实际检查 D1 self-depth summary、D1 metadata、D234 metadata、fixed batch、actor metrics 和 source-run metadata。两条必须一致的模型身份如下：

| K_train | role | checkpoint step | SHA256 |
|---:|---|---:|---|
| 4 | `best` | 900000 | `54fb1ce920e08cf9593acd36eb67f6421f0df35e089cbcb81d95190ca387d628` |
| 8 | `best` | 200000 | `1cd4a2eb7f2f428aa4c51f6860a73ebd20a3bcbcc2c0b84b8a694f42e02b3743` |

两个 K 的 D1 与 D234 都已确认 step、path、config identity 和 SHA256 一致；任何一项不一致都会在 D5/D6 planning 阶段失败。D234 的唯一 fixed batch 为 N=1024、diagnostic seed=18018，指纹为：

`ab7c2e0a6abaccd7634427549025ecf406cab66f702cb4d32ced558047658e70`

现有 D1 作为背景证据仍保持可读，矩阵为：

| K_train \ K_actor_test | 1 | 2 | 4 | 8 |
|---:|---:|---:|---:|---:|
| 1 | 0.87 | 0 | 0 | 0 |
| 2 | 0 | 0.75 | 0 | 0 |
| 4 | 0 | 0 | 0.93 | 0 |
| 8 | 0 | 0 | 0 | 0.15 |

这是既有 artifact 的单训练 seed 描述结果，不是本轮新实验结果。新的 analyzer dry-run 也已成功读取 16 个 D1 rows、810 个 D2/D3/D4 trace rows，并从保存的 per-sample trace 识别到 144 个 D2+ aggregate rows。

## 本轮文件改动

| 文件 | 改动目的 |
|---|---|
| `impls/diagnostics/puzzle_logic.py` | 新增仅供诊断使用的 Puzzle-4x4 logical-state extractor、环境导出的 transition oracle、reverse-BFS exact d* 与 live-environment parity audit。 |
| `tools/m18_d_reference.py` | 新增不可变的 D1/D234 reference contract；绝不重新解析当前 semantic `best`。 |
| `tools/m18_cross_actor_critic.py` | 新增 D6：复用 fixed batch 与 D3 已保存 action 的 within-critic preference 分析。 |
| `tools/m18_paired_rollout_diagnostics.py` | 新增 D5：锁定 checkpoint 的 paired rollout、logical trajectory、episode/paired aggregates。 |
| `tools/analyze_m18_d.py` | 增加纯 post-hoc D2+、D5/D6 formal artifact 汇总、D12--D17 图和限定为六项的最终 hypothesis table。 |
| `tests/integration/test_m18_d_diagnostics.py` | 增加 D2+ identities、D6 margins、D3 saved-action、Puzzle oracle、real-env parity 和 shared-goal manifest tests。 |

未修改的关键区域包括：`impls/agents/gciql.py`、`impls/computation/topologies/single_state.py`、`impls/computation/structured.py`、`impls/computation/readouts.py`、M18 study/configs、normal evaluation/training path、optimizer/checkpoint schema，以及 `ogbench/manipspace/envs/puzzle_env.py`。

## D2+：mean-pooling retained energy

对每个已保存的 sample `i`、iteration `k`，而不是对 aggregate mean，定义：

```text
rho(i,k) = mean_token_rms(i,k)^2 / (state_rms(i,k)^2 + 1e-8)
discarded_energy_fraction(i,k) = 1 - rho(i,k)
```

其中 D2 原始定义满足逐样本恒等式：

```text
state_rms^2 = mean_token_rms^2 + token_variance
```

实现会逐样本验证两种 `rho` 计算方式的一致性、nonzero-state 的 `[0,1]` 数值范围，并在 `state_rms^2 <= 1e-8` 时写入 `NaN`，绝不人为写为 0 或 1。K4 中 `k>4` 被明确标为 depth extrapolation；K8 的 `k=1..8` 保持 within trained depth。

`rho` 只能称为 mean pooling 对 token-state **energy** 的保留比例：它衡量共享 mean-token 成分的平方能量占比。它不测 task information、mutual information、因果重要性或 policy 所用信息。因此即使观察到 `rho` 随 k 降低且 token variance 上升，也只能说“与更多 token-specific energy 被 mean pooling 丢弃一致”，不能说 mean pooling 已被证明导致失败。

D2+ 不 restore checkpoint、不 forward network、不新采样 batch。它纯粹读取现有 `actor_metrics.npz` / `value_metrics.npz` / `critic_metrics.npz` 中的 `state_rms`、`mean_token_rms` 和 `token_variance`，再生成 summary 和可选 per-sample 输出。

## D6：cross-actor × cross-critic preference

### 输入与严格锁定

D6 只读取现有 D234 `fixed_batch.npz`，验证上述完整 fingerprint，并使用该 batch 的 `observations`、`actor_goals` 和 `dataset_actions`。因此不会重新采样 `(s,g)`，也不会把 actor goal 换成其他 goal semantics。

action 定义为：

```text
a_data = fixed_batch.dataset_actions
a4     = K4 actor_metrics.npz 的 clipped_action[:, 4]
a8     = K8 actor_metrics.npz 的 clipped_action[:, 8]
```

工具进一步验证：

- `sample_id` 与 fixed batch 的顺序完全一致；
- action artifact 是 `slot=actor`、`checkpoint_role=best`；
- actor artifact 的 `checkpoint_step` 必须分别等于锁定的 900000 / 200000；
- actor metrics 必须是相应 locked D234 metadata 的同目录 sibling；
- `clipped_action[:, K_train]` 与保存的 `normal_actor_mode_at_train_k` clip 后逐元素最大误差不超过 `1e-6`。

这保证 D6 的 `a4/a8` 就是 D3 当时保存的 final deterministic clipped action，而不是新的 actor forward。

### Critic 与统计定义

Q4 从 K4 checkpoint 恢复并始终以 critic K=4 执行；Q8 从 K8 checkpoint 恢复并始终以 critic K=8 执行。对每个 critic、每个 action 先取双 Q ensemble min：

```text
Qc(a) = min(Qc,1(s,g,a), Qc,2(s,g,a))
```

主指标严格是同一 critic 内的比较：

```text
Delta_Q4_self = Q4(a4) - Q4(a8)
Delta_Q8_self = Q8(a8) - Q8(a4)

P4_self       = Pr[Delta_Q4_self > 1e-6]
P8_self       = Pr[Delta_Q8_self > 1e-6]
P_joint_self  = Pr[Delta_Q4_self > 1e-6 and Delta_Q8_self > 1e-6]
```

另有 `tie_Q4_self`、`tie_Q8_self`（绝对 margin 不超过 `1e-6`）、四个对 `a_data` 的 control margins、每个 action 的 `|Q1-Q2|`、以及仅作 secondary magnitude 参考的 normalized margins。summary 报告 mean、std、median、p10、p90 和 positive fraction。

禁止解释跨 critic 的 raw absolute scale，例如禁止从 `Q4(a4) > Q8(a8)` 推断任何结论。D6 可支持的是 depth-specific within-critic action preference geometry；它不能独立证明某 critic 全局错误，也不能把 ensemble disagreement 解释为 calibrated uncertainty。

## D5：真实 Puzzle-4x4 paired closed-loop rollout

### 真实环境语义审计

审计读取的是当前本地 `ogbench/manipspace/envs/puzzle_env.py`、`manipspace_env.py` 和 live `ogbench.make_env_and_datasets('puzzle-4x4-play-v0', env_only=True)`，没有依据“类似 Lights-Out”的记忆硬编码环境。

得到并用 live environment 验证的事实为：

- 标准 state/goal observation 维度为 83，即 19 个 robot features 加 `16 × 4` button blocks。
- 每个 button block 的前两个元素是严格 binary one-hot；它们唯一编码 `_cur_button_states` 或 `_target_button_states`。button id 是 row-major：`row * 4 + col`。
- `_num_rows=4`、`_num_cols=4`、`_num_buttons=16`、`_num_button_states=2`。
- `PuzzleEnv.post_step` 的有效物理事件是目标 button joint 从 `> -0.02` 跨越到 `<= -0.02`；事件会将该 button 与上下左右 in-bounds cardinal neighbours 全部 modulo-2 toggle。
- 环境 `info` 提供 `prev_button_states`、`button_states` 和 `success`；success 的逻辑条件是当前 16 个 button state 全部等于 target state。
- continuous robot state 不改变这个离散 logical shortest-path graph；但是它属于 policy 输入 goal 的 raw vector，因此仍须在 paired policy input 中严格控制。

审计通过 16 个由真实环境 `pre_step/post_step` 触发的 button transitions：每一例都满足 `oracle(current, pressed_button) == observed_next`，且观测提取与环境内部 state 一致。集成测试还另行通过 8 个 live parity cases。

### exact d*

在 live audit 成功后，工具才启用 exact shortest valid-press distance：将 16 个 binary buttons encode 为 canonical integer，对每个 goal 在真实环境已验证的 transition graph 上做 reverse BFS。全空间为 `2^16=65536` configurations；本 transition system 的单个 reachable component 为 4096 states。对不可达的 `(s,g)`，工具写 unavailable/`None`，不把任何 heuristic 伪称为 d*。

oracle 还验证：`d*(goal,goal)=0`、所有 reachable distance 非负，且 sampled reachable state 至少存在一个有效 press 使 distance 减一。若未来环境 audit 任一关键 parity 失败，D5 会关闭 exact d*，仍保留 logical-state trace，但不会输出名为 d* / shortest distance 的 heuristic。

### 成对初始条件与发现的 goal hazard

每个 `paired_episode_id=taskXX_epYYY` 使用既有 `common_task_episode_v1` protocol，记录并复用同一 task id、task seed、episode/reset seed、actor seed、noise seed 和 episode index。K4 用 native actor/critic K=4；K8 用 native actor/critic K=8；D5 不做 cross-K actor，也不做 cross-critic Q comparison。

实现中发现一个必须修正的真实环境细节：即使 `env.reset(seed=...)` 相同，`info['goal']` 的连续 robot 分量仍可因 Puzzle goal construction 内部的新 `action_space.sample()` 而不同；逻辑 target 相同并不足以保证 policy 的完整 raw goal 输入相同。若直接令两模型各自使用 reset 返回的 raw goal，就会破坏 prompt 所要求的 same goal pairing。

因此 D5 的 root 在 worker 启动前对每一个 paired episode 使用一次真实 `PuzzleEnv.reset` 生成 `paired_goal_manifest.npz`：

- 保存这一次实际发出的完整 raw `info['goal']`、其 logical goal code、anchor initial code 和 manifest fingerprint；
- K4/K8 都字节级读取同一份 raw policy goal，用于 actor 和 own-critic calls；
- 每个 worker 仍以相同 task/reset seed reset 自己的真实环境，并验证 shared policy goal 的 logical target 等于该环境 `reset_info['goal']` 的 logical target；环境 success 始终由该 worker 的真实环境决定；
- paired aggregate 再验证 K4/K8 的 initial observation SHA、goal SHA、logical codes 和所有 seeds 完全一致，否则拒绝汇总。

这不是对 environment semantics 的改动，而是对 diagnostic input pairing 的必要控制。

### D5 可支持与不可识别的量

每 timestep 记录 raw observation/next observation/goal/action（NPZ）、own Q1/Q2/Qmin、reward、success、termination、logical configurations、exact d*（可用时）、distance delta，以及可验证的 single-press event。每 episode 汇总 success、initial/final/minimum d*、net/best logical progress、first progress time、logical/progress/regressive/neutral/no-interaction event counts 与 rates；随后按 task 和 paired K4-minus-K8 差异汇总。

这些指标可以严格支持可观测的 closed-loop logical progress / regression 描述。它们不能从现有连续 policy interface 唯一识别 actor 的“intended button”，因而不能把 no logical interaction 直接归因为 reasoning failure 或 motor-control failure；也不会强行逐 timestep 对齐两条 action 已不同的 trajectory。

## Checkpoint immutability 与输出安全

`tools/m18_d_reference.py` 从完成的 D1 self-depth artifact 和 D234 trace artifact 建立 contract。它直接使用其中记录的 step-specific checkpoint path/SHA，且不调用 semantic-checkpoint resolver。因此即使当前 `best` 指针之后改变，也不会静默替换模型。

D5/D6 均在 restore 前后计算 stable SHA256；每个 restored agent 还记录 parameter tree fingerprint 和 `network.step`，并要求前后完全相同。metadata 显式写入 `evaluation_only=true`、`finetuning=false`、`optimizer_updates=0`。任何 source hash、parameter fingerprint 或 step 变化都会失败。输出只能写在新 diagnostics directory，source run 下不写诊断文件；既有 output directory 或 model directory 存在时拒绝覆盖。

## 已完成的验证

| 验证 | 结果 |
|---|---|
| M18-D diagnostics integration tests | `12 tests`，`22.503s`，`OK`。覆盖 D2+ identity/NaN、D6 margins/rates、D3 action loading、goal manifest、logical oracle 与 real Puzzle parity。 |
| 受影响回归集 | `80 tests`，`79.099s`，`OK`：M18 recurrent compute、M17 modular structured computation、canonical agents、M14 base algorithm、checkpoint lifecycle、reevaluation、computation foundation、SingleState、TwoState。 |
| `python -m compileall -q impls tools tests` | 通过。 |
| Existing D1--D4 readability | locked-reference D5/D6 formal-size dry-run 与 analyzer dry-run 均成功。 |
| Formal D6 dry-run | 成功锁定 N=1024 fixed batch 和两项 checkpoint SHA，且 `/tmp/m18d_20260901_d6_formal_dryrun` 未被创建。 |
| Formal D5 dry-run | 成功计划 5 tasks × 20 episodes = 100 paired conditions，且 `/tmp/m18d_20260901_d5_formal_dryrun` 未被创建。 |
| Analyzer dry-run | 成功；不写 report directory。 |

测试中的 Gymnasium `Box` float64-to-float32 warning 为正常环境 warning，不是失败或数值断言错误。

### tiny smoke（非科学证据）

所有 smoke 输出都在 `/tmp`，不在正式 diagnostics root；正式 root 下当前没有 `closed_loop` 或 `cross_actor_critic` 正式输出目录。

- D5：`task_id=1 × 1 paired episode`，K4/K8 均完成。paired aggregate 通过了同一 initial observation SHA 和同一完整 goal SHA 的检查；shared-goal manifest fingerprint 为 `b93e8cd326ba129f93f6a1753beb051532a5353829fc2eb110a525b31d5755df`，exact d* 可用。两个 worker 都报告 source checkpoint immutable、`optimizer_updates=0`，network step 分别保持 `900001→900001`、`200001→200001`。
- D6：N=16 fixed-batch prefix，输出 per-sample NPZ 和 summary 成功；仍使用完整 batch fingerprint `ab7c…58e70`，D3 action source step 分别为 K4=900000、K8=200000。两个 critic 均报告 immutable source、`optimizer_updates=0`，network steps `900001→900001`、`200001→200001`。

这些 smoke 的 success、d* 或 preference rate 样本量不足，且 smoke-only metadata 会使正式 analyzer 明确忽略它们；本文不把它们作为 M18-D scientific evidence。

## 用户手动执行顺序

下面命令均要求用户自行决定并填写已 review/commit 的 `<USER_REVIEWED_COMMIT_SHA>`。本轮没有也不会代替用户执行 Git 操作。

建议顺序是 D6、D5、final analyzer。D5/D6 的 outputs 均 refuse overwrite；若正式运行中断或失败，请先保留并审计现有 artifact，再由用户决定后续处理，不要盲目删除或覆盖。

### 1. 正式 D6（N=1024）

```bash
cd /home/eai/Research/RLC
/home/eai/Tools/miniforge3/envs/brain_nav/bin/python tools/m18_cross_actor_critic.py \
  --reference-diagnostics-root /data/qijunrong/06-RL/offline-rl/exp/RLC/diagnostics \
  --source-run-root /data/qijunrong/06-RL/offline-rl/exp/RLC/runs \
  --output-root /data/qijunrong/06-RL/offline-rl/exp/RLC/diagnostics \
  --train-ks 4,8 \
  --checkpoint locked-reference \
  --diagnostic-code-commit <USER_REVIEWED_COMMIT_SHA> \
  --execute
```

### 2. 正式 D5（5 tasks × 20 paired episodes）

```bash
cd /home/eai/Research/RLC
/home/eai/Tools/miniforge3/envs/brain_nav/bin/python tools/m18_paired_rollout_diagnostics.py \
  --reference-diagnostics-root /data/qijunrong/06-RL/offline-rl/exp/RLC/diagnostics \
  --source-run-root /data/qijunrong/06-RL/offline-rl/exp/RLC/runs \
  --output-root /data/qijunrong/06-RL/offline-rl/exp/RLC/diagnostics \
  --train-ks 4,8 \
  --checkpoint locked-reference \
  --task-ids 1,2,3,4,5 \
  --episodes-per-task 20 \
  --evaluation-seed 18018 \
  --gpus 0,1 \
  --diagnostic-code-commit <USER_REVIEWED_COMMIT_SHA> \
  --execute
```

`--gpus 0,1` 是 prompt 的默认 physical GPU mapping；请在实际启动前由用户按资源情况手动选择空闲 GPU。

### 3. D2+ 与最终 analyzer：先 dry-run，再 execute

D2+ 没有单独的 network execution command；它由 analyzer 纯 post-hoc 从 D234 saved metrics 推导。请在 D5/D6 均正式完成后使用一个此前不存在的 report directory：

```bash
cd /home/eai/Research/RLC
/home/eai/Tools/miniforge3/envs/brain_nav/bin/python tools/analyze_m18_d.py \
  --diagnostics-root /data/qijunrong/06-RL/offline-rl/exp/RLC/diagnostics \
  --output-dir /data/qijunrong/06-RL/offline-rl/exp/RLC/diagnostics/M18D/reports/checkpoint_locked_final \
  --checkpoint best \
  --dry-run
```

```bash
cd /home/eai/Research/RLC
/home/eai/Tools/miniforge3/envs/brain_nav/bin/python tools/analyze_m18_d.py \
  --diagnostics-root /data/qijunrong/06-RL/offline-rl/exp/RLC/diagnostics \
  --output-dir /data/qijunrong/06-RL/offline-rl/exp/RLC/diagnostics/M18D/reports/checkpoint_locked_final \
  --checkpoint best \
  --execute
```

analyzer 参数中的 `--checkpoint best` 只对应既有 D1/D234 artifact namespace；D5/D6 本身始终从 `locked-reference` source checkpoint identity 读取。最终 report 会保留 D1--D4，并加入 D2+、D5、D6、D12--D17 figures 和六项非因果 hypothesis table：

`H_depth_specialization`、`H_state_instability`、`H_action_instability`、`H_mean_pooling_mismatch`、`H_actor_critic_coadaptation`、`H_closed_loop_progress_failure`。

每项 status 只可能是 `consistent`、`mixed`、`not observed` 或 `insufficient evidence`，不会输出 causal true/false。

## 明确 deferred 的内容

- 不新增 latent/action/norm/cosine post-hoc metric；
- 不实现 terminal-only vs multi-depth supervision、`z+x` vs x-initialized/no-reinjection 等 2×2 intervention；
- 不做 frozen critic、actor-only training、readout intervention 或 architecture repair；
- 不启动 Cube、M19 或新的 K sweep；
- 不启动任何新训练；
- 不对 D5 smoke 或 D6 N=16 smoke 作科学解释。

正式 D5/D6 完成后，后续是否进入 intervention 应由结果决定：strong D6 self-preference 才考虑 frozen-critic/actor-only causal study；strong D2+ retained-energy decline 才考虑 mean readout vs information-preserving readout；D5 logical-progress degradation 才考虑 representation/computation intervention；若 logical progression 正常但环境 success 低，才考虑 policy-readout/robot-action interaction intervention。

## M18-D STOP RULE

> M18-D is considered diagnostically complete after D2+, D5, and D6. No additional post-hoc latent/action/norm/cosine diagnostics will be added unless these analyses uncover an implementation correctness issue. Further hypotheses must be tested through intervention experiments.

因此，在用户完成上述三项正式执行并获得最终 analyzer report 后，应停止 M18-D 的额外 post-hoc 扩展；除非结果暴露实现正确性问题，下一步只能是预先定义的 intervention，而不是继续堆叠描述性诊断。
