# M12-D：Fixed-Critic Policy Realization Diagnosis

## 1. 报告范围与结论

本报告是对 /home/eai/Research/docs/8-24/prompt for codex3.md 的重新执行记录。根据项目约定，本轮所有代码和文档改动均位于主线工作树 /home/eai/Research/RLC；其他 worktree 视为 frozen，本轮没有对其进行修改。所有 Git 操作以及正式实验启动仍由用户手动完成。

本轮完成的是 M12-D 的诊断框架实现、真实 checkpoint 审计、单元测试、checkpoint temporal preflight 和小规模 smoke。没有启动正式训练，没有启动正式 M12-D 全量诊断，也没有修改任何 checkpoint。因此目前没有新的 M12-D 科学结果，不能把 smoke 数值解释为正式实验结论。

当前结论：

- A–L 合约测试：12/12 通过。
- 三个 seed 的 last@1M preflight：通过，3 × 5 个 actor 均成功恢复。
- 800k、900k、1M temporal preflight：9/9 个 seed–step cell 通过，3 个时间点都可恢复。
- seed-0 smoke：通过；B_T、B_DE、B_R、support bank、cross-evaluation、exact objective parity、Qmin 语义和参数不变性均通过。
- K4 shared normal 使用 M12A-C003 attempt 2，即 M12B-R 的正确 artifact；没有误用历史 attempt 1。
- 五个 actor 在每个 seed 都使用同 seed 的 M12A-C001 fixed critic last@1M，并核对 checkpoint SHA256 与 critic subtree fingerprint。
- 正式 M12-D 尚未得到 GO：还需要用户自己完成 Git HEAD/clean 状态和目标运行环境 GPU/EGL 的最终确认，并由用户手动启动正式命令。

## 2. 诊断问题与边界

M12-D 研究的问题是：在完全相同、seed-matched 的 frozen CRL critic 下，不同 actor computation 的性能差异，主要来自：

1. training-support 上的 actor policy extraction；
2. evaluator goal 相对 dataset goal 的变化；
3. 正式 rollout state 相对 dataset state 的变化；
4. actor architecture 在这些输入分布上的 policy realization 差异。

实现严格限制为 checkpoint-only post-hoc diagnosis：

- 不训练 actor 或 critic；
- 不改写已有 checkpoint；
- 不读取 SingleState 内部 z1 到 z4、state dynamics 或 recurrence trace；
- 不增加模型结构，不改变既有 CRL actor/critic 实现；
- 只使用外部可观测的 observation、goal、action、Q1、Q2、Qmin、disagreement 以及正式 evaluator rollout state。

## 3. Primary actor 与 provenance

| 名称 | 配置 | attempt | 研究角色 |
|---|---|---:|---|
| K1SN | M12B-C001 | 0 | K1 shared normal |
| K4SN | M12A-C003 | 2 | K4 shared normal；M12B-R 正确 artifact |
| K4SZ | M12B-C003 | 0 | K4 shared zero |
| D9 | M12B-C006 | 0 | Deep feed-forward |
| Residual | M12B-C007 | 0 | Residual feed-forward |

所有 actor primary checkpoint 均为 last@1M。actor artifact 的运行版本是：

bb2644ccb23ee77a0c08e8b9cded85a57716df67

M12A-C001 frozen critic 的历史来源版本是：

e88aa1adfaa354c7df6d5a74c732363d0e4690b4

这两个版本分别表示 actor artifact 运行版本和 frozen critic 预训练来源版本，不能混为一个代码版本。

### 同 seed frozen critic 核验

| seed | M12A-C001 critic checkpoint SHA256 | critic subtree fingerprint |
|---:|---|---|
| 0 | f801f7521aedc70a0ed182a2a2f2d7765d9faa0e6b7ac623f98ad284926006d5 | 35bfa7630a317e40bae4fbc4f529635c4655f8946975af1e87388d4490bb85b7 |
| 1 | b89f45b1e61436b0ee469471b51f40b517ac6130339951c9086c2a5912281c98 | 1ac0eda0a97b315e1f8a6e48d0b29c70267de85bb54b81c4ba9718f0bbcdc36e |
| 2 | 64def1c398dd59a30b533c1cfa704937d389ca5a5088ba45db66b3f7a8005ab2 | 0c0579ae1f6012e89b2c5ec465dd976baabee0d7f882efbddc4cbe4330423d18 |

审计还核对了 actor 的 completed 状态、环境、seed、source config、attempt、checkpoint 存在性以及 metadata 中的 git_dirty=false。K4SN 的恢复路径明确指向 M12A-C003 ... seed_xxx__attempt_002。

## 4. 实现位置

### 通用诊断层

- impls/diagnostics/banks.py：不可变 bank、manifest、sample index、SHA hash、B_T/B_DE bank；
- impls/diagnostics/checkpoints.py：actor/critic provenance、完整 agent restore、critic fingerprint 配对、temporal checkpoint 选择；
- impls/diagnostics/metrics.py：exact actor objective、action/Q tensor、Q1/Q2/Qmin、disagreement、norm/clipping、pairwise contrast；
- impls/diagnostics/rollout.py：common task/episode seed、正式 evaluator rollout、progress-bin pooling；
- impls/diagnostics/support.py：最多 50,000 个 reference states、dataset mean/std、nearest standardized distance proxy。

### M12-D 编排层

- experiments/M12D_fixed_critic_policy_realization_diagnosis/protocol.yaml
- experiments/M12D_fixed_critic_policy_realization_diagnosis/actors.yaml
- experiments/M12D_fixed_critic_policy_realization_diagnosis/build_training_bank.py
- experiments/M12D_fixed_critic_policy_realization_diagnosis/build_eval_goal_bank.py
- experiments/M12D_fixed_critic_policy_realization_diagnosis/build_support_reference.py
- experiments/M12D_fixed_critic_policy_realization_diagnosis/collect_rollout_bank.py
- experiments/M12D_fixed_critic_policy_realization_diagnosis/evaluate_bank.py
- experiments/M12D_fixed_critic_policy_realization_diagnosis/aggregate.py
- experiments/M12D_fixed_critic_policy_realization_diagnosis/preflight.py
- experiments/M12D_fixed_critic_policy_realization_diagnosis/smoke.py

测试文件：

- tests/diagnostics/test_m12d.py

实现没有新增 impls/experiment/m12d.py，没有加入 if study_id == "M12D" 分支，没有修改既有模型实现。

## 5. 三类 bank 的科学语义

### 5.1 B_T：Training-Support Bank

正式配置为每个 critic seed 10 个 batch、每个 batch 1024 个样本，共 10,240 个样本。实现直接使用现有 GCDataset.sample() 与 canonical CRL actor-goal sampling，保存：

- observations、actions；
- actor goals、value goals；
- dataset index、actor goal index、value goal index；
- batch index、sampling seed、dataset root、source commit、bank hash。

因此 B_T 不是人工近似的 goal sampler，而是对既有 CRL 训练数据语义的可审计重放。

### 5.2 B_DE：Dataset-State / Evaluator-Goal Bank

B_DE 复用 B_T 的 dataset state indices，只替换为正式 evaluator task reset 获得的 goal。当前 antmaze-large-navigate-v0 的 task ID 为 1–5，正式 bank 为所有 task 的 balanced cross-product。若 evaluator reset 不能提供正式 goal，脚本直接失败，不会静默退化。

### 5.3 B_R：Shared Rollout-State Bank

B_R 使用正式 evaluator rollout 语义和 common task/episode seeds。每个 actor 独立产生 origin rollout，然后在每个 origin_actor × task × episode × progress_bin cell 中确定性地选择最接近 bin midpoint 的 state。正式配置为每个 actor/task 20 个 episode，progress bins 为：

[0,.2), [.2,.4), [.4,.6), [.6,.8), [.8,1.0]

如果 cell 缺失，bank 构建直接失败。之后五个 actor 在完全相同的 (s_R, g_eval) 上 cross-evaluate，避免把 origin rollout 差异误当成 actor policy 差异。

## 6. Exact CRL 语义与指标

诊断使用现有 CRL 定义：

~~~text
dist      = actor(s, g)
q_action  = clip(dist.mode(), -1, 1)
q1, q2    = critic(s, g, q_action)
Q_min     = min(q1, q2)
~~~

保存 raw action、clipped action、action L2 norm、clipping indicator、Q1、Q2、Qmin 和 abs(Q1-Q2)。不使用 sampled action Q，不使用 unclipped action Q，也不使用 max(Q1,Q2)。

B_T 主指标直接调用现有 CRLPolicyExtractorAgent.policy_extraction_loss()，并保留 exact batch-level normalization，而不是手工重写近似 objective。输出包含 q loss、BC loss、actor loss、data/policy Q、Q delta、BC log probability、behavior MSE、Q1/Q2/Qmin 和 disagreement。

pairwise contrast 同时记录：

- action divergence 的 L2 mean 与 squared-L2 mean；
- Q delta：Q_right - Q_left；
- 按 left actor mean absolute Q 归一化的 Q delta；
- right actor win rate、tie rate；
- Q1/Q2 disagreement。

### Gap decomposition

正式 aggregate 将输出：

- G_goal = ΔQ_DE - ΔQ_T；
- G_state = ΔQ_R - ΔQ_DE；
- 每个 actor/task/episode/progress 的 rollout 对照；
- support proxy quartile 内的 Q gap 与 critic disagreement；
- seed-level mean、median、sample standard deviation。

其中 seed 是模型统计块；当前只有 3 个 seed，不作显著性检验。Spearman 和 sign agreement 只作 descriptive validity。若 support proximity 增大伴随 critic disagreement 增大，结论必须降级为 critic-OOD inconclusive，不能直接声称 actor generalization 改善。

## 7. 实际验证结果

### 7.1 单元测试

执行：

~~~bash
cd /home/eai/Research/RLC
PYTHONPATH=. /home/eai/Tools/miniforge3/envs/brain_nav/bin/python -m py_compile \
  impls/diagnostics/*.py \
  experiments/M12D_fixed_critic_policy_realization_diagnosis/*.py \
  tests/diagnostics/test_m12d.py
PYTHONPATH=. /home/eai/Tools/miniforge3/envs/brain_nav/bin/python \
  -m unittest tests.diagnostics.test_m12d
~~~

结果：12/12 passed。环境同时打印了 JAX CUDA plugin 因当前会话没有 CUDA device 而回退 CPU 的 warning；这没有导致测试失败。

覆盖内容包括 primary actor/protocol、bank immutability/hash、B_DE state identity、B_R balancing、support determinism、exact objective parity、Qmin/pairwise semantics、K4SN attempt2、D9/Residual config identity、parameter non-mutation、禁止内部 state trace 依赖和 temporal selector。

### 7.2 last@1M preflight

真实恢复 3 个 seed × 5 个 actor，并核验各自的同 seed M12A-C001 critic。结果：PASS。

### 7.3 temporal preflight

真实恢复：

3 seeds × {800000, 900000, 1000000} × 5 actors = 45 个 actor checkpoint

critic 仍固定为对应 seed 的 M12A-C001 last@1M。9 个 seed–step cell 全部 PASS：

- 800k：3/3；
- 900k：3/3；
- 1M：3/3。

这一步只证明 checkpoint 可恢复、provenance 正确；它不是 temporal B_T metric evaluation。

### 7.4 seed-0 smoke

使用临时目录 /tmp/m12d_rlc_smoke_20260824_v2，没有写入正式实验目录。实际规模：

| bank | smoke 规模 | bank hash |
|---|---:|---|
| B_T | 2 × 32 = 64 | 1f8afbc181935464a49b5fc93017ec5faeccc8bc0c613c3cb7c5494fbec11050 |
| B_DE | 5 tasks × 64 = 320 | a4924228fe5d0704bf12d39693bd87160097ada5d20502b3a2ddbc0dfc191620 |
| B_R | 5 actors × 5 tasks × 1 episode × 5 bins = 125 | 42697ee7cf0d2a325e10ea5f32bbe1b94e6588563023d3f9ece1748b5c29867e |
| support | 256 reference states | 201ecb50ca26454a9c4aabac4c6ba8062f76820311a7d6110740539739ee0c92 |

smoke 通过的检查包括 checkpoint pairing、K4SN attempt2、D9/Residual loading、exact GCDataset sampler、B_DE state identity、B_R balancing、五 actor cross-evaluation、Qmin、exact objective parity、finite metrics、support determinism 和 parameter non-mutation。smoke manifest 明确记录：

- formal_training_started: false
- formal_diagnostic_started: false
- status: PASS

### 7.5 evaluate/aggregate 编排 smoke

在 smoke 生成的 B_R 上另行执行了 evaluate_bank.py 和 aggregate.py，成功生成：

- 每个 actor 的 metrics_raw.npz、metrics_rows.csv、metrics_summary.csv；
- pairwise_contrasts.json；
- support_quartile_analysis.csv；
- aggregate_summary.csv、aggregate_pairwise.csv；
- 含 mean/median/std 的 pairwise_means.csv；
- gap_decomposition.csv；
- seed-level validity summary。

## 8. 正式运行前的 GO / NO-GO

### 已通过

- actor checkpoint existence/completion；
- K4SN M12A-C003 attempt2 identity；
- D9/Residual restore；
- same-seed critic SHA256；
- same-seed critic subtree fingerprint；
- actor source commit 与 metadata dirty 状态；
- last@1M restore；
- 800k/900k/1M temporal restore；
- B_T/B_DE/B_R implementation；
- exact CRL objective 与 Qmin semantics；
- no parameter mutation；
- no training、no checkpoint modification、no Git operation。

### 当前状态

实现与 preflight：GO。正式 M12-D：NO-GO/待用户手动确认。

尚待用户自行确认：

1. 主线工作树 /home/eai/Research/RLC 的 HEAD、clean/detached 状态；
2. 正式环境中目标 GPU 可见且 JAX backend 确实为 GPU；
3. EGL/MuJoCo 初始化正常；
4. 输出目录可写且不会覆盖已有 immutable bank/evaluation artifact。

本 agent 不执行上述 Git 命令，也不自动启动正式诊断。

## 9. 用户手动执行的正式命令

以下命令只用于 post-hoc diagnosis，不启动训练。Git 命令由用户自己执行。

~~~bash
cd /home/eai/Research/RLC

# 由用户手动完成；本轮 agent 没有执行 Git 操作
git rev-parse HEAD
git status --porcelain

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONPATH=.
PY=/home/eai/Tools/miniforge3/envs/brain_nav/bin/python
M12D_ROOT=/data/qijunrong/06-RL/offline-rl/exp/RLC/diagnostics/M12D

for seed in 0 1 2; do
  SID=$(printf '%03d' "$seed")
  $PY experiments/M12D_fixed_critic_policy_realization_diagnosis/build_training_bank.py \
    --seed "$seed" \
    --output "$M12D_ROOT/banks/seed_$SID/B_T"
  $PY experiments/M12D_fixed_critic_policy_realization_diagnosis/build_eval_goal_bank.py \
    --seed "$seed" \
    --training-bank "$M12D_ROOT/banks/seed_$SID/B_T" \
    --output "$M12D_ROOT/banks/seed_$SID/B_DE"
  $PY experiments/M12D_fixed_critic_policy_realization_diagnosis/build_support_reference.py \
    --seed "$seed" \
    --output "$M12D_ROOT/banks/seed_$SID/support"
  $PY experiments/M12D_fixed_critic_policy_realization_diagnosis/collect_rollout_bank.py \
    --seed "$seed" \
    --output "$M12D_ROOT/banks/seed_$SID/B_R"
done

# last@1M primary evaluation
for seed in 0 1 2; do
  SID=$(printf '%03d' "$seed")
  for bank in B_T B_DE B_R; do
    $PY experiments/M12D_fixed_critic_policy_realization_diagnosis/evaluate_bank.py \
      --seed "$seed" \
      --bank "$M12D_ROOT/banks/seed_$SID/$bank" \
      --support-bank "$M12D_ROOT/banks/seed_$SID/support" \
      --output "$M12D_ROOT/eval/primary/seed_$SID/$bank"
  done
done

# primary aggregation
$PY experiments/M12D_fixed_critic_policy_realization_diagnosis/aggregate.py \
  --root "$M12D_ROOT/eval/primary" \
  --output "$M12D_ROOT/aggregate/primary"
~~~

若不希望 B_T/B_DE 生成 support proxy，可以只在 B_R evaluation 中传入 support bank；上面的命令将三类 bank 统一传入 support bank，便于保留完整 provenance。

Temporal B_T 只评价 800k、900k、1M actor checkpoint，critic 仍为同 seed M12A-C001 last@1M：

~~~bash
for seed in 0 1 2; do
  SID=$(printf '%03d' "$seed")
  for step in 800000 900000 1000000; do
    $PY experiments/M12D_fixed_critic_policy_realization_diagnosis/evaluate_bank.py \
      --seed "$seed" \
      --bank "$M12D_ROOT/banks/seed_$SID/B_T" \
      --checkpoint-step "$step" \
      --output "$M12D_ROOT/eval/temporal/seed_$SID/B_T_$step"
  done
done
~~~

正式运行中不要复用已经存在的 bank/evaluation output 目录；save_bank 和 evaluation 脚本会拒绝覆盖，以保留 hash 可审计性。

## 10. 最终交接

本轮交付的是位于 RLC 主线的可复用诊断实现与验证记录，不是 M12-D 正式结果报告。正式运行完成后，应基于 aggregate/primary 和 eval/temporal 生成科学结果报告，至少报告：

- 每个 actor 的 B_T、B_DE、B_R 指标；
- 800k/900k/1M temporal B_T；
- action divergence、Q delta、Q disagreement、clipping 与 exact objective；
- G_goal、G_state；
- support proxy quartile 结果及 critic disagreement；
- seed mean/median/std、sign agreement、descriptive Spearman；
- outcome taxonomy：支持 training-support、支持 goal shift、支持 rollout-state shift、critic-OOD inconclusive 或 inconclusive。

在正式 diagnostic 运行前，不应把当前 smoke 的任何 action/Q 数值写成 M12-D 的科学结论。
