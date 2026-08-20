# M11A：CRL Actor × Critic Computation Interaction

日期：2026-08-20  
状态：代码、Study、diagnostic tool 与测试完成；正式 1M 训练尚未启动。  
Git 约束：根据用户要求，本任务没有执行任何 Git 命令；提交、分支、状态检查、diff、commit、push、worktree 等均由用户手动完成。

## 1. 研究目标与科学边界

M11A 要回答：在 CRL 中，actor-side computation 与 critic-side computation 的瓶颈关系更接近 substitution、complementarity，还是 additive/independent。研究使用两种 topology 做 replication：SingleState 与 TwoState。两者不是 M11A 的比较对象；primary factors 是 `topology`、`actor_computation`、`critic_computation`。

本任务严格没有新增 GRU、LSTM、Attention、Transformer、ThreeState、normalization、SwiGLU、RMSNorm、activation、state initialization、credit rule、schedule、residual variant 或 width factor。

## 2. 已审阅的现有架构

实现前审阅了：

- `README.md` 及 M9/M9B/M9B-1M protocol 文档；
- `impls/agents/crl.py`；
- `impls/networks/common.py`；
- `impls/computation/factory.py`、`accounting.py`、`interfaces.py`、MLP primitive、FeedForward/SingleState/TwoState topology 和 credit policies；
- `impls/utils/evaluation.py`；
- `impls/experiment/management.py`、`reevaluation.py`；
- `impls/analysis/`、`tools/reevaluate_checkpoint.py`、`tools/reevaluate_study.py`、`tools/sweep.py`、`scripts/run_study.sh`；
- M9A/M9B/M9B-1M Study/config；
- `tests/computation/`、`tests/experiment/`、`tests/integration/`。

当前 CRL 的计算边界为：

```text
phi(s, a)       <- critic_state slot
psi(g)          <- critic_goal slot
Q(s,a,g) = <phi(s,a), psi(g)> / sqrt(latent_dim)
loss            = contrastive critic objective + DDPG+BC actor objective
```

M11A 没有改 bilinear readout、ensemble 数量、loss、optimizer、dataset sampler、goal sampler、actor readout 或 evaluation policy。

## 3. Hidden-depth incompatibility 的根因

旧 factory 对 recurrent topology 使用了 actor-specific 限制：`len(hidden_dims)==3` 且所有 hidden dims 同宽。actor 默认是 `(512,512,512)`，但 CRL bilinear branch 实际是：

```text
branch_dims = (*value_hidden_dims, latent_dim) = (512, 512, 512, 512)
```

不能通过缩短 `value_hidden_dims`、删除 `latent_dim`、改变 vanilla critic depth 或把 actor 改成四层来绕过，否则会污染既有 baseline。

## 4. `update_depth` 泛化实现

factory 现在允许 recurrent topology 使用不同长度的 branch hidden tuple，只要求最终 branch width 与 `state_dim` 一致。recurrent core 的实际结构由显式 `topology_kwargs.update_depth` 控制：

```text
input_mapping: D_in -> state_dim
update_module: MLP((state_dim,) * update_depth)
```

SingleState 的 recurrent equation 保持为：

```text
x_hidden = input_mapping(x_raw)
z = broadcast(z_init)
repeat iterations:
    z = update_module(z + x_hidden)       # residual=false
```

TwoState 的 equation 和 schedule 保持为：

```text
z_L <- l_update(z_L + z_H + x)
z_H <- h_update(z_H + z_L)
```

默认值与兼容性：

- `update_depth` 必须是正整数；bool、浮点数、0、负数均 fail loudly；
- legacy actor 未声明该字段时默认 2；
- legacy SingleState 仍为 1 个 input mapping Dense + 2 个 shared update Dense；
- legacy TwoState 仍为 1 个 input mapping Dense + 2 个 H update Dense + 2 个 L update Dense；
- M11A critic branch 显式使用 `update_depth=3`；
- state buffer、input injection、residual、credit、state shape 和参数 key 未因泛化而改变。

CRL critic 的 `critic_state` 与 `critic_goal` 使用相同 spec，但仍是独立 Flax module、独立参数和独立 recurrent buffers。由于 CRL critic 是 ensemble，`ensemblize()` 现在同时映射和 split `params`、`buffers` collection；否则 recurrent critic 的 buffer 无法正确初始化或会被错误共享。legacy feed-forward branch 不创建 buffers，参数结构不变。

## 5. 修改与新增文件

修改：

- `impls/computation/factory.py`；
- `impls/computation/topologies/single_state.py`；
- `impls/computation/topologies/two_state.py`；
- `impls/networks/common.py`；
- `impls/computation/accounting.py`；
- `impls/main.py`；
- `impls/utils/evaluation.py`；
- `tests/computation/test_single_state.py`；
- `tests/computation/test_two_state.py`。

新增：

- `experiments/M11A_crl_computation_interaction/study.yaml`；
- `experiments/M11A_crl_computation_interaction/configs/M11A-C001.yaml` 至 `M11A-C007.yaml`；
- `experiments/M11A_crl_computation_interaction/manifest.csv`；
- `experiments/M11A_crl_computation_interaction/diagnostic.yaml`；
- `experiments/M11A_crl_computation_interaction/reevaluation_last.yaml`；
- `impls/analysis/crl_interaction.py`；
- `tools/diagnose_crl_interaction.py`；
- `tests/integration/test_m11a_crl_interaction.py`；
- 本报告。

## 6. M11A 七个正式 configuration

所有配置均为 CRL、`actor_loss=ddpgbc`、`antmaze-large-navigate-v0`、seed 0、1M steps。C001 是唯一 factorial anchor，不额外创建重复 vanilla config。

| config | actor computation | critic computation | topology | actor | critic_state / critic_goal |
|---|---|---|---|---|---|
| M11A-C001 | FF | FF | feedforward | disabled | disabled / disabled |
| M11A-C002 | FF | SingleState | SingleState | disabled | enabled / enabled |
| M11A-C003 | SingleState | FF | SingleState | enabled | disabled / disabled |
| M11A-C004 | SingleState | SingleState | SingleState | enabled | enabled / enabled |
| M11A-C005 | FF | TwoState | TwoState | disabled | enabled / enabled |
| M11A-C006 | TwoState | FF | TwoState | enabled | disabled / disabled |
| M11A-C007 | TwoState | TwoState | TwoState | enabled | enabled / enabled |

C002/C003/C004 使用 SingleState canonical spec；C005/C006/C007 使用 TwoState H2L1 full-BPTT canonical spec。M11A 没有 H2L6、更多 K 或 residual sweep。

## 7. Resolved computation specs

### Actor

| condition | topology | credit | iterations/schedule | residual | state_dim | update_depth | init |
|---|---|---|---|---|---:|---:|---|
| FF | feedforward | direct | one pass | n/a | n/a | n/a | n/a |
| SingleState | single_state | direct | iterations=4 | false | 512 | 2 | normal_buffer, std=1.0 |
| TwoState | two_state | full_bptt | H2L1, H=2/L=1 | false | 512 | 2 | normal_buffer, std=1.0 |

### Critic state/goal branch

`critic_state` 与 `critic_goal` spec 相同，但 module/parameter 独立。

| condition | topology | credit | iterations/schedule | residual | state_dim | update_depth | init |
|---|---|---|---|---|---:|---:|---|
| FF | feedforward | direct | one pass | n/a | n/a | n/a | n/a |
| SingleState | single_state | direct | iterations=4 | false | 512 | 3 | normal_buffer, std=1.0 |
| TwoState | two_state | full_bptt | H2L1, H=2/L=1 | false | 512 | 3 | normal_buffer, std=1.0 |

critic branch 的 `(512,512,512,512)` 通过 1 个 input mapping Dense + 3 个 shared update Dense 表达；没有删除 latent readout，也没有改变 bilinear interaction。

## 8. Protocol 与 provenance

Study protocol 已写入 `study.yaml`：

```text
train_steps       = 1,000,000
batch_size        = 1,024
learning_rate     = 3e-4
log_interval      = 5,000
eval_interval     = 100,000
eval_tasks        = all
eval_episodes     = 20 per task
eval_temperature  = 0
eval_gaussian     = null
save_interval     = 100,000
save_best/last    = true
primary           = last@1M evaluation/overall_success
secondary         = best checkpoint robustness check
```

所有 7 个 formal runs 必须来自同一个 clean commit、同一个 dataset root、同一个 Large 环境和 seed 0，并且从 step 0 重新训练。旧 M9/M9B/M9B-1M runs 不能混入 M11A factorial comparison。

本任务没有启动 M11A formal run，也没有创建 source checkpoint 或 diagnostic source artifact。

## 9. Generic slot accounting

旧的 `actor_parameter_accounting` 字段没有删除或重命名。新增 `computation_slot_accounting` 对每个 enabled slot 记录：slot name、topology、primitive、credit、state_dim、update_depth、iterations 或 H/L cycles、residual、total executions、state init、trainable/core params、buffer elements 以及 input/update parameter counts。

Large 环境 (`observation_dim=29`, `action_dim=8`) 的代表性 accounting 如下；critic branch 数值已包含 ensemble=2。

| slot | branch input | topology | depth | trainable params | core params | buffers | executions |
|---|---:|---|---:|---:|---:|---:|---:|
| actor SingleState | 58 | SS | 2 | 559,624 | 555,520 | 512 | 4 |
| actor TwoState | 58 | TS H2L1 | 2 | 1,084,936 | 1,080,832 | 1,024 | 4 |
| critic_state SingleState | 37 | SS | 3 | 1,614,848 | 1,614,848 | 1,024 | 4 |
| critic_goal SingleState | 29 | SS | 3 | 1,606,656 | 1,606,656 | 1,024 | 4 |
| critic_state TwoState | 37 | TS H2L1 | 3 | 3,190,784 | 3,190,784 | 2,048 | 4 |
| critic_goal TwoState | 29 | TS H2L1 | 3 | 3,182,592 | 3,182,592 | 2,048 | 4 |

FF/FF C001 保持原 CRL vanilla network，不产生 recurrent buffer；同一 configuration 的 actor、critic_state、critic_goal 不共享参数。

## 10. Shared evaluation diagnostic bank

diagnostic spec 是 `experiments/M11A_crl_computation_interaction/diagnostic.yaml`。bank 由 C001 的 `last@1M` checkpoint 通过真实 environment evaluation rollout 生成，不从 train/val offline dataset 采样，不改变 gradient、agent 或 checkpoint。

协议：all five tasks、20 episodes/task、temperature 0、Gaussian null、`common_task_episode_v1`、evaluation seed `20260820`。每条 trajectory 保存 task/episode identity、episode/actor seed、observation、executed action、next observation、done 和 original eval goal。高维数据使用 `.npz`，provenance 使用 `.json`；metadata 记录 source checkpoint path/SHA256、source commit、config fingerprint、environment、protocol、tensor shapes 和 bank hash。

anchor 与 future goal：

```text
anchor_stride      = 25
goal_offset_stride = 25
max_goal_offset    = 200
h ∈ {25,50,75,100,125,150,175,200}
```

仅当 `t+h < trajectory_length` 时构造 `g_h=observation[t+h]`。bank 生成后是 immutable artifact。

## 11. Diagnostic metrics

### 11.1 Performance interaction

primary performance 使用每个 formal run 的 last@1M overall success：

```text
I_S = J(S-CA) - J(S-C) - J(S-A) + J(A)
I_T = J(T-CA) - J(T-C) - J(T-A) + J(A)
```

`I<0` 只能叫 descriptive substitution，`I>0` 只能叫 descriptive complementarity，`I≈0` 只能叫 descriptive additive/independent。单 seed 不能做 statistical significance claim。

### 11.2 Conservative critic value

严格使用当前 CRL ensemble semantics：

```text
q_C(s,a,g) = min(Q1(s,a,g), Q2(s,a,g))
```

### 11.3 `E_eval_temporal`

固定 baseline rollout 中的 `s_t` 与 `a_exec_t`，对同一 anchor 的 `h_i<h_j` 检查：

```text
expected: q(s_t,a_exec_t,g_hi) > q(s_t,a_exec_t,g_hj)

E_eval_temporal =
    sum 1[q(s_t,a_t,g_hi) <= q(s_t,a_t,g_hj)]
    /
    number_of_valid_temporal_pairs
```

lower is better。输出 overall、pair count、per-h、per-(h_i,h_j)、per-task、per-episode 和 tie count。它是 realized temporal-ordering consistency，不是 oracle Q error，也不是 training contrastive loss。

### 11.4 Candidate action pools

所有 action 在相同 `(s_t,g_{t+h})` 上用 deterministic `distribution.mode()` 计算，再 clip 到 `[-1,1]`。禁止 random action、Gaussian perturbation、critic gradient ascent、unrestricted argmax、CEM、random search 和 dataset nearest-neighbor action。

```text
A_x^S = {a_exec, a_A, a_S-C, a_S-A, a_S-CA}
A_x^T = {a_exec, a_A, a_T-C, a_T-A, a_T-CA}
```

`a_exec` 是 C001 rollout 中真实执行的 action；其余 action 来自对应 source checkpoint。candidate pool 单独保存，后续 scoring 不重新随机生成。

### 11.5 `E_ext_gap`

```text
q_max = max_{a∈A_x} q_C(s,a,g)
q_min = min_{a∈A_x} q_C(s,a,g)

E_ext_gap(C,π;x) =
    [q_max - q_C(s,a_π,g)]
    /
    [q_max - q_min + eps]
```

`eps=1e-6` 在 diagnostic spec 中声明，lower is better。

### 11.6 `E_ext_rank`

```text
E_ext_rank(C,π;x) =
    count_{a≠a_π}[q_C(s,a,g) > q_C(s,a_π,g)]
    /
    (|A_x|-1)
```

严格使用 `>`，tie 不算 beat；同步记录 ties、duplicate candidate actions 和 degenerate pool。若 `q_max-q_min<eps`，不删除 sample，记录 `degenerate_pool=true` 和 `degenerate_pool_rate`。

anchor/goal pair 在 episode 内高度相关，因此 bootstrap 以 episode 为 cluster。CI 只表示 evaluation sampling uncertainty，不表示 training-seed uncertainty。

## 12. Critic identity audit

audit 比较以下理论上应共享 critic training path 的组：

```text
FF critic:       C001, C003, C006
SingleState:     C002, C004
TwoState:        C005, C007
```

对 `critic_state(phi)` 与 `critic_goal(psi)` 分别比较 parameter tree structure、shape、dtype、count、exact equality、max absolute difference、allclose tolerance 和 stable SHA256 fingerprint。输出：`diagnostics/M11A-D001/audits/critic_identity.json`。

如果 identity 不成立，工具会报告完整差异，不强行复制参数、不伪造通过、不静默忽略。当前 CRL DDPG+BC actor loss 的既有 critic gradient path 没有在本任务中改成新的 stop-gradient 语义，因此 identity audit 是实际审计，而不是预设必然通过的断言。

## 13. Artifact layout

```text
diagnostics/M11A-D001/
  bank/diagnostic_bank.npz
  bank/bank_metadata.json
  candidates/single_state_candidates.npz
  candidates/two_state_candidates.npz
  candidates/candidate_metadata.json
  audits/critic_identity.json
  scores/evaluator_scores.npz
  scores/extraction_scores.npz
  metrics/evaluator_metrics.csv
  metrics/extraction_metrics.csv
  metrics/interaction_metrics.csv
  score_metadata.json
  summary.json
```

所有 diagnostic 输出写到独立 root，不写回 source run；source run、checkpoint 和 raw training artifacts 只读。bank/candidate/score 对已有输出采用覆盖保护，并记录上游 hash。

## 14. Tests 与实际验证

已实际执行：

1. computation、CRL runtime 和 M11A targeted tests：通过；
2. M11A 7-config doctor：7/7 通过。每个 config 完成 synthetic create、finite update、checkpoint save/restore 和 action equality probe；
3. M11A integration tests：5/5 通过，覆盖 Study matrix、slot wiring、recurrent critic 双分支、独立 buffers、checkpoint restore 和三个 diagnostic formulas；
4. computation + experiment + integration regression：121 tests，121 passed，0 failures，0 errors；
5. 本次修改 Python 文件 `py_compile`：通过。

完整 unittest discovery 运行了 124 个测试，其中 121 个通过；剩余 3 个 error 来自仓库已有 `tests/analysis/test_pairing.py`、`test_plotting.py`、`test_views.py` 导入 `pytest`，当前环境没有 pytest。没有出现这 3 个测试的断言失败，也没有为此安装依赖或修改环境。

尚未执行：7 个 M11A formal 1M runs、shared bank、critic identity audit、E_eval/E_ext scoring，以及任何长时间 GPU 任务。M9/M9B/M9B-1M source artifacts 未被修改。

## 15. 用户手动执行命令

以下命令只提供给用户手动执行；本 agent 不会执行 formal launcher、reevaluation 或 source-dependent diagnostics。

### Doctor

```bash
cd /home/eai/Research/RLC
JAX_PLATFORMS=cpu PYTHONPATH=. \
/home/eai/Tools/miniforge3/envs/brain_nav/bin/python \
tools/diagnose_crl_interaction.py --stage doctor \
  --study experiments/M11A_crl_computation_interaction/study.yaml
```

### Formal dry-run 与 1M campaign

```bash
cd /home/eai/Research/RLC-exp
bash scripts/run_study.sh \
  --study experiments/M11A_crl_computation_interaction/study.yaml \
  --gpus 0,1 \
  --run-root /data/qijunrong/06-RL/offline-rl/exp/RLC/runs \
  --dataset-root /data/qijunrong/06-RL/offline-rl/data/raw_ogbench \
  --train-steps 1000000 --batch-size 1024 --log-interval 5000 \
  --eval-interval 100000 --eval-tasks all --eval-episodes 20 \
  --save-interval 100000 --eval-temperature 0 --dry-run
```

用户完成 worktree/clean-commit/GPU/protocol 确认后，手动将 `--dry-run` 改为 `--execute` 启动 7 个 formal runs。agent 不替用户创建或验证 worktree。

### Last@1M reevaluation

```bash
cd /home/eai/Research/RLC-exp
PYTHONPATH=. python tools/reevaluate_study.py \
  --spec experiments/M11A_crl_computation_interaction/reevaluation_last.yaml \
  --source-run-root /data/qijunrong/06-RL/offline-rl/exp/RLC/runs \
  --reeval-root /data/qijunrong/06-RL/offline-rl/exp/RLC/reevaluations \
  --gpus 0,1 --dry-run
```

用户确认 source runs completed 后，再手动改为 `--execute`。

### Bank、candidate、audit、score、aggregate

```bash
cd /home/eai/Research/RLC-exp
PYTHONPATH=. python tools/diagnose_crl_interaction.py --stage bank \
  --spec experiments/M11A_crl_computation_interaction/diagnostic.yaml \
  --diagnostic-root /data/qijunrong/06-RL/offline-rl/exp/RLC/diagnostics
PYTHONPATH=. python tools/diagnose_crl_interaction.py --stage candidates \
  --spec experiments/M11A_crl_computation_interaction/diagnostic.yaml \
  --diagnostic-root /data/qijunrong/06-RL/offline-rl/exp/RLC/diagnostics
PYTHONPATH=. python tools/diagnose_crl_interaction.py --stage audit \
  --spec experiments/M11A_crl_computation_interaction/diagnostic.yaml \
  --diagnostic-root /data/qijunrong/06-RL/offline-rl/exp/RLC/diagnostics
PYTHONPATH=. python tools/diagnose_crl_interaction.py --stage score \
  --spec experiments/M11A_crl_computation_interaction/diagnostic.yaml \
  --diagnostic-root /data/qijunrong/06-RL/offline-rl/exp/RLC/diagnostics
PYTHONPATH=. python tools/diagnose_crl_interaction.py --stage aggregate \
  --spec experiments/M11A_crl_computation_interaction/diagnostic.yaml \
  --diagnostic-root /data/qijunrong/06-RL/offline-rl/exp/RLC/diagnostics
```

`aggregate` 使用 last@1M 计算 `I_S`、`I_T`，不会把 best checkpoint 偷换成 primary mechanism result；source identity、bank hash 或 checkpoint provenance 不一致时各阶段会 fail loudly。

## 16. Scientific limitations

1. M11A 只有 seed 0；interaction 正负只能是 descriptive result，不能是统计显著结论。
2. 第一阶段只有 Large；Medium/Giant/Stitch 是 deferred validation。
3. last@1M 是 primary，best 只是 secondary robustness check；不能事后只挑峰值定义 interaction。
4. 20-episode training evaluation 有 sampling noise；100-episode reevaluation 只降低 evaluation uncertainty，不替代多 training seeds。
5. critic identity audit 可能暴露既有 DDPG+BC critic-gradient path；本任务没有修改 CRL loss 强制 identity。
6. `E_eval_temporal` 是 rollout 上的 future-goal ordering consistency，不是 oracle Q error。
7. E_ext candidate pool 是受控小 pool，不代表 offline critic 的全局 action optimum。
8. 当前训练日志还没有 state norm、H/L update magnitude 和 gradient attribution；后续机制解释需要新的 approved telemetry run，不能从现有 loss 臆测。

## 17. 完成判定

代码、Study/config、manifest、reevaluation spec、diagnostic bank/candidate/scoring/audit 工具和测试基础设施均已完成；M11A formal training 与 source-dependent diagnostics 按用户要求保留为手动步骤。系统现在可以在不改变 CRL canonical algorithm、不混入旧 M9 runs、不污染 source artifacts 的前提下，检验 substitution、complementarity 和 additivity 三种假设。
