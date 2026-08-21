# M11B Implementation and Preflight Report

日期：2026-08-21  
Study：M11B — Cross-Task and Cross-Algorithm Computation Demand  
状态：实现与 preflight 已完成；正式训练未启动；当前结论为 NO-GO

## 1. 执行边界与结论

本轮严格按照 docs/8-21/prompt for codex2.md 执行了 Study 设计、配置解析、canonical hyperparameter 核验、静态校验、单元/集成测试、doctor、dry-run，以及真实环境维度上的有限 runtime smoke。

本轮没有执行：

- 任何正式 M11B 训练；
- --execute；
- seed 1–4；
- 1M-step campaign；
- 跨域 E_eval/E_ext diagnostics；
- checkpoint mutation 或旧实验重跑；
- TwoState、H2L、residual、K/width/primitive sweep；
- 任何 Git 命令、commit、push、branch、worktree 操作。

实现层面的主要结果：

- Study 严格解析为 34 个 configurations；
- 34 个配置 ID 和环境绑定均通过静态检查；
- CRL 与 HIQL 的 2×2 factorial 结构通过检查；
- SingleState K4 non-residual 语义通过检查；
- Giant/Humanoid/Stitch canonical overrides 通过检查；
- 3 个可用的新环境上的 12 个真实维度 runtime probes 全部通过；
- CRL/HIQL checkpoint save/restore probes 通过；
- dry-run 报告 planned=34，且没有创建任何正式 M11B run 目录；
- M9A、M9B、M11A 及 computation/runtime regression 全部通过；
- 但 antmaze-large-stitch-v0.npz 与 antmaze-large-stitch-v0-val.npz 均缺失。由于 prompt 明确禁止用其他数据集替代，该关键前置条件失败，因此不能建议创建 frozen formal commit 或启动正式 campaign。

## 2. 科学问题与设计边界

M11B 的问题是：

> GCRL 中额外计算的有效位置与边际价值，如何随任务需求及算法分解方式变化？

M11A 研究了 AntMaze-Large-Navigate 中 CRL 的 actor/critic computation placement；M11B 不再搜索 topology，而是把 computation intervention 固定为 SingleState K4，然后比较：

- CRL：actor × critic_state/critic_goal；
- HIQL：high_actor × low_actor；
- task-scale-associated demand：AntMaze Large → Giant；
- embodiment/control-complexity-associated demand：AntMazeMaze → HumanoidMaze；
- stitching-associated demand：Navigate → Stitch。

跨环境比较仅作 descriptive comparison。不能把 Giant 解读为只改变 horizon，也不能把 Humanoid 解读为只改变 action dimension；只有同一 environment 内的 compute-slot factorial 是 controlled intervention。

### 2.1 五个 environment references

| requested ID | underlying OGBench env | 角色 | 数据证据 | resolved canonical ID |
|---|---|---|---|---|
| antmaze-large-navigate-v0 | antmaze-large-v0 | campaign-level calibration anchor | train/val 均存在；observation (29,)、action (8,) | antmaze-large-navigate-v0 |
| antmaze-giant-navigate-v0 | antmaze-giant-v0 | task-scale-associated reference | train/val 均存在；observation (29,)、action (8,) | antmaze-giant-navigate-v0 |
| humanoidmaze-large-navigate-v0 | humanoidmaze-large-v0 | embodiment/control-complexity-associated reference | train/val 均存在；observation (69,)、action (21,) | humanoidmaze-large-navigate-v0 |
| humanoidmaze-giant-navigate-v0 | humanoidmaze-giant-v0 | difficult embodiment plus scale | train/val 均存在；observation (69,)、action (21,) | humanoidmaze-giant-navigate-v0 |
| antmaze-large-stitch-v0 | antmaze-large-v0 已注册 | dataset compositionality/stitching reference | train/val 均缺失 | 未解析；NO-GO |

证据来源是当前 RLC vendored OGBench 注册表、gymnasium registry 和数据根 /data/qijunrong/06-RL/offline-rl/data/raw_ogbench。另通过 Research 树搜索确认没有另一份 antmaze-large-stitch-v0 train/val 数据可以作为当前数据根的隐式来源。现有 antmaze-teleport-stitch-v0 不被视为替代品。

可用数据的具体 header 形状如下：

| environment | train observation | train action | train rows | val rows |
|---|---:|---:|---:|---:|
| antmaze-large-navigate-v0 | (1001000, 29) | (1001000, 8) | 1,001,000 | 100,100 |
| antmaze-giant-navigate-v0 | (1000500, 29) | (1000500, 8) | 1,000,500 | 100,050 |
| humanoidmaze-large-navigate-v0 | (2001000, 69) | (2001000, 21) | 2,001,000 | 200,100 |
| humanoidmaze-giant-navigate-v0 | (4001000, 69) | (4001000, 21) | 4,001,000 | 400,100 |

## 3. Configuration 数量与完整 34-row table

配置数量为：

~~~text
4 new environments × 4 CRL conditions  = 16
4 new environments × 4 HIQL conditions = 16
2 fresh AntMaze-Large baselines       =  2
total                                  = 34
~~~

每个配置文件显式声明其固定 environment；launcher 不再把 M11B 配置错误地与五个环境做笛卡尔积。因此 34 个配置对应 34 个 seed-0 planned runs。

| ID | environment | algorithm | condition | semantic label |
|---|---|---|---|---|
| M11B-C001 | antmaze-large-navigate-v0 | CRL | baseline | CRL-A: FF actor × FF critic |
| M11B-C002 | antmaze-large-navigate-v0 | HIQL | baseline | HIQL-A: FF high × FF low |
| M11B-C003 | antmaze-giant-navigate-v0 | CRL | baseline | CRL-A: FF actor × FF critic |
| M11B-C004 | antmaze-giant-navigate-v0 | CRL | critic_ss | CRL-C: FF actor × SS critic |
| M11B-C005 | antmaze-giant-navigate-v0 | CRL | actor_ss | CRL-P: SS actor × FF critic |
| M11B-C006 | antmaze-giant-navigate-v0 | CRL | actor_critic_ss | CRL-PC: SS actor × SS critic |
| M11B-C007 | antmaze-giant-navigate-v0 | HIQL | baseline | HIQL-A: FF high × FF low |
| M11B-C008 | antmaze-giant-navigate-v0 | HIQL | high_ss | HIQL-H: SS high × FF low |
| M11B-C009 | antmaze-giant-navigate-v0 | HIQL | low_ss | HIQL-L: FF high × SS low |
| M11B-C010 | antmaze-giant-navigate-v0 | HIQL | high_low_ss | HIQL-HL: SS high × SS low |
| M11B-C011 | humanoidmaze-large-navigate-v0 | CRL | baseline | CRL-A: FF actor × FF critic |
| M11B-C012 | humanoidmaze-large-navigate-v0 | CRL | critic_ss | CRL-C: FF actor × SS critic |
| M11B-C013 | humanoidmaze-large-navigate-v0 | CRL | actor_ss | CRL-P: SS actor × FF critic |
| M11B-C014 | humanoidmaze-large-navigate-v0 | CRL | actor_critic_ss | CRL-PC: SS actor × SS critic |
| M11B-C015 | humanoidmaze-large-navigate-v0 | HIQL | baseline | HIQL-A: FF high × FF low |
| M11B-C016 | humanoidmaze-large-navigate-v0 | HIQL | high_ss | HIQL-H: SS high × FF low |
| M11B-C017 | humanoidmaze-large-navigate-v0 | HIQL | low_ss | HIQL-L: FF high × SS low |
| M11B-C018 | humanoidmaze-large-navigate-v0 | HIQL | high_low_ss | HIQL-HL: SS high × SS low |
| M11B-C019 | humanoidmaze-giant-navigate-v0 | CRL | baseline | CRL-A: FF actor × FF critic |
| M11B-C020 | humanoidmaze-giant-navigate-v0 | CRL | critic_ss | CRL-C: FF actor × SS critic |
| M11B-C021 | humanoidmaze-giant-navigate-v0 | CRL | actor_ss | CRL-P: SS actor × FF critic |
| M11B-C022 | humanoidmaze-giant-navigate-v0 | CRL | actor_critic_ss | CRL-PC: SS actor × SS critic |
| M11B-C023 | humanoidmaze-giant-navigate-v0 | HIQL | baseline | HIQL-A: FF high × FF low |
| M11B-C024 | humanoidmaze-giant-navigate-v0 | HIQL | high_ss | HIQL-H: SS high × FF low |
| M11B-C025 | humanoidmaze-giant-navigate-v0 | HIQL | low_ss | HIQL-L: FF high × SS low |
| M11B-C026 | humanoidmaze-giant-navigate-v0 | HIQL | high_low_ss | HIQL-HL: SS high × SS low |
| M11B-C027 | antmaze-large-stitch-v0 | CRL | baseline | CRL-A: FF actor × FF critic |
| M11B-C028 | antmaze-large-stitch-v0 | CRL | critic_ss | CRL-C: FF actor × SS critic |
| M11B-C029 | antmaze-large-stitch-v0 | CRL | actor_ss | CRL-P: SS actor × FF critic |
| M11B-C030 | antmaze-large-stitch-v0 | CRL | actor_critic_ss | CRL-PC: SS actor × SS critic |
| M11B-C031 | antmaze-large-stitch-v0 | HIQL | baseline | HIQL-A: FF high × FF low |
| M11B-C032 | antmaze-large-stitch-v0 | HIQL | high_ss | HIQL-H: SS high × FF low |
| M11B-C033 | antmaze-large-stitch-v0 | HIQL | low_ss | HIQL-L: FF high × SS low |
| M11B-C034 | antmaze-large-stitch-v0 | HIQL | high_low_ss | HIQL-HL: SS high × SS low |

每个 configuration 的 semantic_label 进一步包含 environment 前缀，因此跨环境重复的 condition name 不会形成歧义。

## 4. Factorial 结构

### 4.1 CRL

对四个新环境，CRL 条件为：

| condition | actor | critic_state | critic_goal |
|---|---|---|---|
| CRL-A | FF | FF | FF |
| CRL-C | FF | SingleState K4 | SingleState K4 |
| CRL-P | SingleState K4 | FF | FF |
| CRL-PC | SingleState K4 | SingleState K4 | SingleState K4 |

critic_state 和 critic_goal 是两个独立模块，拥有独立 params 和 independent recurrent buffers。CRL bilinear evaluator、ensemble、actor loss、critic contrastive loss 以及 conservative actor-facing min(Q1,Q2) 语义均未改动。

~~~text
J_A  = FF actor + FF critic
J_C  = FF actor + SS critic
J_P  = SS actor + FF critic
J_PC = SS actor + SS critic

DeltaP = J_P - J_A
DeltaC = J_C - J_A
I_PC   = J_PC - J_P - J_C + J_A
~~~

这些量只作 descriptive complementarity/substitution/approximately additive 描述，不作 statistically significant interaction 结论，因为 Stage 1 只有一个 training seed。

### 4.2 HIQL

HIQL 没有被强行映射到 CRL actor/critic。其 factorial 直接作用于 high/low actor：

| condition | high actor | low actor |
|---|---|---|
| HIQL-A | FF | FF |
| HIQL-H | SingleState K4 | FF |
| HIQL-L | FF | SingleState K4 |
| HIQL-HL | SingleState K4 | SingleState K4 |

~~~text
J_A  = FF high + FF low
J_H  = SS high + FF low
J_L  = FF high + SS low
J_HL = SS high + SS low

DeltaH = J_H - J_A
DeltaL = J_L - J_A
I_HL   = J_HL - J_H - J_L + J_A
~~~

HIQL 的 value objective、expectile/advantage semantics、subgoal construction、target update、high/low readout、action distribution 和 dataset sampling 均保持不变；SingleState 只替换 high/low actor body。

## 5. Frozen computation semantics

所有 recurrent condition 都使用：

~~~text
primitive             = mlp
topology              = single_state
iterations            = 4
residual              = false
input_injection       = z_plus_x
state_dim             = 512
state_init            = normal_buffer
state_init_std        = 1.0
decision scope        = decision-local
update weights        = shared across iterations
~~~

CRL actor、HIQL high actor、HIQL low actor 继承 M11A/M9A actor semantics：

~~~text
update_depth          = 2
layer_norm            = false
update_activate_final = true
~~~

CRL recurrent critic 的修正版 semantics 为：

~~~text
update_depth          = 3
layer_norm            = true
update_activate_final = false
~~~

因此 CRL critic recurrent branch 只替换 computation topology，不重新引入 normalization 或 final-activation confound。所有 FF slot 显式解析为 enabled=false, topology=feedforward。

## 6. Canonical hyperparameter 核验

主要 canonical source：

~~~text
/home/eai/Research/offline-rl/docs/ALGORITHM_HYPERPARAMETERS.md
~~~

该表的 CRL/HIQL task rows 与当前 RLC agent defaults 一致地提供 task-context override；当前 RLC 的 impls/agents/crl.py 和 impls/agents/hiql.py 提供共享 lr、batch、hidden dims、objective 与 dataset defaults。M11B 通过 impls/experiment/m11b.py 解析，配置 YAML 不重复硬编码 34 份 agent override。

### 6.1 CRL resolved overrides

所有 CRL 环境共同保持：

~~~text
lr=3e-4, batch_size=1024
actor_hidden_dims=(512,512,512)
value_hidden_dims=(512,512,512), latent_dim=512
layer_norm=true, actor_loss=ddpgbc, alpha=0.1, const_std=true
dataset_class=GCDataset
value goal mix = cur/traj/random = 0.0/1.0/0.0
value_geom_sample=true, actor_geom_sample=false
gc_negative=false, p_aug=0.0
~~~

| environment | discount | actor goal mix cur/traj/random |
|---|---:|---:|
| antmaze-large-navigate-v0 | 0.99 | 0.0 / 1.0 / 0.0 |
| antmaze-giant-navigate-v0 | 0.995 | 0.0 / 1.0 / 0.0 |
| humanoidmaze-large-navigate-v0 | 0.995 | 0.0 / 1.0 / 0.0 |
| humanoidmaze-giant-navigate-v0 | 0.995 | 0.0 / 1.0 / 0.0 |
| antmaze-large-stitch-v0 | 0.99 | 0.0 / 0.5 / 0.5 |

### 6.2 HIQL resolved overrides

所有 HIQL 环境共同保持：

~~~text
lr=3e-4, batch_size=1024
actor_hidden_dims=(512,512,512)
value_hidden_dims=(512,512,512), layer_norm=true
tau=0.005, expectile=0.7, low_alpha=3.0, high_alpha=3.0
rep_dim=10, low_actor_rep_grad=false, const_std=true
dataset_class=HGCDataset
value goal mix = cur/traj/random = 0.2/0.5/0.3
value_geom_sample=true, actor_geom_sample=false
gc_negative=true, p_aug=0.0
~~~

| environment | discount | subgoal_steps | actor goal mix cur/traj/random |
|---|---:|---:|---:|
| antmaze-large-navigate-v0 | 0.99 | 25 | 0.0 / 1.0 / 0.0 |
| antmaze-giant-navigate-v0 | 0.995 | 25 | 0.0 / 1.0 / 0.0 |
| humanoidmaze-large-navigate-v0 | 0.995 | 100 | 0.0 / 1.0 / 0.0 |
| humanoidmaze-giant-navigate-v0 | 0.995 | 100 | 0.0 / 1.0 / 0.0 |
| antmaze-large-stitch-v0 | 0.99 | 25 | 0.0 / 0.5 / 0.5 |

### 6.3 Prompt expectation 与 resolved value

| expectation in prompt | canonical source value | resolved value | 结论 |
|---|---|---|---|
| Giant/Humanoid likely discount=0.995 | Giant antmaze 0.995；Humanoid large/giant 0.995 | 按 environment profile 解析 | 一致 |
| HIQL Humanoid likely subgoal_steps=100 | Humanoid HIQL rows 100 | large/giant 均为 100 | 一致 |
| Stitch actor goal 0.5/0.5 | Stitch CRL/HIQL rows traj=0.5, random=0.5 | 仅 Stitch 使用 0.0/0.5/0.5 | 一致 |
| Navigate actor goal 1.0/0.0 | Navigate CRL/HIQL rows traj=1.0, random=0.0 | 仅 Navigate 使用 0.0/1.0/0.0 | 一致 |

没有发现需要静默覆盖 canonical source 的 discrepancy。

## 7. Training/evaluation/secondary protocol

所有 34 个未来 formal runs 统一使用：

~~~text
training seed       = 0
train_steps         = 1,000,000
batch_size          = 1,024
learning_rate       = algorithm canonical lr = 3e-4
log_interval        = 5,000
eval_interval       = 100,000
save_interval       = 100,000
eval_tasks          = all
eval_episodes       = 20 per task
eval_temperature    = 0
eval_gaussian       = none
video               = false
primary             = evaluation/overall_success at last@1M
secondary           = best_success, best_step, last3_mean, normalized_eval_auc
~~~

best_success 不能替代 primary last@1M。

### 7.1 normalized evaluation AUC

使用 100k–1M 的 10 个共同 checkpoint：

~~~text
100k, 200k, 300k, ..., 900k, 1M
~~~

固定公式为：

~~~text
normalized_eval_auc
  = 1/(1,000,000 - 100,000)
    * Σ_i [(s_i + s_{i+1}) / 2] * (t_{i+1} - t_i)
~~~

即对 [100k, 1M] 做 trapezoidal area 后除以 900,000，结果仍在 success scale。synthetic test 使用从 0.1 到 1.0 的线性曲线，结果为 0.55，验证通过。

## 8. Hypotheses 与未来 replication

Study documentation 已写入：

- H1：actor/policy-side computation 在 AntMaze-Large 以外可能仍有正向边际价值；
- H2：computation effect magnitude 可能随 task scale 改变，但不预注册单调性；
- H3：useful computation placement 可能随 embodiment/control-complexity-associated demand 改变；
- H4：若 generic critic computation 响应 stitching/cross-trajectory burden，DeltaC_Stitch 相对 Navigate 可能增加；
- H5：CRL actor×critic 与 HIQL high×low 可能呈现不同 allocation pattern。

H4 的关键预测是 relative critic marginal value 增加，不是 DeltaC_Stitch > DeltaP_Stitch。

后续 multi-seed replication 预注册为 seeds 0,1,2,3,4；触发条件包括：|primary Delta| >= 0.10、重要任务间 effect sign reversal、Stitch 与 Navigate critic effect 显著不同、CRL/HIQL placement pattern 变化、Humanoid-Giant 出现 qualitatively new effect，或 paper-level claim 会依赖单一 seed-0。

## 9. Fingerprint 与 runtime metadata

每个配置的 fingerprint payload 至少包含：

~~~text
study_id
config_id
semantic_condition / semantic_label
algorithm
environment
dataset_root
canonical resolved agent config
training seed
training protocol
source commit placeholder/value
~~~

doctor 已为 34 个 configuration 解析并生成稳定 SHA-256 fingerprints。当前因为 Git 操作必须由用户手动完成，source commit 在 doctor/dry-run 中显示为 <manual-user-supplied>；用户审核并手动冻结 commit 后可通过 RLC_SOURCE_COMMIT=<commit> 注入。

未来正式 run 的 runtime_metadata.json 还会保存：study_id、config_id、semantic_condition、algorithm、environment、seed、git_commit、git_dirty、dataset root、resolved config、fingerprint、compute slots，以及 recurrent slot accounting。resolved_config.json 同时包含 launcher protocol、agent config、dataset root、environment 和 seed，使 fingerprint 不只依赖 YAML filename。

## 10. Doctor 结果

执行命令：

~~~bash
PYTHONPATH=. MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  /home/eai/Tools/miniforge3/envs/brain_nav/bin/python \
  tools/m11b_doctor.py \
  --dataset-root /data/qijunrong/06-RL/offline-rl/data/raw_ogbench
~~~

核心结果：

| check | result |
|---|---|
| planned_configs | 34 |
| config structure / ID / semantic label | PASS |
| canonical agent resolution | PASS |
| no TwoState/H2L/residual/K leakage | PASS |
| canonical environment registry | 四个可用 reference + Stitch underlying env registered |
| dataset train/val | 4 个 reference PASS；antmaze-large-stitch-v0 FAIL |
| real shape + action + one finite update | 12/12 probes PASS |
| deterministic action and action bounds | 12/12 PASS |
| checkpoint save/restore | CRL full SS、HIQL full SS 各 1 个 probe，均 PASS |
| AUC synthetic test | PASS，0.55 |
| factorial aggregation synthetic test | PASS |
| formal training started | false |
| final doctor | NO-GO |

真实 runtime probe 矩阵覆盖 3 个可用新环境，每个环境运行：CRL baseline、CRL actor+critic SS、HIQL baseline、HIQL high+low SS，共 12 个 probes。结果均报告实际 observation/action dimensions、deterministic action、action bounds、one finite update 和 updated step 2。

## 11. Dry-run 结果

统一 launcher 命令：

~~~bash
RLC_PYTHON=/home/eai/Tools/miniforge3/envs/brain_nav/bin/python \
OGBENCH_DATASET_DIR=/data/qijunrong/06-RL/offline-rl/data/raw_ogbench \
RLC_SOURCE_COMMIT=<user-manually-reviewed-commit> \
bash scripts/run_study.sh \
  --study experiments/M11B_cross_task_computation/study.yaml \
  --gpus 0,1 \
  --run-root /data/qijunrong/06-RL/offline-rl/exp/RLC/runs \
  --dataset-root /data/qijunrong/06-RL/offline-rl/data/raw_ogbench \
  --train-steps 1000000 \
  --batch-size 1024 \
  --log-interval 5000 \
  --eval-interval 100000 \
  --eval-tasks all \
  --eval-episodes 20 \
  --save-interval 100000 \
  --eval-temperature 0 \
  --dry-run
~~~

实际 dry-run 结果为：

~~~text
Git preflight: skipped for --dry-run
Run-root preflight: skipped for --dry-run (no run directory will be created)
Planned runs: 34
total=34 planned=34 completed=0 failed=0 running=0 retained=0 remaining=34
Mode: --dry-run
dataset_preflight: NO-GO missing=antmaze-large-stitch-v0
~~~

dry-run 打印了每个 configuration 的 ID、semantic condition、algorithm、environment、seed、GPU round-robin allocation、train steps、evaluation protocol 和 canonical run path。测试后确认 /data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M11B 没有新建正式 run 子目录。

--execute 分支仍保留 formal execution 的 Git clean-worktree 检查，但本轮没有调用它；未来由用户本人手动审核 Git 状态、冻结 commit 并执行正式命令。

## 12. Regression/test results

### 12.1 M11B-specific

~~~text
tests.integration.test_m11b_study
tests.experiment.test_sweep
13 tests: OK
~~~

覆盖了：34 config 结构、ID/semantic label 唯一性、CRL/HIQL factorial、SingleState semantics、Giant/Humanoid/Stitch overrides、Humanoid hierarchy fields、Stitch sampling anti-leak、真实 environment/data shape、AUC、factorial aggregation 和 sweep fixed-environment job count。

### 12.2 M9/M11A regression

~~~text
tests.integration.test_m9_single_state_study
tests.integration.test_m9b_two_state_1m_study
tests.integration.test_m11a_crl_interaction
tests.computation.test_single_state
tests.computation.test_foundation
40 tests: OK

tests.integration.test_m9b_two_state_study
tests.integration.test_computation_provenance
tests.integration.test_hiql_smoke
7 tests: OK

tests.integration.test_crl_runtime
13 tests: OK
~~~

这些回归覆盖 M9A SingleState、M9B TwoState、M11A critic primitive parity、buffer handling、gradient boundary、HIQL high/low legacy creation、CRL runtime 和 checkpoint restore 相关路径。bash -n scripts/run_study.sh、相关 Python py_compile 也通过。

运行中 JAX 输出了 CUDA plugin 在当前无可用 CUDA device 时的初始化 warning；runtime probes 以 CPU fallback 完成，未启动任何正式 GPU training。

## 13. Aggregation schema

未来 run 完成后，每个 run 至少提取：

~~~text
final_success       = last@1M evaluation/overall_success
best_success
best_step
last3_mean
normalized_eval_auc
~~~

aggregation 只输出 descriptive quantities，不自动生成强科学语言。CRL table 输出 baseline、actor SS、critic SS、actor+critic SS、Delta actor、Delta critic、interaction；HIQL table 输出 baseline、high SS、low SS、high+low SS、Delta high、Delta low、interaction；cross-task table 输出 environment、algorithm、placement effects、interaction 和 AUC pattern。

M11B fresh Large baselines 与 M11A historical factorial 明确区分：前者是 campaign calibration anchor，后者未来只能作为 historical/contextual prior，不能在 aggregation 中伪装成同一 campaign 的 paired factorial。

## 14. Unresolved issues and final recommendation

### Blocking issue

antmaze-large-stitch-v0 的 train/val 数据缺失：

~~~text
/data/qijunrong/06-RL/offline-rl/data/raw_ogbench/antmaze-large-stitch-v0.npz
/data/qijunrong/06-RL/offline-rl/data/raw_ogbench/antmaze-large-stitch-v0-val.npz
~~~

underlying antmaze-large-v0 environment 虽然已注册，但这不能证明 requested Stitch dataset 存在。antmaze-teleport-stitch-v0 也不能替代 requested Large Stitch。未经用户确认、数据补齐和重新 preflight，不应改变 Study scientific design。

### Final recommendation

**NO-GO for creating the frozen formal M11B commit and NO-GO for formal training at this time.**

代码实现和除 Stitch 外的 preflight 已通过；一旦用户补齐与 requested ID 完全对应的 train/val 数据，应重新运行 doctor 和 dry-run，并确认：

1. antmaze-large-stitch-v0 两个文件存在且 shape/dataset semantics 正确；
2. doctor 达到 34/34 PASS；
3. RLC_SOURCE_COMMIT 由用户手动填入最终审核 commit；
4. 用户自行完成全部 Git 操作后，再由用户手动决定是否启动 34 个 seed-0、1M-step formal runs。

当前报告明确记录：**formal M11B training was NOT started**。

## 15. Post-download update（2026-08-21）

在用户授权后，使用官方 OGBench 下载索引，将以下三类环境的普通 Stitch train/val 数据下载到：

~~~text
/data/qijunrong/06-RL/offline-rl/data/raw_ogbench
~~~

下载范围为 `antmaze`、`pointmaze`、`humanoidmaze` 的普通 `medium/large/giant-stitch-v0` 数据；原有的 `teleport-stitch-v0` 文件未被替换。官方下载器最终复核结果为：

- Stitch 目标文件共 22 个（普通 Stitch 18 个 + 原有 teleport Stitch 4 个）；
- `Need to download 0 files`，即全部文件均与远端 Content-Length 一致；
- 全部 22 个 NPZ 归档 CRC 检查通过，`bad=[]`；
- 目标文件累计大小为 5,905,659,695 bytes（约 5.91 GB）；
- 未保留 `.tmp` 残留文件。

随后重新运行 M11B doctor：

- 静态 doctor：`34/34 PASS`；
- 五个 Study environment reference 均解析成功，包含 `antmaze-large-stitch-v0`；
- Stitch train shape 为 `(1005000, 29)`，action shape 为 `(1005000, 8)`；
- Stitch val shape 为 `(100500, 29)`，action shape 为 `(100500, 8)`；
- checkpoint save/restore、fingerprint、AUC/factorial synthetic checks 全部通过；
- 完整 runtime probes 共 16 个，全部通过 `one_finite_update`、动作边界和确定性检查；
- doctor 最终状态：`GO`，`formal_training_started: false`。

因此，本报告第 14 节中关于 Stitch 数据缺失的 `NO-GO` 结论已被本节 supersede。当前 M11B 已达到“实现与预检通过、可由用户手动审核后决定正式启动”的状态；正式训练仍未启动，且本次没有执行任何 Git 操作。完整 runtime doctor 应使用 `MUJOCO_GL=egl PYOPENGL_PLATFORM=egl` 环境变量；未设置该变量时仅会出现 OpenGL context 预检错误，不代表数据或 Study 配置错误。
