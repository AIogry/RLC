# M12B-R — Missing SS-K4 Shared-Normal Completion

# Preflight Report

日期：2026-08-24  
实验身份：M12B-R  
唯一目标：补齐 M12B active matrix 中缺失的 SS-K4 + shared + normal_buffer  
最终判定：**NO-GO（未启动正式训练）**

## 1. Executive summary

M12B active matrix 已有 K4 shared zero，但没有 K4 shared normal。此前 M12A-C003 的 K4 normal 只是 historical reference，不能代替 M12B active run。

本轮严格限定为：

- 唯一 condition：SS-K4 + shared + normal_buffer
- seed：0、1、2
- formal runs：精确 3 个
- 不重跑 M12B K4 shared zero
- 不运行其它 architecture、task、algorithm 或 seed

科学 preflight 的主要结果均通过：

- M12A-C003 在当前 implementation 下 resolved 为 SS-K4 shared normal。
- 它与已有 M12B-C003 K4 shared zero 的有效 actor spec 除 state_init 外完全一致。
- 真实参数树、参数量、Dense 数量、执行次数和 MAC parity 全部通过。
- seed 0 的真实 production-path smoke 通过。
- 25 个相关 unittest 全部通过。
- dry-run 严格规划 3 个目标 run，未创建正式 artifact。

最终仍为 NO-GO，原因是：

1. 按用户要求，本轮没有执行任何 Git 操作，因此无法实时证明当前 worktree 是 exact target commit、clean 且 detached。
2. JAX CUDA plugin 初始化失败并回退 CPU；GPU 可见性不等于当前 JAX formal runtime 已在 GPU 上验证。

## 2. Source provenance 与 worktree 状态

Prompt 要求的正式 source commit：

bb2644ccb23ee77a0c08e8b9cded85a57716df67

本轮没有执行 git rev-parse、git status、git branch 或其它 Git 操作。因此当前 worktree 的实时状态记录为：

| 项目 | 本轮状态 |
|---|---|
| current HEAD exact target | 未核验 |
| worktree clean | 未核验 |
| detached HEAD | 未核验 |
| Git write operation | 未执行 |

历史 artifact provenance 可作为旁证，但不能代替当前实时 Git 状态：

- 已有 M12B-C003 K4 shared zero 的三个 formal artifact metadata 均记录 commit bb2644... 且 git_dirty=false。
- 旧 M12A-C003 normal reference attempt 1 使用的是历史 commit b3fde3...，它不是本轮目标 source。
- 本轮候选正式 run 必须由用户在 exact bb2644...、clean、detached 的 worktree 中手动启动。

## 3. 为什么只补 3 个 runs

M12B 的 active matrix 实际状态为：

| actor condition | status in M12B |
|---|---|
| K1 shared normal | existing |
| K1 shared zero | existing |
| K4 shared normal | missing；本轮唯一目标 |
| K4 shared zero | existing；本轮禁止重跑 |
| K4 untied normal | existing |
| K4 untied zero | existing |
| Deep FF | existing |
| Residual FF | existing |

因此本轮只需生成 M12B shared SingleState 2×2 matrix 的一个缺失 cell：

| 结构 | normal_buffer | zero_buffer |
|---|---:|---:|
| K1 shared | existing | existing |
| K4 shared | THIS RUN | existing |

本轮不提前解释 2×2 matrix 的数值结果。正式完成后才可计算：

- Δ_K_normal
- Δ_K_zero
- Δ_init_K1
- Δ_init_K4
- I_K×init

旧 M12A-C003 normal 只保留为 prior reference / replication reference，不能和新三个 run 合并为六个 independent seeds。

## 4. M12A-C003 resolved semantics

本轮复用了：

experiments/M12A_frozen_critic_policy_extraction/configs/M12A-C003.yaml

在当前 source tree 的 resolver 下，M12A-C003 的有效 actor spec 为：

| field | resolved value |
|---|---|
| primitive | mlp |
| block | plain |
| topology | single_state |
| parameter_sharing | shared |
| credit | direct |
| iterations | 4 |
| residual | false |
| input_injection | z_plus_x |
| state_dim | 512 |
| state_init | normal_buffer |
| state_init_std | 1.0 |
| update_depth | 2 |
| layer_norm | false |
| update_activate_final | true |

其它 Stage-2 agent identity 也已核验：

- algorithm：CRL
- actor loss：DDPG+BC
- runtime variant：policy_extractor
- training mode：policy_extraction
- actor hidden dims：512, 512, 512
- value hidden dims：512, 512, 512
- learning rate：0.0003
- batch size：1024

M12A-C003 YAML 中没有显式写 parameter_sharing，但当前 SingleState resolver 的默认值是 shared；没有通过新增配置或修改源码实现该语义。

## 5. 与 M12B-C003 K4-zero 的 parity

已有 M12B K4 shared zero reference：

experiments/M12B_actor_computation_structure_isolation/configs/M12B-C003.yaml

对 M12A-C003 和 M12B-C003 进行有效 resolved-agent 对比后，除 state_init 外没有差异：

| field | K4 normal candidate | K4 zero reference |
|---|---|---|
| primitive | mlp | mlp |
| block | plain | plain |
| topology | single_state | single_state |
| parameter_sharing | shared | shared |
| credit | direct | direct |
| iterations | 4 | 4 |
| residual | false | false |
| input_injection | z_plus_x | z_plus_x |
| state_dim | 512 | 512 |
| state_init_std | 1.0 | 1.0 |
| update_depth | 2 | 2 |
| layer_norm | false | false |
| update_activate_final | true | true |
| state_init | normal_buffer | zero_buffer |

因此本轮没有创建 M12B-C008，也没有修改 M12B-C003 或其它 YAML。

## 6. Shared parameterization 与 initialization semantics

真实 agent construction 验证：

- actor parameter tree 使用历史 shared subtree：modules_actor.actor_net.topology.update_module。
- 没有出现 update_modules_0、update_modules_1 等 untied subtree。
- normal_buffer 的 z_init 位于：

modules_actor.actor_net.topology.z_init

- z_init 是 model_state 中的 persistent、non-trainable buffer。
- normal candidate 的 z_init 非零；seed 0 的最大绝对值为 4.6848669052。
- zero reference 的 z_init 精确全零。
- trainable actor params 不因 state_init 改变。
- buffer 未在 initialization 后修改。

同 seed、同 example batch 下，normal/zero 的 actor parameter tree fingerprint 完全相同；model-state 的唯一差异路径就是上述 z_init。

## 7. Parameter 与 computation accounting parity

以下数值由真实初始化后的 parameter tree 和 accounting helper 计算，不是硬编码 truth：

| accounting field | K4 normal | K4 zero |
|---|---:|---:|
| core trainable params | 555,520 | 555,520 |
| actor total trainable params | 559,624 | 559,624 |
| buffer elements | 512 | 512 |
| input mapping params | 30,208 | 30,208 |
| update module params | 525,312 | 525,312 |
| unique Dense layers | 3 | 3 |
| executed Dense layers | 9 | 9 |
| update executions | 4 | 4 |
| sequential depth | 9 | 9 |
| actor body Dense MACs | 2,126,848 | 2,126,848 |
| full actor forward Dense MACs | 2,130,944 | 2,130,944 |
| parameter-tree Dense MACs | 558,080 | 558,080 |
| update module MACs per execution | 524,288 | 524,288 |
| total update-module MACs | 2,097,152 | 2,097,152 |

因此 K4 normal 与 K4 zero 在 trainable parameter count、unique Dense count、executed Dense count 和 forward computation 上完全 parity。

## 8. Stage-2 protocol

正式三个 run 必须使用以下 protocol：

- environment：antmaze-large-navigate-v0
- algorithm：CRL
- runtime：policy_extractor
- training mode：policy_extraction
- actor objective：DDPG+BC
- training steps：1,000,000
- batch size：1024
- learning rate：3e-4
- log interval：5000
- evaluation interval：100000
- evaluation tasks：all
- evaluation episodes：20
- evaluation temperature：0
- Gaussian noise：disabled
- video：disabled
- save interval：100000
- save best：true
- save last：true
- primary endpoint：overall success @ last@1M
- secondary endpoints：normalized trapezoid AUC、best success、best step、last3 mean

M12A study Stage-2 protocol 与 M12B active Stage-2 protocol 的上述字段一致，因此复用 M12A-C003 semantics 不改变 M12B 的 runtime/protocol。正式新 artifact 通过 run_attempt=2 与历史 M12A-C003 artifacts 分离。

## 9. Frozen critic provenance

每个新 seed c 都必须使用 M12A-C001 seed c、attempt 0、critic、last@1M。已有 M12B-C003 metadata 已验证其依赖记录与该规则一致：

| seed | checkpoint path | SHA-256 | critic module fingerprint |
|---:|---|---|---|
| 0 | M12A-C001/antmaze-large-navigate-v0/seed_000/checkpoints/last/params_1000000.pkl | f801f7521aedc70a0ed182a2a2f2d7765d9faa0e6b7ac623f98ad284926006d5 | 35bfa7630a317e40bae4fbc4f529635c4655f8946975af1e87388d4490bb85b7 |
| 1 | M12A-C001/antmaze-large-navigate-v0/seed_001/checkpoints/last/params_1000000.pkl | b89f45b1e61436b0ee469471b51f40b517ac6130339951c9086c2a5912281c98 | 1ac0eda0a97b315e1f8a6e48d0b29c70267de85bb54b81c4ba9718f0bbcdc36e |
| 2 | M12A-C001/antmaze-large-navigate-v0/seed_002/checkpoints/last/params_1000000.pkl | 64def1c398dd59a30b533c1cfa704937d389ca5a5088ba45db66b3f7a8005ab2 | 0c0579ae1f6012e89b2c5ec465dd976baabee0d7f882efbddc4cbe4330423d18 |

每个 checkpoint 都满足：

- checkpoint role = last
- checkpoint step = 1,000,000
- same seed
- source config = M12A-C001
- source attempt = 0
- module = critic

不使用 best、早期 checkpoint、其它 seed、其它 attempt 或 fallback critic。

## 10. RNG 与 batch-stream parity

使用真实 antmaze-large-navigate-v0 dataset 和 Stage-2 GCDataset generation logic，对 seed 0 按 derived seed stream 生成前 10 个 batch，并分别用 normal/zero resolved config 重复生成。

覆盖字段：

- observations
- actions
- actor_goals
- value_goals

前 10 个 batch hash：

~~~text
21b38a58fad7fe5c94b77d0a671ca60b6c50986171895644f777a908b8a2c35d
3ac1655f2eaf739537c174d98caa19f89e25960174a63f6714cca13fc7ade1f0
1e730bc6516e46526d10b70c6cf13c43cf1ce459d096191431350a8185e71166
9b30b97d6a26caff52d8da1a17358cd3062ae851d729afc5e6d3227bd1c87b21
307ddc4b2bce341ed412c3ca7b32507fc129901e5c64b8ed73118b13775530cb
dac421601f8775c63ba8e57ebf92817efa434936f61ecc79de37a36fbb9fd453
f3bdbebfbcbf4057ed3ae8497bff73c0f0f21d275291cf2467e5fa75dee56c0d
3bd96fdb4cb0014b6964836aa76e2bb296b691159b21191f74890c862b4739b7
5b24d7c25435287b251aa49e7ee882b7c72d50a7681e772d0b05087fe41e151f
56ed62ed103d3ef6bb7861bd9a47dc55b2a7c8bd804316b9e96f66d82021484d
~~~

normal 与 zero 两组 10 个 hash 完全相同。由于配置只改变 z_init，actor-goal sampling、dataset RNG、batch sampling 和 evaluation seed derivation 均保持同一逻辑。

## 11. Production-path smoke test

没有使用 formal run directory。使用真实 dataset、M12A-C003 resolved config、CRLPolicyExtractorAgent 和 seed 0 完成：

1. resolve config；
2. 创建 policy extractor；
3. 恢复 seed-matched C001 last@1M critic；
4. 检查 critic SHA/fingerprint；
5. 执行 3 次真实 actor update。

结果：

| check | result |
|---|---|
| checkpoint role/step | last@1M |
| restored critic fingerprint matches expected | pass |
| JAX backend | cpu |
| optimizer steps | 2, 3, 4 |
| all metrics finite | pass |
| actor changes each update | pass |
| critic unchanged each update | pass |
| final critic fingerprint unchanged | pass |

Smoke conclusion：**SMOKE_PASS = true**。

但该 smoke 过程同时报告 JAX CUDA plugin 初始化失败并回退 CPU。它验证了 production code path 和数值逻辑，不能证明正式 GPU runtime 已通过验证。

## 12. Tests

本轮使用 unittest，不依赖未安装的 pytest：

- tests.computation.test_m12b_architecture：5/5 通过。
- tests.integration.test_m12a_frozen_critic：15/15 通过。
- tests.experiment.test_sweep：5/5 通过。

合计窄范围相关测试：25/25 通过。

此外，production-path smoke 也通过。测试和 smoke 均没有创建 formal run artifact。

## 13. Artifact audit 与 selected run_attempt

### 13.1 M12A-C003 historical artifacts

实际扫描结果：

- seed 0/1/2 canonical attempt 0：均存在，但 status=failed。
- seed 0/1/2 attempt 1：均 completed。
- attempt 2：三个 seed 均不存在。
- attempt 1 的历史 source commit：b3fde3f91d89169c02c7604ace80d65bdf8ced25。

因此本轮统一选择：

run_attempt = 2

不会覆盖 attempt 0 或 attempt 1。

### 13.2 Existing M12B-C003 artifacts

- seed 0/1/2 canonical attempt 0：均 completed。
- source commit metadata：bb2644...。
- last@1M 和 best checkpoint 均存在。
- 本轮不重跑、不覆盖、不创建 zero attempt。

## 14. Exact target run paths

由于 prompt 要求不创建新的 model config，且 M12A-C003 是已存在的唯一 executable K4-normal config semantics，本轮新 artifact 使用 M12A-C003 identity 加 attempt 2；科学身份通过本报告标记为 M12B-R missing SS-K4 shared-normal completion。

三个目标路径精确为：

~~~text
/data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M12A/M12A-C003__policy_extraction_single_state_k4_actor/antmaze-large-navigate-v0/seed_000__attempt_002
/data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M12A/M12A-C003__policy_extraction_single_state_k4_actor/antmaze-large-navigate-v0/seed_001__attempt_002
/data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M12A/M12A-C003__policy_extraction_single_state_k4_actor/antmaze-large-navigate-v0/seed_002__attempt_002
~~~

三者在 dry-run 前后均不存在；没有旧 artifact 被覆盖。

## 15. Dry-run output

执行的是 M12A-C003 semantics 的单 config dry-run，run_attempt=2，真实 dataset root，未加 --execute：

~~~text
total=3 planned=3 completed=0 failed=0 running=0 retained=0 remaining=3
[PLANNED] M12A-C003 ... seed=0 ... seed_000__attempt_002
[PLANNED] M12A-C003 ... seed=1 ... seed_001__attempt_002
[PLANNED] M12A-C003 ... seed=2 ... seed_002__attempt_002
~~~

dry-run 结论：

- total = 3
- planned = 3
- completed = 0
- failed = 0
- running = 0
- remaining = 3
- formal_training_started = false
- selected configuration count = 1
- selected seeds = [0, 1, 2]
- other M12B conditions selected = 0

没有 dry-run M12B-C003 zero，也没有选择 K1、untied、FF、D9、其它 task 或其它 algorithm。

## 16. Formal result schema

本轮完成后，主比较应为新 M12B-R K4 shared normal 与已有 M12B K4 shared zero：

Δ_init,c = J(K4 shared normal, seed=c) − J(K4 shared zero, seed=c)

对 c=0,1,2 分别保留：

- final@1M
- normalized trapezoid AUC
- best success
- best step
- last3 mean

primary endpoint 为 final@1M。旧 M12A-C003 normal 只作为 replication comparison，不与新结果合并为 independent seed pool。

完成 K1/K4 2×2 后，再计算 Δ_K_normal、Δ_K_zero、Δ_init_K1、Δ_init_K4 和 I_K×init。本轮不提前解释这些量。

## 17. Suggested formal launch command

以下是唯一建议的正式启动命令。本轮没有执行它；必须由用户完成 Git 手动核验并 review 后手动启动。

~~~bash
cd /home/eai/Research/RLC-M12B-final
env OGBENCH_DATASET_DIR=/data/qijunrong/06-RL/offline-rl/data/raw_ogbench \
  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONPATH=. \
  /home/eai/Tools/miniforge3/envs/brain_nav/bin/python tools/sweep.py \
  --study experiments/M12A_frozen_critic_policy_extraction/study.yaml \
  --run-root /data/qijunrong/06-RL/offline-rl/exp/RLC/runs \
  --configs M12A-C003 \
  --run-attempt 2 \
  --dataset-root /data/qijunrong/06-RL/offline-rl/data/raw_ogbench \
  --gpus 0 \
  --train_steps 1000000 \
  --batch_size 1024 \
  --log_interval 5000 \
  --eval_interval 100000 \
  --eval_tasks all \
  --eval_episodes 20 \
  --eval_temperature 0 \
  --video_episodes 0 \
  --save_interval 100000 \
  --execute
~~~

省略 eval_gaussian 表示使用 parser 的 None，即 disabled。study protocol 会补齐 save_best_checkpoint=true 和 save_last_checkpoint=true。

## 18. Unresolved issues

1. 当前 exact HEAD、clean status 和 detached status 没有被本轮直接核验，因为所有 Git 操作由用户执行。
2. JAX CUDA plugin 当前初始化失败并回退 CPU；正式运行前必须由用户确认 GPU/JAX backend。
3. 新 formal artifact 的 filesystem identity 是 M12A-C003 attempt 2，这是在“不新增 M12B-C008、不修改 YAML、不修改 framework”的约束下复用 M12A-C003 executable semantics 的结果。科学身份是 M12B-R completion，而不是把旧 M12A attempt 1 误认为 M12B active run。

以上问题中，第 1 项和第 2 项足以阻止严格 GO。

## 19. GO / NO-GO

| criterion | status |
|---|---|
| exact source commit bb2644... | NOT VERIFIED |
| clean detached worktree | NOT VERIFIED |
| no source modification | PASS |
| target only SS-K4 shared normal | PASS |
| parameter_sharing=shared | PASS |
| state_init=normal_buffer | PASS |
| iterations=4 | PASS |
| same frozen critic per seed | PASS |
| C001 last@1M | PASS |
| parity with existing K4-zero except state_init | PASS |
| parameter count parity | PASS |
| executed computation/MAC parity | PASS |
| paired batch-stream semantics | PASS |
| production smoke | PASS |
| seeds exactly 0,1,2 | PASS |
| exactly 3 planned runs | PASS |
| no old artifact overwritten | PASS |
| no other condition rerun | PASS |
| no formal execution started | PASS |

严格结论：

**NO-GO。**

科学设计、实现路径、依赖、结构 parity、smoke 和 dry-run 均已满足；但在用户手动完成 Git 状态核验并确认 GPU/JAX runtime 之前，不应启动正式训练。

## 20. Explicit statement

本轮：

- 没有修改源码。
- 没有修改实验 YAML 或 study 文件。
- 没有执行 Git 写操作，也没有执行任何 Git 操作。
- 没有启动正式训练。
- 没有执行 --execute。
- 没有重跑 M12B K4 shared zero。
- 没有运行其它 M12B condition。
