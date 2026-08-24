# M12B-R — K4 Shared Initialization Replication

## Preflight Report

日期：2026-08-24  
范围：M12B K4 shared initialization 的窄范围 confirmatory rerun  
最终判定：**NO-GO（未启动正式训练）**

> **重要勘误（以用户于 2026-08-24 的事实更正为准）：** M12B 实验实际没有运行 SS K4 normal，M12A 实验实际没有运行 SS K4 zero；M12A 的 SS K4 normal checkpoint 也没有被用于 M12B。因此，本报告中把 M12A-C003 normal 作为 M12B normal 对照、并据此规划 M12B-R 的表述，不能作为已经完成的 M12B 对照证据。请优先阅读同目录下的 m12b_r_correction_addendum_2026-08-24.md；该勘误对原报告的实验解释和后续设计具有优先级。

## 1. 执行边界

本次严格按照 docs/8-24/prompt for codex1.md 执行了 provenance audit、配置审计、依赖审计、单元/集成测试、真实数据 production-path smoke test 和 dry-run 规划。

遵循用户此前明确要求：

- 没有执行任何 Git 操作；包括没有执行 git rev-parse、git status、git branch 等命令。
- 没有修改源代码、实验 YAML、study 文件或训练逻辑。
- 没有使用 --execute，没有创建正式 run 目录，没有启动正式训练。
- 不把 dry-run 或 smoke test 计入正式实验结果。

由于用户要求所有 Git 操作由用户手动完成，本报告不能声称当前工作树的实时 HEAD、clean/dirty 状态或 detached 状态已经被本次审计直接核验。按照 prompt 的严格规则，这一点足以使最终判定为 NO-GO。

## 2. 研究问题与重跑范围

M12B-R 只验证一个因素：K4 SingleState 中共享 update module 的初始化方式。

固定内容：

- task：antmaze-large-navigate-v0
- algorithm：CRL
- actor：policy extractor
- critic：DDPG + BC，冻结为同 seed 的 M12A-C001 attempt 0 的 last@1M critic
- actor width：512
- training steps：1M
- batch size：1024
- original learning rate：3e-4
- logging interval：5000
- evaluation：每 100k steps、全部 evaluation episodes、temperature 0、无 Gaussian noise、无 video
- checkpoint：每 100k、best、last
- formal 条件：2 个条件 × seed 0/1/2，共 6 个正式 run

只允许的两个条件：

| 条件 | 配置 | 唯一变量 |
|---|---|---|
| Normal buffer | M12A-C003 | K4 shared normal_buffer |
| Zero buffer | M12B-C003 | K4 shared zero_buffer |

没有增加第三个 condition，也没有改变 K4、网络宽度、训练步数、数据集或评估 protocol。

## 3. Source provenance 与 Git 状态

Prompt 要求正式运行使用 source commit：

bb2644ccb23ee77a0c08e8b9cded85a57716df67

历史正式 M12B metadata 显示，M12B formal runs 使用了该 commit 且 recorded dirty state 为 false。旧 normal reference 的历史 metadata 使用 commit：

b3fde3f91d89169c02c7604ace80d65bdf8ced25

但是，本次没有执行任何 Git 命令，因此以下实时事实未能独立确认：

- 当前 /home/eai/Research/RLC-M12B-final 的 HEAD 是否正好为 bb2644...
- 当前 worktree 是否 clean
- 当前是否 detached HEAD

这不是推测性地把历史 metadata 当作当前状态；本报告明确区分了“历史 formal metadata”与“当前实时 Git 状态”。正式启动前必须由用户手动完成上述核验。

## 4. 旧结果与 attempt 选择

### 4.1 Normal buffer reference

旧 normal 运行目录：

/data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M12A/M12A-C003__policy_extraction_single_state_k4_actor

审计结果：

- attempt 0：失败/无有效 eval，不作为 reference。
- attempt 1：三个 seed 均完成，作为旧 normal reference。
- seed 0：last@1M = 0.93，AUC = 0.832222，best = 0.93 @ 1M。
- seed 1：last@1M = 0.79，AUC = 0.748889，best = 0.85 @ 700k。
- seed 2：last@1M = 0.87，AUC = 0.798333，best = 0.87 @ 900k。

因此 normal 条件的下一个未使用 attempt 为 attempt 2。

### 4.2 Zero buffer reference

旧 zero 运行目录：

/data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M12B/M12B-C003__ss_k4_shared_zero

审计结果：

- attempt 0：三个 seed 均完成，作为旧 zero reference。
- seed 0：last@1M = 0.72，AUC = 0.727778，best = 0.80 @ 800k。
- seed 1：last@1M = 0.69，AUC = 0.582222，best = 0.71 @ 900k。
- seed 2：last@1M = 0.80，AUC = 0.712778，best = 0.81 @ 600k。

因此 zero 条件的下一个未使用 attempt 为 attempt 1。

候选新目录均已做存在性检查，当前不存在；没有覆盖已有结果。

## 5. 配置语义审计

两份 YAML 的有效语义一致，唯一差异为 state_init：

- M12A-C003.yaml：normal_buffer
- M12B-C003.yaml：zero_buffer

共同配置：

- primitive：mlp
- block：plain
- topology：single_state
- credit：direct
- iterations = 4
- residual = false
- input：z_plus_x
- state_dim = 512
- update_depth = 2
- layer_norm = false
- update_activate_final = true
- parameter sharing：shared

源码审计确认 SingleState 的默认 parameter sharing 也是 shared；shared 模式使用单一 update_module。zero buffer 是精确零向量，不是小噪声或随机近零初始化。normal buffer 由 buffer RNG 生成。

## 6. Critic dependency 审计

正式重跑必须使用 M12A-C001 attempt 0、同 seed、last@1M critic；不能使用 best checkpoint，也不能跨 seed。

已核验的 checkpoint SHA 与 module fingerprint：

| seed | critic checkpoint SHA | critic module fingerprint |
|---:|---|---|
| 0 | f801f7521aedc70a0ed182a2a2f2d7765d9faa0e6b7ac623f98ad284926006d5 | 35bfa7630a317e40bae4fbc4f529635c4655f8946975af1e87388d4490bb85b7 |
| 1 | b89f45b1e61436b0ee469471b51f40b517ac6130339951c9086c2a5912281c98 | 1ac0eda0a97b315e1f8a6e48d0b29c70267de85bb54b81c4ba9718f0bbcdc36e |
| 2 | 64def1c398dd59a30b533c1cfa704937d389ca5a5088ba45db66b3f7a8005ab2 | 0c0579ae1f6012e89b2c5ec465dd976baabee0d7f882efbddc4cbe4330423d18 |

production-path smoke 中，每个 seed 都按自己的 M12A-C001 last@1M dependency 恢复；恢复后 normal/zero 两条路径的 critic fingerprint 相同，3 次 actor update 后 critic 参数保持不变。

## 7. Paired initialization 与数据流证据

对每个 seed，使用真实 antmaze-large-navigate-v0.npz、真实 CRLPolicyExtractorAgent 和正式配置走 production path。

结果：

- normal 与 zero 的 actor 初始参数完全相同。
- 两条路径的前 10 个训练 batch 完全相同。
- critic 初始参数完全相同。
- 训练后的 critic 仍保持不变。
- 参数树的唯一差异路径为：

modules_actor.actor_net.topology.z_init

- normal buffer 的 z_init 非零。
- zero buffer 的 z_init 精确为零。
- 每个 seed 的 3 次 actor update 均 finite，actor 参数发生变化，critic 参数不变。

production-path smoke 结论：

ALL_PRODUCTION_SMOKE_PASS=True

这说明本次设计能够在实现层面实现 paired comparison；它不等价于正式实验结果，也不能替代 1M-step formal training。

## 8. 测试结果

可运行的 unittest 结果：

- tests.computation.test_m12b_architecture：5/5 通过。
- tests.integration.test_m12a_frozen_critic.M12AStudyAndDependencyTest：9/9 通过。

合并运行相关 computation/integration 测试：18/20 通过，2 个已有机制测试失败：

1. test_critic_only_loss_is_canonical_critic_loss：出现约 5.96e-08 的 exact array mismatch。
2. test_joint_and_critic_only_critic_trajectories_match_for_100_steps：critic fingerprint mismatch，涉及 joint 与 critic-only 的 100-step trajectory。

这两个失败不属于本次 M12B-R 的 K4 actor paired-initialization 逻辑，且本次没有修改代码，因此不能隐瞒；但在严格科学复现实验的标准下，仍应作为正式启动前的 unresolved regression 风险记录。环境中未安装 pytest，因此使用 unittest 直接运行等价测试集合。

## 9. 正式 run 目录与 attempt 规划

Normal buffer，M12A-C003，attempt 2：

~~~text
/data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M12A/M12A-C003__policy_extraction_single_state_k4_actor/antmaze-large-navigate-v0/seed_000__attempt_002
/data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M12A/M12A-C003__policy_extraction_single_state_k4_actor/antmaze-large-navigate-v0/seed_001__attempt_002
/data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M12A/M12A-C003__policy_extraction_single_state_k4_actor/antmaze-large-navigate-v0/seed_002__attempt_002
~~~

Zero buffer，M12B-C003，attempt 1：

~~~text
/data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M12B/M12B-C003__ss_k4_shared_zero/antmaze-large-navigate-v0/seed_000__attempt_001
/data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M12B/M12B-C003__ss_k4_shared_zero/antmaze-large-navigate-v0/seed_001__attempt_001
/data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M12B/M12B-C003__ss_k4_shared_zero/antmaze-large-navigate-v0/seed_002__attempt_001
~~~

没有发现这些目标目录已存在；dry-run 规划为：

- total = 6
- planned = 6
- completed = 0
- failed = 0
- running = 0
- remaining = 6
- 其他 M12B 条件：0

## 10. Dry-run 与 formal launch 状态

两个 study 分别运行 dry-run，均成功解析出 3 个 seed，合计 6 个正式任务。首次直接调用 sweep 时出现 No module named impls，确认是调用时未设置 PYTHONPATH=., 随后以仓库根目录和正确 PYTHONPATH 重跑成功；这不是训练逻辑或配置错误。

prompt 要求的候选正式命令已审计，但没有执行。在用户完成 Git 状态核验、解决或明确接受测试回归风险、并确认运行环境后，才可由用户手动决定是否启动：

~~~bash
cd /home/eai/Research/RLC-M12B-final
env OGBENCH_DATASET_DIR=/data/qijunrong/06-RL/offline-rl/data/raw_ogbench \
  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONPATH=. \
  /home/eai/Tools/miniforge3/envs/brain_nav/bin/python tools/sweep.py \
  --study experiments/M12A_frozen_critic_policy_extraction/study.yaml \
  --run-root /data/qijunrong/06-RL/offline-rl/exp/RLC/runs \
  --configs M12A-C003 --run-attempt 2 \
  --dataset-root /data/qijunrong/06-RL/offline-rl/data/raw_ogbench \
  --gpus 0 --execute

env OGBENCH_DATASET_DIR=/data/qijunrong/06-RL/offline-rl/data/raw_ogbench \
  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONPATH=. \
  /home/eai/Tools/miniforge3/envs/brain_nav/bin/python tools/sweep.py \
  --study experiments/M12B_actor_computation_structure_isolation/study.yaml \
  --run-root /data/qijunrong/06-RL/offline-rl/exp/RLC/runs \
  --configs M12B-C003 --run-attempt 1 \
  --dataset-root /data/qijunrong/06-RL/offline-rl/data/raw_ogbench \
  --gpus 0 --execute
~~~

上面的命令仅作为 prompt 要求的 launch design 记录；本次没有运行。

## 11. Strict GO/NO-GO 判定

### 已通过

- 研究问题被限制为一个初始化因素。
- 只有 2 条 condition、3 个 seed，共 6 个 run。
- 旧结果与 attempt 选择已审计，未覆盖旧目录。
- 两份配置只在 state_init 上不同。
- critic 使用同 seed、last@1M、精确 SHA/fingerprint。
- actor 初始参数、前 10 个 batch、critic 状态均已配对验证。
- production-path smoke 对 seed 0/1/2 全部通过。
- dry-run 正确规划 6 个任务，未启动 formal run。

### 阻止 GO 的事项

1. 由于用户明确禁止本次执行任何 Git 操作，当前 exact HEAD、clean status、detached status 未被本次直接核验，无法严格证明当前 source 正是目标 commit。
2. 相关测试集合存在 2 个未解决的机制回归失败。
3. 当前 smoke 环境的 JAX CUDA plugin 初始化失败并回退 CPU；因此 smoke 不能证明 formal GPU runtime 已验证。nvidia-smi 能看到 GPU，但这不等价于当前 JAX 进程已使用 GPU。

因此最终结论是：

**NO-GO：本次已完成设计和科学 preflight，但未授权、未执行、也不应在当前证据状态下自动启动 M12B-R 正式训练。**

## 12. 用户手动接管项

在决定正式启动前，用户需要自行完成并记录：

- 验证 /home/eai/Research/RLC-M12B-final 当前 HEAD 为 bb2644ccb23ee77a0c08e8b9cded85a57716df67。
- 验证 worktree clean 且状态符合 prompt 要求。
- 决定如何处理上述 2 个已有机制测试失败。
- 确认正式训练确实使用可用的 GPU/JAX backend。
- 由用户手动执行正式 launch；本 agent 不执行 Git 操作，也不执行 --execute。

本报告生成过程中没有对源代码和实验配置做任何修改。
