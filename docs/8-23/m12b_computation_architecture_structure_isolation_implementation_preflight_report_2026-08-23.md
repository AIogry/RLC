# M12B Computation Architecture + Structure Isolation
# Implementation and Preflight Report

日期：2026-08-23  
项目：RLC 研究平台  
状态：架构重构与启动前预检通过；M12B 正式训练未启动

## 总结判定

本轮不是沿用上一轮 M12B abstraction，而是先审计 rollback 后 repository，再按 Primitive / Block / Topology / Parameter Sharing / Credit ontology 重新实现。

最终判定：

- 架构实现：GO；
- M12A 兼容性与 frozen-critic restore：GO；
- M12B 21-run dry-run：GO；
- M12B 是否已有新增科学结果：NO，尚未训练；
- Git 操作：未执行；
- M12A-D002：未实现。

本轮没有创建正式 M12B run artifact，也没有使用 --execute。

## 1. rollback 后初始 repository 状态

开始编码前进行了只读审计：

- impls/computation/topologies/untied_single_state.py：不存在；
- impls/computation/topologies/residual_stack.py：不存在；
- impls/experiment/m12b.py：不存在；
- 当前 topology 源文件只有 feedforward.py、single_state.py 和 two_state.py；
- impls/computation/blocks/ 原有 MLPMixerBlock，没有 residual MLP block；
- SingleState 仍是 shared-only，物理参数名为 update_module；
- ComputationSpec 原先只有 primitive/topology/credit/topology_kwargs，没有 block 字段；
- M12B 目录源码已被回退，但保留了上一轮运行产生的 stale __pycache__。stale bytecode 未被采用为实现，也不属于源码 topology。

因此，源码 repository 与 prompt 预期的 rollback 状态一致；本轮没有继续使用上一轮的两个错误 topology abstraction。

## 2. 原 computation architecture 审计

审计覆盖 interfaces.py、factory.py、accounting.py、primitives、blocks、topologies、networks/common.py、CRL agents、M12A Study/configs、experiment management、sweep 和既有 tests。

审计结论：原 framework 已有足够小的 ComputationCore / ComputationSpec / factory 边界，不需要引入任意 DAG、复杂 registry 或大规模 DSL。最小合理改动是增加可复用 block schema，并把 parameter sharing 加到 SingleState。

## 3. Primitive 定义

Primitive 描述一次局部 neural transformation 的基本函数族，例如当前 OGBench-compatible MLP。它负责 Dense、activation、LayerNorm 等局部映射，但不负责 recurrent schedule、state lifecycle、iteration count 或 parameter-sharing schedule。

当前 MLP 的原始排列、初始化和默认行为保持不变。

## 4. Block 定义

Block 描述 stateless/local computation unit 的内部 wiring 和 residual composition。新增：

    impls/computation/blocks/residual_mlp.py

ResidualMLPBlock 实现：

    y = x + F(x)

ResidualMLPStack 是由 input projection 和多个独立 residual blocks 组成的 stateless body。Block 不定义 decision-local recurrent state、cross-step state 或 state lifecycle。

## 5. Topology 定义

Topology 只描述 execution/state graph：

- feedforward：无 computation state，body 执行一次；
- single_state：一个 decision-local state，定义 state 初始化、输入注入和迭代 schedule；
- two_state：两个 decision-local state，保留原 H/L schedule 和 credit 语义。

本轮没有新增 topology class。FeedForward 仍然只是接收一个 stateless body、执行一次并返回 ComputationOutput；它不知道 body 是 plain MLP 还是 residual MLP。

## 6. Parameter Sharing 定义

Parameter sharing 描述同一 topology 的多个 execution step 是否复用相同 update 参数：

    shared: z_{k+1} = F_theta(z_k + x_hidden)
    untied: z_{k+1} = F_theta_k(z_k + x_hidden)

二者完全共享 state graph、input injection、K、state lifecycle、update depth、activation、LayerNorm、residual flag 和 credit。差异仅为 parameter tying schedule。

## 7. Credit 定义

Credit 继续表示 gradient 如何穿过 computation graph：direct、full_bptt、one_step。本轮没有改变 credit implementation 或语义。

## 8. 为什么 untied 不应是 topology

shared 和 untied 拥有同一个 SingleState state graph：

    x_hidden = P(x_raw)
    z_0 = z_init
    z_{k+1} = F(z_k + x_hidden)

只有 F 是否被不同 execution step 复用不同。因此把 untied 命名为新 topology 会把参数化方式错误地混成状态图。本轮将其实现为：

    SingleState(parameter_sharing='shared' | 'untied')

没有创建 untied_single_state.py。

## 9. 为什么 residual 不应是 topology

Residual FF 没有 recurrent state、没有 z_init、没有 repeated input injection、没有 cross-decision state；它只是：

    h_0 = P(x)
    h_{k+1} = h_k + F_k(h_k)

因此 residual 是 block/body wiring，而不是 execution/state graph。没有创建 residual topology。

## 10. ComputationSpec 选择

本轮选择方案 A，为 ComputationSpec 增加一等字段：

    primitive: str = 'mlp'
    topology: str = 'feedforward'
    credit: str = 'direct'
    topology_kwargs: Mapping = {}
    block: str = 'plain'
    parameter_sharing: str = 'shared'
    block_kwargs: Mapping = {}

新增字段放在原 positional 字段之后，保留历史调用：

    ComputationSpec('mlp', 'feedforward', 'direct')

旧配置没有 block 时默认为 plain，没有 parameter_sharing 时 SingleState 默认为 shared。这样 scientific ontology 清楚，同时避免为 M12B 增加 study-specific factory 分支。

Residual 由 factory 组合为：

    ResidualMLPStack body -> FeedForward topology -> ComputationCore

这一设计可自然支持未来的 SwiGLU primitive、Mixer block 和其他 stateless block；本轮没有实现这些扩展。

## 11. SingleState shared/untied 最终 API

SingleState 新增：

    parameter_sharing: str = 'shared'

shared 模式继续且只创建：

    topology/update_module

K 次执行复用该 subtree。untied 模式创建：

    topology/update_modules_0
    topology/update_modules_1
    ...
    topology/update_modules_{K-1}

每个模块独立执行一次。默认 shared 保持旧配置行为。

## 12. backward compatibility evidence

architecture tests 已验证旧式 config 与显式 parameter_sharing=shared：

- params tree fingerprint exact same；
- buffers tree fingerprint exact same；
- 相同 RNG/input 下 forward output exact same；
- shared 参数树仍含 update_module，不改名为 update_modules_0。

另外，测试发现新字段若插入旧 positional 参数会破坏兼容性；已将新字段移到原字段之后。修复后 95 项指定回归全部通过。

## 13. M12A checkpoint compatibility evidence

使用当前代码构造 M12A-C003 SingleState K4 actor，并 restore 真实 artifact：

    M12A-C003/.../seed_000__attempt_001/checkpoints/last/params_1000000.pkl

restore 成功，当前网络 step 为 1，actor topology 参数 keys 保持：

    ['input_mapping', 'update_module']

这证明现有 M12A SS-K4 shared checkpoint 不需要迁移到新参数名。

## 14. residual block/body implementation

ResidualMLPBlock 实现 y=x+F(x)，其中 F 是 configurable MLP。ResidualMLPStack 实现：

    h = input_mapping(x)
    for block in residual_blocks:
        h = block(h)

它不接受 recurrent state，不创建 buffer，也没有 z_init。M12B 使用 width=512、4 blocks、每 block 2 Dense、GELU、无 LayerNorm。

## 15. residual FeedForward composition

B009 的 resolved computation spec 是：

    topology: feedforward
    block: residual
    block_kwargs:
      state_dim: 512
      blocks: 4
      block_depth: 2
      layer_norm: false
      block_activate_final: true

因此 B009 的 residual 结构明确属于 FeedForward，而不是新的 topology。

## 16. final M12B 9-condition matrix

| 条件 | 来源/配置 | topology | block | state init | sharing | hidden Dense execution |
|---|---|---|---|---|---|---:|
| B001 | M12A-C002 attempt 1 | feedforward | plain | — | — | 3 |
| B002 | M12B-C001 | single_state | plain | normal | shared | 3 |
| B003 | M12B-C002 | single_state | plain | zero | shared | 3 |
| B004 | M12A-C003 attempt 1 | single_state | plain | normal | shared | 9 |
| B005 | M12B-C003 | single_state | plain | zero | shared | 9 |
| B006 | M12B-C004 | single_state | plain | normal | untied | 9 |
| B007 | M12B-C005 | single_state | plain | zero | untied | 9 |
| B008 | M12B-C006 | feedforward | plain | — | — | 9 |
| B009 | M12B-C007 | feedforward | residual | — | — | 9 |

B001/B004 是 external anchors；新增 active conditions 是 B002、B003、B005、B006、B007、B008、B009。

## 17. 两个 external anchors

预检逐 seed 验证：

- B001：M12A-C002，canonical FF，attempt 1；
- B004：M12A-C003，SingleState K4 shared normal，attempt 1；
- 两者均 completed；
- last@1M 存在；
- environment 为 antmaze-large-navigate-v0；
- CRL + DDPG+BC；
- Stage2 protocol 一致；
- 同一 seed 的 M12A-C001 critic dependency；
- critic checkpoint SHA 和 module fingerprint 一致。

实际 valid attempt 是已有 artifact 中的 1，不是重新猜测或重跑得到的。

## 18. 七个 active configs

| config | 条件 | 结构 |
|---|---|---|
| M12B-C001 | B002 | SingleState K1 shared normal |
| M12B-C002 | B003 | SingleState K1 shared zero |
| M12B-C003 | B005 | SingleState K4 shared zero |
| M12B-C004 | B006 | SingleState K4 untied normal |
| M12B-C005 | B007 | SingleState K4 untied zero |
| M12B-C006 | B008 | FeedForward + original MLP，9 hidden Dense |
| M12B-C007 | B009 | FeedForward + ResidualMLPStack，4×2 block |

## 19. exact 21-run count

    7 active conditions × 3 seeds = 21 new formal runs
    2 external anchors × 3 seeds = 6 reused anchor rows
    9 conceptual conditions × 3 seeds = 27 result-table rows

## 20. trainable params、buffers 和 accounting

Dense bias 计入参数；hidden depth 不计共同 action mean readout；1 MAC = 2 FLOPs。实际 parameter tree accounting 得到：

| 条件 | actor body params | actor total params | buffers | unique Dense | executed Dense | body MAC | full actor MAC | full actor FLOPs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B001 | 555,520 | 559,624 | 0 | 3 | 3 | 553,984 | 558,080 | 1,116,160 |
| B002/B003 | 555,520 | 559,624 | 512 | 3 | 3 | 553,984 | 558,080 | 1,116,160 |
| B004/B005 | 555,520 | 559,624 | 512 | 3 | 9 | 2,126,848 | 2,130,944 | 4,261,888 |
| B006/B007 | 2,131,456 | 2,135,560 | 512 | 9 | 9 | 2,126,848 | 2,130,944 | 4,261,888 |
| B008 | 2,131,456 | 2,135,560 | 0 | 9 | 9 | 2,126,848 | 2,130,944 | 4,261,888 |
| B009 | 2,131,456 | 2,135,560 | 0 | 9 | 9 | 2,126,848 | 2,130,944 | 4,261,888 |

B004/B005 的物理 Dense 只有 3 个，但 shared update 被执行 4 次，因此 executed depth 是 1+4×2=9。B006/B007/B009 具有 9 个独立物理 Dense；B008 直接使用 9 hidden Dense MLP。B006/B007/B008/B009 的 actor body 参数和 MAC exact match。

## 21. initialization pairing evidence

preflight 在 seeds 0/1/2 上比较：

- B002 vs B003：params exact same，仅 z_init 不同；zero buffer 精确为 0；
- B004 vs B005：params exact same，仅 z_init 不同；
- B006 vs B007：params exact same，仅 z_init 不同；
- B002 vs B004：shared normal 下 params 和 buffer exact same；
- B003 vs B005：shared zero 下 params 和 buffer exact same；
- shared K4 只有一个 update module；
- untied K4 有四个独立 update module；
- untied 的四个 update module 均获得有限梯度。

## 22. batch-stream pairing evidence

所有 active config 使用相同 M12A Stage2 dataset protocol 和同一 seed-derived stream。preflight 对每个 config 构造相同 synthetic GCDataset，并比较前 10 个 sampled batch 的 stable hash，结果一致，覆盖 observation/action/actor-goal/value-goal 等 batch 字段。

架构不会修改 dataset RNG 或 evaluation seed schedule。

## 23. frozen critic validation

每个 active config 的 dependency 都是 generic cross-study declaration：

    source study: M12A_frozen_critic_policy_extraction
    source config: M12A-C001
    same seed
    attempt: 0
    module: critic
    checkpoint: last@1M

impls/experiment/management.py 新增的是通用 source_study_path/source_study_id 解析和 ignored_agent_fields 声明能力，没有 M12A/M12B runtime conditional。C006 只声明 actor-side actor_hidden_dims 可忽略，critic 与训练语义仍严格比较。

## 24. tests 与结果

本轮指定回归共 95 项：

    Ran 95 tests in 45.399s
    OK

覆盖：

- computation foundation；
- MLP parity；
- SingleState；
- TwoState；
- checkpoint lifecycle；
- run-attempt；
- experiment management；
- sweep；
- M12A frozen critic；
- computation provenance；
- run-study launcher；
- 新增 M12B architecture tests。

另有 7-config runtime smoke：每个 config 的 CRLPolicyExtractorAgent 初始化、legacy/generic accounting consistency 均为 pass。

JAX 环境输出 CUDA_ERROR_NO_DEVICE 后使用 CPU fallback；这是环境 warning，不是测试失败，也没有触发正式训练。

## 25. M12A regression

M12A 相关验证通过：

- C001 critic pretrain 配置和 dependency resolution；
- C002 canonical FF policy extraction；
- C003 SingleState K4 shared normal construction；
- frozen critic restore；
- source checkpoint last@1M 选择；
- run_attempt behavior；
- 真实 C003 actor checkpoint restore。

没有改变 TwoState，也没有改变 M12A actor 的旧 parameter tree names。

## 26. dry-run output

M12B 专项 preflight：

    planned_runs: 21
    external_anchor_runs: 6
    new_formal_runs: 21
    formal_training_started: false

统一 tools/sweep.py --dry-run：

    total=21 planned=21 completed=0 failed=0 running=0 retained=0 remaining=21

B001/B004 没有进入新增 training plan；没有 M12B formal path 被创建。

## 27. 建议的正式启动命令

以下命令只提供给用户手动 review/执行。本轮没有执行 --execute：

~~~bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
OGBENCH_DATASET_DIR=/data/qijunrong/06-RL/offline-rl/data/raw_ogbench \
PYTHONPATH=. \
/home/eai/Tools/miniforge3/envs/brain_nav/bin/python tools/sweep.py \
  --study experiments/M12B_actor_computation_structure_isolation/study.yaml \
  --configs M12B-C001,M12B-C002,M12B-C003,M12B-C004,M12B-C005,M12B-C006,M12B-C007 \
  --run-attempt 0 \
  --gpus 0,1 \
  --run-root /data/qijunrong/06-RL/offline-rl/exp/RLC/runs \
  --dataset-root /data/qijunrong/06-RL/offline-rl/data/raw_ogbench \
  --execute \
  --train_steps=1000000 \
  --batch_size=1024 \
  --log_interval=5000 \
  --eval_interval=100000 \
  --eval_tasks=all \
  --eval_episodes=20 \
  --save_interval=100000 \
  --eval_temperature=0.0
~~~

该命令仅包含 7 个 active M12B config，不会重新训练 B001/B004 anchor。

## 28. unresolved architectural limitations

当前仍明确 out of scope：

- primitive injection 还不是完全通用的 dependency-injection DSL；
- 尚未实现 SwiGLU、attention-like primitive 或 Mixer 迁移；
- 尚未实现 grouped sharing、部分 tying 或 arbitrary DAG；
- residual block schema 当前服务于 stateless FeedForward body；
- M12A-D002 的 trace、intermediate state、Q_k、gradient alignment 未实现；
- 未新增跨任务、跨算法或其他 recurrent family。

这些限制是有意控制 change surface 的结果，不影响本轮 M12A compatibility 和 M12B intervention。

## 29. 最终 GO / NO-GO criteria

| 条目 | 判定 |
|---|---|
| Primitive / Block / Topology 边界明确 | GO |
| 没有错误的两个新 topology | GO |
| shared/untied 是 SingleState property | GO |
| 旧 SingleState 默认 shared | GO |
| M12A checkpoint compatibility | GO |
| residual 位于 block/body 层 | GO |
| residual 使用 FeedForward | GO |
| Deep FF 使用 FeedForward + existing MLP | GO |
| 无 M12B study-specific runtime module/branch | GO |
| 9 conceptual conditions | GO |
| 2 valid anchors | GO |
| 7 active configs | GO |
| 21 planned runs | GO |
| frozen critics last@1M | GO |
| initialization pairing | GO |
| paired data streams | GO |
| parameter/MAC accounting | GO |
| M12A regression | GO |
| formal dry-run | GO |
| formal training started | NO，按要求未启动 |
| Git operation performed | NO，按要求未执行 |
| M12A-D002 implemented | NO，明确未实现 |

## 30. 明确声明

本轮：

- 没有执行 Git 操作；
- 没有启动 M12B formal training；
- 没有实现 M12A-D002；
- 没有修改、删除或重跑 M12A formal runs；
- 只完成了架构重构、测试、preflight、dry-run 和聚合骨架。

M12B 现在达到“可由用户手动启动正式实验”的状态，但尚未产生可用于科学解释的新增性能结果。

