# M12A Frozen-Critic Policy Extraction
# Implementation and Preflight Report

> 本报告对应 `Research/docs/8-22/prompt for codex2.md`。本轮只完成 implementation、configuration、generic dependency support、tests、smoke、preflight 和 dry-run；没有启动正式训练，也没有执行任何 Git 操作。

## 结论先行

M12A 的代码与实验设计已经完成，关键机制不变量通过 CPU 小规模测试。当前建议为：

- implementation/preflight：GO；
- Stage 1 formal training：等待用户 review 后由用户手动启动；
- Stage 2：当前 NO-GO，原因是所检查的 run root 尚不存在 3 个 completed 的 C001 `last@1M` source Runs；不会自动补跑 source。

正式实验的总设计是 3 个 configurations × 3 个 seeds = 9 个 formal Runs。Stage 1 完成并逐个验证 source artifact 后，Stage 2 才允许规划/启动 6 个 actor extraction Runs。

## 1. Exact scientific question

M11A 中 SingleState recurrent actor 的优势可能来自 actor 本身的计算机制，也可能来自 actor–critic co-adaptation 或 critic non-stationarity。M12A 固定同一个已学习的 CRL critic (Q_{\phi_c}^*\)，分别训练 canonical FF actor 与 SingleState K4 actor，并比较：

\[
\Delta_c = J(\pi_{SS}^{(c)}) - J(\pi_{FF}^{(c)}).
\]

若在相同 frozen critic 下仍有正的 paired gap，才支持 actor-side computational advantage 在固定 evaluator 下持续存在；若 gap 消失，则说明 M11A 优势实质上依赖 joint co-adaptation 和/或 critic non-stationarity。

## 2. Why only AntMaze-Large initially

M12A 是 mechanism isolation experiment，不是 benchmark sweep。第一轮只使用 `antmaze-large-navigate-v0`，避免把 task diversity 与 frozen-critic mechanism 混在一起；M11B 已承担 cross-task generalization。`antmaze-giant-navigate-v0` 已在 Study 中声明为 prespecified confirmatory extension，但没有 active Giant configuration、没有启动 Giant、也没有扩大主表。

## 3. Why seeds = 3

正式 seeds 固定为 `[0, 1, 2]`，每个 critic seed 同时作为 paired actor initialization seed；dataset/evaluation 使用由 seed 派生的独立 deterministic streams。M12A 的目标是严格配对机制比较而不是建立 benchmark 统计主张，因此保留 3 个 critic seeds，并明确不把 20 个 evaluation episodes 当成独立 statistical seeds。

## 4. Why the critic source is last@1M

Stage 1 没有 trained actor policy，不存在有科学依据的 environment success selector。因此 primary source 固定为：

```text
checkpoint_role = last
checkpoint_step = 1_000_000
best_selection = disabled
```

不使用 best、validation contrastive loss、ranking、margin、categorical accuracy 或其它 proxy 选择 critic。这样控制的是固定 critic training budget，而不是每个 seed 的选择器行为。

## 5. Exact three configurations

配置文件位于 [`experiments/M12A_frozen_critic_policy_extraction/`](../../experiments/M12A_frozen_critic_policy_extraction/)。

| Configuration | stage | actor | critic/source |
|---|---|---|---|
| `M12A-C001` | `critic_pretrain` | 不训练 | canonical FF CRL critic-only contrastive training |
| `M12A-C002` | `policy_extraction` | canonical FF CRL actor | same-seed C001 `last@1M` frozen critic |
| `M12A-C003` | `policy_extraction` | SingleState K4 actor | same-seed C001 `last@1M` frozen critic |

C002/C003 的 dependency 均为 `source_config_id: M12A-C001`、`seed_policy: same_seed`、`source_run_attempt: 0`、`checkpoint_role: last`、`checkpoint_step: 1000000`、`module: critic`。

## 6. Exact nine planned formal Runs

Run ontology 遵守 `Study -> Configuration -> Run`：seed 属于 Run，不属于 Configuration。计划矩阵为：

- Stage 1：C001 × seeds `{0,1,2}` = 3 Runs；
- Stage 2：C002 × `{0,1,2}` + C003 × `{0,1,2}` = 6 Runs；
- 合计：9 formal Runs。

C002 seed `c` 与 C003 seed `c` 必须引用完全相同的 C001 seed `c` critic；不能跨 seed、跨 attempt 或 fallback 到其它 checkpoint。

## 7. Study files and architecture boundary

已新增：

- [`study.yaml`](../../experiments/M12A_frozen_critic_policy_extraction/study.yaml)
- [`README.md`](../../experiments/M12A_frozen_critic_policy_extraction/README.md)
- [`M12A-C001.yaml`](../../experiments/M12A_frozen_critic_policy_extraction/configs/M12A-C001.yaml)
- [`M12A-C002.yaml`](../../experiments/M12A_frozen_critic_policy_extraction/configs/M12A-C002.yaml)
- [`M12A-C003.yaml`](../../experiments/M12A_frozen_critic_policy_extraction/configs/M12A-C003.yaml)
- [`preflight.py`](../../experiments/M12A_frozen_critic_policy_extraction/preflight.py)

已确认没有新增 `impls/experiment/m12a.py`，也没有在 `impls/main.py` 或 `impls/experiment/management.py` 中加入 M12A-specific branch；代码只使用 generic `training_mode`、`runtime_variant` 和 dependency semantics。`m11b.py` 没有被本轮 M12A 实现重构或改写。

## 8. Generic experiment-framework changes

[`impls/experiment/management.py`](../../impls/experiment/management.py) 增加了通用的 `resolve_run_dependency()` 与 `validate_source_run_dependency()`，支持 source Run 定位、same-seed/explicit-seed policy、completed 状态检查、semantic checkpoint 解析、SHA 校验、环境/算法/compute compatibility 检查及 module fingerprint 记录。

[`impls/utils/checkpointing.py`](../../impls/utils/checkpointing.py) 增加 deterministic `tree_fingerprint()`、module key resolution 和 checkpoint module fingerprint。[`impls/utils/flax_utils.py`](../../impls/utils/flax_utils.py) 增加从完整 checkpoint 提取单一 module 的 restore helper，保留 target 的 fresh optimizer state。

[`tools/sweep.py`](../../tools/sweep.py) 现在在 dependency preflight 失败时输出 `preflight: NO-GO` 并返回非零状态；它不会自动启动缺少 source 的 runs。`scripts/run_study.sh` 仅扩展了通用 save-best/save-last 参数转发；本轮没有以 `--execute` 调用它。

## 9. Critic-only API and Stage-1 objective

[`impls/agents/crl.py`](../../impls/agents/crl.py) 增加了 additive API：

```python
critic_only_loss(batch, grad_params)
critic_only_update(batch)
```

它直接调用 canonical `contrastive_loss(batch, grad_params, module_name='critic')`。canonical `total_loss()`、`actor_loss()`、`update()` 的默认 joint training semantics 没有被替换为 M12A 专用逻辑；训练循环只通过 generic `training_mode: critic_only` dispatch 到新 API。Stage 1 的 actor 可以被实例化以保持 checkpoint 结构兼容，但不会获得更新。

## 10. Critic-only vs joint parity result

`tests/integration/test_m12a_frozen_critic.py` 的 100-step parity test 使用相同 initialization、相同 batch stream、相同 seed 和同一 optimizer：

- joint CRL 与 critic-only 的 critic subtree fingerprint 完全相同；
- fixed probe batch 的 critic Q 输出逐数组 exact equal；
- 因而 observed `max_abs_param_diff = 0`、fixed-Q output diff = 0（bitwise fingerprint equality）。

这确认了 DDPG+BC actor branch 对 critic 参数的 stop-gradient 与 critic-only objective 在该 invariant 下没有引入 critic trajectory 差异。

## 11. Stage-1 actor-freeze result

100-step critic-only test 中 actor subtree 在更新前后 deterministic fingerprint 完全相同；这是 exact tree equality，不是 tolerance-based “接近不变”。critic-only loss parity test 也确认新增 API 没有改变 canonical critic component。

## 12. Stage-1 evaluation semantics

[`impls/main.py`](../../impls/main.py) 的 `--eval_tasks` 现在严格区分：

- `none`：evaluation disabled，直接返回空 metrics，不调用 environment policy evaluation；
- `all`：评估全部 task；
- 正整数：评估指定数量 task。

Stage 1 protocol 在 Study 中固定为 `eval_tasks: none`、`save_best_checkpoint: false`、`save_last_checkpoint: true`、`save_interval: 100000`。这不是把 `none` 和 `all` 都映射为 `None`。

## 13. CRLPolicyExtractorAgent design

[`impls/agents/crl_policy_extractor.py`](../../impls/agents/crl_policy_extractor.py) 提供窄范围 runtime variant `CRLPolicyExtractorAgent`：

- 复用 `CRLAgent`、`GCActor`、canonical CRL critic 和 action readout；
- 只接受 `actor_loss=ddpgbc`；
- actor optimizer 使用 canonical Adam learning rate；
- critic 参数使用 `optax.set_to_zero()` 的 frozen transform；
- Stage 2 optimizer state 由 target agent fresh initialize，不读取 Stage 1 critic Adam momentum；
- update 仍为 generic actor-only policy extraction update。

runtime identity 是 `algorithm: crl` + `runtime_variant: policy_extractor`，不会把 M12A actor extraction 错标成新的 scientific RL algorithm。

## 14. Frozen critic source dependency and validation

在任何 Stage 2 Run directory 创建前，validator 强制检查：source Run 路径、`status=completed`、config/environment/algorithm/seed/run_attempt、`last` semantic role、step `1_000_000`、checkpoint 文件、checkpoint metadata/index SHA、source resolved agent compatibility、critic computation FF 和 source module 可提取。

不接受 best、早期 numeric checkpoint、其它 seed 或其它 attempt 的 silent fallback。缺失 dependency 时会 fail loudly；Stage 2 不会自动触发 C001。

## 15. SHA and critic fingerprint mechanism

source full checkpoint 仍是普通完整 agent checkpoint。validator 使用 checkpoint lifecycle 中记录的 SHA256 验证 source 文件未被替换，然后从 `network.params` 提取 `critic`/`modules_critic` subtree，记录 `module_fingerprint`。C002/C003 的 dependency record 都包含相同 source Run、source checkpoint SHA 和 critic subtree fingerprint。

target restore 只替换 critic subtree，不恢复 source optimizer state。restore 后保存 `target_module_fingerprint_before`，后续 checkpoint metadata/runtime metadata 会携带 frozen dependency provenance。

## 16. Stage-2 frozen-critic invariant

[`impls/main.py`](../../impls/main.py) 在 source restore 后、每个 `save_interval`/final step、numeric/semantic checkpoint 写入后分别检查 frozen module fingerprint。任何内存或 checkpoint fingerprint 变化都会抛出异常，而不是 warning。M12A mechanism test 还直接构造 critic mutation，确认 invariant 会以 `changed in memory` fail loudly。

## 17. Stage-2 actor objective and metrics

Stage 2 只支持 canonical DDPG+BC：

```text
a_pi = actor.mode(s, g)
q = min(Q1*(s, g, a_pi), Q2*(s, g, a_pi))
q_loss = -mean(q) / stop_gradient(mean(abs(q)) + epsilon)
bc_loss = -alpha * mean(log pi(a_D | s, g))
actor_loss = q_loss + bc_loss
```

`CRLPolicyExtractorAgent` 复用继承自 `CRLAgent` 的同一份 `actor_loss` 实现，而不是复制公式。strict parity test 对 `actor_loss`、`q_loss`、`bc_loss`、`q_mean`、`q_abs_mean`、`bc_log_prob`、`mse`、`std` 全部 exact 对比通过。

额外 frozen-signal metrics 已实现：`frozen/q_data_mean`、`frozen/q_policy_mean`、`frozen/q_delta`。未实现 Qk、Pk、gradient alignment 或 action-update decomposition，也未实现 AWR/frozen-V/M12B。

## 18. Actor architecture specifications

C002 使用 canonical FF CRL actor，hidden dims 为 `(512,512,512)`，actor computation slot 为 disabled/feedforward。

C003 使用 exact SingleState K4：`primitive=mlp`、`topology=single_state`、`credit=direct`、`iterations=4`、`residual=false`、`input_injection=z_plus_x`、`state_dim=512`、`state_init=normal_buffer`、`state_init_std=1.0`、`update_depth=2`、`layer_norm=false`、`update_activate_final=true`。critic state/goal 仍是 canonical FF，未引入 recurrent critic。

## 19. Paired RNG and data-stream design

runtime metadata 记录：`actor_seed=c`、`dataset_seed=derive_seed(c,1)`、`train_data_rng_seed=derive_seed(c,11)`、`evaluation_seed=derive_seed(c,4)`，以及 `sampling_protocol=explicit_derived_seed_v1`。FF/SS actor architecture 不同，因此同 integer seed 不意味着 parameter tree 相同；它保证的是 paired seed policy 和相同的 deterministic stream policy。

在 synthetic `GCDataset` smoke 中，C002/C003 使用相同 raw data、相同 derived data seed，对前 10 个 batches 的 `observations/actions/value_goals/actor_goals` tree fingerprint 逐 batch exact 相同。

## 20. Stage-2 evaluation protocol

Study protocol 固定 `eval_tasks=all`、`eval_episodes=20`、`eval_temperature=0.0`、`eval_gaussian=null`、`video_episodes=0`、`eval_interval=100000`。测试通过 mock evaluation 验证 all task 会对全部 task 调用 20 episodes、temperature 0 和 no Gaussian noise。Primary endpoint 是 final@1M；secondary endpoints 是 normalized evaluation AUC、best success、best step、last3 mean。

## 21. Result structure after formal execution

正式结果应按 critic seed 输出：

| critic seed | FF final | SS final | paired delta | FF AUC | SS AUC | AUC delta |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 待 Stage 2 | 待 Stage 2 | 待计算 | 待 Stage 2 | 待 Stage 2 | 待计算 |
| 1 | 待 Stage 2 | 待 Stage 2 | 待计算 | 待 Stage 2 | 待 Stage 2 | 待计算 |
| 2 | 待 Stage 2 | 待 Stage 2 | 待计算 | 待 Stage 2 | 待 Stage 2 | 待计算 |

同时报告 mean/median/std of paired delta。当前没有正式 M12A performance result；本轮不能把 smoke 或 preflight 结果误报为 scientific performance。

## 22. Tests executed

M12A 专项测试：

```text
JAX_PLATFORMS=cpu PYTHONPATH=. \
  /home/eai/Tools/miniforge3/envs/brain_nav/bin/python \
  -m unittest -v tests.integration.test_m12a_frozen_critic
```

最终专项结果：14 tests，全部 `OK`。覆盖 critic-only loss parity、100-step critic trajectory/Q parity、100-step actor freeze、same-seed dependency、last@1M role、best rejection、source SHA mismatch、critic fingerprint invariant、actor-only update、optimizer isolation、DDPG+BC loss parity、paired first-10-batch stream、Stage-1 none semantics、Stage-2 all evaluation protocol、temporary source-backed Stage-2 plan。

另外通过：

- M12A + sweep + checkpoint lifecycle：18 tests，全部 `OK`；
- M9/M10/M11A/M11B regression batch：41 项中 40 项默认环境通过，1 项因默认 X11/OpenGL context 初始化失败；
- 同一失败环境测试在 `MUJOCO_GL=egl PYOPENGL_PLATFORM=egl` 下单独复测通过；
- CRL runtime：非 real-environment 的 10 项通过，real-data parity 2 项在 EGL 下通过；
- compileall：`impls`、M12A experiment files、scripts、tests 通过。

`pytest` 在当前 brain_nav environment 不可用，因此使用仓库现有 unittest-compatible test runner；这不是 scientific test failure。

## 23. M9/M10/M11A/M11B regression result

M9 single-state、M9B two-state、M9B1M、M10A fixed-budget、M11A CRL interaction、M11B study matrix/provenance 等既有测试均通过。M11B 的默认环境 shape test 仅受无 EGL 的 X11/OpenGL context 影响，切换到项目既有 EGL 设置后通过。M12A 没有修改 M9/M10/M11A/M11B 的 scientific definitions。

## 24. Stage-1 dry-run command and result

实际执行的隔离 dry-run：

```bash
JAX_PLATFORMS=cpu PYTHONPATH=. \
  /home/eai/Tools/miniforge3/envs/brain_nav/bin/python \
  experiments/M12A_frozen_critic_policy_extraction/preflight.py \
  --stage 1 --run-root /tmp/m12a_preflight_runs
```

结果为 `planned_runs: 3`，分别是 C001 seed 000/001/002，且输出 `formal_training_started: false`。另用 `tools/sweep.py --dry-run --configs M12A-C001` 验证了同样的 3-run protocol，未创建 formal Run directory。

## 25. Stage-2 dry-run command and result

实际执行的 source-missing dry-run：

```bash
JAX_PLATFORMS=cpu PYTHONPATH=. \
  /home/eai/Tools/miniforge3/envs/brain_nav/bin/python \
  experiments/M12A_frozen_critic_policy_extraction/preflight.py \
  --stage 2 --run-root /tmp/m12a_preflight_runs
```

结果为 `dependency_preflight: NO-GO`，3 个 seed 的 C001 `runtime_metadata.json` 均不存在。`tools/sweep.py` 对 C002/C003 同样返回 `preflight: NO-GO`，没有自动 source run。

为验证 launcher 的正向分支，在独立 temporary directory 写入了仅用于 smoke 的 3 个 completed source artifacts；之后 Stage-2 preflight 输出 `planned_runs: 6`，C002/C003 各 3 个 seed，且 `formal_training_started: false`。这些 artifacts 不在正式 run root，也不是 formal result。

## 26. Current Stage-2 dependency status

当前正式执行状态是 NO-GO：Stage 1 的三个 C001 completed `last@1M` source artifacts 尚未由用户启动并完成，因此 Stage 2 不可执行。缺失 source 是故意的安全阻断，不是实现失败。完成 Stage 1 后，用户应先检查每个 run 的 status、checkpoint index/metadata、step、SHA 和 critic fingerprint，再重新运行 Stage-2 preflight。

## 27. Giant confirmatory extension

AntMaze-Giant 只在 `study.yaml` 的 `confirmatory_extension` 中标记为 `prespecified_not_active`。本轮没有 Giant active config、没有 Giant dry-run、没有 Giant training，也没有将 Giant 纳入 9-run primary matrix。

## 28. Unresolved issues

1. 没有正式 M12A performance data，因此尚不能回答 `Delta_c` 是否为正。
2. Stage 2 必须等待用户手动完成 Stage 1；本轮不会代为启动。
3. 默认 X11 环境无法初始化 MuJoCo offscreen context；正式环境测试/训练应沿用项目 EGL 配置。
4. 目前只实现 DDPG+BC frozen critic extraction；AWR、frozen V、M12B factorial 和后续 diagnostic 未实现，符合 prompt 的 scope。

## 29. GO / NO-GO recommendation

实现层面的 GO 条件已满足：无 per-study `m12a.py`、无 M12A-specific main/management branch、3 configs/3 seeds/9 planned Runs、canonical FF critic-only、actor freeze、joint parity、Stage-1 none、last@1M-only dependency、SHA/fingerprint、structural actor-only optimizer、DDPG+BC parity、FF/SS exact spec、paired stream 和回归测试均通过。

但对“现在正式启动 M12A”给出的建议是 NO-GO，直到 Stage 1 的三个 source Runs 由用户手动完成并通过 dependency preflight。之后可以在用户 review/commit/frozen worktree 完成后，由用户手动启动 Stage 2。

## 30. Explicit final statement

**No formal M12A training was started.**

本轮没有执行 `--execute`、1M critic training、1M actor training、Git commit、Git push、Git branch、Git worktree 或任何其它 Git 操作。下一步由用户 review、手动完成 Git 相关操作，并手动决定是否启动 Stage 1；本 agent 不会自动推进正式实验。

## 31. Prompt item-by-item completion map

为避免把合并章节误读为遗漏，prompt 要求的 39 项逐项对应如下：

1. scientific question：见第 1 节；
2. AntMaze-Large scope：见第 2 节；
3. three seeds：见第 3 节；
4. fixed last@1M source：见第 4 节；
5. no best selector：见第 4 节；
6. exact three configurations：见第 5 节；
7. exact nine planned Runs：见第 6 节；
8. Study/Configuration/Run ontology：见第 6 节；
9. no `impls/experiment/m12a.py`：见第 7 节；
10. generic framework changes：见第 8 节；
11. `m11b.py` untouched by M12A：见第 7 节和第 23 节；
12. critic-only API：见第 9 节；
13. critic-only/joint parity：见第 10 节；
14. Stage-1 actor freeze：见第 11 节；
15. policy extractor design：见第 13 节；
16. runtime/scientific identity separation：见第 13 节；
17. dependency schema：见第 5 节和第 14 节；
18. source checkpoint validation：见第 14 节；
19. SHA and subtree fingerprint：见第 15 节；
20. frozen-critic invariant：见第 16 节；
21. actor-only optimizer isolation：见第 13 节和第 17 节；
22. canonical DDPG+BC：见第 17 节；
23. C002 FF actor：见第 18 节；
24. C003 exact SS K4：见第 18 节；
25. paired RNG/data design：见第 19 节；
26. paired-batch result：见第 19 节；
27. Stage-1 no-evaluation semantics：见第 12 节；
28. implemented metrics：见第 17 节；
29. executed tests：见第 22 节；
30. M9/M10/M11A/M11B regression：见第 23 节；
31. Stage-1 dry-run command：见第 24 节；
32. Stage-1 planned count = 3：见第 24 节；
33. Stage-2 dry-run command：见第 25 节；
34. Stage-2 planned count = 6 after dependencies：见第 25 节；
35. current Stage-2 dependency status：见第 26 节；
36. Giant extension status：见第 27 节；
37. unresolved issues：见第 28 节；
38. GO/NO-GO recommendation：见第 29 节；
39. explicit no-formal-training statement：见第 30 节。
