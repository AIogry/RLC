# M11A 正式实验前修订与验证报告（codex3）

日期：2026-08-21  
Study：`M11A — CRL Actor × Critic Computation Interaction`  
最终判定：**NO-GO，暂不启动正式 7×1M campaign**

本轮严格按照 `docs/8-20/prompt for codex3.md` 执行。正式 M11A 1M training、formal reevaluation、formal diagnostic bank/scoring 均未启动。没有直接执行任何 Git shell 命令，也没有 commit/branch/worktree/add/push/pull；但允许的 pilot 通过既有 `create_run_context -> _git_metadata` provenance 代码间接调用了 `git rev-parse`/`git status` 来写运行元数据。这一 side effect 在报告中明确披露；今后若要绝对禁止任何 Git 调用，应由用户手动运行 pilot 或先提供不调用 provenance Git 的 pilot 入口。`git status --short`、`git diff --stat` 等检查仍需要由用户手动完成。

## 1. 本轮结论摘要

已完成并通过：

- recurrent critic primitive semantics 修复：critic 继承 vanilla branch 的 `LayerNorm=True` 与 final layer 无 activation 语义；
- actor 旧 M9 语义保持：`LayerNorm=False`、update final activation=True；
- SingleState critic K=1、zero-state、无 residual 的 vanilla MLP forward/gradient/LayerNorm/参数量 parity；
- TwoState critic 的 LayerNorm 与无 final activation manual-reference parity；
- critic identity audit 扩展为 params + recurrent buffers；
- duplicate candidate metric 改为比较 action vectors，不再比较 critic scalar；
- M9A CRL SingleState K4 no-residual 与 M9B CRL TwoState H2L1 full-BPTT actor regression；
- M11A 7-config doctor：7/7 passed；
- computation tests：48/48 passed；
- M9A/M9B/M11A 相关 integration tests：17/17 passed；
- 真实 `antmaze-large-navigate-v0` 短 pilot：最终修订版 6/6 完成，训练/评估/checkpoint restore 均通过。

未满足 GO 条件：

- 独立 GPU 进程的 critic params 不能保证 exact identity；
- 同一配置的跨进程重复 pilot 也出现约 `1e-5` 至 `4e-5` 的 params 差异；
- 因此不能声称 actor topology 改变不会影响独立 formal run 的 critic trajectory；
- 当前环境存在尚未被消除的 GPU/XLA 跨进程数值路径问题。

结论是 NO-GO，而不是代码无法运行。正式 campaign 应在确定性 GPU 执行问题被解释并解决后再由用户手动启动。

## 2. Scientific confound：recurrent critic 与 vanilla primitive 不一致

CRL vanilla bilinear critic branch 使用：

```text
branch_dims = (512, 512, 512, 512)
activate_final = False
layer_norm = True
```

因此 branch 的语义是：

```text
Dense -> GELU -> LayerNorm
Dense -> GELU -> LayerNorm
Dense -> GELU -> LayerNorm
Dense
```

修订前 recurrent factory 存在两个 confound：

1. factory 强制 `layer_norm=False`；
2. SingleState/TwoState update MLP 强制 `activate_final=True`。

这使 M11A 实际比较的不只是 computation topology，而是：

```text
topology + normalization + final activation recipe
```

这会直接污染 actor/critic computation interaction 的科学解释。

## 3. Primitive semantics 修复

### 3.1 新的传递方式

`make_computation_core()` 现在把 caller 的 primitive 语义传入 topology：

```text
caller activate_final -> topology.update_activate_final
caller layer_norm     -> topology.layer_norm
```

SingleState 和 TwoState 只负责 recurrent execution equation，不再覆盖 caller 语义。

内部字段：

```text
update_activate_final: bool = True
layer_norm: bool = False
```

默认 fallback 保留旧的 direct recurrent-core actor 行为；实际 network caller 显式传递 primitive：

| caller | `layer_norm` | `update_activate_final` |
|---|---:|---:|
| GCActor / legacy actor | `False` | `True` |
| CRL `_ComputationBilinearBody` critic | `True` | `False` |

没有新增 M11A scientific factor，也没有把 activation/normalization 暴露为新的实验 placement。

### 3.2 Actor backwards compatibility

GCActor 仍调用 recurrent core 时使用 `activate_final=True`；actor 没有 LayerNorm。因此：

- M9A SingleState actor 仍为 `update_depth=2`、`LayerNorm=False`、update final activation=True；
- M9B TwoState actor 仍为 `update_depth=2`、`LayerNorm=False`、update final activation=True；
- legacy M9 config 不需要新增字段；
- legacy actor parameter tree 中没有新增 LayerNorm subtree；
- checkpoint save/restore 与 deterministic sample action 通过。

### 3.3 Critic branch semantics

CRL recurrent critic branch 现在为：

```text
input_mapping:  Dense -> GELU -> LayerNorm
update depth:   3
update middle:  GELU -> LayerNorm
update final:   Dense，无 activation，无 LayerNorm
```

SingleState 的 `input_mapping` 与 `update_module` 按 vanilla 四层 branch 分解。TwoState 的 `h_update` 和 `l_update` 使用相同 primitive，但保留自身 H/L schedule；TwoState 不被宣称为 vanilla functional decomposition。

Study metadata 已删除错误的：

```yaml
normalization: none_in_recurrent_core
```

并改为 fixed primitive semantics：

```yaml
primitive_semantics:
  actor_recurrent_core: {activation: gelu, layer_norm: false, update_activate_final: true}
  critic_recurrent_core: {activation: gelu, layer_norm: true, update_activate_final: false}
```

## 4. SingleState critic K=1 zero-state parity

测试构造：

```text
vanilla = MLP(
    hidden_dims=(d,d,d,d),
    activate_final=False,
    layer_norm=True,
)

recurrent = SingleState(
    iterations=1,
    residual=False,
    state_init=zero_buffer,
    update_depth=3,
    layer_norm=True,
    update_activate_final=False,
)
```

参数映射：

```text
vanilla Dense_0  -> input_mapping Dense_0
vanilla Dense_1  -> update_module Dense_0
vanilla Dense_2  -> update_module Dense_1
vanilla Dense_3  -> update_module Dense_2

vanilla LayerNorm_0 -> input_mapping LayerNorm_0
vanilla LayerNorm_1 -> update_module LayerNorm_0
vanilla LayerNorm_2 -> update_module LayerNorm_1
```

验证结果：

- forward output shape parity：通过；
- forward numerical parity：通过，`atol=1e-6`；
- gradient parity：通过，`atol=2e-6`；
- LayerNorm 参数 subtree 数量和映射：通过；
- Dense + LayerNorm parameter count parity：通过；
- recurrent zero buffer 不进入 trainable params：通过。

这证明 K=1、zero state、no residual 时 recurrent critic body 可以严格分解 vanilla CRL critic branch。

## 5. TwoState primitive semantics test

TwoState 使用：

```text
update_depth=3
layer_norm=True
update_activate_final=False
h_cycles=2
l_cycles=1
credit=full_bptt
```

测试检查：

- `input_mapping` 有 LayerNorm；
- `h_update` 有两层中间 LayerNorm，最后 Dense 无 LayerNorm；
- `l_update` 同样；
- H/L execution trace 与原 schedule 一致；
- final H representation shape 正确；
- forward finite；
- gradient finite；
- manual reference forward parity 通过。

不把 TwoState 宣称为 vanilla branch 的 functional parity，只验证 primitive semantics 正确传入 H/L update modules。

## 6. DDPG+BC critic gradient boundary

pilot 暴露了另一个必须处理的 deterministic-path 问题：原 `actor_loss` 的 Q 分支在联合 loss graph 中可能使 actor topology 影响 critic update graph。

本轮没有改变：

- Q 数值定义；
- contrastive critic loss；
- BC loss；
- DDPG+BC actor loss 公式；
- optimizer；
- dataset sampling；
- evaluation protocol。

修复方式是：

1. actor Q 分支对 critic params 使用 stop-gradient；
2. 保留 Q 对 `q_actions` 的梯度，使 actor 仍可通过 critic 更新；
3. DDPG+BC 的 critic-loss gradient 与 actor-loss gradient 分别计算，再合并到同一个 optimizer update；
4. critic gradient graph 不再依赖 actor topology 的联合编译图。

新增测试证明 actor Q branch 对 critic params 不产生超过 `1e-6` 的额外梯度。该修改是 gradient-boundary 修复，不是新 loss 或新 representation。

## 7. Generic slot accounting

`computation_slot_accounting` 保留旧字段，并新增/确认记录：

- `layer_norm`；
- `update_activate_final`；
- `update_depth`；
- topology、credit、state initialization、update executions；
- trainable/core params；
- buffers；
- input/update parameter counts。

正式 M11A resolved runtime metadata 预期记录：

| slot | `layer_norm` | `update_activate_final` |
|---|---:|---:|
| actor SingleState/TwoState | false | true |
| critic_state SingleState/TwoState | true | false |
| critic_goal SingleState/TwoState | true | false |

## 8. Critic identity audit 修复

旧 audit 只比较：

```text
network.params['modules_critic'][branch]
```

新 audit 对 `phi` 和 `psi` 分别比较：

```text
A. network.params['modules_critic'][branch]
B. network.model_state['buffers']['modules_critic'][branch]
```

每一部分均记录：

- tree structure；
- paths；
- shape；
- dtype；
- element count；
- exact array equality；
- max absolute difference；
- allclose 与 tolerance；
- stable SHA256 hash。

最终 `same_critic_verification_passed` 只有在 params 和 relevant buffers 均 exact identity 时才为 true。FF critic 没有 buffers 是合法结果；recurrent critic 的 `z_init`/`z_h_init`/`z_l_init` 必须纳入 audit。

audit 不会复制参数、不强制同步、不把 allclose 伪装成 exact。

## 9. Duplicate candidate metric 修复

candidate artifact 保存的是：

```text
[num_pairs, num_candidates, action_dim]
```

修订前 duplicate 标志错误地比较了同一 row 中的 Q scalar。现在直接比较 action vectors：

```python
np.array_equal(action_i, action_j)
```

当前第一版只记录 exact duplicate，不新增 hidden tolerance。`E_ext_gap` 与 `E_ext_rank` 主公式没有改变；duplicate 仍只是 diagnostic quality flag。

新增 synthetic test 覆盖：

- Q scalar 相同但 action vector 不同 -> 不计 duplicate；
- action vector exact 相同但 Q scalar 不同 -> 计 duplicate。

## 10. M9A/M9B actor regression

实际测试配置：

| legacy config | topology | protocol check |
|---|---|---|
| M9A-C007 | SingleState K4 no-residual | update depth 2、无 LayerNorm、checkpoint restore |
| M9B-C001 | TwoState H2L1 full-BPTT | H/L 各两层 Dense、无 LayerNorm、checkpoint restore |

结果：

- legacy config 无需新字段：通过；
- actor parameter tree 未增加 LayerNorm：通过；
- actor forward finite：通过；
- sample action deterministic restore equality：通过；
- SingleState execution/zero-state parity：通过；
- TwoState execution trace/credit regression：通过。

## 11. M11A 7-config doctor

CPU synthetic doctor 对以下 7 个 config 均执行：

1. synthetic create；
2. 一次 finite update；
3. checkpoint save；
4. checkpoint restore；
5. action equality probe。

结果：

| config | topology | finite update | checkpoint restore |
|---|---|---:|---:|
| M11A-C001 | feedforward | PASS | PASS |
| M11A-C002 | SingleState critic | PASS | PASS |
| M11A-C003 | SingleState actor | PASS | PASS |
| M11A-C004 | SingleState actor+critic | PASS | PASS |
| M11A-C005 | TwoState critic | PASS | PASS |
| M11A-C006 | TwoState actor | PASS | PASS |
| M11A-C007 | TwoState actor+critic | PASS | PASS |

总结果：**7/7 passed**。

## 12. 相关测试结果

实际执行：

```text
tests/computation/ discovery:                         48 passed
M9A/M9B/M11A integration selection:                  17 passed
M11A targeted integration + computation selection:    42 passed
M11A doctor:                                           7/7 passed
Python py_compile:                                     passed
```

其中 48 个 computation tests 覆盖：

- SingleState legacy；
- generalized update_depth；
- SingleState K1 actor parity；
- SingleState K1 critic primitive parity；
- LayerNorm 参数映射；
- TwoState legacy；
- TwoState generalized update_depth；
- TwoState LayerNorm/final activation semantics；
- full-BPTT；
- one-step credit；
- execution trace；
- finite gradients。

## 13. Real AntMaze-Large pilot protocol

formal 7×1M 未执行；只执行了独立 `/tmp` pilot：

```text
environment: antmaze-large-navigate-v0
dataset: /data/qijunrong/06-RL/offline-rl/data/raw_ogbench
seed: 0
batch_size: 1024
steps: 1000
GPU: physical GPU 1
eval: all five tasks, one episode/task, temperature=0, no Gaussian noise
run root: /tmp/m11a_preflight_splitgrad.LkkC0n/runs
```

六个 pilot config：

```text
C001, C003   FF critic：actor FF vs actor SingleState
C002, C004   SingleState critic：actor FF vs actor SingleState
C005, C007   TwoState critic：actor FF vs actor TwoState
```

6/6 任务均完成 1000 steps，训练 loss finite，评估完成，best/last/root checkpoint save/restore 通过，recurrent model_state buffer restore 通过。

## 14. Pilot critic params/buffers audit

以下是最终 split-gradient pilot 的 `params_1000.pkl` 审计。每行中的 hash 是对应 branch 的 stable SHA256；`buffers exact` 是 relevant recurrent buffers 的 exact equality。

| pair | branch | params exact | max abs diff | reference hash | compared hash | buffers exact |
|---|---|---:|---:|---|---|---:|
| C001 vs C003 | phi | false | 1.1489e-05 | `9eccb5ff1bc8d62a85535801f179f0591228a0b308d960fb8c392abee55c81c6` | `62513ae5246520711f118fcf56deb9e95da915d3ba22c00c3a2d4c744c417639` | true |
| C001 vs C003 | psi | false | 2.7757e-05 | `e9f537ec0c822f8ba967d67fb7716c3484cec7ac37b5ec0c05c323e0bcc1021c` | `08a6a035f73497a9c9efb84e54180a5be1ad6a34528015cf8ab0fba2e7225bed` | true |
| C002 vs C004 | phi | false | 2.7671e-05 | `8583489da7454e585d0f34f5c4a408c61b2bfa9574d63248a4bbe0608aa19112` | `1338605316bf1e4ebb3e74fdc22b213c105bd400baf33e50169c7f1cdc04f62b` | true |
| C002 vs C004 | psi | false | 1.7408e-05 | `74a046fe80a22318361608ac467a815dc937dbd05805f9bf14f46c633ba8bbb4` | `8cfe3df02b1bd2691be67670533e5116d1b36468baff8c07ff1f45f3e959de6b` | true |
| C005 vs C007 | phi | false | 2.6968e-02 | `4babd2c5dcc38e987bdee2045220887ab8312fb8dbf0c1dcbed69411bd8cda97` | `80ffb970fa796bd840173f48479240785122861451b57cad55d1a46d75c13fe0` | true |
| C005 vs C007 | psi | false | 2.6466e-02 | `234f9eb9ad25dabf059b0f241c0f31db4bf8dffacd115a75ae1847c46f1dab1d` | `6f1109b2ce1c2d84a2c5c4116c63c0a157f2b2c87f7627a227a355cfc9a0ad5a` | true |

所有 pair 的 parameter path/count/shape 兼容；差异不是 tree schema mismatch。recurrent buffers 的 count、path、hash 和 array 均 exact identical。

## 15. RNG coupling 与 divergence 调查

### 15.1 初始化 RNG

对三组 actor-only pair，初始化时 critic `phi` 和 `psi` 参数均 exact identical，max diff=0；recurrent buffers 也 exact identical。

因此没有发现：

- ModuleDict init order 导致的 critic initial RNG coupling；
- params RNG split coupling；
- buffers RNG split coupling；
- actor topology 改变 critic initialization。

### 15.2 Dataset stream

C002/C004 使用相同 seed 构造的首个真实 batch 逐字段 exact identical，包括：

```text
observations
next_observations
actions
value_goals
actor_goals
masks
rewards
terminals
valids
```

同一进程共享同一 batch stream 训练 1000 steps 时，C002/C004 critic `phi`/`psi` params exact identical。

### 15.3 Cross-process GPU behavior

同配置 C002 的独立第二次 1000-step pilot 也出现：

```text
phi max abs diff: 1.20997e-05
psi max abs diff: 3.73181e-05
buffers: exact identical
```

这证明小幅差异不是 actor topology 独有；独立 GPU process 本身即可产生非零 critic parameter difference。最终 split-gradient pilot 中 C005/C007 仍出现约 `2.7e-2` 差异，说明当前 GPU/XLA execution environment 尚未达到 formal identity audit 所需的严格可复现性。

因此目前能诚实得出的解释是：

```text
actor->critic gradient confound: 已修复
initial RNG coupling: 未发现
dataset RNG coupling: 未发现
cross-process GPU/XLA numeric divergence: 仍存在
```

这个环境级问题不能被 `allclose` 或复制参数掩盖。

## 16. Unresolved issues

仍有一个 unresolved scientific/infrastructure issue：

> 在当前 GPU/JAX/XLA 环境中，独立 process 的相同或 actor-only-different run 不能保证 critic params exact equality。

需要用户/实验环境后续明确并解决的方向包括：

- 检查 GPU/XLA deterministic reduction 与 kernel 配置；
- 确认正式运行是否必须固定到同一 deterministic execution mode；
- 评估是否需要在同一 process 中完成 paired actor conditions；
- 不能通过复制 critic 参数、修改 audit 判据或事后同步 buffer 来规避问题。

本轮没有继续扩大到新的 topology、loss、environment、seed 或 candidate search。

## 17. GO/NO-GO 判定

### 已满足

- recurrent critic `layer_norm=True`：是；
- recurrent critic `activate_final=False`：是；
- legacy actor `layer_norm=False`：是；
- legacy actor `activate_final=True`：是；
- M9A regression：通过；
- M9B regression：通过；
- SingleState critic forward parity：通过；
- SingleState critic gradient parity：通过；
- critic LayerNorm parity：通过；
- critic parameter count parity：通过；
- recurrent critic phi/psi independent：通过；
- ensemble recurrent buffers independent：通过；
- checkpoint save/restore：通过；
- identity audit 包含 params：是；
- identity audit 包含 buffers：是；
- duplicate metric 比较 action vector：是；
- 7-config doctor：通过；
- relevant tests：通过；
- real-data pilot 6/6 完成：是；
- formal 1M 未启动：是。

### 未满足

- pilot 没有达到三组独立 process critic params exact identity；
- 同配置重复 pilot 也有非零差异；
- cross-process GPU/XLA divergence 尚未被工程上消除。

因此最终是：

```text
NO-GO for formal M11A 7×1M campaign
```

这不是对 M11A 科学假设的否定，而是正式实验前 deterministic critic-path 尚未达标。用户不应在当前状态下启动 formal 1M。

## 18. 用户后续手动步骤

本轮不提供 `--execute` 启动建议，因为 GO 条件未满足。用户后续应先解决 GPU/XLA deterministic issue，然后手动重新执行：

1. 相关 tests；
2. 7-config doctor；
3. 三组或完整 6-config real-data pilot；
4. params + buffers identity audit；
5. 只有全部满足后，再手动创建 frozen commit/detached worktree；
6. 手动执行 M11A dry-run；
7. 最后才由用户手动启动 7×1M formal campaign。

本 agent 没有直接执行 Git 命令，也没有启动 formal 1M training、formal reevaluation、formal bank 或 formal E_eval/E_ext scoring。pilot 的既有 provenance 代码间接调用 Git 的事实如本报告开头所述。
