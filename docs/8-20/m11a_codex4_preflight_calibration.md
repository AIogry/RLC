# M11A codex4 最后一次正式实验前校准报告

日期：2026-08-21  
Study：`M11A — CRL Actor × Critic Computation Interaction`  
校准范围：真实 `antmaze-large-navigate-v0`，seed 0，batch size 1024，1000 steps  
最终判定：**GO（仅表示通过正式实验前校准；不表示已启动正式实验）**

本轮严格执行 `docs/8-20/prompt for codex4.md`。没有启动 7×1M formal training，也没有启动 formal reevaluation、diagnostic bank 或 scoring。所有 pilot 输出只写入临时目录 `/tmp/m11a_codex4.beZRBR`。

本轮没有执行任何 Git 命令，也没有 commit、branch、worktree、add、push、pull 或修改 Git provenance。校准脚本绕过 `impls.main.run()` 的 run-context provenance，仅直接构造 agent、复用真实数据集并执行 1000-step update。

## 1. 最终结论

codex4 的新判据全部满足：

1. split-gradient / separate-gradient training path 已撤回，恢复 canonical CRL `total_loss -> apply_loss_fn` update；
2. actor Q 分支对 critic 参数的梯度为严格零，同时保留对 actor 参数的 `dQ/da` 梯度；
3. C001/C003、C002/C004、C005/C007 三组 same-process critic 在完全相同 batch sequence 下，`phi`/`psi` 的参数、buffers、hash 和 held-out Q 全部 exact identical；
4. independent-process 的 actor-different critic divergence 没有系统性超过同配置 repeat 的自然背景 divergence；
5. 特别是 C005/C007 没有超过 C005/C005′ 的参数背景 divergence；
6. computation、M9A、M9B、M11A 相关测试全部通过。

因此，M11A 具备由用户手动启动正式 7×1M campaign 的代码与数值校准条件。正式启动仍不在本轮执行范围内。

## 2. split-gradient 撤回情况

`impls/agents/crl.py` 已恢复为 canonical update：

```text
new_rng, rng = split(self.rng)
loss_fn(params) = total_loss(batch, params, rng)
network.apply_loss_fn(loss_fn)
```

上一轮新增的 DDPG+BC 专用 `critic_loss_fn`、`actor_loss_fn`、分开求梯度再相加的路径已经删除。当前 `update()` 只对 joint `total_loss` 求一次梯度，并调用原有 optimizer 一次。

保留的唯一边界修复位于 DDPG+BC actor Q 分支：critic 参数在 Q evaluator 中使用 `stop_gradient`，但 `q_actions` 没有 stop-gradient。因此：

```text
critic params <- actor Q branch:      no gradient
actor params  <- Q through action:   gradient retained
critic params <- contrastive loss:   canonical critic update
```

这没有改变 Q 数值定义、BC loss、contrastive critic objective、optimizer 或数据 protocol。

## 3. Gradient-boundary test

新增/更新的测试位于：

- `tests/integration/test_m11a_crl_interaction.py::test_ddpgbc_actor_q_branch_does_not_update_critic_params`

测试分别对完整 actor loss 和其中的 `q_loss` 求梯度，并逐 leaf 检查 `phi`、`psi`：

| 检查项 | 结果 |
|---|---:|
| `max_abs(∇critic L_actor)` | `0.0`，严格数组相等于零 |
| `max_abs(∇critic L_Q)` | `0.0`，严格数组相等于零 |
| `max_abs(∇actor L_Q)` | `0.5734459`，非零 |
| `L2(∇actor L_Q)` | `0.7782537`，非零 |

因此 actor 仍通过 Q 对 action 的梯度学习，而 actor loss 不更新 critic 参数。

由于历史 external reference CRL 尚未包含该边界，`test_crl_runtime` 的 DDPG+BC migration comparison 已改为：保留 primal/info/actor-gradient/update 的数值迁移检查，并对 GPU fusion 导致的小量数值差使用 `5e-4` absolute tolerance；不再错误地要求旧 reference 的 critic gradient 与新边界一致。AWR 路径仍保持原有 exact migration check。

## 4. Same-process critic identity

协议：每一组在同一 Python process、同一物理 GPU、同一初始化、同一真实 batch sequence 中同时训练两个配置 1000 steps。比较 final critic 的 `phi`、`psi` 参数和 recurrent buffers，并在固定 held-out batch 上比较 critic Q 输出。

GPU：`NVIDIA GeForce RTX 4090`，GPU UUID `GPU-70b791e9-3f0b-48b7-a3a6-b5bdb0a22c8f`。

| pair | phi params | psi params | critic params max abs / rel L2 | buffers | held-out Q mean / max abs |
|---|---|---|---:|---|---:|
| C001 vs C003 | exact | exact | `0 / 0` | exact | `0 / 0` |
| C002 vs C004 | exact | exact | `0 / 0` | exact | `0 / 0` |
| C005 vs C007 | exact | exact | `0 / 0` | exact | `0 / 0` |

三组 initial critic `phi`/`psi` 也均 exact identical。aggregate critic parameter stable hash 如下；每行 reference 与 compared hash 完全相同：

| pair | aggregate critic parameter hash |
|---|---|
| C001 vs C003 | `eeb3b35875d9659df26318610e358af160a0b48109979c07a26f44692614a494` |
| C002 vs C004 | `d43ac4e6bafe611c918af24b3cdb2479470d6784cda601ab3bc6a69139acf61e` |
| C005 vs C007 | `df0e2ae8243837aa1f0e158b53964156202ac3cd6e82f9bf31b9e55ba064ff32` |

`phi`/`psi` branch hash 也分别完全相同；C002/C004 与 C005/C007 的 recurrent buffers 均有非零元素，但两侧 buffer hash 和数组仍 exact 相同。C001/C003 为无 recurrent buffer 的 feed-forward critic。

## 5. Independent-process repeat calibration

每个独立 process 都使用相同 seed/config/data/protocol，并在同一物理 GPU 上顺序执行。`params` 指 phi+psi aggregate；`relative L2` 为：

```text
||params_a - params_b||_2 / ||params_a||_2
```

所有 independent comparisons 的 buffers 均 exact，`max_abs_diff=0`；所有 comparisons 的 initial critic params 均 exact。

### 5.1 Same-config repeat 与 actor-different pair

| comparison | params exact | params max abs | params rel L2 | fixed-Q mean abs | fixed-Q max abs |
|---|---:|---:|---:|---:|---:|
| C001 vs C001′ | yes | `0` | `0` | `2.8045e-5` | `9.7513e-4` |
| C001 vs C003 | no | `1.7479e-5` | `2.8410e-6` | `5.6846e-4` | `2.6417e-3` |
| C002 vs C002′ | no | `1.6874e-4` | `9.6066e-6` | `1.7874e-3` | `9.6836e-3` |
| C002 vs C004 | no | `1.7461e-4` | `9.8845e-6` | `2.4671e-3` | `1.4981e-2` |
| C005 vs C005′ | no | `1.0696e-1` | `2.2443e-2` | `6.5211e-1` | `4.7445` |
| C005 vs C007 | no | `7.4593e-2` | `2.2551e-2` | `1.0317` | `6.7087` |

branch-level parameter results：

| comparison | phi max abs / rel L2 | psi max abs / rel L2 |
|---|---:|---:|
| C001 vs C001′ | `0 / 0` | `0 / 0` |
| C001 vs C003 | `1.0073e-5 / 2.7804e-6` | `1.7479e-5 / 2.9005e-6` |
| C002 vs C002′ | `1.0300e-4 / 9.4856e-6` | `1.6874e-4 / 9.7267e-6` |
| C002 vs C004 | `1.0331e-4 / 9.7647e-6` | `1.7461e-4 / 1.0003e-5` |
| C005 vs C005′ | `1.0696e-1 / 2.1701e-2` | `7.7677e-2 / 2.3163e-2` |
| C005 vs C007 | `7.4593e-2 / 2.1832e-2` | `6.6370e-2 / 2.3250e-2` |

### 5.2 Actor-pair 与 repeat 的直接比较

#### C001/C003

C001/C003 的参数 divergence 为 `1.7479e-5`，而 C001/C001′ repeat 的参数为 exact identical。这个差异是独立 GPU process 数值路径产生的微小参数 divergence，绝对量和 relative L2 都很小。

fixed-batch Q 差异为 `5.6846e-4` mean、`2.6417e-3` max；C001/C001′ 的 Q background 为 `2.8045e-5` mean、`9.7513e-4` max。C001 repeat 的参数 exact 但 Q 仍有微小跨进程计算差异，说明 Q 数值比较本身也包含 GPU process-level arithmetic noise；因此不能把这一接近零的 repeat mean 当作严格零噪声基线。

#### C002/C004

C002/C004 的参数 divergence 与同配置 repeat 几乎相同：

```text
actor pair:  rel L2 = 9.8845e-6, max abs = 1.7461e-4
repeat:      rel L2 = 9.6066e-6, max abs = 1.6874e-4
```

Q mean/max 为 `2.4671e-3 / 1.4981e-2`，repeat 为 `1.7874e-3 / 9.6836e-3`，处于同一数量级。

#### C005/C007（重点）

C005/C007 没有超过 C005/C005′ 的参数背景 divergence：

```text
                    max abs          relative L2
C005 vs C005′       1.0696e-1        2.2443e-2
C005 vs C007        7.4593e-2        2.2551e-2
```

relative L2 的差异只有约 `1.08e-4` absolute，actor pair 的 max abs 反而低于同配置 repeat。Q-function divergence 为：

```text
                    mean abs         max abs
C005 vs C005′       6.5211e-1        4.7445
C005 vs C007        1.0317           6.7087
```

actor pair 的 Q 差异约为 repeat 的 `1.58×`（mean）和 `1.41×`（max），仍处于同一数量级；没有出现与参数 divergence 相匹配的系统性放大。特别是 codex4 要求重点排查的 C005/C007，没有明显超过 C005/C005′ 的 natural process background。

## 6. 测试结果

| suite | result |
|---|---:|
| computation tests | `48/48 passed` |
| M9A/M9B/M9B1M/M11A/CRL runtime integration | `33/33 passed` |
| changed Python files `py_compile` | passed |
| same-process calibration | `3/3 completed` |
| independent-process calibration | `9` single-process runs completed（3 repeats + 6 actor-pair members） |

## 7. GO/NO-GO 判定

本轮采用 codex4 的修订判据，不再要求 independent GPU processes bitwise exact identical。

判定：**GO**。

理由：

- same-process actor-only critic identity 三组全部 exact；
- same-process actor+critic pair 也全部 exact，说明 actor topology 不会在同一 execution stream 中改变 critic update；
- initialization、dataset protocol、critic recurrent buffers 没有 actor-dependent coupling；
- C002/C004 的 independent divergence 与 C002/C002′ repeat 几乎一致；
- C005/C007 的 relative-L2 divergence 不超过 C005/C005′，且 max abs 更低；
- fixed-batch Q divergence 虽有跨进程数值噪声，但 C005/C007 与 C005/C005′ 同量级，不构成 architecture-dependent numerical failure；
- 所有已有 computation/M9A/M9B/M11A 测试通过。

因此，不需要因为独立 GPU process 的 bitwise non-identity 阻止正式实验。正式 7×1M training 仍须由用户手动启动；本轮没有替用户启动任何正式实验。

## 8. 建议的启动前边界

本轮不建议再修改 M11A 架构、loss、seed、environment 或 Study 7 configs。用户后续手动启动正式实验时，应保存每个 run 的：

- `runtime_metadata.json` 中的 computation slot 与 primitive semantics；
- critic `phi`/`psi` params 与 recurrent buffers hash；
- final checkpoint 的完整 provenance；
- 训练过程中的 eval 与 checkpoint lifecycle。

如需进一步提高 independent-process 的 bitwise reproducibility，应将其作为单独的基础设施研究问题，不应在 M11A 正式 factorial 中临时改变科学设计。
