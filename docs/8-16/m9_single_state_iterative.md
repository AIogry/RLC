# M9A：Single-State Iterative Actor Computation

日期：2026-08-16

## 结论与范围

M9A 的第一版实现已经完成为一个可审计、可配置、可恢复的 actor-only
computation vertical slice。当前只实现 decision-local single-state topology，
没有引入 HRM 的双状态、跨 environment step memory、SwiGLU、RMSNorm、gating、
adaptive halting 或新的 credit policy。

本次工作完成了代码、Study 配置、runtime provenance、参数 accounting、测试和
短 smoke 验证；没有启动完整的 52-run scientific sweep，也没有冻结正式训练
protocol 或报告 scientific performance 结论。

## 1. 前置审计

### Baseline hidden width

在实现前核对了官方 OGBench baseline
`/home/eai/Research/offline_rl_baselines/ogbench` 以及 RLC 当前 agent config：

| agent/slot | official actor hidden dims | RLC hidden dims |
|---|---:|---:|
| HIQL `high_actor` | `(512, 512, 512)` | `(512, 512, 512)` |
| HIQL `low_actor` | `(512, 512, 512)` | `(512, 512, 512)` |
| CRL `actor` | `(512, 512, 512)` | `(512, 512, 512)` |

因此 M9 的 `state_dim=512` 是由 RLC/baseline actor hidden width 决定的，
不是从 HRM 的 hidden size 倒推出来的。canonical Study 对 actor hidden dims
做了 fail-fast 校验，避免通过 `--width` 或 `--depth` 改变科学变量。

### Real AntMaze shapes

使用 medium/large 的真实 Dataset sample 做了初始化与 shape audit，没有训练：

| slot | raw actor input | actor readout output | hidden body |
|---|---:|---:|---:|
| HIQL `high_actor` | `58` (`state + goal`) | `10` (`rep_dim`) | `(512,512,512)` |
| HIQL `low_actor` | `39` (`state + encoded subgoal`) | `8` (action) | `(512,512,512)` |
| CRL `actor` | `58` (`state + goal`) | `8` (action) | `(512,512,512)` |

这两个环境的 observation/action shape 均为 `(1,29)` 与 `(1,8)`。HIQL high
actor 的 output `10` 是 subgoal representation 维度，不是 hidden width；CRL
的 `latent_dim=512` 也不是 actor hidden width。

### HRM-mini 对照审计

审计了 `/home/eai/Research/hrm-mini/arch/rt.py`、`arch/hrm.py`、
`arch/layers.py` 和 tuned configs。RT 的核心模式是：输入 embedding 得到
`x`，从 persistent `z` 开始，重复执行共享 core `z = core(z + x)`，最后返回
更新后的状态；tuned config 使用 hidden size `512`、cycles `7`。HRM 的 `z_init`
是 `nn.Buffer(trunc_normal_init_(...))`，并有 persistent carry。

M9 只借鉴“同一输入上的共享迭代 update”这一抽象。M9 使用独立的 normal
buffer、`K ∈ {1,2,4}` 和 decision-local reset，不复制 HRM 的双状态或其它
架构语义。

## 2. SingleState 定义

对 raw actor input `x ∈ R^{D_in}`，先执行一个输入映射：

```text
x_hidden = InputMLP(x)              # D_in -> 512, Dense + GELU
z_0 = broadcast(z_init)             # [512], non-trainable buffer
z_{t+1} = F(z_t + x_hidden)          # non-residual
z_{t+1} = z_t + F(z_t + x_hidden)    # residual
```

其中：

- `F` 是一个物理上唯一、所有 iteration 共享参数的 `Dense(512) + GELU +
  Dense(512) + GELU` update module；
- `K=1,2,4` 只改变同一个 module 的执行次数，不复制参数；
- 每次 `__call__` 都从保存的 `z_init` broadcast 一个 local state，调用之间
  不写回 buffer，也不跨 environment step 传递 state；
- `z_init` 在 Flax `buffers` collection 中，由独立 `buffers` RNG 初始化，保存在
  checkpoint 中，不进入 optimizer；
- 当前 M9 禁止 layer norm，保持 actor computation 的 MLP 语义；
- `direct` credit 使用普通 JAX differentiation，因此 K 个内部 update 采用
  full BPTT，且没有 truncation。

`impls/computation/topologies/single_state.py` 只处理 representation
transformation，不知道 HIQL/CRL 的 loss、goal semantics 或 readout semantics。
actor 的 distribution/readout 仍由各自 network 保留。

### K=1 baseline decomposition

当 `z_init=0` 且 non-residual、`K=1` 时：

```text
InputMLP (D -> 512) + F (512 -> 512 -> 512)
```

正好对应原始 actor 的三个 hidden layers；最后的 actor readout 仍由原网络
提供。因此这个 decomposition 被作为 baseline zero-state 单元测试。

## 3. Flax state 与 checkpoint

`TrainState` 新增通用 `model_state` 字段，baseline 没有非参数 collection 时
仍使用 `{}`。Network initialization 保留 legacy `params` RNG path，额外的
`buffers` RNG 只用于 `z_init`，不会扰动 baseline 参数初始化。

checkpoint 会保存完整 agent state，包括 params、optimizer、RNG 和
`model_state/buffers`。restore 对 M9 前的旧 checkpoint 自动补空的
`model_state={}`，保证 baseline checkpoint 兼容。

HIQL 的 high/low actor 按官方语义继续是两个独立的 GCActor。M9 只在选中的
slot 上放置 SingleState；没有人为引入 shared actor core。`high_actor+low_actor`
placement 会创建两个独立的参数 subtree 与两个独立的 `z_init` buffer。

## 4. 参数 accounting

设 `h=512`、raw input 为 `D`、readout output 为 `A`。原始 actor 的可训练
参数量为：

```text
P_vanilla = (D*h + h) + 2*(h*h + h) + (h*A + A)
```

SingleState 的 trainable 参数量相同：输入映射和共享 update module 合计
原始 body 的参数，readout 仍在 actor network 中；额外增加的只有 `h=512`
个 non-trainable buffer elements。K=1、K=2、K=4 和 residual/non-residual
均不复制 update 参数。

当前已实现的 accounting metadata 对每个 actor slot 记录：
`trainable_params`、`core_trainable_params`、`vanilla_actor_trainable_params`、
`buffer_elements`、`state_dim`、`iterations` 和 `shared_update_executions`。

对应真实 shape 的静态期望值为：

| slot | `D` | `A` | trainable params | core params | buffer elements |
|---|---:|---:|---:|---:|---:|
| HIQL high actor | 58 | 10 | 560650 | 555520 | 512 |
| HIQL low actor | 39 | 8 | 549896 | 545792 | 512 |
| CRL actor | 58 | 8 | 559624 | 555520 | 512 |

## 5. Placement 与 M9A 配置矩阵

`experiments/M9A_single_state_iteration/` 现在包含 26 个 configuration：

| group | configurations |
|---|---:|
| HIQL baseline | 1 |
| CRL baseline | 1 |
| CRL actor：`K=1,2,4` × no-residual/residual | 6 |
| HIQL high actor：`K=1,2,4` × no-residual/residual | 6 |
| HIQL low actor：`K=1,2,4` × no-residual/residual | 6 |
| HIQL high + low actor：`K=1,2,4` × no-residual/residual | 6 |
| **total** | **26** |

每个 configuration 都显式写出 `agent_overrides`，由 override 真正控制
`compute.actor` 或 HIQL 的 `compute.high_actor`/`compute.low_actor`，而不是
通过 slug 推断。Study 环境为：

```text
antmaze-medium-navigate-v0
antmaze-large-navigate-v0
```

seed 为 `[0]`，所以 manifest 共有 `26 × 2 × 1 = 52` 个 planned rows。
`tools/sweep.py --dry-run` 会显示全部 52 个路径；默认不执行，只有显式
`--execute` 才会启动，并且每个 GPU 同时只派发一个 job。已完成的 run 会跳过，
failed/invalid/running 状态会保留并不会被静默重跑。

## 6. 代码与文件

- `impls/computation/topologies/single_state.py`：SingleState topology；
- `impls/computation/factory.py`：`topology_kwargs` 与参数校验；
- `impls/utils/flax_utils.py`：TrainState model state 和 checkpoint 兼容；
- `impls/agents/hiql.py`、`impls/agents/crl.py`：独立 buffer RNG 与 state 提取；
- `impls/main.py`：Study overrides、canonical fail-fast、metadata、accounting；
- `tools/run.py`、`tools/sweep.py`：canonical launch 与 GPU worker；
- `experiments/M9A_single_state_iteration/`：26 个配置、manifest、aggregation；
- `tests/computation/test_single_state.py`：topology、RNG、immutability、共享、
  residual、gradient 与 K=1 decomposition tests；
- `tests/integration/test_m9_single_state_study.py`：配置矩阵、slot placement、
  high/low 独立 buffers 与 canonical path tests。

## 7. 验证记录

已完成的验证分为以下层次：

1. 单元层：SingleState dimensions、buffer seed 独立性、forward 不变性、
   residual/non-residual 方程、K 参数共享、full-BPTT gradient 和 zero-state
   K=1 decomposition；
2. 真实 CPU vertical slice：CRL actor `K=1` non-residual，真实
   `antmaze-medium-navigate-v0` Dataset，1 update、finite loss、evaluation、
   checkpoint save/restore action/value probe；
3. 真实 shape audit：HIQL high/low 与 CRL actor 的 medium/large 输入输出维度；
4. GPU smoke：只覆盖短 smoke，不代表正式训练结论。5 个 smoke 均使用真实
   `antmaze-medium-navigate-v0` 数据、`train_steps=1`、`batch_size=8`，并且
   metadata 记录 `jax_backend=gpu`、`cuda:0`：

   - CRL actor `K=1` non-residual：pass，loss `2.721940`；
   - HIQL high actor `K=1` non-residual：pass，loss `78.750275`；
   - HIQL low actor `K=1` non-residual：pass，loss `77.774712`；
   - HIQL high+low actor `K=1` non-residual：pass，loss `78.871475`；
   - HIQL high+low actor `K=4` residual：pass，loss `90.371124`。

   每个 smoke 都完成 finite update、evaluation 和 checkpoint save/restore
   action/value probe。一次 sandbox 内的 CUDA 尝试因 `CUDA_ERROR_NO_DEVICE`
   回退到 CPU；受控环境执行后 GPU backend 正常可见，故最终 GPU 记录以
   上述 `jax_backend=gpu` 结果为准。

完整 CPU regression 为 `65/65 PASS`，包含既有回归、SingleState topology
和 M9A Study placement 测试。M9A manifest 验证为 `52` 个 planned rows，
`aggregated.csv` 当前没有伪造的 numeric result。

当前没有根据短 smoke 推断 success improvement，也没有宣称任何 K 或 residual
variant 优于 baseline。

## 8. Protocol 记录与后续边界

当前 RLC runtime 中已经存在的共同设置为：

| setting | current runtime value | role in this milestone |
|---|---:|---|
| HIQL/CRL optimizer learning rate | `3e-4` | baseline agent default |
| HIQL/CRL batch size | `1024` | baseline agent default |
| launcher `train_steps` default | `1000` | infrastructure smoke/default only |
| launcher eval/save interval default | `1000` | infrastructure smoke/default only |
| launcher eval tasks/episodes default | `1 / 1` | lightweight runtime check |
| M9A planned seeds | `[0]` | matrix placeholder, not statistical claim |

本次 1-step smoke 为了快速验证使用了显式 `batch_size=8`、
`eval_tasks=1`、`eval_episodes=1` 和 `save_interval=1`，这些参数不是正式
实验协议。

正式 M9A 的建议 protocol 是：所有 baseline 与 SingleState configuration
使用同一环境、同一 dataset split、同一 optimizer/batch/lr、同一训练预算、
同一 evaluation/checkpoint schedule；只改变 Study 中声明的 actor placement、
`K` 和 residual。HIQL baseline 每个环境只运行一次，CRL baseline 与 CRL
actor placement 使用同样的 runtime 流程。最终 primary metric 使用 manifest
中声明的 `evaluation/overall_success`，并在正式报告前增加确认后的多 seed
设置和统计汇总。

以上是 protocol 约束建议，不是我替研究者冻结的 paper-facing 数值。正式
`train_steps`、`eval_interval`、`eval_tasks`、`eval_episodes`、
`save_interval`、seed 数量、GPU allocation、retry policy 和统计分析仍需
单独确认；在确认前不启动完整 52-run Study。
