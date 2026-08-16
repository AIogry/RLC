# Milestone: HIQL computation and runtime validation

日期：2026-08-12

## Milestone conclusion

本日完成了 RLC HIQL computation migration 的真实 runtime 闭环，并完成了
GPU 上 1000 步的 controlled strict parity diagnostic。

当前结论是：

> HIQL computationized baseline 已达到 local integration validated；此前
> native legacy/computation 训练 loss gap 可以归因于两种不同 Flax
> parameter-tree scope 导致的 native initialization 差异，而不是
> computation migration 的数学或训练更新错误。

这不是正式 baseline reproduction，也没有启动 CRL、长训练或多 seed 实验。

## 今天完成的内容

### 1. HIQL computation slots

保持当前已经验证的三个 computation slots：

```text
high_actor = MLP + FeedForward + Direct
low_actor  = MLP + FeedForward + Direct
value      = MLP + FeedForward + Direct
```

同时保持：

- `goal_rep`、actor distribution/readout 和 value scalar readout 的原有语义；
- HIQL value loss、expectile loss、actor loss 和 Polyak target update；
- 两个 value ensemble members 的独立参数；
- `target_value` 自动镜像 online `value` architecture，不增加独立 slot。

### 2. 真实 OGBench runtime vertical slice

已打通：

```text
RLC/ogbench
  -> Dataset / GCDataset / HGCDataset
  -> HIQLAgent
  -> computation slots
  -> agent.update
  -> evaluation
  -> CSV logging
  -> checkpoint save/load
```

runtime 使用 RLC 自己的 canonical `ogbench`，没有用 upstream OGBench package
覆盖 `RLC/ogbench`。

Dataset runtime 保留了：

- trajectory boundaries 和 terminal indices；
- current/future/random goal sampling；
- `value_goals`；
- `low_actor_goals`；
- `high_actor_goals`；
- `high_actor_targets`；
- `rewards`、`masks`、`next_observations`；
- `subgoal_steps` 语义。

同时采用了 CoGHP fixed runtime 中已经验证过的显式 NumPy RNG stream 和
独立 evaluation seed stream。

### 3. 真实数据测试

测试数据集为：

```text
antmaze-medium-navigate-v0
```

确认实际导入的 benchmark package 为：

```text
/home/eai/Research/RLC/ogbench/__init__.py
```

真实数据测试结果：

- 同 seed 连续 20 个 `HGCDataset` batch：所有 required fields bitwise identical，
  最大误差 `0.0`；
- sampled indices 在数据范围内；
- future goals 不跨 episode；
- high-level targets 满足 `min(index + subgoal_steps, high_goal)`；
- batch shape、dtype 和 finite 检查通过；
- 真实 batch 上 legacy/computation strict parity N=20 通过。

### 4. CPU regression 和 trainer smoke

完整 CPU 测试：

```text
25/25 PASS
```

其中原有 22 个测试全部保持通过，新增 3 个真实 runtime 测试。

legacy 和 computation 两种 native trainer mode 都完成了：

```text
1000 steps
antmaze-medium-navigate-v0
batch_size=8
hidden width=6
depth=2
```

两种模式均完成了：

- JIT update；
- finite loss/diagnostics；
- evaluation hook；
- CSV logging；
- checkpoint save/load；
- action/value restore equality probe。

短 smoke 的 task-1 success 均为 `0.0`，不作为算法结果解释。

## GPU N=1000 controlled diagnostic

### 实验设置

使用完全相同的：

- 真实 OGBench sampled batch sequence；
- semantic online parameters；
- target parameters；
- optimizer state；
- agent RNG。

GPU/backend：

```text
JAX backend: gpu
Devices: cuda:0, cuda:1
GPU: 2 x NVIDIA GeForce RTX 4090
```

### Matched-initialization 结果

连续 1000 步后：

| 指标 | 最大绝对误差 |
|---|---:|
| total loss | `0.0` |
| value loss | `0.0` |
| high actor loss | `0.0` |
| low actor loss | `0.0` |
| semantic online parameters | `0.0` |
| target value parameters | `0.0` |
| optimizer state | `0.0` |
| agent RNG | `0.0` |
| `grad/max` | `0.0` |
| `grad/min` | `0.0` |
| `grad/norm` | `6.103515625e-05` |

`grad/norm` 的非零误差是 raw pytree leaf ordering 改变后的聚合 reduction
差异；没有观察到 loss、semantic parameter 或 target parameter 的训练轨迹
divergence。

first divergence：

```text
exact divergence: none
float32 tolerance divergence: none
semantic parameter divergence: none
```

### Native step-0 诊断

相同 seed、未做 semantic parameter graft 的 native initialization 已经不同：

| Loss | Legacy | Computation |
|---|---:|---:|
| total | `16.81831932` | `11.03039646` |
| value | `0.39081892` | `0.31889442` |
| high actor | `5.56704044` | `3.43273759` |
| low actor | `10.86046028` | `7.27876472` |

native step-0 total-loss absolute error 为 `5.78792286`。

native semantic initial parameters 的最大绝对误差为：

```text
online parameters: 1.65875196
target value parameters: 1.65875196
```

因此此前 native 1000-step loss gap 在 step 0 就已经存在；matched
initialization 的 GPU N=1000 实验又保持 exact parity，支持以下判断：

> native loss gap 主要来自不同 parameter-tree scope 造成的 initialization
> 差异，而不是 computation implementation 或 HIQL update migration 的错误。

## 相关文件

主要 runtime 文件：

- `impls/main.py`
- `impls/utils/datasets.py`
- `impls/utils/env_utils.py`
- `impls/utils/evaluation.py`
- `impls/utils/flax_utils.py`
- `impls/utils/log_utils.py`
- `impls/utils/reproducibility.py`

主要测试和诊断：

- `tests/integration/test_hiql_real_runtime.py`
- `tests/integration/test_hiql_smoke.py`
- `tools/diagnose_hiql_gpu_parity.py`

主要文档：

- `docs/runtime_migration_audit.md`
- `docs/hiql_runtime_integration.md`
- `docs/architecture_decisions.md`
- `docs/status.md`

## 明确未完成/未开始的工作

- 没有迁移 CRL 或 CoGHP；
- 没有修改 computation abstraction 或 HIQL loss；
- 没有修改 goal sampling semantics；
- 没有进行 success-rate comparison；
- 没有运行 100k/1M steps、multi-seed 或正式 baseline reproduction；
- 没有开始下一个 milestone。
