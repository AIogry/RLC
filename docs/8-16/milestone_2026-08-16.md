# Milestone：M9 计算架构与正式实验基础设施

日期：2026-08-16

## 总结

本阶段完成了 RLC M9A SingleState、M9B TwoState 的 actor computation
vertical slice，并将项目推进到可以进行正式实验冻结与 dry-run 审计的状态。

本阶段没有启动完整 M9A/M9B scientific sweep，没有生成正式性能结论，也没有
修改 HIQL、CRL、SingleState 或 TwoState 的科学定义。

## 一、M9A SingleState

M9A 实现了 decision-local SingleState actor computation：

- `K ∈ {1, 2, 4}`；
- shared MLP update module；
- residual/non-residual 两种更新；
- full-BPTT direct credit；
- normal non-trainable buffer 初始化；
- state 每次 decision reset，不跨 environment step carry；
- checkpoint 保存并恢复 `model_state/buffers`。

HIQL high actor、low actor 仍然是官方语义下的两个独立 GCActor。高低 actor
同时启用时使用独立参数树与独立 buffer，没有引入 shared core。

M9A Study inventory：

```text
HIQL baseline                         1 config
CRL baseline                          1 config
CRL actor：K=1/2/4 × residual/nores   6 configs
HIQL high actor                       6 configs
HIQL low actor                        6 configs
HIQL high+low actor                   6 configs
总计                                  26 configs
```

两个 AntMaze 环境、seed 0 共同展开为 `26 × 2 = 52` 个 planned Run identities。

## 二、M9B TwoState

M9B 在 actor-only computation 基础上实现了 decision-local TwoState：

- 独立 H/L GELU MLP update；
- `H2L1` 与 `H2L6` 两种 schedule；
- `full_bptt` 与 `one_step` 两种 credit policy；
- H/L 独立 normal buffers；
- 最终 actor representation 使用 `z_H`；
- 不引入 HRM 的 Transformer、Attention、SwiGLU、RMSNorm 或跨环境 carry。

M9B Study inventory：

```text
CRL actor                         4 configs
HIQL high actor                   4 configs
HIQL low actor                    4 configs
HIQL high+low actor               4 configs
总计                              16 configs
```

16 个 configuration 在两个环境、seed 0 下展开为 `32` 个 planned Run identities。
M9B 通过 `baseline_reference` 引用 M9A 的 `M9A-C001` HIQL baseline 和
`M9A-C002` CRL baseline，不重复定义 baseline。

M9A 与 M9B 合计为：

```text
84 planned Run identities
```

## 三、正式实验 protocol 冻结

第一轮正式实验共同设置为：

```text
batch_size       = 1024
learning_rate    = 3e-4
log_interval     = 5000
eval_interval    = 100000
eval_tasks       = all
eval_episodes    = 20
eval_temperature = 0
eval_gaussian    = None
seed             = 0
environments     = antmaze-medium-navigate-v0
                   antmaze-large-navigate-v0
```

Baseline（HIQL vanilla、CRL vanilla）：

```text
train_steps   = 1,000,000
save_interval = 1,000,000
```

SingleState/TwoState exploration variants：

```text
train_steps   = 500,000
save_interval = 500,000
```

正式结果的主要 matched-budget comparison 定义为：

```text
Variant @ 500k
vs corresponding Vanilla Baseline @ 500k
```

Baseline 继续训练到 1M，并保留 `Baseline @ 1M` 作为完整训练 reference，不能
把 `Variant @ 500k` 与 `Baseline @ 1M` 作为主要等预算比较。

正式 evaluation points 为：

```text
Variant @ 500k：100k, 200k, 300k, 400k, 500k
Baseline @ 1M：100k, 200k, ..., 900k, 1M
```

当前 RLC 主循环没有额外的 step-1 evaluation。Baseline 的 1M checkpoint 和
variant 的 500k checkpoint 都表示各自训练终点保存，不代表不同的 optimizer
或 optimization protocol。

完整 protocol 记录见
[`docs/experiment_execution.md`](experiment_execution.md)。

## 四、正式 experiment infrastructure

完成并冻结了 Study → Configuration → Run 的正式执行规范：

- canonical Run path 不含 timestamp，禁止 silent overwrite；
- `resolved_config.json` 保存最终配置及稳定 SHA-256 fingerprint；
- `runtime_metadata.json` 保存 Git、dataset、JAX device、protocol、compute
  slots、accounting 和 lifecycle status；
- failed/interrupted Run 保留 partial artifacts，不自动覆盖或静默重试；
- formal execution 要求 clean Git worktree 和 frozen commit；
- 建议通过 Git detached worktree 运行正式 sweep。

新增/完善的通用 launcher 能力：

```text
--configs ID1,ID2,...
--exclude-configs ID1,ID2,...
--dry-run
--execute
--log-interval N
```

`tools/sweep.py` 现在支持：

- config include/exclude filtering；
- 未知 config ID 与 include/exclude 冲突 fail fast；
- 每个 physical GPU 一个 persistent worker；
- 动态 free-GPU queue，避免同一 GPU 同时运行多个 Run；
- dataset train/validation 文件 preflight；
- filtered Study 的统一 summary、dry-run 和 execute inventory。

正式 launcher 文件为 [`scripts/run_study.sh`](../scripts/run_study.sh)，执行
规范为 [`docs/experiment_execution.md`](experiment_execution.md)。

## 五、验证结果

本阶段完成：

- M9A/M9B manifest inventory：`52` / `32` planned rows；
- M9A baseline filtering dry-run：`4` runs；
- M9A exploration filtering dry-run：`48` runs；
- M9B full dry-run：`32` runs；
- 实际 OGBench dataset train/validation preflight 通过；
- 新增 filtering、scheduler、launcher mock tests：`7/7 PASS`；
- 完整 CPU regression：`84/84 PASS`；
- `bash -n scripts/run_study.sh`、`compileall`、`git diff --check` 全部通过。

此前的 M9A 五个与 M9B 五个 GPU 1-step real-data smoke 均通过，并完成 finite
update、evaluation、checkpoint restore 和 provenance 检查。这些 smoke 只验证
runtime vertical slice，不属于正式 scientific results。

## 六、当前状态与边界

当前 HEAD 为：

```text
f30b64bf81e1738235eef4f213d3019820ee918a
8-16 RLC: fix formal experiment launcher
```

当前正式实验尚未启动。后续启动前仍需：

1. 确认并保留 frozen commit；
2. 使用 clean detached worktree；
3. 执行 M9A/M9B dry-run 并核对 inventory；
4. 确认 dataset、run root、双 GPU 和 baseline reuse 条件；
5. 仅在 checklist 完成后使用 `--execute`。

seed 0 仍是 exploratory seed。当前结果不能描述为 statistically significant、
robust improvement 或 final paper result；正式结构筛选后还需要 multi-seed
confirmation。
