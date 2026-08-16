# Milestone：M8 Computation Foundation 与 M9 Experiment Management Foundation

日期：2026-08-15

## Milestone 结论

今天完成了 RLC 的 computation foundation 收尾，以及 M9 scientific
exploration 开始前的 experiment management foundation。当前项目已经能够
用稳定的 Study → Configuration → Run 身份管理实验，并自动保存 resolved
configuration、runtime provenance、训练/评估 CSV、summary 和 checkpoints。

当前结论：

> RLC 已具备可复现、可聚合、可追溯的文件型实验管理基础；M9A 的
> single-state iteration 目前只建立了 planned Study，没有实现 topology，
> 没有启动 scientific training。

本 milestone 不包含 GPU scientific experiment、长时间训练、SingleState、
RNN、HRM、multi-state 或新的 computation semantics。

## 一、M8 Computation Foundation

### 1. Ontology 与边界

冻结并记录以下 computation ontology：

```text
Operator → Primitive → Block
```

同时明确以下独立维度：

- State Structure
- Topology
- Execution Schedule
- Parameter Reuse
- Credit Structure

Computation 仍然位于 Agent / Network / Computation / Compute Slot 边界内，
没有把 scientific algorithm semantics 混入 experiment configuration。

### 2. State contract

`ComputationCore` 现在接受可选 `state`，并将 state 语义委托给 topology。
当前 `FeedForward` topology 明确拒绝非空 state，保持现有无状态行为。

没有实现 single-state iteration、跨 environment step state、RNN 或 HRM。
CoGHP 的 autoregressive 行为仍然保留在其原有 network/algorithm semantics
中。

### 3. MLPMixerBlock parity

新增独立的 computation-side `MLPMixerBlock`，严格镜像官方 CoGHP
`MixerBlock` 的参数结构、参数数量、forward 输出和 full gradient。
验证覆盖多个 token、embedding 和 hidden-dimension 配置。CoGHP 生产路径
仍继续使用原有 `impls/networks/coghp.py` 中的官方 reference block，尚未
切换 production import。

### 4. Baseline defaults 与 provenance

CRL 的 actor、critic state/goal、AWR value state/goal computation slots 默认
均关闭，`--computation` 仍作为显式迁移快捷方式。HIQL、CRL、CoGHP 的原有
network、loss、RNG、target/update 和参数边界没有被 experiment layer 改变。

M8 的 `runtime_metadata.json` 保留原有 `computation` 和 `compute_slots`
字段，并作为 M9 provenance 的基础。

## 二、Experiment Management Foundation

### 1. Study → Configuration → Run

新增 [impls/experiment/management.py](../impls/experiment/management.py)，
定义：

- Study：一个 scientific question；
- Configuration：一组 scientific factors，不包含 seed；
- Run：Configuration + Environment + Seed + Git commit。

Configuration 使用稳定 `config_id` 和可读 slug，并拒绝依赖
`final/new/best/v2/try2` 等可变名称。

### 2. 稳定 Run identity

canonical Run 路径为：

```text
runs/<study_id>/<config_id>__<slug>/<environment>/seed_<NNN>/
```

该路径不使用 timestamp 作为实验身份。相同 identity 重复创建时
fail-fast。旧的 `--save_dir` 和 `runs/legacy/` 仅保留为 debug/compatibility
路径，不作为 canonical scientific Run。

### 3. Run artifacts

每个 Run 自动保留：

```text
resolved_config.json
runtime_metadata.json
train.csv
eval.csv
summary.json
checkpoints/
```

`resolved_config.json` 包含 Study、Configuration、launcher 参数和 resolved
agent config。

`runtime_metadata.json` 包含：

```text
study_id
config_id
config_slug
algorithm
environment
seed
git_commit
git_dirty
start_time
end_time
hostname
jax_backend
jax_device_descriptions
dataset_identity
dataset_dir
computation
compute_slots
status
```

同时保留 M8 兼容字段 `agent`、`ogbench_module` 等。

### 4. Lifecycle 与 failure retention

支持以下状态：

```text
planned / running / completed / failed / aborted / invalid
```

失败或中断时不会删除 partial artifacts，会保留 metadata、CSV、checkpoint，
并写入 `failure.json` 与 failure reason。

`summary.json` 从显式的 `eval.csv` success 字段生成：

```text
evaluation/overall_success
overall_success
success
```

字段不存在时保持 `null`，不进行推测。

### 5. Manifest 与 aggregation

新增：

- [tools/manifest.py](../tools/manifest.py)
- [tools/aggregate_results.py](../tools/aggregate_results.py)
- [tools/summarize.py](../tools/summarize.py)
- [tools/run.py](../tools/run.py)

Manifest 至少包含：

```text
study_id, config_id, slug, algorithm, placement, topology, block,
iterations, residual, environment, seed, git_commit, status, run_dir,
final_success, best_success, best_step
```

Aggregation 只读取 manifest/CSV，按 `config_id + environment` 输出
`count / mean / std`，不会修改 raw artifacts，也没有引入 SQL、W&B、MLflow
或数据库服务。

## 三、M9A planned Study

新增：

```text
experiments/M9A_single_state_iteration/
├── study.yaml
├── configs/
│   ├── M9A-C001.yaml
│   ├── M9A-C002.yaml
│   └── M9A-C003.yaml
├── manifest.csv
└── aggregated.csv
```

M9A scientific intent：

- Question：decision-local single-state iterative computation 是否改善 GCRL；
- Primary factors：internal iterations `K`、residual vs non-residual；
- Fixed：decision-local state、non-learned state、`z+x` input injection、MLP
  update module、shared parameters、不跨 environment step；
- Deferred：state initialization alternatives、其他 injection、gating、
  normalization recipe、multi-state、HRM。

三个 configuration 当前均标记为 `executable: false`。通过 launcher 运行时会
明确拒绝启动，以防在 SingleState implementation 尚不存在时误运行。

示例 planned manifest row：

```text
M9A,M9A-C002,k2_no_residual,,,,,2,False,
antmaze-medium-navigate-v0,0,,planned,
runs/M9A/M9A-C002__k2_no_residual/antmaze-medium-navigate-v0/seed_000,,,
```

由于当前没有 completed runs，M9A 的 `aggregated.csv` 只有表头，没有伪造
任何 mean/std 结果。

## 四、文档与项目配置

新增或更新：

- [docs/experiment_management.md](experiment_management.md)
- [docs/experiment_management_audit.md](experiment_management_audit.md)
- [README.md](../README.md)
- `.gitignore`：忽略 `runs/` 和旧 `exp/` artifact roots；
- `pyproject.toml`：声明 `pyyaml` 依赖；
- `impls/utils/flax_utils.py`：兼容 canonical `checkpoints/` 子目录恢复。

原则明确为：

> filenames are for readability; metadata is the source of experimental truth.

## 五、验证结果

### Experiment management tests

共 `8/8 PASS`，覆盖 Study/config parsing、stable run identity、duplicate
identity fail-fast、resolved config JSON serialization、Git metadata helper、
runtime metadata、synthetic eval summary、manifest construction、failed run
retention、aggregation mean/std，以及缺失 success 字段时的透明处理。

### Existing regression

最终 CPU discovery：

```text
Ran 54 tests
OK
```

其中包含既有 HIQL、CRL、CoGHP computation/runtime regression，以及新的
experiment-management tests。真实 `antmaze-medium-navigate-v0` 数据上的
CRL 1-step CPU runtime smoke 也通过，生成了完整 run artifacts，并通过
checkpoint action/value save/restore probe。

此外通过了：

- `git diff --check`；
- Python `py_compile`；
- M9A planned launcher guard；
- manifest 和 aggregation CLI smoke。

## 六、未完成与后续工作

本 milestone 明确未做：

- SingleState iterative topology 实现；
- state initialization / injection / gating / normalization scientific study；
- M9A 正式训练、多 seed 运行和成功率比较；
- GPU scientific experiment；
- W&B、MLflow、SQL 或其他外部 tracking service；
- compute-budget scalar；
- 修改 HIQL/CRL/CoGHP canonical semantics。

下一阶段应先在独立的 M9 implementation milestone 中实现并 parity-validate
SingleState，再允许 M9A configuration 从 planned 转为 executable。

## Milestone 状态

**Experiment management foundation：integration validated locally。**

这表示管理、metadata、manifest、summary、aggregation 和现有 baseline
regression 已在本地验证；不表示 M9 scientific hypothesis 已经验证。
