# Reevaluation analysis pipeline

本项目的 reevaluation 分析采用“不可变原始结果 + 可重建派生结果”的方式。分析程序只读取 reevaluation 目录，不会覆盖 episode、task summary、checkpoint 或训练产物。

## 目录与入口

通用实现位于：

- `impls/analysis/schema.py`：分析规格、canonical schema、图表注册表和 spec fingerprint；
- `impls/analysis/loaders.py`：campaign、manifest、配置 YAML、run metadata、episode/task/summary 的只读加载与一致性验证；
- `impls/analysis/statistics.py`：训练 seed 层面的均值、population SD、sample SD 以及严格 paired join；
- `impls/analysis/allocation.py`：allocation、task、task-group、reference delta 和 paired comparison 派生表；
- `impls/analysis/plotting.py`：集中式绘图风格和稳定图表 ID；
- `tools/analyze_reevaluation.py`：dry-run、smoke 和 formal execute 入口。

M10A-A001 的规格为 `experiments/M10A_fixed_budget_placement/analyses/M10A-A001.yaml`。其中 reference 是可配置的 `M10A-C002`，标签为 `within_parameterization_reference`；task group 只声明 `focal_task=[2]` 与 `remaining_tasks=[1,3,4,5]`，程序不会把任何 task 硬编码为“困难”或“长时程”。

## canonical table

`canonical_results.csv` 是 long-format。每个 episode 有 `metric=episode_success`，每个 task 有 `metric=task_success_rate`，每个 config/seed 有 `metric=overall_success`。公共字段至少包括：

`study_id, reevaluation_id, config_id, config_slug, environment, training_seed, checkpoint_step, checkpoint_sha256, budget, k_high, k_low, high_fraction, low_fraction, task_id, task_name, metric, value`。

`K_H`、`K_L`、`B` 和 allocation fraction 必须来自配置元数据；如果配置元数据存在显式值，loader 会校验 manifest、metadata 和比例关系，不会从 slug 推断。没有 allocation 参数的外部 baseline 可以保留空值，并从 allocation response 表中排除，但仍会保留在 canonical table 与 reference delta 中。

## 统计口径

- 训练 seed 是模型层面的独立重复单位；每个 seed 先得到一个 task/overall 指标，再跨 seed 汇总。
- `mean` 是 seed-level arithmetic mean；`population_sd` 分母为 `n`；`sample_sd` 分母为 `n-1`，`n<2` 时定义为 0。
- episode sampling uncertainty 只作为原始 reevaluation summary 的信息保留，不与 training-seed variability 混合，也不把 episode 行池化成独立模型重复。
- 图中保留原始 seed 点；均值的误差条默认使用 training-seed population SD。
- reference delta 定义为同一 `training_seed`、同一 `task_id` 的 target success 减 reference success，并随后跨 seed 汇总。
- paired comparison 严格按 `training_seed + task_id + paired_episode_id` 连接；任一侧缺失或重复都会失败，且不会使用 adaptive allocator。

## 图表注册表

每个稳定 ID 都生成 PDF、PNG 和“绘图实际使用的原始/统计数据”CSV：

1. `allocation_response_overall`
2. `allocation_response_by_task`
3. `allocation_response_focal_remaining`
4. `allocation_seed_consistency`
5. `task_delta_vs_reference`

横轴为 `high_fraction`，预算通过语义稳定的 B=2/B=5/B=16 样式区分；B=5 与 B=16 明确作为不同 budget series。图中不拟合曲线，也不插入未测量 allocation 点。

## 执行方式

只读验证：

```bash
PYTHONPATH=. python3 tools/analyze_reevaluation.py \
  --spec experiments/M10A_fixed_budget_placement/analyses/M10A-A001.yaml \
  --repo-root . --dry-run
```

正式执行要求 git worktree clean：

```bash
PYTHONPATH=. python3 tools/analyze_reevaluation.py \
  --spec experiments/M10A_fixed_budget_placement/analyses/M10A-A001.yaml \
  --repo-root . --execute \
  --output-root /data/qijunrong/06-RL/offline-rl/exp/RLC/analyses
```

在开发或测试阶段，可以显式使用 `--smoke` 允许 dirty worktree；这不会改变源 reevaluation，但 metadata 会记录 `execution_mode=smoke` 和 dirty 状态：

```bash
PYTHONPATH=. python3 tools/analyze_reevaluation.py \
  --spec experiments/M10A_fixed_budget_placement/analyses/M10A-A001.yaml \
  --repo-root . --execute --smoke \
  --output-root /tmp/rlc-analysis-smoke
```

