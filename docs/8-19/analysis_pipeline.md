# Reevaluation analysis pipeline

本项目的 reevaluation 分析采用“不可变原始结果 + 可重建派生结果”的方式。分析程序只读取 reevaluation 目录，不会覆盖 episode、task summary、checkpoint 或训练产物。

## 目录与入口

通用实现位于：

- `impls/analysis/schema.py`：分析规格、canonical schema、figure/view 注册表、axis range 校验和 spec fingerprint；
- `impls/analysis/loaders.py`：campaign、manifest、配置 YAML、run metadata、episode/task/summary 的只读加载与一致性验证；
- `impls/analysis/statistics.py`：训练 seed 层面的均值、population SD、sample SD 以及严格 paired join；
- `impls/analysis/allocation.py`：allocation、task、task-group、reference delta 和 paired comparison 派生表；
- `impls/analysis/plotting.py`：集中式绘图风格、稳定 figure ID、声明式 view 应用和范围安全校验；
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

## Figure ID 与 view ID

`figure_id` 表示图所表达的科学语义；`view_id` 表示同一科学数据的视觉尺度或 panel arrangement。二者正交，输出使用稳定格式：

```text
<figure_id>__<view_id>.pdf
<figure_id>__<view_id>.png
<figure_id>__<view_id>.csv
```

因此 full 与 zoom 是同一 figure 的不同 view，而不是新的 statistical analysis。所有 view 使用完全相同的 scientific rows、seed、mean、SD、reference 和 task grouping；zoom 只改变 axis range，split view 只改变 panel arrangement。

当前稳定 figure ID 为：

1. `allocation_response_overall`
2. `allocation_response_by_task`
3. `allocation_response_focal_remaining`
4. `allocation_seed_consistency`
5. `task_delta_vs_reference`

M10A-A001 当前声明的 views 为：

- overall：`full=[0.0,1.0]`、`zoom=[0.75,0.95]`；
- seed consistency：`full=[0.0,1.0]`、`zoom=[0.75,0.95]`；
- by task：保留五 panel 共用的 `full=[0.0,1.0]`；
- focal/remaining：`full=[0.0,1.0]`，以及由 YAML task groups 驱动的 `split_zoom`：`focal_task=[0.15,0.90]`、`remaining_tasks=[0.84,0.98]`；
- task delta：`full=[-0.75,0.75]`，以及 `split_zoom`：`focal_task=[-0.70,0.70]`、`remaining_tasks=[-0.18,0.18]`。

task-delta 正式数据的 seed-level 范围为 `[-0.63,0.68]`，所以 full view 采用固定、对称的 `[-0.75,0.75]`。remaining-task delta 的正式最小值为 `-0.16`，因此 prompt 中建议的 `[-0.15,0.15]` 会造成 silent clipping；最终声明为 `[-0.18,0.18]`，并保留固定余量。

## Declarative axis ranges

所有正式 axis range 必须预先写入 analysis YAML。schema 会验证 `y_min < y_max`、finite 数值、success-rate 范围在 `[0,1]` 内，以及 delta 范围在 `[-1,1]` 内。未知 figure、view、panel 或字段会直接失败。

正式绘图禁止使用 data-driven auto-zoom，例如根据 `min(data)` 或 `max(data)` 动态生成范围。这样不同实验可以比较相同视觉尺度，也避免根据当前结果事后夸大差异。如果数据点或 error-bar 区间超出声明范围，pipeline 会报告具体 figure/view、panel、config、seed、task 和字段，并停止绘图，不会自动扩大范围或静默裁剪。

`full` view 是完整 context；zoom view 会在图中标注 `Zoomed y-axis`。`analysis_metadata.json` 会记录每个 rendered view 的 `figure_id`、`view_id`、`y_axis`、`panel_layout`、`y_axis_truncated`、natural data range、source table 和输出文件。

focal/remaining 与 delta 的 split view 使用两个横向 panel。panel membership 只来自 YAML 的 `task_groups`，generic plotting code 不硬编码 task2，也不将 task 标记为 hard/easy/long-horizon。delta split 两个 panel 都保留 `y=0` reference line，remaining panel 仍保留 task1/task3/task4/task5 的独立 task identity 和颜色。

横轴为 `high_fraction`，预算通过语义稳定的 B=2/B=5/B=16 样式区分；B=5 与 B=16 明确作为不同 budget series。图中不拟合曲线，也不插入未测量 allocation 点。

每个 rendered view 都有同名 CSV。full/zoom CSV 的 scientific rows 完全相同，不因 zoom 过滤数据；split panel CSV 额外增加 `panel_id`，仅表达 panel membership，不改变任何 metric value。

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

如果 canonical analysis output 已存在，程序默认拒绝覆盖。确认目标目录只包含可重建的派生分析结果后，可使用安全 rebuild：

```bash
PYTHONPATH=. python3 tools/analyze_reevaluation.py \
  --spec experiments/M10A_fixed_budget_placement/analyses/M10A-A001.yaml \
  --repo-root . --execute --rebuild \
  --output-root /data/qijunrong/06-RL/offline-rl/exp/RLC/analyses
```

`--rebuild` 只允许在目标目录存在且其中 `analysis_metadata.json` 的 study/analysis/reevaluation provenance 匹配时重建；它不会触碰 source reevaluation。

在开发或测试阶段，可以显式使用 `--smoke` 允许 dirty worktree；这不会改变源 reevaluation，但 metadata 会记录 `execution_mode=smoke` 和 dirty 状态：

```bash
PYTHONPATH=. python3 tools/analyze_reevaluation.py \
  --spec experiments/M10A_fixed_budget_placement/analyses/M10A-A001.yaml \
  --repo-root . --execute --smoke \
  --output-root /tmp/rlc-analysis-smoke
```

旧的 analysis spec 如果仍使用 `figures: [figure_id, ...]` 列表，会向后兼容地为每个 figure 生成声明式 `full` view；success figure 默认 `[0,1]`，delta figure 默认 `[-1,1]`。新的 spec 应使用 mapping 形式显式声明 views。

未来 M10B/M11 只需在各自 analysis YAML 中声明相同 figure ID 的 views，或在 schema 支持的范围内加入新的稳定 view ID；统计表和 scientific data 不需要复制，plotting code 也不需要加入实验特定的 autoscale 分支。
