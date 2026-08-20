# RLC Research Handoff

本文件是另一个模型接手 RLC 时的第一阅读入口。当前项目不是一个可以任意重构的普通代码库，而是一个带有 baseline-fidelity、实验 provenance 和科学问题约束的研究平台。

## 当前状态

- 当前研究平台：RLC — RL Computation；
- 当前主线：HIQL、CRL、Vanilla CoGHP 的 computation boundary 已建立，M9/M10 实验和 reevaluation infrastructure 已形成；
- 当前正式分析：`M10A-A001`，来源为不可变的 `M10A-R001`；
- 当前 M10A formal analysis output：`/data/qijunrong/06-RL/offline-rl/exp/RLC/analyses/M10A/M10A-A001/`；
- 当前代码版本和 analysis provenance 必须以 `tools/handoff_doctor.py` 的实时检查为准，不要手工猜测；
- M10A 研究结果是描述性结果，不应仅凭 3 个 training seeds 做强 significance claim。

## 接手阅读顺序

1. `README.md`；
2. `handoff/design_invariants.md`；
3. `handoff/research_questions.md`；
4. `handoff/experiment_registry.yaml`；
5. `handoff/data_ledger.yaml`；
6. `docs/8-19/analysis_pipeline.md`；
7. 当前 study 的 `study.yaml`、config YAML 和 reevaluation YAML；
8. 只有在确定修改范围后，才阅读对应的 `impls/agents/`、`impls/networks/`、`impls/computation/` 或 `impls/experiment/` 文件。

接手后首先执行：

```bash
cd /home/eai/Research/RLC
PYTHONPATH=. python3 tools/handoff_doctor.py
```

在 doctor 通过、研究问题和不变量得到确认之前，不得启动训练、reevaluation 或修改 scientific data。

## 不可变对象

以下对象只读：

- `/data/qijunrong/06-RL/offline-rl/data/raw_ogbench/`；
- `/data/qijunrong/06-RL/offline-rl/exp/RLC/runs/`；
- `/data/qijunrong/06-RL/offline-rl/exp/RLC/reevaluations/`；
- checkpoint 文件；
- `episode_results.csv`、`task_summary.csv`、`summary.json`；
- 已冻结 analysis 的 canonical scientific fields。

analysis output 是可重建的 derived artifact。重建时必须使用已有的 provenance guard，不得手工删除 source reevaluation，也不得修改 canonical scientific values 来适应图表。

## 规范性要求

- Agent、Network、Computation、Compute Slot 的边界必须保持清晰；
- baseline 的 readout、goal encoding、loss、target update、dataset semantics 不得被实验性改动悄悄改变；
- HIQL 官方语义是两个独立的 high/low GCActor，不要默认引入 shared-core HIQL；
- training seed 是模型级独立重复单位，episode 不能直接池化为独立模型重复；
- allocation、topology、primitive、credit、training duration 等 scientific factors 必须在 Study/config 中显式声明；
- 图表 axis range 必须由 analysis YAML 声明，禁止 data-driven auto-zoom；
- 所有正式运行必须从 clean worktree、明确 protocol 和可追溯 checkpoint 开始；
- 任何“结论”必须区分：代码验证、实验观察、统计结论和待验证假设。

## 当前建议的下一步

1. 运行 `tools/handoff_doctor.py`，确认代码和 M10A formal analysis provenance 一致；
2. 提交 handoff 文档并创建 handoff tag；
3. 完成 Python/JAX/Flax/OGBench 环境 manifest；
4. 为 M9B 设计 training-duration/checkpoint-curve follow-up，区分 TwoState 方法效应和优化未充分训练；
5. 新模型先做只读分析或测试，再申请任何训练/reevaluation 扩展。

## 禁止的默认行为

- 不要从 plot 外观推断新的统计结论；
- 不要把 `M10A-C001` external baseline 和 `M10A-C002` within-parameterization reference 混为同一概念；
- 不要从 config slug 反推显式 scientific factor；
- 不要选择“当前最好”的 config 作为事后 reference；
- 不要在 dirty worktree 下冒充 formal result；
- 不要把 smoke result 写成 formal result；
- 不要因为另一个模型的默认偏好而改变项目既有拓扑或参数共享语义。
