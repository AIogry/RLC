# RLC architecture audit

审计日期：2026-08-11

## 结论摘要

RLC 当前处于“runtime 已放入、算法层尚未迁移”的状态。`RLC/ogbench` 是可运行时参考实现，`RLC/impls`、`RLC/tests` 和配置目录此前主要是项目骨架。第一阶段应保留这份 runtime，不从旧 OGBench 仓库覆盖它；算法层只按 slot 渐进迁移。

本地参考代码实际位于 `offline_rl_baselines/ogbench`，而不是文件中写的根目录 `/offline_rl_baselines/ogbench`。CoGHP 的 canonical runtime 位于 `CoGHP/ogbench`。

## 当前目录状态

当前 RLC 的重要部分如下：

```text
RLC/
├── impls/                 # 算法、网络、computation 的目标位置；初始文件为空
├── configs/               # agents/experiments/sweeps，尚无配置内容
├── tests/                 # reference/computation/integration，尚无测试内容
├── tools/                 # run/sweep/summarize/compare 占位
├── autoresearch/          # AGENTS.md、contract.yaml、campaigns
├── docs/                  # 初始文档占位
├── ogbench/               # 已复制的 canonical 环境/runtime
└── pyproject.toml
```

`RLC/ogbench` 目前包含 OGBench 的 locomaze、manipspace、powderworld、relabel 和 dataset loading 代码；它不是算法实现，也不应与 `impls/utils` 中的训练 dataset wrapper 混为一谈。

## 来源与保留策略

### 应保留：RLC canonical runtime

逐文件比较显示：

```text
RLC/ogbench == CoGHP/ogbench
```

这份代码包含此前 CoGHP 项目中使用的 runtime 修复。与官方 `offline_rl_baselines/ogbench/ogbench` 相比，重要差异包括：

- `utils.py` 的默认数据集目录由 `OGBENCH_DATASET_DIR` 环境变量（及本地 canonical fallback）控制；
- `make_env_and_datasets` 的接口已收敛到当前训练流程，不再暴露旧的 `dataset_path`、`dataset_only` 和 `cur_env` 参数；
- locomaze、manipspace 等环境文件保留了 CoGHP 版本的 seed/runtime 修复。

因此，后续训练应导入 `RLC/ogbench`，而不是把官方旧 runtime 复制进 RLC 覆盖现有代码。

### 作为 algorithm reference 阅读，但不整体复制

`offline_rl_baselines/ogbench/impls` 提供 CRL、HIQL、GCBC、GCIQL、GCIVL、QRL、SAC 及其网络、dataset wrapper、evaluation 和 checkpoint 代码。第一阶段只需要以 CRL/HIQL 为语义参考；不应无差别复制所有 agent。

应优先复用或精确对齐的内容：

- `MLP` 的层顺序、GELU、`activate_final`、LayerNorm 位置和默认初始化；
- `GCEncoder` 的 state/goal/concat 组合语义；
- `GCActor` 的 Gaussian/Categorical distribution head 语义；
- HIQL 的 value、advantage、AWR 和 target update 语义；
- CRL 的 contrastive loss、bilinear value 语义。

不应在 computation 层复制的内容：

- goal sampling 和 reward/mask 构造；
- expectile/AWR/contrastive loss；
- Gaussian distribution construction、`LengthNormalize` 和最终 task-specific readout；
- OGBench environment registration 和 dataset relabeling。

## 当前 training/evaluation/data/RNG/checkpoint 路径

官方 baseline 的 `impls/main.py` 执行顺序是：

1. `utils.env_utils.make_env_and_datasets` 创建环境并通过 `ogbench.make_env_and_datasets` 加载数据；
2. `Dataset`、`GCDataset` 或 `HGCDataset` 负责 batch 和 goal sampling；
3. `random.seed` 与 `np.random.seed` 初始化 Python/NumPy 随机流，agent 使用 JAX `PRNGKey`；
4. `Agent.create` 初始化 Flax `ModuleDict` 和 Optax optimizer；
5. `TrainState.apply_loss_fn` 计算梯度并更新参数；
6. `evaluate` 负责 seeded rollout；
7. `save_agent` 将 Flax state dict 以 `params_<epoch>.pkl` 写入实验目录。

官方默认数据下载目录是 `~/.ogbench/data`。CoGHP/RLC canonical runtime 使用 `OGBENCH_DATASET_DIR` 覆盖目录并保持本地实验数据布局。RLC 当前尚未有自己的 `main.py` 训练入口，因此本轮不重写这条流程；迁移代码应保持这些算法和 runtime 边界。

## 哪些文件不应继续复制

- 不要复制官方 `ogbench/` 覆盖 `RLC/ogbench/`；canonical runtime 以 RLC 当前版本为准。
- 不要复制所有 OGBench agent 作为 RLC agent；Agent 表示学习算法，不表示计算结构组合。
- 不要将每个 computation 变体写成 `HRMHIQLAgent`、`MixerCRLAgent` 等组合类。
- 不要把 `GCActor` 的 distribution head、`GCValue` 的 final readout、`LengthNormalize` 或 loss 视为可替换 computation body。
- 不要把 seed、dataset sampling、evaluation 的修复重新实现成全局 `np.random` 状态；后续应沿用 canonical runtime 的复现语义。

## 推荐最终结构

```text
RLC/
├── impls/
│   ├── agents/             # CRL/HIQL/CoGHP：只表达 learning algorithm
│   ├── networks/           # encoder、actor/value 输入输出语义和 readout
│   ├── computation/        # primitive、topology、credit、accounting
│   ├── utils/              # dataset/evaluation/runtime adapters
│   └── main.py
├── ogbench/                # canonical environment/runtime；独立于 algorithm reference
├── configs/                # algorithm + per-slot computation specs
├── tests/                  # reference/computation/integration parity
├── docs/
└── tools/
```

第一阶段只实现 `computation/primitives/mlp.py`、`computation/topologies/feedforward.py` 和 `computation/credit/direct.py`，并通过 `HIQL.low_actor` 验证 slot 注入。其它 topology/primitive 文件在真正需要时再加入。

