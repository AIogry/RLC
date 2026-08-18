# M9 实验结果完整报告

日期：2026-08-18

## 说明与数据原则

本报告覆盖 M9A SingleState 与 M9B TwoState 的全部正式运行产物。原始数据与分析严格分开。

- 原始附录中的 eval.csv 与 summary.json 代码块直接读取自运行目录，保留原始 header、行顺序、step 顺序和浮点文本。
- 原始 train.csv 不做平滑、采样、均值或重写；由于总文件量较大，报告列出每个文件的原始路径、大小、行数和 SHA-256，文件本体保留在原实验目录。
- 分析章节中的均值、差值、best、波动和分组表均是派生展示，不替代原始附录。

原始实验根目录：

```text
/data/qijunrong/06-RL/offline-rl/exp/RLC/runs
```

正式运行 metadata 记录的工作树为 /home/eai/Research/RLC-exp，所有 84 个 run 的 Git commit 为：

```text
f30b64bf81e1738235eef4f213d3019820ee918a
```

## 1. 执行完整性

| 项目 | 数量/结果 |
|---|---:|
| M9A runs | 52 |
| M9B runs | 32 |
| total runs | 84 |
| completed summaries | 84 |
| eval.csv | 84 |
| summary.json | 84 |
| runtime_metadata.json | 84 |
| resolved_config.json | 84 |
| train.csv | 84 |
| params_500000.pkl | 80 |
| params_1000000.pkl | 4 |

### 1.1 metadata 一致性

- git_commit raw unique values: ['f30b64bf81e1738235eef4f213d3019820ee918a']
- git_dirty raw unique values: ['False']
- jax_backend raw unique values: ['gpu']
- dataset_dir raw unique values: ['/data/qijunrong/06-RL/offline-rl/data/raw_ogbench']
- seed raw unique values: ['0']

Raw training protocol JSON values:

```text
4 runs: {"batch_size": 1024, "eval_episodes": 20, "eval_gaussian": null, "eval_interval": 100000, "eval_tasks": null, "eval_temperature": 0.0, "save_interval": 1000000, "train_steps": 1000000, "video_episodes": 0}
80 runs: {"batch_size": 1024, "eval_episodes": 20, "eval_gaussian": null, "eval_interval": 100000, "eval_tasks": null, "eval_temperature": 0.0, "save_interval": 500000, "train_steps": 500000, "video_episodes": 0}
```

### 1.2 管理层文件状态

实际 run 目录中的 84 个 summary 均为 completed；但当前两个 Study 的管理文件没有回填：

- experiments/M9A_single_state_iteration/manifest.csv 仍为 planned；
- experiments/M9B_two_state/manifest.csv 仍为 planned；
- 两个 aggregated.csv 仍只有表头。

下面的数值分析直接以运行目录中的原始 eval.csv 和 summary.json 为准。

## 2. 正式协议

协议见 docs/experiment_execution.md:125。

| field | value |
|---|---|
| seed | 0 |
| environments | antmaze-medium-navigate-v0, antmaze-large-navigate-v0 |
| batch_size | 1024 |
| learning_rate | 3e-4 |
| log_interval | 5000 |
| eval_interval | 100000 |
| eval_episodes | 20 per task |
| eval_tasks | all；实际包含 task1–task5 |
| eval_temperature | 0 |
| eval_gaussian | None |
| vanilla budget | 1,000,000 steps |
| M9A/M9B variant budget | 500,000 steps |

主比较口径：variant @500k 对比相同 algorithm 的 vanilla baseline @500k；baseline @1M 是独立的完整训练参考。

## 3. 方法边界

M9A 是 decision-local SingleState，K=1/2/4 重复共享 update module；M9B 是独立 H/L update module。

| M9B schedule | L executions | H executions | total executions |
|---|---:|---:|---:|
| H2L1 | 2 | 2 | 4 |
| H2L6 | 12 | 2 | 14 |

full_bptt 保留 warm-up state 的梯度；one_step 在最终 L→H pair 前对累积状态使用 stop-gradient。

| actor | vanilla params | TwoState params | ratio |
|---|---:|---:|---:|
| CRL actor | 559624 | 1084936 | 1.938 |
| HIQL high actor | 560650 | 1085962 | 1.938 |
| HIQL low actor | 549896 | 1075208 | 1.956 |

相同 optimizer steps 不等价于相同 FLOPs、wall-clock 或内部 update 次数。

## 4. 原始 run 索引

| study | config | slug | environment | seed | final step | final success | best success | best step | run dir |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| M9A | M9A-C001 | hiql_vanilla | antmaze-large-navigate-v0 | 0 | 1000000 | 0.8700000000000001 | 0.8700000000000001 | 1000000 | M9A/M9A-C001__hiql_vanilla/antmaze-large-navigate-v0/seed_000 |
| M9A | M9A-C001 | hiql_vanilla | antmaze-medium-navigate-v0 | 0 | 1000000 | 0.96 | 0.9800000000000001 | 800000 | M9A/M9A-C001__hiql_vanilla/antmaze-medium-navigate-v0/seed_000 |
| M9A | M9A-C002 | crl_vanilla | antmaze-large-navigate-v0 | 0 | 1000000 | 0.62 | 0.77 | 900000 | M9A/M9A-C002__crl_vanilla/antmaze-large-navigate-v0/seed_000 |
| M9A | M9A-C002 | crl_vanilla | antmaze-medium-navigate-v0 | 0 | 1000000 | 0.93 | 0.99 | 400000 | M9A/M9A-C002__crl_vanilla/antmaze-medium-navigate-v0/seed_000 |
| M9A | M9A-C003 | crl_actor_k1_nores | antmaze-large-navigate-v0 | 0 | 500000 | 0.8399999999999999 | 0.85 | 400000 | M9A/M9A-C003__crl_actor_k1_nores/antmaze-large-navigate-v0/seed_000 |
| M9A | M9A-C003 | crl_actor_k1_nores | antmaze-medium-navigate-v0 | 0 | 500000 | 0.9400000000000001 | 0.9400000000000001 | 500000 | M9A/M9A-C003__crl_actor_k1_nores/antmaze-medium-navigate-v0/seed_000 |
| M9A | M9A-C004 | crl_actor_k1_res | antmaze-large-navigate-v0 | 0 | 500000 | 0.86 | 0.86 | 500000 | M9A/M9A-C004__crl_actor_k1_res/antmaze-large-navigate-v0/seed_000 |
| M9A | M9A-C004 | crl_actor_k1_res | antmaze-medium-navigate-v0 | 0 | 500000 | 0.89 | 0.9099999999999999 | 400000 | M9A/M9A-C004__crl_actor_k1_res/antmaze-medium-navigate-v0/seed_000 |
| M9A | M9A-C005 | crl_actor_k2_nores | antmaze-large-navigate-v0 | 0 | 500000 | 0.8700000000000001 | 0.89 | 400000 | M9A/M9A-C005__crl_actor_k2_nores/antmaze-large-navigate-v0/seed_000 |
| M9A | M9A-C005 | crl_actor_k2_nores | antmaze-medium-navigate-v0 | 0 | 500000 | 0.96 | 0.9800000000000001 | 300000 | M9A/M9A-C005__crl_actor_k2_nores/antmaze-medium-navigate-v0/seed_000 |
| M9A | M9A-C006 | crl_actor_k2_res | antmaze-large-navigate-v0 | 0 | 500000 | 0.8699999999999999 | 0.9 | 400000 | M9A/M9A-C006__crl_actor_k2_res/antmaze-large-navigate-v0/seed_000 |
| M9A | M9A-C006 | crl_actor_k2_res | antmaze-medium-navigate-v0 | 0 | 500000 | 0.9099999999999999 | 0.9700000000000001 | 400000 | M9A/M9A-C006__crl_actor_k2_res/antmaze-medium-navigate-v0/seed_000 |
| M9A | M9A-C007 | crl_actor_k4_nores | antmaze-large-navigate-v0 | 0 | 500000 | 0.9099999999999999 | 0.9200000000000002 | 400000 | M9A/M9A-C007__crl_actor_k4_nores/antmaze-large-navigate-v0/seed_000 |
| M9A | M9A-C007 | crl_actor_k4_nores | antmaze-medium-navigate-v0 | 0 | 500000 | 0.97 | 0.9800000000000001 | 300000 | M9A/M9A-C007__crl_actor_k4_nores/antmaze-medium-navigate-v0/seed_000 |
| M9A | M9A-C008 | crl_actor_k4_res | antmaze-large-navigate-v0 | 0 | 500000 | 0.9099999999999999 | 0.9299999999999999 | 300000 | M9A/M9A-C008__crl_actor_k4_res/antmaze-large-navigate-v0/seed_000 |
| M9A | M9A-C008 | crl_actor_k4_res | antmaze-medium-navigate-v0 | 0 | 500000 | 0.95 | 0.96 | 300000 | M9A/M9A-C008__crl_actor_k4_res/antmaze-medium-navigate-v0/seed_000 |
| M9A | M9A-C009 | hiql_high_k1_nores | antmaze-large-navigate-v0 | 0 | 500000 | 0.8300000000000001 | 0.9100000000000001 | 400000 | M9A/M9A-C009__hiql_high_k1_nores/antmaze-large-navigate-v0/seed_000 |
| M9A | M9A-C009 | hiql_high_k1_nores | antmaze-medium-navigate-v0 | 0 | 500000 | 0.9800000000000001 | 0.9800000000000001 | 500000 | M9A/M9A-C009__hiql_high_k1_nores/antmaze-medium-navigate-v0/seed_000 |
| M9A | M9A-C010 | hiql_high_k1_res | antmaze-large-navigate-v0 | 0 | 500000 | 0.9099999999999999 | 0.9099999999999999 | 500000 | M9A/M9A-C010__hiql_high_k1_res/antmaze-large-navigate-v0/seed_000 |
| M9A | M9A-C010 | hiql_high_k1_res | antmaze-medium-navigate-v0 | 0 | 500000 | 0.99 | 0.99 | 500000 | M9A/M9A-C010__hiql_high_k1_res/antmaze-medium-navigate-v0/seed_000 |
| M9A | M9A-C011 | hiql_high_k2_nores | antmaze-large-navigate-v0 | 0 | 500000 | 0.9199999999999999 | 0.9199999999999999 | 500000 | M9A/M9A-C011__hiql_high_k2_nores/antmaze-large-navigate-v0/seed_000 |
| M9A | M9A-C011 | hiql_high_k2_nores | antmaze-medium-navigate-v0 | 0 | 500000 | 0.95 | 0.96 | 300000 | M9A/M9A-C011__hiql_high_k2_nores/antmaze-medium-navigate-v0/seed_000 |
| M9A | M9A-C012 | hiql_high_k2_res | antmaze-large-navigate-v0 | 0 | 500000 | 0.85 | 0.85 | 300000 | M9A/M9A-C012__hiql_high_k2_res/antmaze-large-navigate-v0/seed_000 |
| M9A | M9A-C012 | hiql_high_k2_res | antmaze-medium-navigate-v0 | 0 | 500000 | 0.9399999999999998 | 0.99 | 100000 | M9A/M9A-C012__hiql_high_k2_res/antmaze-medium-navigate-v0/seed_000 |
| M9A | M9A-C013 | hiql_high_k4_nores | antmaze-large-navigate-v0 | 0 | 500000 | 0.85 | 0.93 | 300000 | M9A/M9A-C013__hiql_high_k4_nores/antmaze-large-navigate-v0/seed_000 |
| M9A | M9A-C013 | hiql_high_k4_nores | antmaze-medium-navigate-v0 | 0 | 500000 | 0.9199999999999999 | 0.9800000000000001 | 300000 | M9A/M9A-C013__hiql_high_k4_nores/antmaze-medium-navigate-v0/seed_000 |
| M9A | M9A-C014 | hiql_high_k4_res | antmaze-large-navigate-v0 | 0 | 500000 | 0.9400000000000001 | 0.9400000000000001 | 400000 | M9A/M9A-C014__hiql_high_k4_res/antmaze-large-navigate-v0/seed_000 |
| M9A | M9A-C014 | hiql_high_k4_res | antmaze-medium-navigate-v0 | 0 | 500000 | 0.95 | 0.99 | 400000 | M9A/M9A-C014__hiql_high_k4_res/antmaze-medium-navigate-v0/seed_000 |
| M9A | M9A-C015 | hiql_low_k1_nores | antmaze-large-navigate-v0 | 0 | 500000 | 0.8400000000000001 | 0.8400000000000001 | 500000 | M9A/M9A-C015__hiql_low_k1_nores/antmaze-large-navigate-v0/seed_000 |
| M9A | M9A-C015 | hiql_low_k1_nores | antmaze-medium-navigate-v0 | 0 | 500000 | 0.97 | 0.97 | 500000 | M9A/M9A-C015__hiql_low_k1_nores/antmaze-medium-navigate-v0/seed_000 |
| M9A | M9A-C016 | hiql_low_k1_res | antmaze-large-navigate-v0 | 0 | 500000 | 0.85 | 0.9200000000000002 | 400000 | M9A/M9A-C016__hiql_low_k1_res/antmaze-large-navigate-v0/seed_000 |
| M9A | M9A-C016 | hiql_low_k1_res | antmaze-medium-navigate-v0 | 0 | 500000 | 0.96 | 0.96 | 300000 | M9A/M9A-C016__hiql_low_k1_res/antmaze-medium-navigate-v0/seed_000 |
| M9A | M9A-C017 | hiql_low_k2_nores | antmaze-large-navigate-v0 | 0 | 500000 | 0.89 | 0.89 | 500000 | M9A/M9A-C017__hiql_low_k2_nores/antmaze-large-navigate-v0/seed_000 |
| M9A | M9A-C017 | hiql_low_k2_nores | antmaze-medium-navigate-v0 | 0 | 500000 | 0.95 | 0.97 | 200000 | M9A/M9A-C017__hiql_low_k2_nores/antmaze-medium-navigate-v0/seed_000 |
| M9A | M9A-C018 | hiql_low_k2_res | antmaze-large-navigate-v0 | 0 | 500000 | 0.86 | 0.9 | 400000 | M9A/M9A-C018__hiql_low_k2_res/antmaze-large-navigate-v0/seed_000 |
| M9A | M9A-C018 | hiql_low_k2_res | antmaze-medium-navigate-v0 | 0 | 500000 | 0.96 | 0.97 | 300000 | M9A/M9A-C018__hiql_low_k2_res/antmaze-medium-navigate-v0/seed_000 |
| M9A | M9A-C019 | hiql_low_k4_nores | antmaze-large-navigate-v0 | 0 | 500000 | 0.93 | 0.93 | 500000 | M9A/M9A-C019__hiql_low_k4_nores/antmaze-large-navigate-v0/seed_000 |
| M9A | M9A-C019 | hiql_low_k4_nores | antmaze-medium-navigate-v0 | 0 | 500000 | 0.9400000000000001 | 0.97 | 200000 | M9A/M9A-C019__hiql_low_k4_nores/antmaze-medium-navigate-v0/seed_000 |
| M9A | M9A-C020 | hiql_low_k4_res | antmaze-large-navigate-v0 | 0 | 500000 | 0.89 | 0.93 | 300000 | M9A/M9A-C020__hiql_low_k4_res/antmaze-large-navigate-v0/seed_000 |
| M9A | M9A-C020 | hiql_low_k4_res | antmaze-medium-navigate-v0 | 0 | 500000 | 0.9400000000000001 | 0.96 | 300000 | M9A/M9A-C020__hiql_low_k4_res/antmaze-medium-navigate-v0/seed_000 |
| M9A | M9A-C021 | hiql_high_low_k1_nores | antmaze-large-navigate-v0 | 0 | 500000 | 0.9200000000000002 | 0.9200000000000002 | 500000 | M9A/M9A-C021__hiql_high_low_k1_nores/antmaze-large-navigate-v0/seed_000 |
| M9A | M9A-C021 | hiql_high_low_k1_nores | antmaze-medium-navigate-v0 | 0 | 500000 | 0.97 | 0.9700000000000001 | 400000 | M9A/M9A-C021__hiql_high_low_k1_nores/antmaze-medium-navigate-v0/seed_000 |
| M9A | M9A-C022 | hiql_high_low_k1_res | antmaze-large-navigate-v0 | 0 | 500000 | 0.82 | 0.93 | 400000 | M9A/M9A-C022__hiql_high_low_k1_res/antmaze-large-navigate-v0/seed_000 |
| M9A | M9A-C022 | hiql_high_low_k1_res | antmaze-medium-navigate-v0 | 0 | 500000 | 0.93 | 0.97 | 100000 | M9A/M9A-C022__hiql_high_low_k1_res/antmaze-medium-navigate-v0/seed_000 |
| M9A | M9A-C023 | hiql_high_low_k2_nores | antmaze-large-navigate-v0 | 0 | 500000 | 0.8300000000000001 | 0.8300000000000001 | 300000 | M9A/M9A-C023__hiql_high_low_k2_nores/antmaze-large-navigate-v0/seed_000 |
| M9A | M9A-C023 | hiql_high_low_k2_nores | antmaze-medium-navigate-v0 | 0 | 500000 | 0.9199999999999999 | 0.96 | 200000 | M9A/M9A-C023__hiql_high_low_k2_nores/antmaze-medium-navigate-v0/seed_000 |
| M9A | M9A-C024 | hiql_high_low_k2_res | antmaze-large-navigate-v0 | 0 | 500000 | 0.8800000000000001 | 0.9400000000000001 | 300000 | M9A/M9A-C024__hiql_high_low_k2_res/antmaze-large-navigate-v0/seed_000 |
| M9A | M9A-C024 | hiql_high_low_k2_res | antmaze-medium-navigate-v0 | 0 | 500000 | 0.9 | 0.99 | 100000 | M9A/M9A-C024__hiql_high_low_k2_res/antmaze-medium-navigate-v0/seed_000 |
| M9A | M9A-C025 | hiql_high_low_k4_nores | antmaze-large-navigate-v0 | 0 | 500000 | 0.79 | 0.9199999999999999 | 400000 | M9A/M9A-C025__hiql_high_low_k4_nores/antmaze-large-navigate-v0/seed_000 |
| M9A | M9A-C025 | hiql_high_low_k4_nores | antmaze-medium-navigate-v0 | 0 | 500000 | 0.9800000000000001 | 0.9800000000000001 | 400000 | M9A/M9A-C025__hiql_high_low_k4_nores/antmaze-medium-navigate-v0/seed_000 |
| M9A | M9A-C026 | hiql_high_low_k4_res | antmaze-large-navigate-v0 | 0 | 500000 | 0.8700000000000001 | 0.93 | 200000 | M9A/M9A-C026__hiql_high_low_k4_res/antmaze-large-navigate-v0/seed_000 |
| M9A | M9A-C026 | hiql_high_low_k4_res | antmaze-medium-navigate-v0 | 0 | 500000 | 0.97 | 0.97 | 300000 | M9A/M9A-C026__hiql_high_low_k4_res/antmaze-medium-navigate-v0/seed_000 |
| M9B | M9B-C001 | crl_actor_h2l1_full_bptt | antmaze-large-navigate-v0 | 0 | 500000 | 0.8400000000000001 | 0.9 | 300000 | M9B/M9B-C001__crl_actor_h2l1_full_bptt/antmaze-large-navigate-v0/seed_000 |
| M9B | M9B-C001 | crl_actor_h2l1_full_bptt | antmaze-medium-navigate-v0 | 0 | 500000 | 0.9399999999999998 | 0.9800000000000001 | 400000 | M9B/M9B-C001__crl_actor_h2l1_full_bptt/antmaze-medium-navigate-v0/seed_000 |
| M9B | M9B-C002 | crl_actor_h2l1_one_step | antmaze-large-navigate-v0 | 0 | 500000 | 0.86 | 0.86 | 500000 | M9B/M9B-C002__crl_actor_h2l1_one_step/antmaze-large-navigate-v0/seed_000 |
| M9B | M9B-C002 | crl_actor_h2l1_one_step | antmaze-medium-navigate-v0 | 0 | 500000 | 0.96 | 0.96 | 500000 | M9B/M9B-C002__crl_actor_h2l1_one_step/antmaze-medium-navigate-v0/seed_000 |
| M9B | M9B-C003 | crl_actor_h2l6_full_bptt | antmaze-large-navigate-v0 | 0 | 500000 | 0.89 | 0.9099999999999999 | 400000 | M9B/M9B-C003__crl_actor_h2l6_full_bptt/antmaze-large-navigate-v0/seed_000 |
| M9B | M9B-C003 | crl_actor_h2l6_full_bptt | antmaze-medium-navigate-v0 | 0 | 500000 | 0.95 | 0.96 | 400000 | M9B/M9B-C003__crl_actor_h2l6_full_bptt/antmaze-medium-navigate-v0/seed_000 |
| M9B | M9B-C004 | crl_actor_h2l6_one_step | antmaze-large-navigate-v0 | 0 | 500000 | 0.7 | 0.7 | 500000 | M9B/M9B-C004__crl_actor_h2l6_one_step/antmaze-large-navigate-v0/seed_000 |
| M9B | M9B-C004 | crl_actor_h2l6_one_step | antmaze-medium-navigate-v0 | 0 | 500000 | 0.8400000000000001 | 0.8400000000000001 | 500000 | M9B/M9B-C004__crl_actor_h2l6_one_step/antmaze-medium-navigate-v0/seed_000 |
| M9B | M9B-C005 | hiql_high_h2l1_full_bptt | antmaze-large-navigate-v0 | 0 | 500000 | 0.89 | 0.9099999999999999 | 400000 | M9B/M9B-C005__hiql_high_h2l1_full_bptt/antmaze-large-navigate-v0/seed_000 |
| M9B | M9B-C005 | hiql_high_h2l1_full_bptt | antmaze-medium-navigate-v0 | 0 | 500000 | 0.97 | 1.0 | 100000 | M9B/M9B-C005__hiql_high_h2l1_full_bptt/antmaze-medium-navigate-v0/seed_000 |
| M9B | M9B-C006 | hiql_high_h2l1_one_step | antmaze-large-navigate-v0 | 0 | 500000 | 0.93 | 0.93 | 400000 | M9B/M9B-C006__hiql_high_h2l1_one_step/antmaze-large-navigate-v0/seed_000 |
| M9B | M9B-C006 | hiql_high_h2l1_one_step | antmaze-medium-navigate-v0 | 0 | 500000 | 0.99 | 0.99 | 200000 | M9B/M9B-C006__hiql_high_h2l1_one_step/antmaze-medium-navigate-v0/seed_000 |
| M9B | M9B-C007 | hiql_high_h2l6_full_bptt | antmaze-large-navigate-v0 | 0 | 500000 | 0.8800000000000001 | 0.8800000000000001 | 500000 | M9B/M9B-C007__hiql_high_h2l6_full_bptt/antmaze-large-navigate-v0/seed_000 |
| M9B | M9B-C007 | hiql_high_h2l6_full_bptt | antmaze-medium-navigate-v0 | 0 | 500000 | 0.99 | 0.99 | 200000 | M9B/M9B-C007__hiql_high_h2l6_full_bptt/antmaze-medium-navigate-v0/seed_000 |
| M9B | M9B-C008 | hiql_high_h2l6_one_step | antmaze-large-navigate-v0 | 0 | 500000 | 0.85 | 0.85 | 500000 | M9B/M9B-C008__hiql_high_h2l6_one_step/antmaze-large-navigate-v0/seed_000 |
| M9B | M9B-C008 | hiql_high_h2l6_one_step | antmaze-medium-navigate-v0 | 0 | 500000 | 0.9700000000000001 | 0.9700000000000001 | 500000 | M9B/M9B-C008__hiql_high_h2l6_one_step/antmaze-medium-navigate-v0/seed_000 |
| M9B | M9B-C009 | hiql_low_h2l1_full_bptt | antmaze-large-navigate-v0 | 0 | 500000 | 0.73 | 0.7700000000000001 | 400000 | M9B/M9B-C009__hiql_low_h2l1_full_bptt/antmaze-large-navigate-v0/seed_000 |
| M9B | M9B-C009 | hiql_low_h2l1_full_bptt | antmaze-medium-navigate-v0 | 0 | 500000 | 0.95 | 0.97 | 400000 | M9B/M9B-C009__hiql_low_h2l1_full_bptt/antmaze-medium-navigate-v0/seed_000 |
| M9B | M9B-C010 | hiql_low_h2l1_one_step | antmaze-large-navigate-v0 | 0 | 500000 | 0.8099999999999999 | 0.9199999999999999 | 400000 | M9B/M9B-C010__hiql_low_h2l1_one_step/antmaze-large-navigate-v0/seed_000 |
| M9B | M9B-C010 | hiql_low_h2l1_one_step | antmaze-medium-navigate-v0 | 0 | 500000 | 0.89 | 0.95 | 300000 | M9B/M9B-C010__hiql_low_h2l1_one_step/antmaze-medium-navigate-v0/seed_000 |
| M9B | M9B-C011 | hiql_low_h2l6_full_bptt | antmaze-large-navigate-v0 | 0 | 500000 | 0.8 | 0.8299999999999998 | 300000 | M9B/M9B-C011__hiql_low_h2l6_full_bptt/antmaze-large-navigate-v0/seed_000 |
| M9B | M9B-C011 | hiql_low_h2l6_full_bptt | antmaze-medium-navigate-v0 | 0 | 500000 | 0.96 | 1.0 | 200000 | M9B/M9B-C011__hiql_low_h2l6_full_bptt/antmaze-medium-navigate-v0/seed_000 |
| M9B | M9B-C012 | hiql_low_h2l6_one_step | antmaze-large-navigate-v0 | 0 | 500000 | 0.8 | 0.8 | 500000 | M9B/M9B-C012__hiql_low_h2l6_one_step/antmaze-large-navigate-v0/seed_000 |
| M9B | M9B-C012 | hiql_low_h2l6_one_step | antmaze-medium-navigate-v0 | 0 | 500000 | 0.9800000000000001 | 0.9800000000000001 | 300000 | M9B/M9B-C012__hiql_low_h2l6_one_step/antmaze-medium-navigate-v0/seed_000 |
| M9B | M9B-C013 | hiql_high_low_h2l1_full_bptt | antmaze-large-navigate-v0 | 0 | 500000 | 0.9099999999999999 | 0.95 | 400000 | M9B/M9B-C013__hiql_high_low_h2l1_full_bptt/antmaze-large-navigate-v0/seed_000 |
| M9B | M9B-C013 | hiql_high_low_h2l1_full_bptt | antmaze-medium-navigate-v0 | 0 | 500000 | 0.89 | 0.96 | 100000 | M9B/M9B-C013__hiql_high_low_h2l1_full_bptt/antmaze-medium-navigate-v0/seed_000 |
| M9B | M9B-C014 | hiql_high_low_h2l1_one_step | antmaze-large-navigate-v0 | 0 | 500000 | 0.8299999999999998 | 0.85 | 300000 | M9B/M9B-C014__hiql_high_low_h2l1_one_step/antmaze-large-navigate-v0/seed_000 |
| M9B | M9B-C014 | hiql_high_low_h2l1_one_step | antmaze-medium-navigate-v0 | 0 | 500000 | 0.9299999999999999 | 0.96 | 300000 | M9B/M9B-C014__hiql_high_low_h2l1_one_step/antmaze-medium-navigate-v0/seed_000 |
| M9B | M9B-C015 | hiql_high_low_h2l6_full_bptt | antmaze-large-navigate-v0 | 0 | 500000 | 0.85 | 0.9199999999999999 | 200000 | M9B/M9B-C015__hiql_high_low_h2l6_full_bptt/antmaze-large-navigate-v0/seed_000 |
| M9B | M9B-C015 | hiql_high_low_h2l6_full_bptt | antmaze-medium-navigate-v0 | 0 | 500000 | 0.95 | 0.96 | 300000 | M9B/M9B-C015__hiql_high_low_h2l6_full_bptt/antmaze-medium-navigate-v0/seed_000 |
| M9B | M9B-C016 | hiql_high_low_h2l6_one_step | antmaze-large-navigate-v0 | 0 | 500000 | 0.8 | 0.89 | 200000 | M9B/M9B-C016__hiql_high_low_h2l6_one_step/antmaze-large-navigate-v0/seed_000 |
| M9B | M9B-C016 | hiql_high_low_h2l6_one_step | antmaze-medium-navigate-v0 | 0 | 500000 | 0.9400000000000001 | 0.96 | 300000 | M9B/M9B-C016__hiql_high_low_h2l6_one_step/antmaze-medium-navigate-v0/seed_000 |

## 5. 分析：baseline learning curve

| baseline | environment | 100k | 200k | 300k | 400k | 500k | 600k | 700k | 800k | 900k | 1M |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| HIQL vanilla | antmaze-medium-navigate-v0 | 0.96 | 0.97 | 0.96 | 0.97 | 0.9700000000000001 | 0.95 | 0.95 | 0.9800000000000001 | 0.9400000000000001 | 0.96 |
| HIQL vanilla | antmaze-large-navigate-v0 | 0.7 | 0.8300000000000001 | 0.8299999999999998 | 0.85 | 0.8099999999999999 | 0.78 | 0.8099999999999999 | 0.85 | 0.8300000000000001 | 0.8700000000000001 |
| CRL vanilla | antmaze-medium-navigate-v0 | 0.8 | 0.8300000000000001 | 0.9099999999999999 | 0.99 | 0.96 | 0.95 | 0.96 | 0.9399999999999998 | 0.97 | 0.93 |
| CRL vanilla | antmaze-large-navigate-v0 | 0.5499999999999999 | 0.65 | 0.7 | 0.6 | 0.75 | 0.64 | 0.65 | 0.64 | 0.77 | 0.62 |

Baseline observation:

- HIQL 500k→1M：Medium 0.97→0.96，Large 0.81→0.87。
- CRL 500k→1M：Medium 0.96→0.93，Large 0.75→0.62。
- 因此 1M 不是所有 baseline 的单调收敛终点。

## 6. 分析：M9A SingleState 全配置

| config | slug | placement | Medium final | Large final | mean final | Medium best@step | Large best@step | mean(best-final) |
|---|---|---|---:|---:|---:|---|---|---:|
| M9A-C003 | crl_actor_k1_nores | CRL actor | 0.94 | 0.84 | 0.890 | 0.94@500000 | 0.85@400000 | +0.005 |
| M9A-C004 | crl_actor_k1_res | CRL actor | 0.89 | 0.86 | 0.875 | 0.91@400000 | 0.86@500000 | +0.010 |
| M9A-C005 | crl_actor_k2_nores | CRL actor | 0.96 | 0.87 | 0.915 | 0.98@300000 | 0.89@400000 | +0.020 |
| M9A-C006 | crl_actor_k2_res | CRL actor | 0.91 | 0.87 | 0.890 | 0.97@400000 | 0.90@400000 | +0.045 |
| M9A-C007 | crl_actor_k4_nores | CRL actor | 0.97 | 0.91 | 0.940 | 0.98@300000 | 0.92@400000 | +0.010 |
| M9A-C008 | crl_actor_k4_res | CRL actor | 0.95 | 0.91 | 0.930 | 0.96@300000 | 0.93@300000 | +0.015 |
| M9A-C009 | hiql_high_k1_nores | HIQL high_actor | 0.98 | 0.83 | 0.905 | 0.98@500000 | 0.91@400000 | +0.040 |
| M9A-C010 | hiql_high_k1_res | HIQL high_actor | 0.99 | 0.91 | 0.950 | 0.99@500000 | 0.91@500000 | +0.000 |
| M9A-C011 | hiql_high_k2_nores | HIQL high_actor | 0.95 | 0.92 | 0.935 | 0.96@300000 | 0.92@500000 | +0.005 |
| M9A-C012 | hiql_high_k2_res | HIQL high_actor | 0.94 | 0.85 | 0.895 | 0.99@100000 | 0.85@300000 | +0.025 |
| M9A-C013 | hiql_high_k4_nores | HIQL high_actor | 0.92 | 0.85 | 0.885 | 0.98@300000 | 0.93@300000 | +0.070 |
| M9A-C014 | hiql_high_k4_res | HIQL high_actor | 0.95 | 0.94 | 0.945 | 0.99@400000 | 0.94@400000 | +0.020 |
| M9A-C015 | hiql_low_k1_nores | HIQL low_actor | 0.97 | 0.84 | 0.905 | 0.97@500000 | 0.84@500000 | +0.000 |
| M9A-C016 | hiql_low_k1_res | HIQL low_actor | 0.96 | 0.85 | 0.905 | 0.96@300000 | 0.92@400000 | +0.035 |
| M9A-C017 | hiql_low_k2_nores | HIQL low_actor | 0.95 | 0.89 | 0.920 | 0.97@200000 | 0.89@500000 | +0.010 |
| M9A-C018 | hiql_low_k2_res | HIQL low_actor | 0.96 | 0.86 | 0.910 | 0.97@300000 | 0.90@400000 | +0.025 |
| M9A-C019 | hiql_low_k4_nores | HIQL low_actor | 0.94 | 0.93 | 0.935 | 0.97@200000 | 0.93@500000 | +0.015 |
| M9A-C020 | hiql_low_k4_res | HIQL low_actor | 0.94 | 0.89 | 0.915 | 0.96@300000 | 0.93@300000 | +0.030 |
| M9A-C021 | hiql_high_low_k1_nores | HIQL high_actor+low_actor | 0.97 | 0.92 | 0.945 | 0.97@400000 | 0.92@500000 | +0.000 |
| M9A-C022 | hiql_high_low_k1_res | HIQL high_actor+low_actor | 0.93 | 0.82 | 0.875 | 0.97@100000 | 0.93@400000 | +0.075 |
| M9A-C023 | hiql_high_low_k2_nores | HIQL high_actor+low_actor | 0.92 | 0.83 | 0.875 | 0.96@200000 | 0.83@300000 | +0.020 |
| M9A-C024 | hiql_high_low_k2_res | HIQL high_actor+low_actor | 0.90 | 0.88 | 0.890 | 0.99@100000 | 0.94@300000 | +0.075 |
| M9A-C025 | hiql_high_low_k4_nores | HIQL high_actor+low_actor | 0.98 | 0.79 | 0.885 | 0.98@400000 | 0.92@400000 | +0.065 |
| M9A-C026 | hiql_high_low_k4_res | HIQL high_actor+low_actor | 0.97 | 0.87 | 0.920 | 0.97@300000 | 0.93@200000 | +0.030 |

### 6.1 M9A representative comparison

| placement | config | Medium/Large final | mean | mean delta vs baseline @500k |
|---|---|---:|---:|---:|
| CRL actor | M9A-C007 | 0.97/0.91 | 0.940 | +0.085 |
| HIQL high_actor | M9A-C010 | 0.99/0.91 | 0.950 | +0.060 |
| HIQL low_actor | M9A-C019 | 0.94/0.93 | 0.935 | +0.045 |
| HIQL high_actor+low_actor | M9A-C021 | 0.97/0.92 | 0.945 | +0.055 |

M9A interpretation:

- CRL 中 K=4 no-residual 是最清晰的正向信号。
- HIQL 各 placement 的最佳 K/residual 不一致，没有统一单调规律。
- 主要提升来自 Large，Medium 存在 ceiling effect。

## 7. 分析：M9B TwoState 全配置

| config | slug | placement | schedule | credit | Medium final | Large final | mean final | mean delta vs baseline @500k |
|---|---|---|---|---|---:|---:|---:|---:|
| M9B-C001 | crl_actor_h2l1_full_bptt | CRL actor | H2L1 | full_bptt | 0.94 | 0.84 | 0.890 | +0.035 |
| M9B-C002 | crl_actor_h2l1_one_step | CRL actor | H2L1 | one_step | 0.96 | 0.86 | 0.910 | +0.055 |
| M9B-C003 | crl_actor_h2l6_full_bptt | CRL actor | H2L6 | full_bptt | 0.95 | 0.89 | 0.920 | +0.065 |
| M9B-C004 | crl_actor_h2l6_one_step | CRL actor | H2L6 | one_step | 0.84 | 0.70 | 0.770 | -0.085 |
| M9B-C005 | hiql_high_h2l1_full_bptt | HIQL high_actor | H2L1 | full_bptt | 0.97 | 0.89 | 0.930 | +0.040 |
| M9B-C006 | hiql_high_h2l1_one_step | HIQL high_actor | H2L1 | one_step | 0.99 | 0.93 | 0.960 | +0.070 |
| M9B-C007 | hiql_high_h2l6_full_bptt | HIQL high_actor | H2L6 | full_bptt | 0.99 | 0.88 | 0.935 | +0.045 |
| M9B-C008 | hiql_high_h2l6_one_step | HIQL high_actor | H2L6 | one_step | 0.97 | 0.85 | 0.910 | +0.020 |
| M9B-C009 | hiql_low_h2l1_full_bptt | HIQL low_actor | H2L1 | full_bptt | 0.95 | 0.73 | 0.840 | -0.050 |
| M9B-C010 | hiql_low_h2l1_one_step | HIQL low_actor | H2L1 | one_step | 0.89 | 0.81 | 0.850 | -0.040 |
| M9B-C011 | hiql_low_h2l6_full_bptt | HIQL low_actor | H2L6 | full_bptt | 0.96 | 0.80 | 0.880 | -0.010 |
| M9B-C012 | hiql_low_h2l6_one_step | HIQL low_actor | H2L6 | one_step | 0.98 | 0.80 | 0.890 | +0.000 |
| M9B-C013 | hiql_high_low_h2l1_full_bptt | HIQL high_actor+low_actor | H2L1 | full_bptt | 0.89 | 0.91 | 0.900 | +0.010 |
| M9B-C014 | hiql_high_low_h2l1_one_step | HIQL high_actor+low_actor | H2L1 | one_step | 0.93 | 0.83 | 0.880 | -0.010 |
| M9B-C015 | hiql_high_low_h2l6_full_bptt | HIQL high_actor+low_actor | H2L6 | full_bptt | 0.95 | 0.85 | 0.900 | +0.010 |
| M9B-C016 | hiql_high_low_h2l6_one_step | HIQL high_actor+low_actor | H2L6 | one_step | 0.94 | 0.80 | 0.870 | -0.020 |

### 7.1 M9B schedule/credit 对比

| placement | H2L1 full | H2L1 one-step | H2L6 full | H2L6 one-step |
|---|---:|---:|---:|---:|
| CRL actor | 0.890 (0.94/0.84) | 0.910 (0.96/0.86) | 0.920 (0.95/0.89) | 0.770 (0.84/0.70) |
| HIQL high_actor | 0.930 (0.97/0.89) | 0.960 (0.99/0.93) | 0.935 (0.99/0.88) | 0.910 (0.97/0.85) |
| HIQL low_actor | 0.840 (0.95/0.73) | 0.850 (0.89/0.81) | 0.880 (0.96/0.80) | 0.890 (0.98/0.80) |
| HIQL high_actor+low_actor | 0.900 (0.89/0.91) | 0.880 (0.93/0.83) | 0.900 (0.95/0.85) | 0.870 (0.94/0.80) |

### 7.2 M9B learning curves

| config | Medium 100k→500k | Large 100k→500k |
|---|---|---|
| M9B-C001 | `0.9200000000000002, 0.89, 0.9400000000000001, 0.9800000000000001, 0.9399999999999998` | `0.8, 0.85, 0.9, 0.89, 0.8400000000000001` |
| M9B-C002 | `0.76, 0.95, 0.95, 0.9099999999999999, 0.96` | `0.62, 0.76, 0.8099999999999999, 0.77, 0.86` |
| M9B-C003 | `0.9400000000000001, 0.95, 0.95, 0.96, 0.95` | `0.8299999999999998, 0.82, 0.8099999999999999, 0.9099999999999999, 0.89` |
| M9B-C004 | `0.6699999999999999, 0.74, 0.74, 0.76, 0.8400000000000001` | `0.4, 0.4699999999999999, 0.48999999999999994, 0.55, 0.7` |
| M9B-C005 | `1.0, 0.9800000000000001, 0.9800000000000001, 0.9800000000000001, 0.97` | `0.79, 0.8299999999999998, 0.8800000000000001, 0.9099999999999999, 0.89` |
| M9B-C006 | `0.9800000000000001, 0.99, 0.95, 0.9199999999999999, 0.99` | `0.7200000000000001, 0.85, 0.8800000000000001, 0.93, 0.93` |
| M9B-C007 | `0.9199999999999999, 0.99, 0.9399999999999998, 0.99, 0.99` | `0.72, 0.7899999999999999, 0.8400000000000001, 0.86, 0.8800000000000001` |
| M9B-C008 | `0.9099999999999999, 0.96, 0.97, 0.95, 0.9700000000000001` | `0.7, 0.78, 0.8099999999999999, 0.8399999999999999, 0.85` |
| M9B-C009 | `0.8800000000000001, 0.9200000000000002, 0.9400000000000001, 0.97, 0.95` | `0.7100000000000001, 0.7, 0.76, 0.7700000000000001, 0.73` |
| M9B-C010 | `0.9, 0.9299999999999999, 0.95, 0.9399999999999998, 0.89` | `0.75, 0.8100000000000002, 0.8600000000000001, 0.9199999999999999, 0.8099999999999999` |
| M9B-C011 | `0.95, 1.0, 0.95, 0.93, 0.96` | `0.62, 0.78, 0.8299999999999998, 0.6900000000000001, 0.8` |
| M9B-C012 | `0.8699999999999999, 0.9400000000000001, 0.9800000000000001, 0.93, 0.9800000000000001` | `0.65, 0.72, 0.75, 0.71, 0.8` |
| M9B-C013 | `0.96, 0.96, 0.95, 0.93, 0.89` | `0.72, 0.9299999999999999, 0.8800000000000001, 0.95, 0.9099999999999999` |
| M9B-C014 | `0.8700000000000001, 0.9099999999999999, 0.96, 0.9400000000000001, 0.9299999999999999` | `0.7699999999999999, 0.77, 0.85, 0.79, 0.8299999999999998` |
| M9B-C015 | `0.95, 0.95, 0.96, 0.95, 0.95` | `0.74, 0.9199999999999999, 0.8400000000000001, 0.9, 0.85` |
| M9B-C016 | `0.9, 0.9199999999999999, 0.96, 0.9199999999999999, 0.9400000000000001` | `0.7899999999999998, 0.89, 0.8800000000000001, 0.8699999999999999, 0.8` |

TwoState 收敛性判断：

- C004 Large 0.40, 0.47, 0.49, 0.55, 0.70，仍明显上升，支持预算不足假设。
- C012 Large 0.65, 0.72, 0.75, 0.71, 0.80，也未显示稳定平台。
- C006 high_actor one-step final 0.99/0.93，说明 TwoState 不是整体不可学习。
- C003 CRL H2L6 full-BPTT final 0.95/0.89，说明 H2L6 在 full-BPTT 下可以工作。
- C004 与 C003 的差异说明 H2L6 one-step 还存在明显 credit assignment 风险。

## 8. 分析：task-level

| group | task1 | task2 | task3 | task4 | task5 | overall mean |
|---|---:|---:|---:|---:|---:|---:|
| CRL baseline | 0.975 | 0.500 | 0.975 | 0.975 | 0.850 | 0.855 |
| CRL M9A variants | 0.946 | 0.846 | 0.917 | 0.879 | 0.946 | 0.907 |
| CRL M9B H2L1 | 0.912 | 0.850 | 0.900 | 0.962 | 0.875 | 0.900 |
| CRL M9B H2L6 | 0.938 | 0.812 | 0.763 | 0.812 | 0.900 | 0.845 |
| HIQL baseline | 0.950 | 0.675 | 0.925 | 0.950 | 0.950 | 0.890 |
| HIQL high M9B H2L1 | 0.925 | 0.950 | 0.975 | 0.938 | 0.938 | 0.945 |
| HIQL low M9B H2L1 | 0.812 | 0.688 | 0.925 | 0.912 | 0.887 | 0.845 |
| HIQL low M9B H2L6 | 0.812 | 0.825 | 0.938 | 0.888 | 0.963 | 0.885 |

Large task2 是 baseline 最困难的维度之一；M9A CRL variants 对 task2 有较明显改善，而 M9B H2L6 one-step 的低表现伴随多个困难 task 同时下降。

## 9. 分析：运行时间与训练日志

| group | n | mean min | median min | min | max |
|---|---:|---:|---:|---:|---:|
| ('M9A', 'CRL actor') | 12 | 28.8 | 27.5 | 26.0 | 35.4 |
| ('M9A', 'HIQL high_actor') | 12 | 35.0 | 35.4 | 29.1 | 38.7 |
| ('M9A', 'HIQL high_actor+low_actor') | 12 | 32.4 | 32.7 | 29.1 | 36.7 |
| ('M9A', 'HIQL low_actor') | 12 | 36.9 | 37.4 | 32.8 | 40.4 |
| ('M9B', 'CRL actor', 'H2L1', 'full_bptt') | 2 | 34.9 | 34.9 | 34.7 | 35.1 |
| ('M9B', 'CRL actor', 'H2L1', 'one_step') | 2 | 31.4 | 31.4 | 30.8 | 32.0 |
| ('M9B', 'CRL actor', 'H2L6', 'full_bptt') | 2 | 37.8 | 37.8 | 37.6 | 38.0 |
| ('M9B', 'CRL actor', 'H2L6', 'one_step') | 2 | 31.6 | 31.6 | 31.4 | 31.7 |
| ('M9B', 'HIQL high_actor', 'H2L1', 'full_bptt') | 2 | 32.9 | 32.9 | 32.6 | 33.1 |
| ('M9B', 'HIQL high_actor', 'H2L1', 'one_step') | 2 | 31.7 | 31.7 | 31.3 | 32.1 |
| ('M9B', 'HIQL high_actor', 'H2L6', 'full_bptt') | 2 | 41.6 | 41.6 | 41.1 | 42.0 |
| ('M9B', 'HIQL high_actor', 'H2L6', 'one_step') | 2 | 34.9 | 34.9 | 34.5 | 35.2 |
| ('M9B', 'HIQL high_actor+low_actor', 'H2L1', 'full_bptt') | 2 | 36.1 | 36.1 | 35.1 | 37.0 |
| ('M9B', 'HIQL high_actor+low_actor', 'H2L1', 'one_step') | 2 | 33.7 | 33.7 | 32.5 | 34.9 |
| ('M9B', 'HIQL high_actor+low_actor', 'H2L6', 'full_bptt') | 2 | 53.2 | 53.2 | 52.2 | 54.2 |
| ('M9B', 'HIQL high_actor+low_actor', 'H2L6', 'one_step') | 2 | 38.6 | 38.6 | 38.0 | 39.1 |
| ('M9B', 'HIQL low_actor', 'H2L1', 'full_bptt') | 2 | 33.4 | 33.4 | 32.1 | 34.6 |
| ('M9B', 'HIQL low_actor', 'H2L1', 'one_step') | 2 | 32.0 | 32.0 | 31.5 | 32.5 |
| ('M9B', 'HIQL low_actor', 'H2L6', 'full_bptt') | 2 | 41.8 | 41.8 | 40.7 | 42.9 |
| ('M9B', 'HIQL low_actor', 'H2L6', 'one_step') | 2 | 35.0 | 35.0 | 33.3 | 36.6 |

H2L6 full-BPTT，尤其 HIQL high+low，实际 wall-clock 高于 H2L1；当前 protocol 只匹配 optimizer steps。

train.csv 中的 actor loss、BC loss、Q/value loss、gradient norm、validation metrics 和时间字段没有在本报告中平滑或平均；请直接读取第 12 节索引所对应的原始 CSV。

## 10. 科学结论与限制

### 数据支持的结论

1. M9A SingleState 在 seed 0 上有正向信号，最突出的是 CRL K=4 no-residual。
2. M9A 主要改善 Large，Medium 受 ceiling effect 影响。
3. TwoState 不是整体失败；HIQL high_actor H2L1 one-step 与 CRL H2L6 full-BPTT 均较强。
4. TwoState 对 placement、schedule、credit 高度敏感。
5. CRL H2L6 one-step 是最明确的失败条件，且同时具有训练不足与 credit assignment 的证据。

### 不能直接推出的结论

- 不能说 TwoState 本身无效。
- 不能说所有低结果都只是 500k 不足。
- 不能说 H2L6 或 full-BPTT 普遍优于其它条件。
- 不能声称统计显著性或多 seed 鲁棒性。

限制：所有结果只有 seed 0；每环境一次 run；每 task 20 episodes；best checkpoint 存在事后选择偏差；两环境均值只是描述性摘要。

## 11. 后续建议

1. 从 500k checkpoint 继续训练 C003、C004、C006、C007、C011、C012、C013、C015 至 1M。
2. 追加 compute-matched 或 wall-clock-matched 对比。
3. 对代表性条件增加至少 3 个 seeds，最好 5 个。
4. 记录 H/L state norm、每个 update 的 gradient norm、warm-up/final pair 梯度比例和每 task success。
5. 回填 manifest、重新生成 aggregated.csv，并记录 dataset hash。

## 12. 原始 train.csv 文件索引

train.csv 文件本体未复制、未修改。以下是原始路径与完整性信息。

| study | config | environment | bytes | lines | sha256 | raw path |
|---|---|---|---:|---:|---|---|
| M9A | M9A-C001 | antmaze-large-navigate-v0 | 115596 | 201 | 802fc6f2efc2bb50bea044be4a6535324a2197337d6e2c98cbc3bc1b0eb545bc | /data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M9A/M9A-C001__hiql_vanilla/antmaze-large-navigate-v0/seed_000/train.csv |
| M9A | M9A-C001 | antmaze-medium-navigate-v0 | 115570 | 201 | c23225bd678358c1f2ced8d0e13ccdf2afe0d5f69e519b3779e4e93e3a02707f | /data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M9A/M9A-C001__hiql_vanilla/antmaze-medium-navigate-v0/seed_000/train.csv |
| M9A | M9A-C002 | antmaze-large-navigate-v0 | 145363 | 201 | d0dff42243c5965519415044cb4786db5d5d3e0656851facb80f7b464e0a94e3 | /data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M9A/M9A-C002__crl_vanilla/antmaze-large-navigate-v0/seed_000/train.csv |
| M9A | M9A-C002 | antmaze-medium-navigate-v0 | 145246 | 201 | e5c04a8fae6a125940d0a0ab0c04250f33008cd499a53c2a0c1afba5fdef8e2e | /data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M9A/M9A-C002__crl_vanilla/antmaze-medium-navigate-v0/seed_000/train.csv |
| M9A | M9A-C003 | antmaze-large-navigate-v0 | 73215 | 101 | 6c2dcafd23dbf970544dbf3b0db4d2aafdb96f8705af26ce0b5cee970e103e3f | /data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M9A/M9A-C003__crl_actor_k1_nores/antmaze-large-navigate-v0/seed_000/train.csv |
| M9A | M9A-C003 | antmaze-medium-navigate-v0 | 73065 | 101 | 730fb913f06658f0d51bf0315d649de6b60cf4e0450dd3f006d523b9387b8479 | /data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M9A/M9A-C003__crl_actor_k1_nores/antmaze-medium-navigate-v0/seed_000/train.csv |
| M9A | M9A-C004 | antmaze-large-navigate-v0 | 73203 | 101 | 7d66e152e085d9963ad6095f525072c11dcc2f24292810c83a7c560ffddae61c | /data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M9A/M9A-C004__crl_actor_k1_res/antmaze-large-navigate-v0/seed_000/train.csv |
| M9A | M9A-C004 | antmaze-medium-navigate-v0 | 72991 | 101 | 2fe1c113f8a7df6a4e6311586d502ad418c79dbb80e24a1acd3f311deed92adc | /data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M9A/M9A-C004__crl_actor_k1_res/antmaze-medium-navigate-v0/seed_000/train.csv |
| M9A | M9A-C005 | antmaze-large-navigate-v0 | 73215 | 101 | 887428a0b89104043e35b08d257c025899b1eb5c996bed210fad4c48fa6925d8 | /data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M9A/M9A-C005__crl_actor_k2_nores/antmaze-large-navigate-v0/seed_000/train.csv |
| M9A | M9A-C005 | antmaze-medium-navigate-v0 | 73048 | 101 | 7056580aa4f5e759e9eb99709e76d8014b3c321cb56b18e37a658ccac49c451b | /data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M9A/M9A-C005__crl_actor_k2_nores/antmaze-medium-navigate-v0/seed_000/train.csv |
| M9A | M9A-C006 | antmaze-large-navigate-v0 | 73166 | 101 | 0d4d07d710eb7f98058a2aba303bdb65766bfbcd9a654f1b90843fa551fdfeef | /data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M9A/M9A-C006__crl_actor_k2_res/antmaze-large-navigate-v0/seed_000/train.csv |
| M9A | M9A-C006 | antmaze-medium-navigate-v0 | 72934 | 101 | b9bf752e0f9177be2e3794e0021244937f6ea6b62092360b4b287b0f6dc665f1 | /data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M9A/M9A-C006__crl_actor_k2_res/antmaze-medium-navigate-v0/seed_000/train.csv |
| M9A | M9A-C007 | antmaze-large-navigate-v0 | 73148 | 101 | 3fb434d301d0cbed62a9199ff5e8452e62b87f51f225e13b934fc0bf6c46a30d | /data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M9A/M9A-C007__crl_actor_k4_nores/antmaze-large-navigate-v0/seed_000/train.csv |
| M9A | M9A-C007 | antmaze-medium-navigate-v0 | 72878 | 101 | daf6ae6df23cc5cf79c082802f3580f4c6d09966573b4e18b736dd9554f35ca8 | /data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M9A/M9A-C007__crl_actor_k4_nores/antmaze-medium-navigate-v0/seed_000/train.csv |
| M9A | M9A-C008 | antmaze-large-navigate-v0 | 73186 | 101 | 031590b7d8e01687cb25fbeca1bfae391eaf1117d11d195be87cdcea570afd89 | /data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M9A/M9A-C008__crl_actor_k4_res/antmaze-large-navigate-v0/seed_000/train.csv |
| M9A | M9A-C008 | antmaze-medium-navigate-v0 | 72878 | 101 | ca879e65d0159b3e47bc27935a2168454532764566e33d30dbc7f8ee5d5c218a | /data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M9A/M9A-C008__crl_actor_k4_res/antmaze-medium-navigate-v0/seed_000/train.csv |
| M9A | M9A-C009 | antmaze-large-navigate-v0 | 58102 | 101 | fc997ebcc3cf7dd7847d76b7d8e0853239faea0811caba7e36816876ace9b82a | /data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M9A/M9A-C009__hiql_high_k1_nores/antmaze-large-navigate-v0/seed_000/train.csv |
| M9A | M9A-C009 | antmaze-medium-navigate-v0 | 58058 | 101 | d298d54e55f3e4221d8f7f52892ddf62d087555c473d9d9aa3f92a0120a724da | /data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M9A/M9A-C009__hiql_high_k1_nores/antmaze-medium-navigate-v0/seed_000/train.csv |
| M9A | M9A-C010 | antmaze-large-navigate-v0 | 58085 | 101 | acfc93b96afd06ab51135acba0f37eb7bc49cbc0058394f04da721ddab52b79f | /data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M9A/M9A-C010__hiql_high_k1_res/antmaze-large-navigate-v0/seed_000/train.csv |
| M9A | M9A-C010 | antmaze-medium-navigate-v0 | 58099 | 101 | 0032aa05b10c063b24589f37fdf8773991baec534b3af23e9ba48930a91ad2c8 | /data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M9A/M9A-C010__hiql_high_k1_res/antmaze-medium-navigate-v0/seed_000/train.csv |
| M9A | M9A-C011 | antmaze-large-navigate-v0 | 58129 | 101 | 7b9d80f85504066badafb39da63271f64d59ee2ca8a322dea98aa7e4e9fa550b | /data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M9A/M9A-C011__hiql_high_k2_nores/antmaze-large-navigate-v0/seed_000/train.csv |
| M9A | M9A-C011 | antmaze-medium-navigate-v0 | 58117 | 101 | 71858be3d1b6e33cbc4b9d01cc36193ba06d0226d884884e46c72798098c369b | /data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M9A/M9A-C011__hiql_high_k2_nores/antmaze-medium-navigate-v0/seed_000/train.csv |
| M9A | M9A-C012 | antmaze-large-navigate-v0 | 58187 | 101 | ceaecaee175ae776026c18169b8c51a8ac589657ee25fd9b1608bc794c9f3084 | /data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M9A/M9A-C012__hiql_high_k2_res/antmaze-large-navigate-v0/seed_000/train.csv |
| M9A | M9A-C012 | antmaze-medium-navigate-v0 | 58065 | 101 | d59ce29fdd444d08a55acb4b802091b0777fa1a90a09329f602e55359b71c8bb | /data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M9A/M9A-C012__hiql_high_k2_res/antmaze-medium-navigate-v0/seed_000/train.csv |
| M9A | M9A-C013 | antmaze-large-navigate-v0 | 58136 | 101 | 55350b107d85a7fcf2ed1cfaa7461ee588ac084a09627f0454096a38f094f83d | /data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M9A/M9A-C013__hiql_high_k4_nores/antmaze-large-navigate-v0/seed_000/train.csv |
| M9A | M9A-C013 | antmaze-medium-navigate-v0 | 58125 | 101 | eb673ffd82d358786445307ecfad07aa45715782766df47d3dc8c6c6781388cb | /data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M9A/M9A-C013__hiql_high_k4_nores/antmaze-medium-navigate-v0/seed_000/train.csv |
| M9A | M9A-C014 | antmaze-large-navigate-v0 | 58134 | 101 | 7bb59b5be3fe00f8f90efa271b3ce90bb0179bc5ebe15cf661414e863904271a | /data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M9A/M9A-C014__hiql_high_k4_res/antmaze-large-navigate-v0/seed_000/train.csv |
| M9A | M9A-C014 | antmaze-medium-navigate-v0 | 58050 | 101 | 45eae5e2a40e3250f29520ba502c3d1f542deec42712454adc6446858854eaea | /data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M9A/M9A-C014__hiql_high_k4_res/antmaze-medium-navigate-v0/seed_000/train.csv |
| M9A | M9A-C015 | antmaze-large-navigate-v0 | 58100 | 101 | b4f7854701248cbffcbf7a40b9e8f50f5b776d043fa10829bc5498c25dd30a9d | /data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M9A/M9A-C015__hiql_low_k1_nores/antmaze-large-navigate-v0/seed_000/train.csv |
| M9A | M9A-C015 | antmaze-medium-navigate-v0 | 58065 | 101 | 8259c040c3cfcb704c3e3b64c2a6316fd5dea288cdf0ff6d77bcb7838ef5923c | /data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M9A/M9A-C015__hiql_low_k1_nores/antmaze-medium-navigate-v0/seed_000/train.csv |
| M9A | M9A-C016 | antmaze-large-navigate-v0 | 58090 | 101 | 233fc4e756e4d54555bc0cbf4d0b525c53e2826741e405714c260b8ddab2ea37 | /data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M9A/M9A-C016__hiql_low_k1_res/antmaze-large-navigate-v0/seed_000/train.csv |
| M9A | M9A-C016 | antmaze-medium-navigate-v0 | 58066 | 101 | 443069c45846590f4a23500a5af71dd312d6bdd1c747c533755a7c1f7c550f4f | /data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M9A/M9A-C016__hiql_low_k1_res/antmaze-medium-navigate-v0/seed_000/train.csv |
| M9A | M9A-C017 | antmaze-large-navigate-v0 | 58070 | 101 | eccb83dfe257c5b38c3f8b73cb9ce168218fc36b0341c8916506f62639a3cbdf | /data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M9A/M9A-C017__hiql_low_k2_nores/antmaze-large-navigate-v0/seed_000/train.csv |
| M9A | M9A-C017 | antmaze-medium-navigate-v0 | 58014 | 101 | 30bef3509f6dcb116157b9895477a86ac94c721fd312b618acd3c61c1af65bd8 | /data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M9A/M9A-C017__hiql_low_k2_nores/antmaze-medium-navigate-v0/seed_000/train.csv |
| M9A | M9A-C018 | antmaze-large-navigate-v0 | 58108 | 101 | 7ad6de5f6c09f212327da4875cf1d103bc8be01901b7413baf2efe5b57164a0a | /data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M9A/M9A-C018__hiql_low_k2_res/antmaze-large-navigate-v0/seed_000/train.csv |
| M9A | M9A-C018 | antmaze-medium-navigate-v0 | 58098 | 101 | f5a48c023165330f2c1b80da80311dd4e9b89dd3e279dd7563c9efdb92c3cd91 | /data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M9A/M9A-C018__hiql_low_k2_res/antmaze-medium-navigate-v0/seed_000/train.csv |
| M9A | M9A-C019 | antmaze-large-navigate-v0 | 58174 | 101 | e014defa0c807d3c515d02f25e4a7841e053dd47331a0060081eb9b8313acf4b | /data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M9A/M9A-C019__hiql_low_k4_nores/antmaze-large-navigate-v0/seed_000/train.csv |
| M9A | M9A-C019 | antmaze-medium-navigate-v0 | 57985 | 101 | 58d377fed016d15568598eff56b09163b1dac3859f97f3b6d9fde8bc7b4fbe8e | /data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M9A/M9A-C019__hiql_low_k4_nores/antmaze-medium-navigate-v0/seed_000/train.csv |
| M9A | M9A-C020 | antmaze-large-navigate-v0 | 58179 | 101 | f2436c148b4d89f78b9e5621132558e9204ca1ac5a43c48b0939dca9a4c0aae2 | /data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M9A/M9A-C020__hiql_low_k4_res/antmaze-large-navigate-v0/seed_000/train.csv |
| M9A | M9A-C020 | antmaze-medium-navigate-v0 | 58134 | 101 | a1437a859edf7c73d2f4fce3426f7f2d2ff6d1ad217d9067c4b6a87dc4661595 | /data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M9A/M9A-C020__hiql_low_k4_res/antmaze-medium-navigate-v0/seed_000/train.csv |
| M9A | M9A-C021 | antmaze-large-navigate-v0 | 58172 | 101 | 6c65bd39568f59727273a10d25f52e6b011157453be6e7a106116e708d42bc91 | /data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M9A/M9A-C021__hiql_high_low_k1_nores/antmaze-large-navigate-v0/seed_000/train.csv |
| M9A | M9A-C021 | antmaze-medium-navigate-v0 | 58114 | 101 | 51dd635f6ab567fc789d90f990314b73a2afd0681d93ea38a84419fe3f3172d6 | /data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M9A/M9A-C021__hiql_high_low_k1_nores/antmaze-medium-navigate-v0/seed_000/train.csv |
| M9A | M9A-C022 | antmaze-large-navigate-v0 | 58143 | 101 | 98fda24779a96581a3262356eb463376d6d8a21d8c70fe64f52df305a922dcc9 | /data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M9A/M9A-C022__hiql_high_low_k1_res/antmaze-large-navigate-v0/seed_000/train.csv |
| M9A | M9A-C022 | antmaze-medium-navigate-v0 | 58129 | 101 | b892bb5b1c3f82feb50a698a38d5e2a85d6ac96d1334870419933f0d8f5178bb | /data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M9A/M9A-C022__hiql_high_low_k1_res/antmaze-medium-navigate-v0/seed_000/train.csv |
| M9A | M9A-C023 | antmaze-large-navigate-v0 | 58125 | 101 | d91c89927f04ee9dd395bca9dcf7c50264bf24b97416fb48f91450a8eb388959 | /data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M9A/M9A-C023__hiql_high_low_k2_nores/antmaze-large-navigate-v0/seed_000/train.csv |
| M9A | M9A-C023 | antmaze-medium-navigate-v0 | 58071 | 101 | 8f35af3ed25f607e22233affb363c8a79ccc7b70646ba0929cc627979fce2722 | /data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M9A/M9A-C023__hiql_high_low_k2_nores/antmaze-medium-navigate-v0/seed_000/train.csv |
| M9A | M9A-C024 | antmaze-large-navigate-v0 | 58162 | 101 | 8db549129bdec63e5dce584809b7a6c4b7a88346c373422f78e302b0a4ae3110 | /data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M9A/M9A-C024__hiql_high_low_k2_res/antmaze-large-navigate-v0/seed_000/train.csv |
| M9A | M9A-C024 | antmaze-medium-navigate-v0 | 58062 | 101 | 203b1ca0096b7ff27151b7949594ecdf09ab53ffe6239af8b27a16e15d9fd83c | /data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M9A/M9A-C024__hiql_high_low_k2_res/antmaze-medium-navigate-v0/seed_000/train.csv |
| M9A | M9A-C025 | antmaze-large-navigate-v0 | 58067 | 101 | a2b0eeb62fa03c29a35cafb50f1efe8cbcb77efe988d466bb8a8fc9a19f6e61e | /data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M9A/M9A-C025__hiql_high_low_k4_nores/antmaze-large-navigate-v0/seed_000/train.csv |
| M9A | M9A-C025 | antmaze-medium-navigate-v0 | 58121 | 101 | 7723c9c99a025f5a750422652525e2a56316bd3627cec7c51448149f36848b46 | /data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M9A/M9A-C025__hiql_high_low_k4_nores/antmaze-medium-navigate-v0/seed_000/train.csv |
| M9A | M9A-C026 | antmaze-large-navigate-v0 | 58198 | 101 | f28ee7d04beb0e3ee8942405eb00b5e8c0c522a8082e54a7ee047d1412f09758 | /data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M9A/M9A-C026__hiql_high_low_k4_res/antmaze-large-navigate-v0/seed_000/train.csv |
| M9A | M9A-C026 | antmaze-medium-navigate-v0 | 58109 | 101 | ed5923cbb50018389bd3430dd3cbf7687f776622a4bbfe076da391d9076f4973 | /data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M9A/M9A-C026__hiql_high_low_k4_res/antmaze-medium-navigate-v0/seed_000/train.csv |
| M9B | M9B-C001 | antmaze-large-navigate-v0 | 73193 | 101 | fddaa29ac633fcf2e724d104f2ab9a817aac5e7d72dad96acb054160bf306f43 | /data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M9B/M9B-C001__crl_actor_h2l1_full_bptt/antmaze-large-navigate-v0/seed_000/train.csv |
| M9B | M9B-C001 | antmaze-medium-navigate-v0 | 72925 | 101 | a056ef9e1c57a125f4d3b00eba44ad52d26e7bc1cfb96f0a9fa83d87e8a2ba98 | /data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M9B/M9B-C001__crl_actor_h2l1_full_bptt/antmaze-medium-navigate-v0/seed_000/train.csv |
| M9B | M9B-C002 | antmaze-large-navigate-v0 | 73192 | 101 | 9a797648d7cad5688dcca17844991b50dcd8175a464564b9bfd26dae55e39871 | /data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M9B/M9B-C002__crl_actor_h2l1_one_step/antmaze-large-navigate-v0/seed_000/train.csv |
| M9B | M9B-C002 | antmaze-medium-navigate-v0 | 72938 | 101 | c39e8cafc7c1c120697e40d2c07839a2380af9a14e602ac843b0ce0e531aa760 | /data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M9B/M9B-C002__crl_actor_h2l1_one_step/antmaze-medium-navigate-v0/seed_000/train.csv |
| M9B | M9B-C003 | antmaze-large-navigate-v0 | 73131 | 101 | 208e58b46cb9e6343f88795fcf20d3dd496de5e1e34a3bd80bd2beb52da2f285 | /data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M9B/M9B-C003__crl_actor_h2l6_full_bptt/antmaze-large-navigate-v0/seed_000/train.csv |
| M9B | M9B-C003 | antmaze-medium-navigate-v0 | 72870 | 101 | e64c8d0131d8b844e5600fda3d3f5e819ab76dd25cd6cbcb6a2de15f932ca65e | /data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M9B/M9B-C003__crl_actor_h2l6_full_bptt/antmaze-medium-navigate-v0/seed_000/train.csv |
| M9B | M9B-C004 | antmaze-large-navigate-v0 | 73115 | 101 | e7be2aed6257b39b1254a9e1831993cfbf0ac75a4c29caf6060452a3ff9f17b5 | /data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M9B/M9B-C004__crl_actor_h2l6_one_step/antmaze-large-navigate-v0/seed_000/train.csv |
| M9B | M9B-C004 | antmaze-medium-navigate-v0 | 72886 | 101 | d35b5a3e4cfc70f17d54b53a64714c355ed752451f6987a91491d31f27c3c992 | /data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M9B/M9B-C004__crl_actor_h2l6_one_step/antmaze-medium-navigate-v0/seed_000/train.csv |
| M9B | M9B-C005 | antmaze-large-navigate-v0 | 58097 | 101 | 512546e70f810467cd3bbe9522c5aaa608fce17664b194ec4ef7f3108473f749 | /data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M9B/M9B-C005__hiql_high_h2l1_full_bptt/antmaze-large-navigate-v0/seed_000/train.csv |
| M9B | M9B-C005 | antmaze-medium-navigate-v0 | 58139 | 101 | 2166c1b4c69fc45d7accd445be8dcc507829fec5a65378049158fa4dd54d1690 | /data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M9B/M9B-C005__hiql_high_h2l1_full_bptt/antmaze-medium-navigate-v0/seed_000/train.csv |
| M9B | M9B-C006 | antmaze-large-navigate-v0 | 58121 | 101 | 864ad06625a43ef4a57871487e137c3bf9a66f324ec101581f7bfad3fd36a853 | /data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M9B/M9B-C006__hiql_high_h2l1_one_step/antmaze-large-navigate-v0/seed_000/train.csv |
| M9B | M9B-C006 | antmaze-medium-navigate-v0 | 58119 | 101 | c92ed72d240f313e1ccb65c1fa6eb8b6af90ae98cc6f5915d2dfe1e71ed05fe9 | /data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M9B/M9B-C006__hiql_high_h2l1_one_step/antmaze-medium-navigate-v0/seed_000/train.csv |
| M9B | M9B-C007 | antmaze-large-navigate-v0 | 58090 | 101 | b5251b6b42fab3054a54c75473f0edc67180efacf4741f514b0187c3bcdf0c9f | /data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M9B/M9B-C007__hiql_high_h2l6_full_bptt/antmaze-large-navigate-v0/seed_000/train.csv |
| M9B | M9B-C007 | antmaze-medium-navigate-v0 | 58172 | 101 | 371603f2f973713e90d5c79ae2dca7e4b289e3931cbb8543314db250de71500e | /data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M9B/M9B-C007__hiql_high_h2l6_full_bptt/antmaze-medium-navigate-v0/seed_000/train.csv |
| M9B | M9B-C008 | antmaze-large-navigate-v0 | 58072 | 101 | 1e2752f1ab074c5be0877224707260d19de0baa23b491621fd71ef09dcdaea73 | /data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M9B/M9B-C008__hiql_high_h2l6_one_step/antmaze-large-navigate-v0/seed_000/train.csv |
| M9B | M9B-C008 | antmaze-medium-navigate-v0 | 58089 | 101 | 4756ae5ff34e6860c20f6f2f8f48ff11d520d3fdfb2c2029b5dfbdf6b01bf0ae | /data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M9B/M9B-C008__hiql_high_h2l6_one_step/antmaze-medium-navigate-v0/seed_000/train.csv |
| M9B | M9B-C009 | antmaze-large-navigate-v0 | 58117 | 101 | 960109bb4b70d972ddef71efe4b857efe613bdfee3edfd6c1c29e72d42617ab1 | /data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M9B/M9B-C009__hiql_low_h2l1_full_bptt/antmaze-large-navigate-v0/seed_000/train.csv |
| M9B | M9B-C009 | antmaze-medium-navigate-v0 | 58108 | 101 | 3ae6bbaac21b6fc49183ee5480e6c4a7f5fd294f09eff623016fefed560f3274 | /data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M9B/M9B-C009__hiql_low_h2l1_full_bptt/antmaze-medium-navigate-v0/seed_000/train.csv |
| M9B | M9B-C010 | antmaze-large-navigate-v0 | 58163 | 101 | 51e70d947e50490e268d2d59d6250933b13c2892d18d6ffc5f6fe6a5484c82e0 | /data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M9B/M9B-C010__hiql_low_h2l1_one_step/antmaze-large-navigate-v0/seed_000/train.csv |
| M9B | M9B-C010 | antmaze-medium-navigate-v0 | 58106 | 101 | 174993136b21b246b5fc4c3da450279ec73cef67fea59aac94dafd826253c97b | /data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M9B/M9B-C010__hiql_low_h2l1_one_step/antmaze-medium-navigate-v0/seed_000/train.csv |
| M9B | M9B-C011 | antmaze-large-navigate-v0 | 58092 | 101 | 443fce3085d72c8ddeea3cb5c695bd9f3dc769eae3ea55eec5df2798362193e2 | /data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M9B/M9B-C011__hiql_low_h2l6_full_bptt/antmaze-large-navigate-v0/seed_000/train.csv |
| M9B | M9B-C011 | antmaze-medium-navigate-v0 | 58171 | 101 | 590ab3c1a93f246e1ba96f485adb14d14e3f1bb38e8a70a1ae4321852c31fbcb | /data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M9B/M9B-C011__hiql_low_h2l6_full_bptt/antmaze-medium-navigate-v0/seed_000/train.csv |
| M9B | M9B-C012 | antmaze-large-navigate-v0 | 57992 | 101 | 7abecd9a2ebe1123ac4921645672fcc4f3d666de2fe9faa4b0cbd2dadab2d07f | /data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M9B/M9B-C012__hiql_low_h2l6_one_step/antmaze-large-navigate-v0/seed_000/train.csv |
| M9B | M9B-C012 | antmaze-medium-navigate-v0 | 58110 | 101 | 71f9524f301ef92b5bb72258d637dd1f967898040505557f5e833b5dfca2612a | /data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M9B/M9B-C012__hiql_low_h2l6_one_step/antmaze-medium-navigate-v0/seed_000/train.csv |
| M9B | M9B-C013 | antmaze-large-navigate-v0 | 58192 | 101 | a2f54b6d8b9834382f528775468af375a84ef84dd8ed6d9d0077b464908d7090 | /data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M9B/M9B-C013__hiql_high_low_h2l1_full_bptt/antmaze-large-navigate-v0/seed_000/train.csv |
| M9B | M9B-C013 | antmaze-medium-navigate-v0 | 58155 | 101 | e6adc7cc71ed5e1bb63c9433dffcd9ac122f0ee003e4cc2d775c0117e3b2ede5 | /data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M9B/M9B-C013__hiql_high_low_h2l1_full_bptt/antmaze-medium-navigate-v0/seed_000/train.csv |
| M9B | M9B-C014 | antmaze-large-navigate-v0 | 58026 | 101 | 7d77f668addcb051595daaad3e30a0e86aef4b49d6e0ba98be4d6b83e6824b41 | /data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M9B/M9B-C014__hiql_high_low_h2l1_one_step/antmaze-large-navigate-v0/seed_000/train.csv |
| M9B | M9B-C014 | antmaze-medium-navigate-v0 | 58093 | 101 | e403f7ad6db07e9e839c8bcdd1fe037468ccb4d825345c16eb95459df4d110ef | /data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M9B/M9B-C014__hiql_high_low_h2l1_one_step/antmaze-medium-navigate-v0/seed_000/train.csv |
| M9B | M9B-C015 | antmaze-large-navigate-v0 | 58088 | 101 | f382bd9d92bd0ec12cc9aae87f247c042a576b2f7b022e61e1ef29f58bd7034b | /data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M9B/M9B-C015__hiql_high_low_h2l6_full_bptt/antmaze-large-navigate-v0/seed_000/train.csv |
| M9B | M9B-C015 | antmaze-medium-navigate-v0 | 58083 | 101 | b77acdfee67fbc0994edef400bad334d184d6d55c1e4ee5878187383e48b2698 | /data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M9B/M9B-C015__hiql_high_low_h2l6_full_bptt/antmaze-medium-navigate-v0/seed_000/train.csv |
| M9B | M9B-C016 | antmaze-large-navigate-v0 | 57992 | 101 | cc2f58e69e73bf064fd8a42a3070706b217a7067ba4b4da6ca94a8ff05faa63c | /data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M9B/M9B-C016__hiql_high_low_h2l6_one_step/antmaze-large-navigate-v0/seed_000/train.csv |
| M9B | M9B-C016 | antmaze-medium-navigate-v0 | 58008 | 101 | 616f90a9e04f3b5fdab70a9fb39866e5e21bc892c73b01f2332ce5f714d3358f | /data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M9B/M9B-C016__hiql_high_low_h2l6_one_step/antmaze-medium-navigate-v0/seed_000/train.csv |

## 13. 原始数据附录 A：eval.csv

以下代码块逐字来自原始 eval.csv 文件。

### M9A

#### M9A/M9A-C001__hiql_vanilla/antmaze-large-navigate-v0/seed_000/eval.csv

```csv
evaluation/task1_success,evaluation/task2_success,evaluation/task3_success,evaluation/task4_success,evaluation/task5_success,evaluation/overall_success,step
0.95,0.0,1.0,0.55,1.0,0.7,100000
0.9,0.35,1.0,1.0,0.9,0.8300000000000001,200000
0.95,0.35,0.9,1.0,0.95,0.8299999999999998,300000
0.9,0.4,1.0,0.95,1.0,0.85,400000
0.9,0.35,0.9,0.95,0.95,0.8099999999999999,500000
0.75,0.3,0.95,0.95,0.95,0.78,600000
1.0,0.25,0.85,1.0,0.95,0.8099999999999999,700000
0.8,0.6,0.9,0.95,1.0,0.85,800000
0.8,0.75,0.8,0.8,1.0,0.8300000000000001,900000
1.0,0.45,1.0,0.95,0.95,0.8700000000000001,1000000

```

#### M9A/M9A-C001__hiql_vanilla/antmaze-medium-navigate-v0/seed_000/eval.csv

```csv
evaluation/task1_success,evaluation/task2_success,evaluation/task3_success,evaluation/task4_success,evaluation/task5_success,evaluation/overall_success,step
1.0,1.0,0.85,0.95,1.0,0.96,100000
1.0,0.95,0.95,0.95,1.0,0.97,200000
0.9,0.95,0.95,1.0,1.0,0.96,300000
0.95,0.95,0.95,1.0,1.0,0.97,400000
1.0,1.0,0.95,0.95,0.95,0.9700000000000001,500000
0.95,1.0,0.9,0.95,0.95,0.95,600000
0.9,0.95,0.95,0.95,1.0,0.95,700000
0.95,1.0,1.0,0.95,1.0,0.9800000000000001,800000
1.0,0.9,0.95,0.95,0.9,0.9400000000000001,900000
0.95,0.95,0.9,1.0,1.0,0.96,1000000

```

#### M9A/M9A-C002__crl_vanilla/antmaze-large-navigate-v0/seed_000/eval.csv

```csv
evaluation/task1_success,evaluation/task2_success,evaluation/task3_success,evaluation/task4_success,evaluation/task5_success,evaluation/overall_success,step
0.75,0.0,0.9,0.95,0.15,0.5499999999999999,100000
0.9,0.05,0.85,0.95,0.5,0.65,200000
1.0,0.0,1.0,0.95,0.55,0.7,300000
0.8,0.05,0.95,0.85,0.35,0.6,400000
1.0,0.1,0.95,0.95,0.75,0.75,500000
0.9,0.1,0.9,0.75,0.55,0.64,600000
0.8,0.15,0.95,0.85,0.5,0.65,700000
0.85,0.2,0.95,0.95,0.25,0.64,800000
0.85,0.5,0.9,1.0,0.6,0.77,900000
0.85,0.0,0.85,1.0,0.4,0.62,1000000

```

#### M9A/M9A-C002__crl_vanilla/antmaze-medium-navigate-v0/seed_000/eval.csv

```csv
evaluation/task1_success,evaluation/task2_success,evaluation/task3_success,evaluation/task4_success,evaluation/task5_success,evaluation/overall_success,step
0.95,0.95,0.9,0.2,1.0,0.8,100000
0.95,1.0,0.7,0.6,0.9,0.8300000000000001,200000
0.8,0.95,0.9,0.9,1.0,0.9099999999999999,300000
1.0,1.0,1.0,1.0,0.95,0.99,400000
0.95,0.9,1.0,1.0,0.95,0.96,500000
1.0,0.9,0.95,0.9,1.0,0.95,600000
1.0,0.9,0.95,0.95,1.0,0.96,700000
0.85,0.95,0.9,1.0,1.0,0.9399999999999998,800000
1.0,0.95,0.9,1.0,1.0,0.97,900000
0.95,0.85,0.95,0.95,0.95,0.93,1000000

```

#### M9A/M9A-C003__crl_actor_k1_nores/antmaze-large-navigate-v0/seed_000/eval.csv

```csv
evaluation/task1_success,evaluation/task2_success,evaluation/task3_success,evaluation/task4_success,evaluation/task5_success,evaluation/overall_success,step
0.95,0.4,0.95,0.8,0.9,0.7999999999999999,100000
0.7,0.15,0.9,0.9,1.0,0.73,200000
0.8,0.35,0.7,0.9,0.9,0.73,300000
0.95,0.8,0.95,0.7,0.85,0.85,400000
0.9,0.75,0.9,0.65,1.0,0.8399999999999999,500000

```

#### M9A/M9A-C003__crl_actor_k1_nores/antmaze-medium-navigate-v0/seed_000/eval.csv

```csv
evaluation/task1_success,evaluation/task2_success,evaluation/task3_success,evaluation/task4_success,evaluation/task5_success,evaluation/overall_success,step
0.95,0.9,0.35,0.3,0.95,0.6900000000000001,100000
1.0,0.9,0.55,0.6,1.0,0.8100000000000002,200000
0.95,1.0,0.7,0.8,1.0,0.89,300000
1.0,0.85,0.75,0.95,0.95,0.9,400000
1.0,0.95,0.8,1.0,0.95,0.9400000000000001,500000

```

#### M9A/M9A-C004__crl_actor_k1_res/antmaze-large-navigate-v0/seed_000/eval.csv

```csv
evaluation/task1_success,evaluation/task2_success,evaluation/task3_success,evaluation/task4_success,evaluation/task5_success,evaluation/overall_success,step
0.85,0.45,0.6,0.9,0.1,0.58,100000
0.85,0.85,0.75,0.8,0.25,0.7,200000
0.9,0.55,0.8,0.9,0.15,0.6599999999999999,300000
1.0,0.7,1.0,0.9,0.35,0.79,400000
0.85,0.75,0.9,0.85,0.95,0.86,500000

```

#### M9A/M9A-C004__crl_actor_k1_res/antmaze-medium-navigate-v0/seed_000/eval.csv

```csv
evaluation/task1_success,evaluation/task2_success,evaluation/task3_success,evaluation/task4_success,evaluation/task5_success,evaluation/overall_success,step
0.9,0.95,0.55,0.05,1.0,0.6900000000000001,100000
1.0,0.95,0.2,0.65,1.0,0.76,200000
1.0,1.0,0.4,0.9,1.0,0.86,300000
0.95,0.95,0.7,1.0,0.95,0.9099999999999999,400000
0.85,0.85,0.8,0.95,1.0,0.89,500000

```

#### M9A/M9A-C005__crl_actor_k2_nores/antmaze-large-navigate-v0/seed_000/eval.csv

```csv
evaluation/task1_success,evaluation/task2_success,evaluation/task3_success,evaluation/task4_success,evaluation/task5_success,evaluation/overall_success,step
1.0,0.1,0.7,0.75,0.75,0.6599999999999999,100000
0.9,0.35,0.8,0.8,0.75,0.72,200000
1.0,0.45,0.9,1.0,1.0,0.8699999999999999,300000
0.95,0.7,0.95,0.95,0.9,0.89,400000
0.9,0.7,1.0,0.85,0.9,0.8700000000000001,500000

```

#### M9A/M9A-C005__crl_actor_k2_nores/antmaze-medium-navigate-v0/seed_000/eval.csv

```csv
evaluation/task1_success,evaluation/task2_success,evaluation/task3_success,evaluation/task4_success,evaluation/task5_success,evaluation/overall_success,step
0.95,0.95,0.35,0.9,1.0,0.8300000000000001,100000
0.85,0.9,0.65,0.95,0.95,0.86,200000
0.95,1.0,1.0,1.0,0.95,0.9800000000000001,300000
0.9,0.95,0.9,0.95,1.0,0.9400000000000001,400000
1.0,0.9,0.9,1.0,1.0,0.96,500000

```

#### M9A/M9A-C006__crl_actor_k2_res/antmaze-large-navigate-v0/seed_000/eval.csv

```csv
evaluation/task1_success,evaluation/task2_success,evaluation/task3_success,evaluation/task4_success,evaluation/task5_success,evaluation/overall_success,step
0.9,0.7,0.9,0.9,0.35,0.75,100000
0.95,0.25,0.9,1.0,0.3,0.6799999999999999,200000
0.95,0.65,0.8,1.0,0.85,0.85,300000
1.0,0.75,1.0,0.85,0.9,0.9,400000
0.95,0.75,0.95,0.9,0.8,0.8699999999999999,500000

```

#### M9A/M9A-C006__crl_actor_k2_res/antmaze-medium-navigate-v0/seed_000/eval.csv

```csv
evaluation/task1_success,evaluation/task2_success,evaluation/task3_success,evaluation/task4_success,evaluation/task5_success,evaluation/overall_success,step
0.85,0.9,0.55,0.1,1.0,0.6799999999999999,100000
0.9,0.95,0.75,0.6,0.95,0.8300000000000001,200000
0.95,0.95,0.95,0.95,1.0,0.96,300000
1.0,1.0,0.95,0.95,0.95,0.9700000000000001,400000
0.95,0.85,0.9,0.85,1.0,0.9099999999999999,500000

```

#### M9A/M9A-C007__crl_actor_k4_nores/antmaze-large-navigate-v0/seed_000/eval.csv

```csv
evaluation/task1_success,evaluation/task2_success,evaluation/task3_success,evaluation/task4_success,evaluation/task5_success,evaluation/overall_success,step
0.85,0.05,1.0,0.85,0.6,0.67,100000
0.85,0.7,0.9,0.9,0.9,0.85,200000
1.0,0.4,1.0,0.95,0.9,0.85,300000
0.95,0.85,0.95,0.95,0.9,0.9200000000000002,400000
1.0,0.85,1.0,0.85,0.85,0.9099999999999999,500000

```

#### M9A/M9A-C007__crl_actor_k4_nores/antmaze-medium-navigate-v0/seed_000/eval.csv

```csv
evaluation/task1_success,evaluation/task2_success,evaluation/task3_success,evaluation/task4_success,evaluation/task5_success,evaluation/overall_success,step
0.9,1.0,0.65,0.45,1.0,0.8,100000
0.95,1.0,0.9,1.0,0.95,0.96,200000
1.0,1.0,1.0,1.0,0.9,0.9800000000000001,300000
0.95,1.0,1.0,0.95,0.95,0.9700000000000001,400000
1.0,0.95,0.95,0.95,1.0,0.97,500000

```

#### M9A/M9A-C008__crl_actor_k4_res/antmaze-large-navigate-v0/seed_000/eval.csv

```csv
evaluation/task1_success,evaluation/task2_success,evaluation/task3_success,evaluation/task4_success,evaluation/task5_success,evaluation/overall_success,step
0.9,0.8,0.9,0.95,0.85,0.8799999999999999,100000
0.95,0.25,1.0,0.95,0.75,0.78,200000
0.95,0.85,0.9,0.95,1.0,0.9299999999999999,300000
1.0,0.8,0.95,0.9,0.9,0.9099999999999999,400000
1.0,0.9,0.95,0.75,0.95,0.9099999999999999,500000

```

#### M9A/M9A-C008__crl_actor_k4_res/antmaze-medium-navigate-v0/seed_000/eval.csv

```csv
evaluation/task1_success,evaluation/task2_success,evaluation/task3_success,evaluation/task4_success,evaluation/task5_success,evaluation/overall_success,step
1.0,0.85,0.8,0.85,1.0,0.9,100000
0.8,1.0,0.65,0.9,0.95,0.86,200000
0.9,1.0,0.95,1.0,0.95,0.96,300000
1.0,0.85,0.95,1.0,0.95,0.95,400000
0.95,0.95,0.95,0.95,0.95,0.95,500000

```

#### M9A/M9A-C009__hiql_high_k1_nores/antmaze-large-navigate-v0/seed_000/eval.csv

```csv
evaluation/task1_success,evaluation/task2_success,evaluation/task3_success,evaluation/task4_success,evaluation/task5_success,evaluation/overall_success,step
0.8,0.05,0.85,0.9,0.85,0.6900000000000001,100000
1.0,0.35,0.9,0.8,0.9,0.7899999999999999,200000
0.9,0.55,1.0,0.95,0.9,0.8600000000000001,300000
0.95,0.8,0.95,0.85,1.0,0.9100000000000001,400000
0.9,0.4,0.95,0.95,0.95,0.8300000000000001,500000

```

#### M9A/M9A-C009__hiql_high_k1_nores/antmaze-medium-navigate-v0/seed_000/eval.csv

```csv
evaluation/task1_success,evaluation/task2_success,evaluation/task3_success,evaluation/task4_success,evaluation/task5_success,evaluation/overall_success,step
0.85,1.0,0.9,0.65,1.0,0.8800000000000001,100000
0.95,0.95,1.0,0.95,1.0,0.97,200000
1.0,0.9,1.0,0.95,0.9,0.95,300000
0.9,0.95,0.95,1.0,1.0,0.96,400000
1.0,1.0,0.95,1.0,0.95,0.9800000000000001,500000

```

#### M9A/M9A-C010__hiql_high_k1_res/antmaze-large-navigate-v0/seed_000/eval.csv

```csv
evaluation/task1_success,evaluation/task2_success,evaluation/task3_success,evaluation/task4_success,evaluation/task5_success,evaluation/overall_success,step
0.8,0.25,0.9,0.75,0.95,0.7300000000000001,100000
0.9,0.4,0.9,1.0,0.95,0.8300000000000001,200000
0.9,0.6,0.95,0.95,1.0,0.8800000000000001,300000
0.85,0.85,1.0,0.95,0.85,0.9,400000
0.95,0.9,0.9,0.95,0.85,0.9099999999999999,500000

```

#### M9A/M9A-C010__hiql_high_k1_res/antmaze-medium-navigate-v0/seed_000/eval.csv

```csv
evaluation/task1_success,evaluation/task2_success,evaluation/task3_success,evaluation/task4_success,evaluation/task5_success,evaluation/overall_success,step
1.0,0.85,0.95,0.9,1.0,0.9399999999999998,100000
0.95,0.8,1.0,1.0,1.0,0.95,200000
1.0,0.95,0.9,0.95,0.95,0.95,300000
0.9,0.95,0.95,0.85,1.0,0.93,400000
1.0,1.0,0.95,1.0,1.0,0.99,500000

```

#### M9A/M9A-C011__hiql_high_k2_nores/antmaze-large-navigate-v0/seed_000/eval.csv

```csv
evaluation/task1_success,evaluation/task2_success,evaluation/task3_success,evaluation/task4_success,evaluation/task5_success,evaluation/overall_success,step
0.95,0.0,0.95,0.85,0.8,0.71,100000
0.75,0.15,0.9,0.95,0.85,0.72,200000
0.8,0.6,1.0,1.0,0.9,0.86,300000
1.0,0.8,0.9,0.95,0.75,0.8800000000000001,400000
0.85,0.85,0.95,1.0,0.95,0.9199999999999999,500000

```

#### M9A/M9A-C011__hiql_high_k2_nores/antmaze-medium-navigate-v0/seed_000/eval.csv

```csv
evaluation/task1_success,evaluation/task2_success,evaluation/task3_success,evaluation/task4_success,evaluation/task5_success,evaluation/overall_success,step
0.9,0.9,0.8,0.95,1.0,0.9099999999999999,100000
1.0,1.0,1.0,0.9,0.85,0.95,200000
1.0,0.9,1.0,0.95,0.95,0.96,300000
0.85,1.0,1.0,0.95,1.0,0.96,400000
0.95,0.95,0.95,1.0,0.9,0.95,500000

```

#### M9A/M9A-C012__hiql_high_k2_res/antmaze-large-navigate-v0/seed_000/eval.csv

```csv
evaluation/task1_success,evaluation/task2_success,evaluation/task3_success,evaluation/task4_success,evaluation/task5_success,evaluation/overall_success,step
0.9,0.35,0.95,0.9,0.95,0.8099999999999999,100000
0.9,0.25,0.95,0.95,0.85,0.78,200000
0.95,0.6,0.85,0.9,0.95,0.85,300000
0.9,0.15,1.0,0.95,0.95,0.79,400000
0.85,0.75,0.9,0.9,0.85,0.85,500000

```

#### M9A/M9A-C012__hiql_high_k2_res/antmaze-medium-navigate-v0/seed_000/eval.csv

```csv
evaluation/task1_success,evaluation/task2_success,evaluation/task3_success,evaluation/task4_success,evaluation/task5_success,evaluation/overall_success,step
1.0,1.0,1.0,0.95,1.0,0.99,100000
0.95,0.95,0.95,0.9,1.0,0.95,200000
1.0,1.0,1.0,0.95,0.95,0.9800000000000001,300000
1.0,0.95,0.9,0.95,1.0,0.96,400000
1.0,0.95,0.85,0.9,1.0,0.9399999999999998,500000

```

#### M9A/M9A-C013__hiql_high_k4_nores/antmaze-large-navigate-v0/seed_000/eval.csv

```csv
evaluation/task1_success,evaluation/task2_success,evaluation/task3_success,evaluation/task4_success,evaluation/task5_success,evaluation/overall_success,step
0.95,0.25,0.85,0.95,0.95,0.79,100000
0.95,0.5,0.95,1.0,1.0,0.8800000000000001,200000
0.85,0.9,0.95,0.95,1.0,0.93,300000
0.9,0.9,1.0,0.9,0.8,0.9,400000
0.8,0.9,0.95,0.9,0.7,0.85,500000

```

#### M9A/M9A-C013__hiql_high_k4_nores/antmaze-medium-navigate-v0/seed_000/eval.csv

```csv
evaluation/task1_success,evaluation/task2_success,evaluation/task3_success,evaluation/task4_success,evaluation/task5_success,evaluation/overall_success,step
0.95,0.85,0.85,0.55,1.0,0.8400000000000001,100000
1.0,0.95,0.95,0.95,1.0,0.97,200000
0.95,1.0,1.0,0.95,1.0,0.9800000000000001,300000
1.0,1.0,0.9,1.0,1.0,0.9800000000000001,400000
0.95,0.95,0.85,0.85,1.0,0.9199999999999999,500000

```

#### M9A/M9A-C014__hiql_high_k4_res/antmaze-large-navigate-v0/seed_000/eval.csv

```csv
evaluation/task1_success,evaluation/task2_success,evaluation/task3_success,evaluation/task4_success,evaluation/task5_success,evaluation/overall_success,step
0.9,0.5,0.95,0.85,0.65,0.7699999999999999,100000
1.0,0.75,0.95,0.85,0.9,0.89,200000
0.8,0.8,0.9,0.95,0.9,0.8700000000000001,300000
1.0,0.8,1.0,1.0,0.9,0.9400000000000001,400000
0.95,0.9,0.95,1.0,0.9,0.9400000000000001,500000

```

#### M9A/M9A-C014__hiql_high_k4_res/antmaze-medium-navigate-v0/seed_000/eval.csv

```csv
evaluation/task1_success,evaluation/task2_success,evaluation/task3_success,evaluation/task4_success,evaluation/task5_success,evaluation/overall_success,step
1.0,0.9,1.0,0.85,0.95,0.9400000000000001,100000
0.8,0.95,0.95,0.9,1.0,0.9199999999999999,200000
1.0,1.0,0.95,0.95,1.0,0.9800000000000001,300000
1.0,1.0,0.95,1.0,1.0,0.99,400000
0.95,1.0,0.9,0.95,0.95,0.95,500000

```

#### M9A/M9A-C015__hiql_low_k1_nores/antmaze-large-navigate-v0/seed_000/eval.csv

```csv
evaluation/task1_success,evaluation/task2_success,evaluation/task3_success,evaluation/task4_success,evaluation/task5_success,evaluation/overall_success,step
0.9,0.3,0.8,0.75,0.85,0.72,100000
0.9,0.25,1.0,1.0,1.0,0.8300000000000001,200000
0.95,0.15,0.95,1.0,0.95,0.8,300000
0.9,0.2,0.9,0.85,1.0,0.77,400000
0.9,0.45,1.0,1.0,0.85,0.8400000000000001,500000

```

#### M9A/M9A-C015__hiql_low_k1_nores/antmaze-medium-navigate-v0/seed_000/eval.csv

```csv
evaluation/task1_success,evaluation/task2_success,evaluation/task3_success,evaluation/task4_success,evaluation/task5_success,evaluation/overall_success,step
0.95,0.9,0.95,1.0,0.8,0.9199999999999999,100000
0.95,0.95,0.95,0.9,1.0,0.95,200000
0.95,0.95,0.9,0.9,0.95,0.9299999999999999,300000
1.0,0.85,0.85,1.0,1.0,0.9400000000000001,400000
1.0,0.9,1.0,1.0,0.95,0.97,500000

```

#### M9A/M9A-C016__hiql_low_k1_res/antmaze-large-navigate-v0/seed_000/eval.csv

```csv
evaluation/task1_success,evaluation/task2_success,evaluation/task3_success,evaluation/task4_success,evaluation/task5_success,evaluation/overall_success,step
0.75,0.05,0.85,0.75,0.75,0.63,100000
0.95,0.25,0.85,0.8,0.95,0.76,200000
1.0,0.45,1.0,1.0,0.95,0.8800000000000001,300000
0.85,0.85,1.0,0.95,0.95,0.9200000000000002,400000
0.85,0.6,0.95,0.9,0.95,0.85,500000

```

#### M9A/M9A-C016__hiql_low_k1_res/antmaze-medium-navigate-v0/seed_000/eval.csv

```csv
evaluation/task1_success,evaluation/task2_success,evaluation/task3_success,evaluation/task4_success,evaluation/task5_success,evaluation/overall_success,step
0.95,0.95,0.8,0.9,0.9,0.9,100000
0.95,0.95,0.9,0.9,0.95,0.9299999999999999,200000
0.95,1.0,0.9,0.95,1.0,0.96,300000
0.9,0.95,0.9,0.9,1.0,0.93,400000
0.95,1.0,1.0,0.9,0.95,0.96,500000

```

#### M9A/M9A-C017__hiql_low_k2_nores/antmaze-large-navigate-v0/seed_000/eval.csv

```csv
evaluation/task1_success,evaluation/task2_success,evaluation/task3_success,evaluation/task4_success,evaluation/task5_success,evaluation/overall_success,step
0.75,0.05,0.9,0.85,0.95,0.7,100000
0.95,0.35,1.0,0.8,0.9,0.7999999999999999,200000
0.95,0.7,0.9,0.9,0.85,0.86,300000
0.85,0.55,0.95,0.85,0.9,0.82,400000
0.85,0.85,0.9,1.0,0.85,0.89,500000

```

#### M9A/M9A-C017__hiql_low_k2_nores/antmaze-medium-navigate-v0/seed_000/eval.csv

```csv
evaluation/task1_success,evaluation/task2_success,evaluation/task3_success,evaluation/task4_success,evaluation/task5_success,evaluation/overall_success,step
0.9,1.0,0.95,0.95,0.9,0.9400000000000001,100000
1.0,1.0,1.0,0.85,1.0,0.97,200000
0.9,1.0,0.9,0.95,0.9,0.93,300000
0.9,1.0,1.0,0.95,1.0,0.97,400000
1.0,0.95,0.95,0.9,0.95,0.95,500000

```

#### M9A/M9A-C018__hiql_low_k2_res/antmaze-large-navigate-v0/seed_000/eval.csv

```csv
evaluation/task1_success,evaluation/task2_success,evaluation/task3_success,evaluation/task4_success,evaluation/task5_success,evaluation/overall_success,step
0.85,0.05,1.0,0.9,0.95,0.75,100000
1.0,0.25,0.95,0.95,0.95,0.8200000000000001,200000
0.95,0.4,1.0,0.9,0.85,0.82,300000
0.95,0.8,0.95,0.85,0.95,0.9,400000
0.95,0.6,0.95,0.9,0.9,0.86,500000

```

#### M9A/M9A-C018__hiql_low_k2_res/antmaze-medium-navigate-v0/seed_000/eval.csv

```csv
evaluation/task1_success,evaluation/task2_success,evaluation/task3_success,evaluation/task4_success,evaluation/task5_success,evaluation/overall_success,step
0.95,1.0,0.9,0.95,1.0,0.96,100000
0.95,1.0,0.85,0.95,1.0,0.95,200000
1.0,0.95,0.95,0.95,1.0,0.97,300000
0.95,0.8,0.75,0.95,0.9,0.8700000000000001,400000
0.9,1.0,0.95,0.95,1.0,0.96,500000

```

#### M9A/M9A-C019__hiql_low_k4_nores/antmaze-large-navigate-v0/seed_000/eval.csv

```csv
evaluation/task1_success,evaluation/task2_success,evaluation/task3_success,evaluation/task4_success,evaluation/task5_success,evaluation/overall_success,step
0.7,0.05,0.95,0.95,0.85,0.7,100000
0.95,0.35,1.0,0.95,0.95,0.8400000000000001,200000
0.9,0.85,0.95,0.95,0.95,0.9200000000000002,300000
0.95,0.7,0.9,0.85,1.0,0.8800000000000001,400000
1.0,0.7,1.0,1.0,0.95,0.93,500000

```

#### M9A/M9A-C019__hiql_low_k4_nores/antmaze-medium-navigate-v0/seed_000/eval.csv

```csv
evaluation/task1_success,evaluation/task2_success,evaluation/task3_success,evaluation/task4_success,evaluation/task5_success,evaluation/overall_success,step
0.95,1.0,0.85,0.75,0.95,0.9,100000
1.0,0.9,1.0,0.95,1.0,0.97,200000
0.95,0.95,0.9,0.95,0.95,0.9400000000000001,300000
0.85,0.85,1.0,0.9,0.95,0.9099999999999999,400000
0.9,0.95,0.95,0.95,0.95,0.9400000000000001,500000

```

#### M9A/M9A-C020__hiql_low_k4_res/antmaze-large-navigate-v0/seed_000/eval.csv

```csv
evaluation/task1_success,evaluation/task2_success,evaluation/task3_success,evaluation/task4_success,evaluation/task5_success,evaluation/overall_success,step
0.75,0.1,1.0,0.75,0.85,0.6900000000000001,100000
0.9,0.3,0.95,0.85,0.75,0.75,200000
0.9,0.8,1.0,0.95,1.0,0.93,300000
0.7,0.8,1.0,0.95,0.95,0.8800000000000001,400000
0.95,0.8,0.95,0.9,0.85,0.89,500000

```

#### M9A/M9A-C020__hiql_low_k4_res/antmaze-medium-navigate-v0/seed_000/eval.csv

```csv
evaluation/task1_success,evaluation/task2_success,evaluation/task3_success,evaluation/task4_success,evaluation/task5_success,evaluation/overall_success,step
0.95,0.9,1.0,0.7,1.0,0.9099999999999999,100000
0.95,0.9,1.0,0.9,0.95,0.9400000000000001,200000
0.95,0.95,1.0,0.95,0.95,0.96,300000
0.95,0.9,0.95,1.0,0.95,0.95,400000
0.95,0.9,0.95,0.95,0.95,0.9400000000000001,500000

```

#### M9A/M9A-C021__hiql_high_low_k1_nores/antmaze-large-navigate-v0/seed_000/eval.csv

```csv
evaluation/task1_success,evaluation/task2_success,evaluation/task3_success,evaluation/task4_success,evaluation/task5_success,evaluation/overall_success,step
0.8,0.25,1.0,0.9,0.75,0.74,100000
0.95,0.3,0.9,0.95,0.8,0.7799999999999999,200000
0.8,0.45,0.9,0.9,0.9,0.7899999999999999,300000
0.75,0.25,0.95,0.95,0.9,0.76,400000
1.0,0.75,1.0,0.95,0.9,0.9200000000000002,500000

```

#### M9A/M9A-C021__hiql_high_low_k1_nores/antmaze-medium-navigate-v0/seed_000/eval.csv

```csv
evaluation/task1_success,evaluation/task2_success,evaluation/task3_success,evaluation/task4_success,evaluation/task5_success,evaluation/overall_success,step
0.95,0.85,0.9,1.0,0.9,0.9199999999999999,100000
0.9,0.9,0.85,0.95,1.0,0.9199999999999999,200000
1.0,0.95,0.95,0.75,1.0,0.93,300000
1.0,1.0,0.95,1.0,0.9,0.9700000000000001,400000
0.95,0.95,1.0,1.0,0.95,0.97,500000

```

#### M9A/M9A-C022__hiql_high_low_k1_res/antmaze-large-navigate-v0/seed_000/eval.csv

```csv
evaluation/task1_success,evaluation/task2_success,evaluation/task3_success,evaluation/task4_success,evaluation/task5_success,evaluation/overall_success,step
0.75,0.0,0.9,0.85,0.9,0.6799999999999999,100000
0.9,0.2,1.0,0.95,0.85,0.78,200000
0.95,0.5,0.9,1.0,0.9,0.85,300000
1.0,0.9,0.85,0.9,1.0,0.93,400000
0.85,0.6,0.85,0.85,0.95,0.82,500000

```

#### M9A/M9A-C022__hiql_high_low_k1_res/antmaze-medium-navigate-v0/seed_000/eval.csv

```csv
evaluation/task1_success,evaluation/task2_success,evaluation/task3_success,evaluation/task4_success,evaluation/task5_success,evaluation/overall_success,step
1.0,0.95,0.9,1.0,1.0,0.97,100000
0.95,1.0,0.8,0.85,0.95,0.9099999999999999,200000
0.85,0.95,0.9,0.95,1.0,0.9299999999999999,300000
0.95,0.9,0.85,0.85,0.95,0.9,400000
0.9,0.9,0.95,0.9,1.0,0.93,500000

```

#### M9A/M9A-C023__hiql_high_low_k2_nores/antmaze-large-navigate-v0/seed_000/eval.csv

```csv
evaluation/task1_success,evaluation/task2_success,evaluation/task3_success,evaluation/task4_success,evaluation/task5_success,evaluation/overall_success,step
0.95,0.05,0.9,0.85,0.85,0.72,100000
0.9,0.45,1.0,0.95,0.85,0.8299999999999998,200000
0.85,0.55,0.9,0.95,0.9,0.8300000000000001,300000
0.9,0.2,1.0,0.95,0.95,0.8,400000
0.75,0.5,1.0,0.95,0.95,0.8300000000000001,500000

```

#### M9A/M9A-C023__hiql_high_low_k2_nores/antmaze-medium-navigate-v0/seed_000/eval.csv

```csv
evaluation/task1_success,evaluation/task2_success,evaluation/task3_success,evaluation/task4_success,evaluation/task5_success,evaluation/overall_success,step
0.95,0.95,0.8,0.75,0.95,0.8800000000000001,100000
0.95,0.9,1.0,0.95,1.0,0.96,200000
0.9,0.95,0.9,0.85,0.9,0.9,300000
0.9,0.85,1.0,1.0,1.0,0.95,400000
0.9,0.95,0.95,0.95,0.85,0.9199999999999999,500000

```

#### M9A/M9A-C024__hiql_high_low_k2_res/antmaze-large-navigate-v0/seed_000/eval.csv

```csv
evaluation/task1_success,evaluation/task2_success,evaluation/task3_success,evaluation/task4_success,evaluation/task5_success,evaluation/overall_success,step
0.8,0.0,0.9,0.75,0.85,0.66,100000
0.9,0.4,0.85,1.0,0.95,0.82,200000
0.95,0.75,1.0,1.0,1.0,0.9400000000000001,300000
0.8,0.8,1.0,1.0,0.9,0.9,400000
0.95,0.8,0.9,0.85,0.9,0.8800000000000001,500000

```

#### M9A/M9A-C024__hiql_high_low_k2_res/antmaze-medium-navigate-v0/seed_000/eval.csv

```csv
evaluation/task1_success,evaluation/task2_success,evaluation/task3_success,evaluation/task4_success,evaluation/task5_success,evaluation/overall_success,step
1.0,0.95,1.0,1.0,1.0,0.99,100000
0.95,1.0,1.0,0.85,1.0,0.9600000000000002,200000
0.9,0.95,0.95,0.95,1.0,0.95,300000
1.0,1.0,0.95,0.9,1.0,0.97,400000
0.95,0.85,1.0,0.95,0.75,0.9,500000

```

#### M9A/M9A-C025__hiql_high_low_k4_nores/antmaze-large-navigate-v0/seed_000/eval.csv

```csv
evaluation/task1_success,evaluation/task2_success,evaluation/task3_success,evaluation/task4_success,evaluation/task5_success,evaluation/overall_success,step
0.9,0.35,0.9,0.9,1.0,0.8099999999999999,100000
0.9,0.6,0.9,0.95,0.95,0.86,200000
0.8,0.75,0.95,0.95,1.0,0.89,300000
1.0,0.7,0.95,0.95,1.0,0.9199999999999999,400000
0.75,0.45,0.9,0.85,1.0,0.79,500000

```

#### M9A/M9A-C025__hiql_high_low_k4_nores/antmaze-medium-navigate-v0/seed_000/eval.csv

```csv
evaluation/task1_success,evaluation/task2_success,evaluation/task3_success,evaluation/task4_success,evaluation/task5_success,evaluation/overall_success,step
1.0,0.85,1.0,0.85,1.0,0.9400000000000001,100000
0.95,0.9,0.95,0.95,1.0,0.95,200000
0.95,0.95,1.0,0.95,1.0,0.97,300000
0.95,0.95,1.0,1.0,1.0,0.9800000000000001,400000
0.95,1.0,0.95,1.0,1.0,0.9800000000000001,500000

```

#### M9A/M9A-C026__hiql_high_low_k4_res/antmaze-large-navigate-v0/seed_000/eval.csv

```csv
evaluation/task1_success,evaluation/task2_success,evaluation/task3_success,evaluation/task4_success,evaluation/task5_success,evaluation/overall_success,step
0.85,0.25,0.9,0.9,0.95,0.7699999999999999,100000
1.0,0.8,0.95,0.95,0.95,0.93,200000
0.9,0.9,1.0,1.0,0.85,0.9299999999999999,300000
0.95,0.75,0.95,0.9,0.9,0.89,400000
0.75,0.85,0.9,0.95,0.9,0.8700000000000001,500000

```

#### M9A/M9A-C026__hiql_high_low_k4_res/antmaze-medium-navigate-v0/seed_000/eval.csv

```csv
evaluation/task1_success,evaluation/task2_success,evaluation/task3_success,evaluation/task4_success,evaluation/task5_success,evaluation/overall_success,step
0.9,1.0,1.0,0.9,1.0,0.96,100000
0.85,1.0,1.0,1.0,0.95,0.96,200000
0.95,0.95,1.0,1.0,0.95,0.97,300000
1.0,0.95,0.95,0.95,0.95,0.96,400000
1.0,0.9,0.95,1.0,1.0,0.97,500000

```

### M9B

#### M9B/M9B-C001__crl_actor_h2l1_full_bptt/antmaze-large-navigate-v0/seed_000/eval.csv

```csv
evaluation/task1_success,evaluation/task2_success,evaluation/task3_success,evaluation/task4_success,evaluation/task5_success,evaluation/overall_success,step
0.95,0.65,0.75,0.85,0.8,0.8,100000
0.85,0.5,0.95,0.95,1.0,0.85,200000
0.95,0.8,0.9,1.0,0.85,0.9,300000
0.9,0.9,0.9,0.8,0.95,0.89,400000
0.85,0.75,0.9,0.9,0.8,0.8400000000000001,500000

```

#### M9B/M9B-C001__crl_actor_h2l1_full_bptt/antmaze-medium-navigate-v0/seed_000/eval.csv

```csv
evaluation/task1_success,evaluation/task2_success,evaluation/task3_success,evaluation/task4_success,evaluation/task5_success,evaluation/overall_success,step
0.95,0.95,1.0,0.8,0.9,0.9200000000000002,100000
1.0,0.9,0.8,0.9,0.85,0.89,200000
0.85,0.95,0.95,0.95,1.0,0.9400000000000001,300000
1.0,1.0,1.0,0.9,1.0,0.9800000000000001,400000
0.95,0.95,0.95,1.0,0.85,0.9399999999999998,500000

```

#### M9B/M9B-C002__crl_actor_h2l1_one_step/antmaze-large-navigate-v0/seed_000/eval.csv

```csv
evaluation/task1_success,evaluation/task2_success,evaluation/task3_success,evaluation/task4_success,evaluation/task5_success,evaluation/overall_success,step
0.9,0.3,1.0,0.65,0.25,0.62,100000
0.95,0.55,0.75,0.75,0.8,0.76,200000
1.0,0.6,0.85,0.75,0.85,0.8099999999999999,300000
0.9,0.7,0.8,0.8,0.65,0.77,400000
0.85,0.75,0.9,0.95,0.85,0.86,500000

```

#### M9B/M9B-C002__crl_actor_h2l1_one_step/antmaze-medium-navigate-v0/seed_000/eval.csv

```csv
evaluation/task1_success,evaluation/task2_success,evaluation/task3_success,evaluation/task4_success,evaluation/task5_success,evaluation/overall_success,step
0.95,0.85,0.6,0.55,0.85,0.76,100000
1.0,0.95,0.9,0.9,1.0,0.95,200000
0.95,0.95,0.85,1.0,1.0,0.95,300000
0.95,0.9,0.95,0.95,0.8,0.9099999999999999,400000
1.0,0.95,0.85,1.0,1.0,0.96,500000

```

#### M9B/M9B-C003__crl_actor_h2l6_full_bptt/antmaze-large-navigate-v0/seed_000/eval.csv

```csv
evaluation/task1_success,evaluation/task2_success,evaluation/task3_success,evaluation/task4_success,evaluation/task5_success,evaluation/overall_success,step
0.95,0.75,0.95,0.95,0.55,0.8299999999999998,100000
0.85,0.65,0.9,0.75,0.95,0.82,200000
0.95,0.45,0.8,0.9,0.95,0.8099999999999999,300000
1.0,0.8,0.9,1.0,0.85,0.9099999999999999,400000
1.0,0.75,0.85,0.9,0.95,0.89,500000

```

#### M9B/M9B-C003__crl_actor_h2l6_full_bptt/antmaze-medium-navigate-v0/seed_000/eval.csv

```csv
evaluation/task1_success,evaluation/task2_success,evaluation/task3_success,evaluation/task4_success,evaluation/task5_success,evaluation/overall_success,step
1.0,0.85,1.0,0.9,0.95,0.9400000000000001,100000
0.95,0.95,0.9,1.0,0.95,0.95,200000
1.0,0.9,0.95,0.9,1.0,0.95,300000
1.0,0.95,0.95,0.9,1.0,0.96,400000
0.85,1.0,1.0,0.95,0.95,0.95,500000

```

#### M9B/M9B-C004__crl_actor_h2l6_one_step/antmaze-large-navigate-v0/seed_000/eval.csv

```csv
evaluation/task1_success,evaluation/task2_success,evaluation/task3_success,evaluation/task4_success,evaluation/task5_success,evaluation/overall_success,step
0.75,0.0,0.9,0.3,0.05,0.4,100000
0.6,0.15,0.95,0.35,0.3,0.4699999999999999,200000
0.7,0.2,0.75,0.65,0.15,0.48999999999999994,300000
0.8,0.35,0.7,0.7,0.2,0.55,400000
0.95,0.5,0.65,0.65,0.75,0.7,500000

```

#### M9B/M9B-C004__crl_actor_h2l6_one_step/antmaze-medium-navigate-v0/seed_000/eval.csv

```csv
evaluation/task1_success,evaluation/task2_success,evaluation/task3_success,evaluation/task4_success,evaluation/task5_success,evaluation/overall_success,step
0.9,0.85,0.55,0.25,0.8,0.6699999999999999,100000
1.0,0.95,0.65,0.55,0.55,0.74,200000
0.75,0.9,0.5,0.8,0.75,0.74,300000
0.85,0.85,0.45,0.8,0.85,0.76,400000
0.95,1.0,0.55,0.75,0.95,0.8400000000000001,500000

```

#### M9B/M9B-C005__hiql_high_h2l1_full_bptt/antmaze-large-navigate-v0/seed_000/eval.csv

```csv
evaluation/task1_success,evaluation/task2_success,evaluation/task3_success,evaluation/task4_success,evaluation/task5_success,evaluation/overall_success,step
0.9,0.25,0.9,0.95,0.95,0.79,100000
0.9,0.5,0.95,0.9,0.9,0.8299999999999998,200000
0.8,0.7,1.0,0.95,0.95,0.8800000000000001,300000
0.85,0.85,1.0,1.0,0.85,0.9099999999999999,400000
0.75,0.95,1.0,0.9,0.85,0.89,500000

```

#### M9B/M9B-C005__hiql_high_h2l1_full_bptt/antmaze-medium-navigate-v0/seed_000/eval.csv

```csv
evaluation/task1_success,evaluation/task2_success,evaluation/task3_success,evaluation/task4_success,evaluation/task5_success,evaluation/overall_success,step
1.0,1.0,1.0,1.0,1.0,1.0,100000
1.0,1.0,0.95,0.95,1.0,0.9800000000000001,200000
1.0,0.95,1.0,0.95,1.0,0.9800000000000001,300000
0.9,1.0,1.0,1.0,1.0,0.9800000000000001,400000
1.0,1.0,0.95,0.9,1.0,0.97,500000

```

#### M9B/M9B-C006__hiql_high_h2l1_one_step/antmaze-large-navigate-v0/seed_000/eval.csv

```csv
evaluation/task1_success,evaluation/task2_success,evaluation/task3_success,evaluation/task4_success,evaluation/task5_success,evaluation/overall_success,step
1.0,0.05,0.9,0.85,0.8,0.7200000000000001,100000
0.95,0.4,1.0,0.95,0.95,0.85,200000
0.8,0.75,0.95,0.95,0.95,0.8800000000000001,300000
1.0,0.75,1.0,1.0,0.9,0.93,400000
0.95,0.85,0.95,0.95,0.95,0.93,500000

```

#### M9B/M9B-C006__hiql_high_h2l1_one_step/antmaze-medium-navigate-v0/seed_000/eval.csv

```csv
evaluation/task1_success,evaluation/task2_success,evaluation/task3_success,evaluation/task4_success,evaluation/task5_success,evaluation/overall_success,step
0.95,1.0,1.0,0.95,1.0,0.9800000000000001,100000
1.0,1.0,0.95,1.0,1.0,0.99,200000
1.0,0.95,0.9,0.95,0.95,0.95,300000
0.95,1.0,0.85,0.9,0.9,0.9199999999999999,400000
1.0,1.0,1.0,1.0,0.95,0.99,500000

```

#### M9B/M9B-C007__hiql_high_h2l6_full_bptt/antmaze-large-navigate-v0/seed_000/eval.csv

```csv
evaluation/task1_success,evaluation/task2_success,evaluation/task3_success,evaluation/task4_success,evaluation/task5_success,evaluation/overall_success,step
0.8,0.2,0.8,0.95,0.85,0.72,100000
0.85,0.25,1.0,0.95,0.9,0.7899999999999999,200000
0.8,0.65,0.95,1.0,0.8,0.8400000000000001,300000
0.85,0.75,0.95,0.85,0.9,0.86,400000
0.9,0.8,0.9,1.0,0.8,0.8800000000000001,500000

```

#### M9B/M9B-C007__hiql_high_h2l6_full_bptt/antmaze-medium-navigate-v0/seed_000/eval.csv

```csv
evaluation/task1_success,evaluation/task2_success,evaluation/task3_success,evaluation/task4_success,evaluation/task5_success,evaluation/overall_success,step
0.9,0.9,0.9,0.9,1.0,0.9199999999999999,100000
1.0,1.0,1.0,0.95,1.0,0.99,200000
1.0,1.0,0.9,0.95,0.85,0.9399999999999998,300000
1.0,0.95,1.0,1.0,1.0,0.99,400000
1.0,1.0,0.95,1.0,1.0,0.99,500000

```

#### M9B/M9B-C008__hiql_high_h2l6_one_step/antmaze-large-navigate-v0/seed_000/eval.csv

```csv
evaluation/task1_success,evaluation/task2_success,evaluation/task3_success,evaluation/task4_success,evaluation/task5_success,evaluation/overall_success,step
0.75,0.0,0.9,0.95,0.9,0.7,100000
0.9,0.25,0.9,0.95,0.9,0.78,200000
1.0,0.25,0.95,0.9,0.95,0.8099999999999999,300000
0.9,0.5,1.0,0.95,0.85,0.8399999999999999,400000
0.95,0.5,0.95,0.9,0.95,0.85,500000

```

#### M9B/M9B-C008__hiql_high_h2l6_one_step/antmaze-medium-navigate-v0/seed_000/eval.csv

```csv
evaluation/task1_success,evaluation/task2_success,evaluation/task3_success,evaluation/task4_success,evaluation/task5_success,evaluation/overall_success,step
1.0,0.85,0.85,0.9,0.95,0.9099999999999999,100000
0.95,1.0,1.0,1.0,0.85,0.96,200000
1.0,1.0,0.95,0.9,1.0,0.97,300000
1.0,0.85,1.0,0.9,1.0,0.95,400000
1.0,0.95,1.0,0.95,0.95,0.9700000000000001,500000

```

#### M9B/M9B-C009__hiql_low_h2l1_full_bptt/antmaze-large-navigate-v0/seed_000/eval.csv

```csv
evaluation/task1_success,evaluation/task2_success,evaluation/task3_success,evaluation/task4_success,evaluation/task5_success,evaluation/overall_success,step
0.95,0.05,0.85,0.85,0.85,0.7100000000000001,100000
0.95,0.0,0.85,0.9,0.8,0.7,200000
0.9,0.1,0.95,0.95,0.9,0.76,300000
0.9,0.15,0.9,0.95,0.95,0.7700000000000001,400000
0.7,0.25,0.95,0.9,0.85,0.73,500000

```

#### M9B/M9B-C009__hiql_low_h2l1_full_bptt/antmaze-medium-navigate-v0/seed_000/eval.csv

```csv
evaluation/task1_success,evaluation/task2_success,evaluation/task3_success,evaluation/task4_success,evaluation/task5_success,evaluation/overall_success,step
0.95,1.0,0.7,0.8,0.95,0.8800000000000001,100000
0.95,0.9,1.0,0.85,0.9,0.9200000000000002,200000
0.85,0.95,0.95,1.0,0.95,0.9400000000000001,300000
0.95,0.95,1.0,0.95,1.0,0.97,400000
0.95,1.0,0.95,0.9,0.95,0.95,500000

```

#### M9B/M9B-C010__hiql_low_h2l1_one_step/antmaze-large-navigate-v0/seed_000/eval.csv

```csv
evaluation/task1_success,evaluation/task2_success,evaluation/task3_success,evaluation/task4_success,evaluation/task5_success,evaluation/overall_success,step
0.9,0.1,0.9,0.9,0.95,0.75,100000
0.8,0.45,0.95,0.95,0.9,0.8100000000000002,200000
0.9,0.55,0.95,1.0,0.9,0.8600000000000001,300000
0.95,0.85,0.95,0.9,0.95,0.9199999999999999,400000
0.8,0.6,0.95,0.9,0.8,0.8099999999999999,500000

```

#### M9B/M9B-C010__hiql_low_h2l1_one_step/antmaze-medium-navigate-v0/seed_000/eval.csv

```csv
evaluation/task1_success,evaluation/task2_success,evaluation/task3_success,evaluation/task4_success,evaluation/task5_success,evaluation/overall_success,step
0.9,0.95,0.9,0.85,0.9,0.9,100000
1.0,1.0,0.9,0.9,0.85,0.9299999999999999,200000
0.95,1.0,1.0,0.9,0.9,0.95,300000
0.95,0.9,0.95,0.9,1.0,0.9399999999999998,400000
0.8,0.9,0.85,0.95,0.95,0.89,500000

```

#### M9B/M9B-C011__hiql_low_h2l6_full_bptt/antmaze-large-navigate-v0/seed_000/eval.csv

```csv
evaluation/task1_success,evaluation/task2_success,evaluation/task3_success,evaluation/task4_success,evaluation/task5_success,evaluation/overall_success,step
0.65,0.1,0.9,0.8,0.65,0.62,100000
0.9,0.35,0.85,0.85,0.95,0.78,200000
0.95,0.5,0.95,0.9,0.85,0.8299999999999998,300000
0.9,0.2,0.85,0.75,0.75,0.6900000000000001,400000
0.85,0.45,0.95,0.75,1.0,0.8,500000

```

#### M9B/M9B-C011__hiql_low_h2l6_full_bptt/antmaze-medium-navigate-v0/seed_000/eval.csv

```csv
evaluation/task1_success,evaluation/task2_success,evaluation/task3_success,evaluation/task4_success,evaluation/task5_success,evaluation/overall_success,step
1.0,1.0,0.8,0.95,1.0,0.95,100000
1.0,1.0,1.0,1.0,1.0,1.0,200000
0.9,1.0,0.9,0.95,1.0,0.95,300000
0.9,0.95,0.9,0.9,1.0,0.93,400000
0.95,1.0,0.95,0.95,0.95,0.96,500000

```

#### M9B/M9B-C012__hiql_low_h2l6_one_step/antmaze-large-navigate-v0/seed_000/eval.csv

```csv
evaluation/task1_success,evaluation/task2_success,evaluation/task3_success,evaluation/task4_success,evaluation/task5_success,evaluation/overall_success,step
0.85,0.05,0.9,0.65,0.8,0.65,100000
0.75,0.1,0.95,0.95,0.85,0.72,200000
1.0,0.2,0.95,0.75,0.85,0.75,300000
0.65,0.4,0.85,0.85,0.8,0.71,400000
0.45,0.85,0.95,0.85,0.9,0.8,500000

```

#### M9B/M9B-C012__hiql_low_h2l6_one_step/antmaze-medium-navigate-v0/seed_000/eval.csv

```csv
evaluation/task1_success,evaluation/task2_success,evaluation/task3_success,evaluation/task4_success,evaluation/task5_success,evaluation/overall_success,step
0.75,0.9,0.9,0.8,1.0,0.8699999999999999,100000
1.0,0.9,0.9,0.95,0.95,0.9400000000000001,200000
0.95,1.0,0.95,1.0,1.0,0.9800000000000001,300000
0.9,0.85,1.0,1.0,0.9,0.93,400000
1.0,1.0,0.9,1.0,1.0,0.9800000000000001,500000

```

#### M9B/M9B-C013__hiql_high_low_h2l1_full_bptt/antmaze-large-navigate-v0/seed_000/eval.csv

```csv
evaluation/task1_success,evaluation/task2_success,evaluation/task3_success,evaluation/task4_success,evaluation/task5_success,evaluation/overall_success,step
0.75,0.2,0.9,0.85,0.9,0.72,100000
1.0,0.9,1.0,0.9,0.85,0.9299999999999999,200000
0.9,0.8,0.9,0.85,0.95,0.8800000000000001,300000
0.95,0.9,1.0,1.0,0.9,0.95,400000
0.9,0.85,1.0,0.9,0.9,0.9099999999999999,500000

```

#### M9B/M9B-C013__hiql_high_low_h2l1_full_bptt/antmaze-medium-navigate-v0/seed_000/eval.csv

```csv
evaluation/task1_success,evaluation/task2_success,evaluation/task3_success,evaluation/task4_success,evaluation/task5_success,evaluation/overall_success,step
1.0,0.95,0.95,0.9,1.0,0.96,100000
1.0,0.95,1.0,0.9,0.95,0.96,200000
0.95,0.95,0.95,0.9,1.0,0.95,300000
1.0,1.0,0.9,0.8,0.95,0.93,400000
0.95,0.85,0.85,0.9,0.9,0.89,500000

```

#### M9B/M9B-C014__hiql_high_low_h2l1_one_step/antmaze-large-navigate-v0/seed_000/eval.csv

```csv
evaluation/task1_success,evaluation/task2_success,evaluation/task3_success,evaluation/task4_success,evaluation/task5_success,evaluation/overall_success,step
0.85,0.3,0.9,0.9,0.9,0.7699999999999999,100000
0.85,0.45,0.9,0.9,0.75,0.77,200000
0.85,0.7,0.9,0.9,0.9,0.85,300000
0.9,0.25,0.9,0.95,0.95,0.79,400000
0.8,0.8,0.95,0.8,0.8,0.8299999999999998,500000

```

#### M9B/M9B-C014__hiql_high_low_h2l1_one_step/antmaze-medium-navigate-v0/seed_000/eval.csv

```csv
evaluation/task1_success,evaluation/task2_success,evaluation/task3_success,evaluation/task4_success,evaluation/task5_success,evaluation/overall_success,step
1.0,1.0,0.75,0.7,0.9,0.8700000000000001,100000
0.75,0.85,1.0,0.95,1.0,0.9099999999999999,200000
0.95,0.9,0.95,1.0,1.0,0.96,300000
0.95,0.95,0.9,1.0,0.9,0.9400000000000001,400000
0.9,1.0,0.95,0.95,0.85,0.9299999999999999,500000

```

#### M9B/M9B-C015__hiql_high_low_h2l6_full_bptt/antmaze-large-navigate-v0/seed_000/eval.csv

```csv
evaluation/task1_success,evaluation/task2_success,evaluation/task3_success,evaluation/task4_success,evaluation/task5_success,evaluation/overall_success,step
0.95,0.05,0.95,0.95,0.8,0.74,100000
0.95,0.9,0.95,0.85,0.95,0.9199999999999999,200000
0.9,0.8,0.7,0.85,0.95,0.8400000000000001,300000
0.9,0.9,0.9,0.95,0.85,0.9,400000
0.85,0.65,0.95,0.95,0.85,0.85,500000

```

#### M9B/M9B-C015__hiql_high_low_h2l6_full_bptt/antmaze-medium-navigate-v0/seed_000/eval.csv

```csv
evaluation/task1_success,evaluation/task2_success,evaluation/task3_success,evaluation/task4_success,evaluation/task5_success,evaluation/overall_success,step
0.95,1.0,0.95,0.9,0.95,0.95,100000
0.95,0.95,1.0,0.95,0.9,0.95,200000
0.9,1.0,1.0,0.95,0.95,0.96,300000
1.0,0.95,0.95,0.9,0.95,0.95,400000
0.9,0.9,0.95,1.0,1.0,0.95,500000

```

#### M9B/M9B-C016__hiql_high_low_h2l6_one_step/antmaze-large-navigate-v0/seed_000/eval.csv

```csv
evaluation/task1_success,evaluation/task2_success,evaluation/task3_success,evaluation/task4_success,evaluation/task5_success,evaluation/overall_success,step
1.0,0.4,0.95,0.9,0.7,0.7899999999999998,100000
0.95,0.6,0.95,1.0,0.95,0.89,200000
0.95,0.8,0.9,0.85,0.9,0.8800000000000001,300000
0.9,0.95,0.95,0.8,0.75,0.8699999999999999,400000
0.8,0.6,0.95,0.85,0.8,0.8,500000

```

#### M9B/M9B-C016__hiql_high_low_h2l6_one_step/antmaze-medium-navigate-v0/seed_000/eval.csv

```csv
evaluation/task1_success,evaluation/task2_success,evaluation/task3_success,evaluation/task4_success,evaluation/task5_success,evaluation/overall_success,step
0.85,0.9,0.95,0.85,0.95,0.9,100000
0.95,1.0,0.8,0.9,0.95,0.9199999999999999,200000
0.95,0.95,0.95,0.95,1.0,0.96,300000
0.85,0.85,1.0,0.9,1.0,0.9199999999999999,400000
0.95,0.9,0.9,0.95,1.0,0.9400000000000001,500000

```

## 14. 原始数据附录 B：summary.json

以下代码块逐字来自原始 summary.json 文件。

### M9A

#### M9A/M9A-C001__hiql_vanilla/antmaze-large-navigate-v0/seed_000/summary.json

```json
{
  "best_step": 1000000,
  "best_success": 0.8700000000000001,
  "final_success": 0.8700000000000001,
  "status": "completed",
  "success_column": "evaluation/overall_success"
}

```

#### M9A/M9A-C001__hiql_vanilla/antmaze-medium-navigate-v0/seed_000/summary.json

```json
{
  "best_step": 800000,
  "best_success": 0.9800000000000001,
  "final_success": 0.96,
  "status": "completed",
  "success_column": "evaluation/overall_success"
}

```

#### M9A/M9A-C002__crl_vanilla/antmaze-large-navigate-v0/seed_000/summary.json

```json
{
  "best_step": 900000,
  "best_success": 0.77,
  "final_success": 0.62,
  "status": "completed",
  "success_column": "evaluation/overall_success"
}

```

#### M9A/M9A-C002__crl_vanilla/antmaze-medium-navigate-v0/seed_000/summary.json

```json
{
  "best_step": 400000,
  "best_success": 0.99,
  "final_success": 0.93,
  "status": "completed",
  "success_column": "evaluation/overall_success"
}

```

#### M9A/M9A-C003__crl_actor_k1_nores/antmaze-large-navigate-v0/seed_000/summary.json

```json
{
  "best_step": 400000,
  "best_success": 0.85,
  "final_success": 0.8399999999999999,
  "status": "completed",
  "success_column": "evaluation/overall_success"
}

```

#### M9A/M9A-C003__crl_actor_k1_nores/antmaze-medium-navigate-v0/seed_000/summary.json

```json
{
  "best_step": 500000,
  "best_success": 0.9400000000000001,
  "final_success": 0.9400000000000001,
  "status": "completed",
  "success_column": "evaluation/overall_success"
}

```

#### M9A/M9A-C004__crl_actor_k1_res/antmaze-large-navigate-v0/seed_000/summary.json

```json
{
  "best_step": 500000,
  "best_success": 0.86,
  "final_success": 0.86,
  "status": "completed",
  "success_column": "evaluation/overall_success"
}

```

#### M9A/M9A-C004__crl_actor_k1_res/antmaze-medium-navigate-v0/seed_000/summary.json

```json
{
  "best_step": 400000,
  "best_success": 0.9099999999999999,
  "final_success": 0.89,
  "status": "completed",
  "success_column": "evaluation/overall_success"
}

```

#### M9A/M9A-C005__crl_actor_k2_nores/antmaze-large-navigate-v0/seed_000/summary.json

```json
{
  "best_step": 400000,
  "best_success": 0.89,
  "final_success": 0.8700000000000001,
  "status": "completed",
  "success_column": "evaluation/overall_success"
}

```

#### M9A/M9A-C005__crl_actor_k2_nores/antmaze-medium-navigate-v0/seed_000/summary.json

```json
{
  "best_step": 300000,
  "best_success": 0.9800000000000001,
  "final_success": 0.96,
  "status": "completed",
  "success_column": "evaluation/overall_success"
}

```

#### M9A/M9A-C006__crl_actor_k2_res/antmaze-large-navigate-v0/seed_000/summary.json

```json
{
  "best_step": 400000,
  "best_success": 0.9,
  "final_success": 0.8699999999999999,
  "status": "completed",
  "success_column": "evaluation/overall_success"
}

```

#### M9A/M9A-C006__crl_actor_k2_res/antmaze-medium-navigate-v0/seed_000/summary.json

```json
{
  "best_step": 400000,
  "best_success": 0.9700000000000001,
  "final_success": 0.9099999999999999,
  "status": "completed",
  "success_column": "evaluation/overall_success"
}

```

#### M9A/M9A-C007__crl_actor_k4_nores/antmaze-large-navigate-v0/seed_000/summary.json

```json
{
  "best_step": 400000,
  "best_success": 0.9200000000000002,
  "final_success": 0.9099999999999999,
  "status": "completed",
  "success_column": "evaluation/overall_success"
}

```

#### M9A/M9A-C007__crl_actor_k4_nores/antmaze-medium-navigate-v0/seed_000/summary.json

```json
{
  "best_step": 300000,
  "best_success": 0.9800000000000001,
  "final_success": 0.97,
  "status": "completed",
  "success_column": "evaluation/overall_success"
}

```

#### M9A/M9A-C008__crl_actor_k4_res/antmaze-large-navigate-v0/seed_000/summary.json

```json
{
  "best_step": 300000,
  "best_success": 0.9299999999999999,
  "final_success": 0.9099999999999999,
  "status": "completed",
  "success_column": "evaluation/overall_success"
}

```

#### M9A/M9A-C008__crl_actor_k4_res/antmaze-medium-navigate-v0/seed_000/summary.json

```json
{
  "best_step": 300000,
  "best_success": 0.96,
  "final_success": 0.95,
  "status": "completed",
  "success_column": "evaluation/overall_success"
}

```

#### M9A/M9A-C009__hiql_high_k1_nores/antmaze-large-navigate-v0/seed_000/summary.json

```json
{
  "best_step": 400000,
  "best_success": 0.9100000000000001,
  "final_success": 0.8300000000000001,
  "status": "completed",
  "success_column": "evaluation/overall_success"
}

```

#### M9A/M9A-C009__hiql_high_k1_nores/antmaze-medium-navigate-v0/seed_000/summary.json

```json
{
  "best_step": 500000,
  "best_success": 0.9800000000000001,
  "final_success": 0.9800000000000001,
  "status": "completed",
  "success_column": "evaluation/overall_success"
}

```

#### M9A/M9A-C010__hiql_high_k1_res/antmaze-large-navigate-v0/seed_000/summary.json

```json
{
  "best_step": 500000,
  "best_success": 0.9099999999999999,
  "final_success": 0.9099999999999999,
  "status": "completed",
  "success_column": "evaluation/overall_success"
}

```

#### M9A/M9A-C010__hiql_high_k1_res/antmaze-medium-navigate-v0/seed_000/summary.json

```json
{
  "best_step": 500000,
  "best_success": 0.99,
  "final_success": 0.99,
  "status": "completed",
  "success_column": "evaluation/overall_success"
}

```

#### M9A/M9A-C011__hiql_high_k2_nores/antmaze-large-navigate-v0/seed_000/summary.json

```json
{
  "best_step": 500000,
  "best_success": 0.9199999999999999,
  "final_success": 0.9199999999999999,
  "status": "completed",
  "success_column": "evaluation/overall_success"
}

```

#### M9A/M9A-C011__hiql_high_k2_nores/antmaze-medium-navigate-v0/seed_000/summary.json

```json
{
  "best_step": 300000,
  "best_success": 0.96,
  "final_success": 0.95,
  "status": "completed",
  "success_column": "evaluation/overall_success"
}

```

#### M9A/M9A-C012__hiql_high_k2_res/antmaze-large-navigate-v0/seed_000/summary.json

```json
{
  "best_step": 300000,
  "best_success": 0.85,
  "final_success": 0.85,
  "status": "completed",
  "success_column": "evaluation/overall_success"
}

```

#### M9A/M9A-C012__hiql_high_k2_res/antmaze-medium-navigate-v0/seed_000/summary.json

```json
{
  "best_step": 100000,
  "best_success": 0.99,
  "final_success": 0.9399999999999998,
  "status": "completed",
  "success_column": "evaluation/overall_success"
}

```

#### M9A/M9A-C013__hiql_high_k4_nores/antmaze-large-navigate-v0/seed_000/summary.json

```json
{
  "best_step": 300000,
  "best_success": 0.93,
  "final_success": 0.85,
  "status": "completed",
  "success_column": "evaluation/overall_success"
}

```

#### M9A/M9A-C013__hiql_high_k4_nores/antmaze-medium-navigate-v0/seed_000/summary.json

```json
{
  "best_step": 300000,
  "best_success": 0.9800000000000001,
  "final_success": 0.9199999999999999,
  "status": "completed",
  "success_column": "evaluation/overall_success"
}

```

#### M9A/M9A-C014__hiql_high_k4_res/antmaze-large-navigate-v0/seed_000/summary.json

```json
{
  "best_step": 400000,
  "best_success": 0.9400000000000001,
  "final_success": 0.9400000000000001,
  "status": "completed",
  "success_column": "evaluation/overall_success"
}

```

#### M9A/M9A-C014__hiql_high_k4_res/antmaze-medium-navigate-v0/seed_000/summary.json

```json
{
  "best_step": 400000,
  "best_success": 0.99,
  "final_success": 0.95,
  "status": "completed",
  "success_column": "evaluation/overall_success"
}

```

#### M9A/M9A-C015__hiql_low_k1_nores/antmaze-large-navigate-v0/seed_000/summary.json

```json
{
  "best_step": 500000,
  "best_success": 0.8400000000000001,
  "final_success": 0.8400000000000001,
  "status": "completed",
  "success_column": "evaluation/overall_success"
}

```

#### M9A/M9A-C015__hiql_low_k1_nores/antmaze-medium-navigate-v0/seed_000/summary.json

```json
{
  "best_step": 500000,
  "best_success": 0.97,
  "final_success": 0.97,
  "status": "completed",
  "success_column": "evaluation/overall_success"
}

```

#### M9A/M9A-C016__hiql_low_k1_res/antmaze-large-navigate-v0/seed_000/summary.json

```json
{
  "best_step": 400000,
  "best_success": 0.9200000000000002,
  "final_success": 0.85,
  "status": "completed",
  "success_column": "evaluation/overall_success"
}

```

#### M9A/M9A-C016__hiql_low_k1_res/antmaze-medium-navigate-v0/seed_000/summary.json

```json
{
  "best_step": 300000,
  "best_success": 0.96,
  "final_success": 0.96,
  "status": "completed",
  "success_column": "evaluation/overall_success"
}

```

#### M9A/M9A-C017__hiql_low_k2_nores/antmaze-large-navigate-v0/seed_000/summary.json

```json
{
  "best_step": 500000,
  "best_success": 0.89,
  "final_success": 0.89,
  "status": "completed",
  "success_column": "evaluation/overall_success"
}

```

#### M9A/M9A-C017__hiql_low_k2_nores/antmaze-medium-navigate-v0/seed_000/summary.json

```json
{
  "best_step": 200000,
  "best_success": 0.97,
  "final_success": 0.95,
  "status": "completed",
  "success_column": "evaluation/overall_success"
}

```

#### M9A/M9A-C018__hiql_low_k2_res/antmaze-large-navigate-v0/seed_000/summary.json

```json
{
  "best_step": 400000,
  "best_success": 0.9,
  "final_success": 0.86,
  "status": "completed",
  "success_column": "evaluation/overall_success"
}

```

#### M9A/M9A-C018__hiql_low_k2_res/antmaze-medium-navigate-v0/seed_000/summary.json

```json
{
  "best_step": 300000,
  "best_success": 0.97,
  "final_success": 0.96,
  "status": "completed",
  "success_column": "evaluation/overall_success"
}

```

#### M9A/M9A-C019__hiql_low_k4_nores/antmaze-large-navigate-v0/seed_000/summary.json

```json
{
  "best_step": 500000,
  "best_success": 0.93,
  "final_success": 0.93,
  "status": "completed",
  "success_column": "evaluation/overall_success"
}

```

#### M9A/M9A-C019__hiql_low_k4_nores/antmaze-medium-navigate-v0/seed_000/summary.json

```json
{
  "best_step": 200000,
  "best_success": 0.97,
  "final_success": 0.9400000000000001,
  "status": "completed",
  "success_column": "evaluation/overall_success"
}

```

#### M9A/M9A-C020__hiql_low_k4_res/antmaze-large-navigate-v0/seed_000/summary.json

```json
{
  "best_step": 300000,
  "best_success": 0.93,
  "final_success": 0.89,
  "status": "completed",
  "success_column": "evaluation/overall_success"
}

```

#### M9A/M9A-C020__hiql_low_k4_res/antmaze-medium-navigate-v0/seed_000/summary.json

```json
{
  "best_step": 300000,
  "best_success": 0.96,
  "final_success": 0.9400000000000001,
  "status": "completed",
  "success_column": "evaluation/overall_success"
}

```

#### M9A/M9A-C021__hiql_high_low_k1_nores/antmaze-large-navigate-v0/seed_000/summary.json

```json
{
  "best_step": 500000,
  "best_success": 0.9200000000000002,
  "final_success": 0.9200000000000002,
  "status": "completed",
  "success_column": "evaluation/overall_success"
}

```

#### M9A/M9A-C021__hiql_high_low_k1_nores/antmaze-medium-navigate-v0/seed_000/summary.json

```json
{
  "best_step": 400000,
  "best_success": 0.9700000000000001,
  "final_success": 0.97,
  "status": "completed",
  "success_column": "evaluation/overall_success"
}

```

#### M9A/M9A-C022__hiql_high_low_k1_res/antmaze-large-navigate-v0/seed_000/summary.json

```json
{
  "best_step": 400000,
  "best_success": 0.93,
  "final_success": 0.82,
  "status": "completed",
  "success_column": "evaluation/overall_success"
}

```

#### M9A/M9A-C022__hiql_high_low_k1_res/antmaze-medium-navigate-v0/seed_000/summary.json

```json
{
  "best_step": 100000,
  "best_success": 0.97,
  "final_success": 0.93,
  "status": "completed",
  "success_column": "evaluation/overall_success"
}

```

#### M9A/M9A-C023__hiql_high_low_k2_nores/antmaze-large-navigate-v0/seed_000/summary.json

```json
{
  "best_step": 300000,
  "best_success": 0.8300000000000001,
  "final_success": 0.8300000000000001,
  "status": "completed",
  "success_column": "evaluation/overall_success"
}

```

#### M9A/M9A-C023__hiql_high_low_k2_nores/antmaze-medium-navigate-v0/seed_000/summary.json

```json
{
  "best_step": 200000,
  "best_success": 0.96,
  "final_success": 0.9199999999999999,
  "status": "completed",
  "success_column": "evaluation/overall_success"
}

```

#### M9A/M9A-C024__hiql_high_low_k2_res/antmaze-large-navigate-v0/seed_000/summary.json

```json
{
  "best_step": 300000,
  "best_success": 0.9400000000000001,
  "final_success": 0.8800000000000001,
  "status": "completed",
  "success_column": "evaluation/overall_success"
}

```

#### M9A/M9A-C024__hiql_high_low_k2_res/antmaze-medium-navigate-v0/seed_000/summary.json

```json
{
  "best_step": 100000,
  "best_success": 0.99,
  "final_success": 0.9,
  "status": "completed",
  "success_column": "evaluation/overall_success"
}

```

#### M9A/M9A-C025__hiql_high_low_k4_nores/antmaze-large-navigate-v0/seed_000/summary.json

```json
{
  "best_step": 400000,
  "best_success": 0.9199999999999999,
  "final_success": 0.79,
  "status": "completed",
  "success_column": "evaluation/overall_success"
}

```

#### M9A/M9A-C025__hiql_high_low_k4_nores/antmaze-medium-navigate-v0/seed_000/summary.json

```json
{
  "best_step": 400000,
  "best_success": 0.9800000000000001,
  "final_success": 0.9800000000000001,
  "status": "completed",
  "success_column": "evaluation/overall_success"
}

```

#### M9A/M9A-C026__hiql_high_low_k4_res/antmaze-large-navigate-v0/seed_000/summary.json

```json
{
  "best_step": 200000,
  "best_success": 0.93,
  "final_success": 0.8700000000000001,
  "status": "completed",
  "success_column": "evaluation/overall_success"
}

```

#### M9A/M9A-C026__hiql_high_low_k4_res/antmaze-medium-navigate-v0/seed_000/summary.json

```json
{
  "best_step": 300000,
  "best_success": 0.97,
  "final_success": 0.97,
  "status": "completed",
  "success_column": "evaluation/overall_success"
}

```

### M9B

#### M9B/M9B-C001__crl_actor_h2l1_full_bptt/antmaze-large-navigate-v0/seed_000/summary.json

```json
{
  "best_step": 300000,
  "best_success": 0.9,
  "final_success": 0.8400000000000001,
  "status": "completed",
  "success_column": "evaluation/overall_success"
}

```

#### M9B/M9B-C001__crl_actor_h2l1_full_bptt/antmaze-medium-navigate-v0/seed_000/summary.json

```json
{
  "best_step": 400000,
  "best_success": 0.9800000000000001,
  "final_success": 0.9399999999999998,
  "status": "completed",
  "success_column": "evaluation/overall_success"
}

```

#### M9B/M9B-C002__crl_actor_h2l1_one_step/antmaze-large-navigate-v0/seed_000/summary.json

```json
{
  "best_step": 500000,
  "best_success": 0.86,
  "final_success": 0.86,
  "status": "completed",
  "success_column": "evaluation/overall_success"
}

```

#### M9B/M9B-C002__crl_actor_h2l1_one_step/antmaze-medium-navigate-v0/seed_000/summary.json

```json
{
  "best_step": 500000,
  "best_success": 0.96,
  "final_success": 0.96,
  "status": "completed",
  "success_column": "evaluation/overall_success"
}

```

#### M9B/M9B-C003__crl_actor_h2l6_full_bptt/antmaze-large-navigate-v0/seed_000/summary.json

```json
{
  "best_step": 400000,
  "best_success": 0.9099999999999999,
  "final_success": 0.89,
  "status": "completed",
  "success_column": "evaluation/overall_success"
}

```

#### M9B/M9B-C003__crl_actor_h2l6_full_bptt/antmaze-medium-navigate-v0/seed_000/summary.json

```json
{
  "best_step": 400000,
  "best_success": 0.96,
  "final_success": 0.95,
  "status": "completed",
  "success_column": "evaluation/overall_success"
}

```

#### M9B/M9B-C004__crl_actor_h2l6_one_step/antmaze-large-navigate-v0/seed_000/summary.json

```json
{
  "best_step": 500000,
  "best_success": 0.7,
  "final_success": 0.7,
  "status": "completed",
  "success_column": "evaluation/overall_success"
}

```

#### M9B/M9B-C004__crl_actor_h2l6_one_step/antmaze-medium-navigate-v0/seed_000/summary.json

```json
{
  "best_step": 500000,
  "best_success": 0.8400000000000001,
  "final_success": 0.8400000000000001,
  "status": "completed",
  "success_column": "evaluation/overall_success"
}

```

#### M9B/M9B-C005__hiql_high_h2l1_full_bptt/antmaze-large-navigate-v0/seed_000/summary.json

```json
{
  "best_step": 400000,
  "best_success": 0.9099999999999999,
  "final_success": 0.89,
  "status": "completed",
  "success_column": "evaluation/overall_success"
}

```

#### M9B/M9B-C005__hiql_high_h2l1_full_bptt/antmaze-medium-navigate-v0/seed_000/summary.json

```json
{
  "best_step": 100000,
  "best_success": 1.0,
  "final_success": 0.97,
  "status": "completed",
  "success_column": "evaluation/overall_success"
}

```

#### M9B/M9B-C006__hiql_high_h2l1_one_step/antmaze-large-navigate-v0/seed_000/summary.json

```json
{
  "best_step": 400000,
  "best_success": 0.93,
  "final_success": 0.93,
  "status": "completed",
  "success_column": "evaluation/overall_success"
}

```

#### M9B/M9B-C006__hiql_high_h2l1_one_step/antmaze-medium-navigate-v0/seed_000/summary.json

```json
{
  "best_step": 200000,
  "best_success": 0.99,
  "final_success": 0.99,
  "status": "completed",
  "success_column": "evaluation/overall_success"
}

```

#### M9B/M9B-C007__hiql_high_h2l6_full_bptt/antmaze-large-navigate-v0/seed_000/summary.json

```json
{
  "best_step": 500000,
  "best_success": 0.8800000000000001,
  "final_success": 0.8800000000000001,
  "status": "completed",
  "success_column": "evaluation/overall_success"
}

```

#### M9B/M9B-C007__hiql_high_h2l6_full_bptt/antmaze-medium-navigate-v0/seed_000/summary.json

```json
{
  "best_step": 200000,
  "best_success": 0.99,
  "final_success": 0.99,
  "status": "completed",
  "success_column": "evaluation/overall_success"
}

```

#### M9B/M9B-C008__hiql_high_h2l6_one_step/antmaze-large-navigate-v0/seed_000/summary.json

```json
{
  "best_step": 500000,
  "best_success": 0.85,
  "final_success": 0.85,
  "status": "completed",
  "success_column": "evaluation/overall_success"
}

```

#### M9B/M9B-C008__hiql_high_h2l6_one_step/antmaze-medium-navigate-v0/seed_000/summary.json

```json
{
  "best_step": 500000,
  "best_success": 0.9700000000000001,
  "final_success": 0.9700000000000001,
  "status": "completed",
  "success_column": "evaluation/overall_success"
}

```

#### M9B/M9B-C009__hiql_low_h2l1_full_bptt/antmaze-large-navigate-v0/seed_000/summary.json

```json
{
  "best_step": 400000,
  "best_success": 0.7700000000000001,
  "final_success": 0.73,
  "status": "completed",
  "success_column": "evaluation/overall_success"
}

```

#### M9B/M9B-C009__hiql_low_h2l1_full_bptt/antmaze-medium-navigate-v0/seed_000/summary.json

```json
{
  "best_step": 400000,
  "best_success": 0.97,
  "final_success": 0.95,
  "status": "completed",
  "success_column": "evaluation/overall_success"
}

```

#### M9B/M9B-C010__hiql_low_h2l1_one_step/antmaze-large-navigate-v0/seed_000/summary.json

```json
{
  "best_step": 400000,
  "best_success": 0.9199999999999999,
  "final_success": 0.8099999999999999,
  "status": "completed",
  "success_column": "evaluation/overall_success"
}

```

#### M9B/M9B-C010__hiql_low_h2l1_one_step/antmaze-medium-navigate-v0/seed_000/summary.json

```json
{
  "best_step": 300000,
  "best_success": 0.95,
  "final_success": 0.89,
  "status": "completed",
  "success_column": "evaluation/overall_success"
}

```

#### M9B/M9B-C011__hiql_low_h2l6_full_bptt/antmaze-large-navigate-v0/seed_000/summary.json

```json
{
  "best_step": 300000,
  "best_success": 0.8299999999999998,
  "final_success": 0.8,
  "status": "completed",
  "success_column": "evaluation/overall_success"
}

```

#### M9B/M9B-C011__hiql_low_h2l6_full_bptt/antmaze-medium-navigate-v0/seed_000/summary.json

```json
{
  "best_step": 200000,
  "best_success": 1.0,
  "final_success": 0.96,
  "status": "completed",
  "success_column": "evaluation/overall_success"
}

```

#### M9B/M9B-C012__hiql_low_h2l6_one_step/antmaze-large-navigate-v0/seed_000/summary.json

```json
{
  "best_step": 500000,
  "best_success": 0.8,
  "final_success": 0.8,
  "status": "completed",
  "success_column": "evaluation/overall_success"
}

```

#### M9B/M9B-C012__hiql_low_h2l6_one_step/antmaze-medium-navigate-v0/seed_000/summary.json

```json
{
  "best_step": 300000,
  "best_success": 0.9800000000000001,
  "final_success": 0.9800000000000001,
  "status": "completed",
  "success_column": "evaluation/overall_success"
}

```

#### M9B/M9B-C013__hiql_high_low_h2l1_full_bptt/antmaze-large-navigate-v0/seed_000/summary.json

```json
{
  "best_step": 400000,
  "best_success": 0.95,
  "final_success": 0.9099999999999999,
  "status": "completed",
  "success_column": "evaluation/overall_success"
}

```

#### M9B/M9B-C013__hiql_high_low_h2l1_full_bptt/antmaze-medium-navigate-v0/seed_000/summary.json

```json
{
  "best_step": 100000,
  "best_success": 0.96,
  "final_success": 0.89,
  "status": "completed",
  "success_column": "evaluation/overall_success"
}

```

#### M9B/M9B-C014__hiql_high_low_h2l1_one_step/antmaze-large-navigate-v0/seed_000/summary.json

```json
{
  "best_step": 300000,
  "best_success": 0.85,
  "final_success": 0.8299999999999998,
  "status": "completed",
  "success_column": "evaluation/overall_success"
}

```

#### M9B/M9B-C014__hiql_high_low_h2l1_one_step/antmaze-medium-navigate-v0/seed_000/summary.json

```json
{
  "best_step": 300000,
  "best_success": 0.96,
  "final_success": 0.9299999999999999,
  "status": "completed",
  "success_column": "evaluation/overall_success"
}

```

#### M9B/M9B-C015__hiql_high_low_h2l6_full_bptt/antmaze-large-navigate-v0/seed_000/summary.json

```json
{
  "best_step": 200000,
  "best_success": 0.9199999999999999,
  "final_success": 0.85,
  "status": "completed",
  "success_column": "evaluation/overall_success"
}

```

#### M9B/M9B-C015__hiql_high_low_h2l6_full_bptt/antmaze-medium-navigate-v0/seed_000/summary.json

```json
{
  "best_step": 300000,
  "best_success": 0.96,
  "final_success": 0.95,
  "status": "completed",
  "success_column": "evaluation/overall_success"
}

```

#### M9B/M9B-C016__hiql_high_low_h2l6_one_step/antmaze-large-navigate-v0/seed_000/summary.json

```json
{
  "best_step": 200000,
  "best_success": 0.89,
  "final_success": 0.8,
  "status": "completed",
  "success_column": "evaluation/overall_success"
}

```

#### M9B/M9B-C016__hiql_high_low_h2l6_one_step/antmaze-medium-navigate-v0/seed_000/summary.json

```json
{
  "best_step": 300000,
  "best_success": 0.96,
  "final_success": 0.9400000000000001,
  "status": "completed",
  "success_column": "evaluation/overall_success"
}

```

## 15. 其它原始文件定位

每个 run 的完整 provenance、resolved config、训练日志和 checkpoint 均保留在：

```text
/data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M9A
/data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M9B
```

具体文件名：

```text
runtime_metadata.json
resolved_config.json
train.csv
checkpoints/params_500000.pkl 或 checkpoints/params_1000000.pkl
```

## 16. 只读复核命令

```bash
find /data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M9A /data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M9B -name eval.csv | sort | wc -l
find /data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M9A /data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M9B -name summary.json -print0 | xargs -0 jq -r .status | sort | uniq -c
sed -n "1,20p" /data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M9B/M9B-C003__crl_actor_h2l6_full_bptt/antmaze-large-navigate-v0/seed_000/eval.csv
```

本次只新增本 Markdown 报告，不修改代码、配置、manifest、aggregated.csv 或任何原始实验文件。
