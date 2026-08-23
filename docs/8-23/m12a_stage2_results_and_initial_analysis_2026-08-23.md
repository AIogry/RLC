# M12A Stage 2 实验结果与初步分析

## 1. 摘要结论

M12A Stage 2 的 6 个正式 Run 已完成：C002（feed-forward actor）和 C003（SingleState-K4 actor）各 3 个 seed，均使用同 seed 的 M12A-C001 `last@1M` frozen critic。6 个 `attempt_1` Run 的状态均为 `completed`。

在当前预注册的配对设计下，C003 在 3 个 seed 上都优于 C002：

- `final@1M`：C003 `0.863 ± 0.070`，C002 `0.670 ± 0.125`，配对差值 `+0.193`；
- `best`：C003 `0.883 ± 0.042`，C002 `0.740 ± 0.087`，配对差值 `+0.143`；
- normalized evaluation AUC：C003 `0.793 ± 0.042`，C002 `0.586 ± 0.107`，配对差值 `+0.207`。

并且，在 100k–1M 的全部 10 个评估时点上，C003 的三 seed 均值都高于 C002。当前证据支持如下谨慎结论：

> 在 `antmaze-large-navigate-v0`、固定同 seed CRL critic、1M 训练步和当前 DDPG+BC actor objective 下，SingleState-K4 actor 相比 canonical FF actor 展现出稳定的描述性优势。

这仍然不是跨环境、跨 critic、跨更多 seed 的普适性结论。样本量只有 3 个 paired seeds，每次评估每个 task 只有 20 episodes，因此本报告不把结果表述为已完成统计显著性检验的结论。

本报告读取已有 Stage 2 `attempt_1` 产物；本轮没有启动新的训练，也没有执行任何 Git 操作。

## 2. 实验设计与比较对象

M12A 的问题是：在同一个固定学习到的 CRL critic 下，SingleState-K4 actor 是否仍然优于 canonical feed-forward actor。

| 项目 | 设定 |
|---|---|
| Study | M12A Frozen-Critic Policy Extraction |
| 环境 | `antmaze-large-navigate-v0` |
| Stage 1 | C001，canonical FF CRL critic-only，1,000,000 updates |
| Stage 2 control | C002，canonical FF actor |
| Stage 2 treatment | C003，SingleState actor，K=4，`residual=false` |
| critic | 每个 seed 使用同 seed C001 的 `last@1M` critic，训练期间冻结 |
| actor objective | canonical CRL DDPG+BC |
| batch size | 1024 |
| Stage 2 train steps | 1,000,000 |
| eval interval | 100,000 |
| evaluation | 5 tasks × 20 episodes/task，temperature=0，Gaussian=None |
| primary endpoint | `final@1M` |
| secondary endpoints | normalized AUC、best、best step、last3 mean |
| seeds | 0、1、2 |

因此 C002 是本 Study 内的 FF actor control，C003 是唯一改变 actor computation 的 treatment。C001 本身不是一个 Stage 2 policy-performance baseline：它只负责提供 frozen critic，Stage 1 按协议关闭 evaluation。

配置和协议文件：

- [M12A study.yaml](../../experiments/M12A_frozen_critic_policy_extraction/study.yaml)
- [M12A-C001.yaml](../../experiments/M12A_frozen_critic_policy_extraction/configs/M12A-C001.yaml)
- [M12A-C002.yaml](../../experiments/M12A_frozen_critic_policy_extraction/configs/M12A-C002.yaml)
- [M12A-C003.yaml](../../experiments/M12A_frozen_critic_policy_extraction/configs/M12A-C003.yaml)

## 3. 运行状态与 provenance 核查

### 3.1 Stage 2 运行状态

6 个正式 Stage 2 目录均为 `run_attempt=1`，状态为 `completed`，均有 `eval.csv`、`summary.json`、`resolved_config.json` 和 semantic checkpoints。

基于实际运行目录的 sweep dry-run 状态为：

```text
total=6 planned=0 completed=6 failed=0 running=0 retained=6 remaining=0
```

这表示 attempt-1 的 6 个计划任务已经全部完成。预执行的设计测试也验证了：在 source runs 可用时，Stage 2 应规划恰好 6 个 runs；本轮实际没有额外创建新的 formal run。

### 3.2 same-seed frozen critic 配对

C002/C003 的每个同 seed pair 都解析到相同的 C001 checkpoint SHA 和 critic module fingerprint：

| seed | C001 checkpoint | checkpoint SHA256 | critic fingerprint | C002/C003 是否完全相同 |
|---:|---|---|---|---|
| 0 | `last@1M` | `f801f7521aedc70a0ed182a2a2f2d7765d9faa0e6b7ac623f98ad284926006d5` | `35bfa7630a317e40bae4fbc4f529635c4655f8946975af1e87388d4490bb85b7` | 是 |
| 1 | `last@1M` | `b89f45b1e61436b0ee469471b51f40b517ac6130339951c9086c2a5912281c98` | `1ac0eda0a97b315e1f8a6e48d0b29c70267de85bb54b81c4ba9718f0bbcdc36e` | 是 |
| 2 | `last@1M` | `64def1c398dd59a30b533c1cfa704937d389ca5a5088ba45db66b3f7a8005ab2` | `0c0579ae1f6012e89b2c5ec465dd976baabee0d7f882efbddc4cbe4330423d18` | 是 |

每个 Stage 2 runtime metadata 还记录了 `source_run_attempt=0`、`checkpoint_role=last`、`checkpoint_step=1000000`，没有跨 seed fallback。

### 3.3 attempt-0 失败目录的处理

原先 C002/C003 各 3 个 `attempt_0` 目录均保留，状态仍为 `failed`，失败原因均为 restore 后参数 PyTree 容器不一致：target optimizer 侧为普通 `dict`，restore 后参数树出现 `FrozenDict`。这些目录没有被删除或覆盖；它们是失败 provenance，而不是有效实验结果。

### 3.4 Stage 1 `save_best_checkpoint` provenance

已有 C001 的 `runtime_metadata.json` 仍记录 `save_best_checkpoint=true`，与 M12A Stage 1 protocol 的 `false` 不一致。但三个 C001 的 checkpoint index 都明确显示：

```text
best = null
best_step = null
last_step = 1000000
```

Stage 2 实际只使用了上述 `last@1M` checkpoint，且 SHA 和 critic fingerprint 均通过验证。因此该不一致是旧 launcher 的 provenance 记录问题，不改变 source checkpoint 的 scientific identity，也没有导致 best checkpoint 被选用。C001 不需要因这一 metadata 问题重跑。

通用 launcher 已增加按 Configuration `protocol_stage` 读取 Study protocol 默认值的逻辑：未来 Stage 1 会自动传递 `--no-save-best-checkpoint`，显式命令行参数仍优先。相关修复和 PyTree 修复的历史说明见：[M12A Stage-2 PyTree 修复报告](m12a_stage2_pytree_fix_report.md)。

## 4. 逐 seed 结果

本报告中的 `AUC` 是 Study 预注册的 normalized evaluation AUC：对 100k、200k、…、1M 的 `evaluation/overall_success` 做梯形积分，再除以 `1,000,000-100,000`。`last3` 是 800k、900k、1M 三个评估点的均值。

| config | actor | seed | final@1M | best | best step | normalized AUC | last3 mean |
|---|---|---:|---:|---:|---:|---:|---:|
| C002 | FF | 0 | 0.630 | 0.700 | 700k | 0.503 | 0.593 |
| C002 | FF | 1 | 0.570 | 0.680 | 800k | 0.548 | 0.623 |
| C002 | FF | 2 | 0.810 | 0.840 | 900k | 0.707 | 0.827 |
| C003 | SS-K4 | 0 | 0.930 | 0.930 | 1M | 0.832 | 0.883 |
| C003 | SS-K4 | 1 | 0.790 | 0.850 | 700k | 0.749 | 0.807 |
| C003 | SS-K4 | 2 | 0.870 | 0.870 | 900k | 0.798 | 0.857 |

### 4.1 跨 seed 汇总

以下为 3 个 seed 的均值 ± sample SD；SD 只用于描述当前 seed dispersion，不应被解读为可靠的总体方差估计。

| metric | C002 FF actor | C003 SS-K4 actor | C003 − C002 |
|---|---:|---:|---:|
| final@1M | 0.670 ± 0.125 | 0.863 ± 0.070 | +0.193 |
| best | 0.740 ± 0.087 | 0.883 ± 0.042 | +0.143 |
| normalized AUC | 0.586 ± 0.107 | 0.793 ± 0.042 | +0.207 |
| last3 mean | 0.681 ± 0.127 | 0.849 ± 0.039 | +0.168 |

配对 endpoint 差值 `C003-C002` 为：

| metric | seed 0 | seed 1 | seed 2 | paired mean |
|---|---:|---:|---:|---:|
| final@1M | +0.300 | +0.220 | +0.060 | +0.193 |
| best | +0.230 | +0.170 | +0.030 | +0.143 |
| normalized AUC | +0.329 | +0.201 | +0.091 | +0.207 |
| last3 mean | +0.290 | +0.183 | +0.030 | +0.168 |

三个 paired seed 的所有 endpoint 差值均为正；但由于 `n=3`，这应作为配对设计下的方向一致性证据，而不是充分的统计检验依据。

## 5. 学习曲线比较

下表是每个评估时点跨 3 个 seed 的均值。C003 在全部 10 个时点都高于 C002：

| step | C002 mean | C003 mean | C003 − C002 |
|---:|---:|---:|---:|
| 100k | 0.367 | 0.567 | +0.200 |
| 200k | 0.437 | 0.740 | +0.303 |
| 300k | 0.513 | 0.793 | +0.280 |
| 400k | 0.590 | 0.780 | +0.190 |
| 500k | 0.563 | 0.823 | +0.260 |
| 600k | 0.617 | 0.770 | +0.153 |
| 700k | 0.663 | 0.833 | +0.170 |
| 800k | 0.693 | 0.830 | +0.137 |
| 900k | 0.680 | 0.853 | +0.173 |
| 1M | 0.670 | 0.863 | +0.193 |

这使得 C003 的优势不只是由 1M 单个 endpoint 的偶然波动造成：normalized AUC 也高出 `0.207`。不过曲线仍有明显非单调性，尤其 C002 的 seed 0/1 和 C003 的 seed 1 在达到峰值后出现回落。

## 6. task-level 结果

下表是 1M final evaluation 的 task 均值 ± sample SD。task 名称沿用环境输出的 `task1`–`task5`，没有将其擅自解释为某种固定难度排序。

| task | C002 FF | C003 SS-K4 | C003 − C002 |
|---|---:|---:|---:|
| task1 | 0.817 ± 0.104 | 0.867 ± 0.104 | +0.050 |
| task2 | 0.667 ± 0.161 | 0.850 ± 0.173 | +0.183 |
| task3 | 0.850 ± 0.050 | 0.867 ± 0.058 | +0.017 |
| task4 | 0.583 ± 0.275 | 0.917 ± 0.058 | +0.333 |
| task5 | 0.433 ± 0.202 | 0.817 ± 0.076 | +0.383 |

初步看，C003 的主要收益集中在 task4 和 task5，其次是 task2；task1 和 task3 在 C002 下已经相对较高，因此可见增益较小，存在一定 ceiling effect 的可能性。不过由于每个 task 只有 20 episodes，这一 task-level pattern 仍需要更高评估预算或复现实验确认。

## 7. 初步科学分析

### 7.1 结果与研究问题一致

M12A 的控制逻辑是固定 critic，只改变 actor computation。当前 provenance 检查确认：

1. C002 和 C003 使用同一个 seed-matched C001 critic；
2. 同 seed 的 checkpoint SHA 和 critic fingerprint 完全相同；
3. Stage 2 的 actor topology 只有 FF 与 SS-K4 的差异；
4. C002/C003 使用相同的 seed policy 和数据流设计；
5. critic 在训练中保持 frozen，修复回归测试也确认 restore 后 critic fingerprint 不变。

因此，观察到的 C003 优势与“SingleState-K4 actor computation 带来更好的 policy extraction”这一机制假设相容。由于这是固定 critic 下的 actor-side isolation，解释范围比普通端到端算法比较更窄，也更适合支持机制层面的初步判断。

### 7.2 优势具有早期出现和持续性

C003 在 100k 已经比 C002 高 `0.200`，在 200k 达到均值差 `0.303`；之后优势虽有波动，但到 1M 仍为 `0.193`。因此当前结果不是“只有训练末端某一个 checkpoint 较好”，而是同时反映在 early learning 和整条 evaluation curve 上。

### 7.3 C003 的 seed 稳定性初步更好，但并非完全稳定

C003 的 final 跨 seed SD 为 `0.070`，低于 C002 的 `0.125`；AUC SD 为 `0.042`，也低于 C002 的 `0.107`。此外，best 到 final 的平均回落幅度约为：

- C002：`0.740 - 0.670 = 0.070`；
- C003：`0.883 - 0.863 = 0.020`。

这说明 C003 在当前三个 critic seed 上不仅平均性能更高，late-stage retention 也初步更好。但 C003 seed 1 仍从 best `0.85@700k` 回落到 final `0.79`，所以不能把它描述为单调或完全稳定的优化过程。

### 7.4 存在明显的 critic-seed × actor-topology interaction 线索

final 配对增益从 seed 0 的 `+0.30`、seed 1 的 `+0.22` 降到 seed 2 的 `+0.06`；AUC 也从 `+0.329`、`+0.201` 降到 `+0.091`。这提示 SingleState-K4 的收益可能依赖于 frozen critic 的具体参数状态，而不是一个与 critic 无关的固定常数。

这不是负面结果：M12A 的 paired frozen-critic 设计正好允许我们看到这种 interaction。但它意味着后续若要把“SS-K4 普遍优于 FF”作为强结论，必须增加 critic seeds，而不能只重复同一个 critic。

### 7.5 task-level 增益集中在部分任务

task4/task5 的 final 平均增益分别为 `+0.333` 和 `+0.383`，数值上明显高于 task1/task3。由于当前文件中只有匿名 task index，不能进一步把这种差异归因到地图结构、轨迹长度或数据覆盖率。合理的下一步是把 task index 与环境的任务定义、成功轨迹统计和 critic value quality 对齐，而不是仅凭 success 曲线推断机制。

## 8. 需要避免的过度结论

当前结果不能支持以下更强表述：

- 不能声称 SingleState-K4 在所有 OGBench 环境或所有 critic 上都优于 FF；
- 不能声称该优势已经达到统计显著；`n=3` 且每次 task evaluation 只有 20 episodes；
- 不能把 C003 的绝对分数直接当作相对于独立训练 CRL/HIQL baseline 的增益；M12A 的正式比较是固定 critic 下的 C002 vs C003；
- 不能把 `best` 结果替代预注册的 `final@1M` endpoint；本报告同时列出两者，是为了呈现 late-stage stability；
- 不能仅凭 task4/task5 的提升断言 SingleState-K4 学到了某种具体的长程规划机制；当前还没有对应的 representation、value calibration 或 trajectory-level diagnostic。

## 9. 对后续 M12A 工作的建议

当前 M12A-Core 的 Stage 2 结果已经足以形成一轮有意义的机制信号，不建议为了修复 bug 或旧 provenance metadata 而重跑 C001 或覆盖现有 attempt-0/attempt-1 目录。

如果需要把结论从“初步证据”提升到“更稳健的实验结论”，优先级建议为：

1. 在保持同一 fixed-critic、same-seed、same protocol 的条件下增加 critic seeds，优先验证 seed 2 中较小的 C003 增益是否普遍存在；
2. 对现有 final/best checkpoints 增加独立 reevaluation episodes，减少每个 task 20 episodes 带来的 Bernoulli evaluation noise；
3. 按 Study 中已预设的顺序考虑 `antmaze-giant-navigate-v0` confirmatory extension；不能把它与当前 AntMaze-Large 结果混为同一个 primary analysis；
4. 补充 task identity、trajectory success、critic score/value calibration 等诊断，用于解释 task4/task5 的差异；
5. 若研究目标转为普适 actor-computation 结论，再新建独立 study 纳入更多 actor topology、环境和 seeds，不改变本次 M12A-Core 的语义。

## 10. 工程与验证记录

本轮对应的 infrastructure 修复和验证如下：

- [`restore_module_from_checkpoint`](../../impls/utils/flax_utils.py) 现在保留 target parameter PyTree 的根和递归 mapping container；插入 source subtree 前按 target subtree 结构 coercion，并显式验证 restore 前后 `tree_structure` 相同；
- restore 只替换参数，不带入 source optimizer state，target optimizer state 保持 fresh；
- M12A integration tests：`15 tests, OK`，覆盖 FF 和 SS-K4 restore 后真实 `agent.update`、10-step smoke、actor changed、critic exact frozen、optimizer state unchanged、finite metrics；
- sweep protocol tests：`5 tests, OK`，覆盖 `protocol_stage` 对 `save_best_checkpoint`/`save_last_checkpoint` 的传播；
- Stage 2 实际状态 dry-run：`total=6 planned=0 completed=6 failed=0 running=0`；
- 未执行 Git commit、push、branch、reset、checkout 或其他 Git 操作；
- 未删除或覆盖六个失败的 `attempt_0` 目录；
- 本轮没有启动新的正式训练。

## 11. 数据文件索引

正式结果位于：

```text
/data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M12A/
```

其中：

- `M12A-C002__policy_extraction_ff_actor/.../seed_000__attempt_001/` 至 `seed_002__attempt_001/` 为 FF actor control；
- `M12A-C003__policy_extraction_single_state_k4_actor/.../seed_000__attempt_001/` 至 `seed_002__attempt_001/` 为 SS-K4 actor；
- 每个正式目录中的 `eval.csv` 是本报告的曲线数据源；
- `summary.json` 提供 final/best/best step；
- `runtime_metadata.json` 提供 run status、frozen dependency、checkpoint SHA 和 module fingerprint；
- `checkpoints/index.json` 提供 semantic best/last checkpoint identity。
