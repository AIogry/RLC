# M26 — Puzzle goal-coordinate mechanism diagnostics

2026-09-19：**10 defined / 8 initially selected / 2 unscheduled matched references**。
本轮仅配置、CPU验证与人工发布交接；没有启动 GPU smoke 或正式训练。

完整科学设计见 [EXPERIMENT_DESIGN](../../docs/research/experiments/M26/EXPERIMENT_DESIGN.md)，
证据见 [M24 继承审计](../../docs/research/experiments/M26/M24_INHERITANCE_AUDIT.md)，
测试与门槛见 [交接](../../docs/research/experiments/M26/EXPERIMENT_HANDOFF.md)。
**按阶段人工执行**：[Git → 冻结 → GPU smoke → 正式启动手册](../../docs/research/experiments/M26/MANUAL_OPERATIONS.md)。

## 唯一正式范围

Study `M26`，环境 `puzzle-4x5-play-v0`，training seeds `[0]`。
固定 GCIQL DDPG+BC / alpha=0.4 / Mixer-L2 / batch1024 / 1M updates。
全部使用 v2 `token_aux_v1`：每 token 是 current4 + goal4 + aux1 = 9D。
value_side 同时作用于 V、Q、target Q。

| ID | actor | value_side | 变换 | distance |
| --- | --- | --- | --- | --- |
| M26-C001 | residual | residual | identity | zero |
| M26-C002 | operation | operation | identity | zero |
| M26-C003 | residual | residual | identity | sum(x)/20 |
| M26-C004 | operation | operation | identity | sum(x)/20 |
| M26-C005 | operation | operation | perm_v1 | zero |
| M26-C006 | operation | operation | perm_v1 | sum(x)/20 |
| M26-C007 | operation | operation | dense_v1 | zero |
| M26-C008 | operation | operation | dense_v1 | sum(x)/20 |
| M26-C009 | operation | residual | identity | zero |
| M26-C010 | residual | operation | identity | zero |

默认只选 C003–C010；C001/C002 都可执行但不在首批。选择通过现有 `--configs`，
不改 YAML、不伪造 completed、不复用 checkpoint、不 warm start。
历史 M24 是 v1/8D，不能当作这两个 v2 匹配参照。首批8组完成不等于全 Study 完成。

全部5个task，每task20 episodes，temperature0、gaussian=None、video0；
log5000 / eval100000 / save100000，保存 best/last。
主终点是 **last@1M** 的各 task success 和 Tasks2–5 的 HardTaskMean；best 仅为运行设施。
单 training seed、单 P、单 B，只用于方向选择，不证明 learned operation 或跨任务泛化。

## 运行边界

GPU1、一个 scheduler、每卡两个独立进程，每个进程完整 batch1024。
Git 写操作和实验启动由用户完成；训练源码必须是本目录所在 `RLC-M26` 的 clean detached
固定提交，不使用当前未提交配置直接训练。工程报告在仓库外，不进入 `runs/M26`。

`tools/study_gpu_smoke.py` 默认只打印计划；需人工显式 `--execute`，先 C008 单进程，
再 C007/C008 双进程。短评估为全部5tasks各1episode，记录编译/训练/评估抽样显存峰值；
不是正式100episode评估，不是全1M资源或性能保证。GPU smoke 尚未执行。
