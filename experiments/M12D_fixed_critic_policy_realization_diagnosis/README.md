# M12-D — Fixed-Critic Policy Realization Diagnosis

本 study 只对已有 actor checkpoint 做 post-hoc diagnosis，不训练、不改写 checkpoint、不执行 Git，也不读取 SingleState 内部 z/state dynamics。

Primary actor names 在 protocol 中声明，通用诊断层不依赖固定五类模型。当前 primary set 是 K1SN、K4SN（M12B-R，M12A-C003 attempt2）、K4SZ、D9、Residual；每个 seed 的 actor 必须配对到同 seed 的 M12A-C001 critic last@1M。

B_T 使用 exact GCDataset actor-goal sampler；B_DE 重用 B_T dataset states 并替换为 formal evaluator goals；B_R 用 common task/episode seeds 收集五个 origin rollout，按 actor/task/episode/progress bin 平衡，再在相同 state-goal 上 cross-evaluate。

正式命令由用户手动执行；默认 smoke 只写入临时目录。

