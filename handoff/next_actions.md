# Handoff Next Actions

## P0：冻结并验证

- [ ] 确认 `git status --short` 为空；
- [ ] 确认当前 HEAD 已推送到 `origin/main`；
- [ ] 运行 `PYTHONPATH=. python3 tools/handoff_doctor.py`；
- [ ] 保存 doctor JSON 输出作为 handoff snapshot；
- [ ] 检查 `/data` mount、OGBench dataset、runs、reevaluation 和 analysis output 可读。

## P1：环境可复现性

- [ ] 在正式运行机器生成 `python -VV`、`pip freeze`、JAX backend/devices 和 conda history；
- [ ] 将 environment manifest 与 machine/mount 信息绑定；
- [ ] 确认 CPU analysis tests 与 GPU runtime tests 使用的解释器不同之处已记录。

## P2：研究延续

- [ ] 对 M9B 增加 training-duration/checkpoint-curve follow-up；
- [ ] 分析 TwoState 的优化轨迹、state norm、loss 和 success 曲线；
- [ ] 保持同 seeds、same protocol、same checkpoint accounting；
- [ ] M10A 后续实验不得把当前 best allocation 事后升级为新的 reference；
- [ ] 如研究 shared-core HIQL，必须新建独立 study/config factor，不得改变当前 official-HIQL baseline 语义。

## P3：长期工程扩展

- [ ] 将环境依赖从宽松版本约束逐步收敛为可审计 lock/manifest；
- [ ] 将 analysis figure registry 和 semantic style registry 进一步配置化；
- [ ] 为新 study 提供统一 Study → Config → Run → Reevaluation → Analysis 模板；
- [ ] 保持 README、docs 日期目录和实际文件路径同步。
