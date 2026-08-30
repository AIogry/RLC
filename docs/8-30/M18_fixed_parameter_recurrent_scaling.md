# M18：固定参数量的 Puzzle recurrent computation scaling

日期：2026-08-30  
状态：正式 Study、配置、preflight、聚合和 M18-D checkpoint-only diagnostic 已准备；未启动任何 M18 正式训练，未执行任何 Git 操作。

## 1. 科学问题与边界

M18 研究的问题是：在固定 representation、固定 MLP-Mixer block capacity、固定 trainable parameter count 的前提下，增加共享 recurrent computation budget `K` 是否改变 Puzzle policy learning？

唯一的正式 manipulated factor 是 `K`。本轮不是：

- 更大网络是否更好；
- `L × K` architecture search；
- alpha sweep；
- Flat baseline 复现；
- alternative readout/state initialization/residual/input-injection 比较；
- adaptive test-time computation、ACT 或 learned halting。

训练仅使用 GCIQL、`puzzle-4x4-play-v0`、seed 0、actor+value+critic 全部 structured。因为只有一个 seed，所有后续差值都只能作描述性解释；不得进行显著性检验、宣称单调/普适 scaling law，或称某个 K statistically optimal。

## 2. M17 前提的实际审计

编码前重新读取了当前主线的 M17 report、representation/computation/factory/accounting/slot/GCIQL/main 路径、M17 integration/real-Puzzle tests、M16A/B/C Study 与 doctor 基础设施、`scripts/run_study.sh`、`tools/sweep.py` 及实验 provenance 代码。结论如下。

| 检查项 | 当前实际状态 |
|---|---|
| Puzzle production path | `factory.py` 对 `structure=puzzle_tokens` 构造 `StructuredComputationBody(adapter, core, readout)`；保留的 `PuzzleStructuredBody` 仅为 legacy reference oracle。 |
| representation ownership | `StructuredRepresentation` parameter-free；`PuzzleTokenAdapter` 只拥有 button/index/context projection，不拥有 Mixer、K 或 fusion。 |
| recurrent block unit | `MLPMixerStack(num_blocks=L)` 内部 L 个 Mixer block 互不共享；M18 固定 `L=2`。 |
| K sharing | structured `SingleState` 将一个 external `MLPMixerStack` clone 为唯一 `update_module`，每次 iteration 复用它，不创建 `update_block_0...`。 |
| canonical state contract | `identity` mapping、`zero_buffer`、`z_plus_x`、topology `residual=false`、`shared`、`direct`。 |
| FF/SS correctness | 当前 `tests.integration.test_m17_modular_structured_computation` 已实际运行，7 tests PASS；其中覆盖 `FF(L)==SS(L,K=1)`、梯度/参数/MAC/depth parity。 |
| three GCIQL slots | actor/value/critic 的 descriptor、factory 与 M17 test 均使用 modular structured path；critic action 只进入 context branch。 |
| runtime alpha authority | `configuration.agent_overrides.alpha` 经 `_make_config()` 合并为 GCIQL 实际 `config['alpha']`。 |
| M16C factual scope | 实际完成的新 run 只有 alpha `0.1/0.2/0.5/0.7`；M16A/M16B 提供 `0.3/1.0` anchors。没有 alpha `0.4` run。 |

M18 因此不改动 M17 framework、GCIQL loss、target/Polyak、goal sampling、dataset、augmentation、action distribution 或 evaluation policy。

## 3. 固定的数学语义

令 Puzzle adapter 为 `A_phi`，其 token 输出为 `X`；令完整的 L-layer Mixer block 为：

```text
B_theta^(L)(X) = M_theta_L(...M_theta_2(M_theta_1(X)))
```

M18 固定 `L=2`，并使用 M17 canonical structured SingleState：

```text
Z^0 = 0
Z^(k+1) = B_theta^(2)(Z^k + X),     k = 0, ..., K-1
```

其中同一个 `B_theta^(2)` 在所有 iteration 上共享参数。`K` 从不创建额外的 Mixer stack：

| K | unique Mixer layers（单一路径） | executed Mixer layers（单一路径） |
|---:|---:|---:|
| 1 | 2 | 2 |
| 2 | 2 | 4 |
| 4 | 2 | 8 |
| 8 | 2 | 16 |

`topology residual=false` 的意思是 `Z^(k+1)=update`，而不是 `Z^k+update`。它不关闭每个 `MLPMixerBlock` 内部的 token-mixing 和 channel-mixing residual。

state 是 decision-local computation state：每次 actor/value/critic forward 都从 fresh broadcast 的 `z_init` 开始；不会跨 environment timestep 传递。

## 4. 正式训练矩阵

| Config ID | L | K | alpha | Environment | Seed | Placement |
|---|---:|---:|---:|---|---:|---|
| `M18-4x4-L2-K1` | 2 | 1 | 0.4 | Puzzle-4x4 Play | 0 | actor+value+critic |
| `M18-4x4-L2-K2` | 2 | 2 | 0.4 | Puzzle-4x4 Play | 0 | actor+value+critic |
| `M18-4x4-L2-K4` | 2 | 4 | 0.4 | Puzzle-4x4 Play | 0 | actor+value+critic |
| `M18-4x4-L2-K8` | 2 | 8 | 0.4 | Puzzle-4x4 Play | 0 | actor+value+critic |

每个 slot 均显式记录：

```yaml
structure: puzzle_tokens
block: mlp_mixer
block_kwargs: {num_blocks: 2, token_hidden_dim: 64, channel_hidden_dim: 256, tm_mode: none}
topology: single_state
topology_kwargs:
  iterations: K
  input_mapping: identity
  state_dim: 128
  state_init: zero_buffer
  state_init_std: 1.0
  input_injection: z_plus_x
  residual: false
  parameter_sharing: shared
readout: mean_context
credit: direct
```

representation 固定为 16 个 button token（`robot_dim=19`、`button_feature_dim=4`、`token_dim=128`、index embedding enabled）；Mixer hidden widths 固定为 token 64、channel 256、`tm_mode=none`。readout 保持 M17/M16 的 mean token summary + robot/context fusion，不改变 activation 或 LayerNorm semantics。

## 5. alpha=0.4 的 provenance

`alpha=0.4` 是固定 nuisance/training hyperparameter，不是 M18 的研究维度，更不是已证明的最优值。它的唯一定位是：M16C 在 Puzzle-4x4 S002 seed-0 下识别出的当前较好 `0.3–0.5` 区域的中点 operating point。

- 不得写成 `optimal alpha`、`best alpha` 或 `alpha*=0.4`；
- M16C 没有训练 `0.4`；
- K1@0.4 同时是 recurrent curve anchor 和 alpha operating-point calibration anchor；
- 即使 K1 中途不佳，K2/K4/K8 仍必须完整训练至 1M，除非出现 NaN/OOM/基础设施或数据错误；
- 不在本轮添加 `alpha × K` sweep。

## 6. 协议、终点与解释规则

训练协议沿用 canonical Puzzle GCIQL：1M steps、batch 1024、lr 3e-4、discount .99、expectile .9、tau .005、DDPG+BC、const std、每 5k log、每 100k eval、all 5 tasks、每 task 20 episodes、temperature 0、no Gaussian noise、no video、每 100k save、best/last semantic checkpoints 都保存。

- Primary：`evaluation/overall_success` at 1M。
- Secondary：best success、best step、last-3 mean、normalized AUC。
- AUC：100k–1M 的 10 个 evaluation checkpoints，trapezoidal integration / 900k。
- 保留每个 evaluation task 的 final success。

后验结果按以下预注册边界描述：

| Pattern | 可支持的描述 | 不可推出 |
|---|---|---|
| K1<K2<K4<K8 | fixed capacity 下与 positive recurrent-computation scaling trend 一致 | 更多计算普遍导致更好 RL |
| K1<K2<K4>K8 | 当前 transition 下存在有限的 useful recurrent-computation range 的证据 | 某 K 具有统计最优性 |
| 近似平坦 | 当前 transition/readout/optimization 下额外 shared compute 的收益有限 | recurrence 无用 |
| K1>K2>K4>K8 | 当前 repeated-input transition 下更深执行有害 | recurrence 一般无用 |

## 7. 工具与可复现性

新增文件：

- `experiments/M18_puzzle_recurrent_compute_scaling/study.yaml`：Study、假设、failure policy、M18-D 边界。
- `experiments/.../configs/M18-4x4-L2-K{1,2,4,8}.yaml`：四个唯一正式 config。
- `tools/m18_doctor.py`：non-executing preflight；解析 resolved runtime config、构造 actor/value/critic、检查 K invariance、MAC/depth 和 output collision。
- `tools/analyze_m18.py`：训练结束后的 raw curve、task endpoint、AUC、descriptive delta 与 Markdown/CSV/JSON 汇总。
- `tools/m18_cross_k_eval.py`：M18-D checkpoint-only `K_train × K_test` worker/queue/aggregation tool。
- `tests/integration/test_m18_recurrent_compute_scaling.py`：配置、doctor、dry-run、same-tree K override restore、aggregation，以及源 checkpoint 不可变的 M18-D 五任务 lifecycle smoke tests。

doctor 的 actor-path accounting 是本轮 launch gate：params、unique Mixer layers、buffer 必须在 K 上恒定；executed Mixer layers 必须为 `2K`，actor body MAC 与 executed depth 必须随 K 增长。critic 是 two-member ensemble，故其 physical layer count 会是单个 critic member 的两倍；所有报告同时保留 actor 单一路径和 slot-level physical accounting，避免混淆。

实际 M18 doctor（当前 Terra 工作区、正式 output root 不存在）结果如下。`Params(total)` 是 online GCIQL actor/value/critic 合计；`Actor params/MAC/depth/buffer` 是单一 actor slot 口径，因而可以直接验证固定 L 下的 K invariance。

| K | Params(total) | Actor params | Actor unique Mixer | Actor executed Mixer | Actor body Dense MAC | Actor executed depth | Actor buffer elems |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1,112,712 | 278,437 | 2 | 2 | 2,773,760 | 10 | 128 |
| 2 | 1,112,712 | 278,437 | 2 | 4 | 5,395,200 | 18 | 128 |
| 4 | 1,112,712 | 278,437 | 2 | 8 | 10,638,080 | 34 | 128 |
| 8 | 1,112,712 | 278,437 | 2 | 16 | 21,123,840 | 66 | 128 |

## 8. M18-D：cross-K test-time depth probe

M18-D 在训练完成后对每个 semantic `last` checkpoint 运行 4 个 test-time K：

```text
             K_test
           1   2   4   8
K_train 1  x   x   x   x
        2  x   x   x   x
        4  x   x   x   x
        8  x   x   x   x
```

工具的安全 contract：

1. 仅验证 completed、clean、provenance-consistent M18 source run；
2. 从 resolved source config 构造同一 parameter shape 的 agent；
3. 唯一覆写 actor/value/critic 三个 `topology_kwargs.iterations`；
4. 检查 source 与 target 的 trainable parameter counts 相等，然后 restore checkpoint；
5. 只调用 episode evaluation，明确记录 `training_or_optimizer_updates: 0`；
6. 只写入独立的 `.../diagnostics/M18D/.../Ktest_*` subtree，拒绝覆盖已有 diagnostic output；
7. 输出 16-cell matrix 与 task-level raw episodes，不修改 source checkpoint。

默认使用 `last`，对应 M18 的 final@1M primary endpoint。对角线是 trained-depth evaluation；上三角存在 inference-depth distribution shift；下三角是 reduced-compute/early-exit robustness probe。它不是 finetuning、adaptive compute、ACT、learned halting 或 test-time scaling proof。

已在临时目录完成最小生命周期验证：构造 provenance-complete 的 `K_train=1` semantic `last` checkpoint，以 `K_test=8` restore，并在五个 Puzzle task 各评估 1 episode；结果写入独立 temporary diagnostic subtree，源 checkpoint 的 SHA-256 在前后完全一致。该 smoke 只验证工具 contract，不构成正式 M18-D 数据。

当前不实施 intermediate `Z^1...Z^K` norm/delta trace。M17 的现有 forward 保持不额外 materialize intermediate state，避免为了诊断改变训练 forward/loss；若 K8 出现退化，该 state-dynamics diagnostic 应作为后续独立的低侵入研究任务。

## 9. 用户手动流程（命令以当前真实 CLI 为准）

以下 Git 操作均由用户手动执行；本次准备没有执行它们。

```bash
# 1) 在主仓库 review 后，由用户自行 add/commit，并记下提交 SHA。
#    下面用 <M18_COMMIT_SHA> 表示该 SHA。

# 2) 由用户创建 detached frozen experiment worktree。
git worktree add --detach /home/eai/Research/RLC-m18-exp <M18_COMMIT_SHA>

# 3) 在 frozen worktree 配置运行环境。
cd /home/eai/Research/RLC-m18-exp
export RLC_PYTHON=/home/eai/Tools/miniforge3/envs/brain_nav/bin/python
export OGBENCH_DATASET_DIR=/data/qijunrong/06-RL/offline-rl/data/raw_ogbench
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# 4) 先运行 non-executing M18 doctor。
"$RLC_PYTHON" tools/m18_doctor.py \
  --study experiments/M18_puzzle_recurrent_compute_scaling/study.yaml \
  --dataset-root "$OGBENCH_DATASET_DIR" \
  --run-root /data/qijunrong/06-RL/offline-rl/exp/RLC/runs \
  --gpus 0,1

# 5) 再 dry-run：此模式不会做 Git preflight、不会创建 run、不会训练。
RLC_SOURCE_COMMIT=<M18_COMMIT_SHA> bash scripts/run_study.sh \
  --study experiments/M18_puzzle_recurrent_compute_scaling/study.yaml \
  --gpus 0,1 \
  --run-root /data/qijunrong/06-RL/offline-rl/exp/RLC/runs \
  --dataset-root "$OGBENCH_DATASET_DIR" \
  --train-steps 1000000 --batch-size 1024 --log-interval 5000 \
  --eval-interval 100000 --eval-tasks all --eval-episodes 20 \
  --save-interval 100000 --eval-temperature 0 \
  --save-best-checkpoint --save-last-checkpoint \
  --dry-run

# 6) 用户确认并自行执行正式训练。--execute 会由 launcher 检查 clean Git worktree。
bash scripts/run_study.sh \
  --study experiments/M18_puzzle_recurrent_compute_scaling/study.yaml \
  --gpus 0,1 \
  --run-root /data/qijunrong/06-RL/offline-rl/exp/RLC/runs \
  --dataset-root "$OGBENCH_DATASET_DIR" \
  --train-steps 1000000 --batch-size 1024 --log-interval 5000 \
  --eval-interval 100000 --eval-tasks all --eval-episodes 20 \
  --save-interval 100000 --eval-temperature 0 \
  --save-best-checkpoint --save-last-checkpoint \
  --execute
```

`run_study.sh` 当前对 `--video-episodes` 仅接受正整数；不传该 option 时 `impls.main` 的实际默认值为 `video_episodes=0`，并会记录在 resolved launcher config。未传 `--eval-gaussian` 时实际默认值为 `None`。这两个 omission 不改变 M18 冻结协议。

训练全部完成后，用户可手动运行：

```bash
# 正式曲线与 endpoint 汇总（只读 run artifacts；输出目录由用户选择）。
"$RLC_PYTHON" tools/analyze_m18.py \
  --study experiments/M18_puzzle_recurrent_compute_scaling/study.yaml \
  --run-root /data/qijunrong/06-RL/offline-rl/exp/RLC/runs \
  --output-dir docs/8-30/M18_results

# M18-D 先检查完整 4 x 4 plan；不会运行 evaluation。
"$RLC_PYTHON" tools/m18_cross_k_eval.py \
  --study experiments/M18_puzzle_recurrent_compute_scaling/study.yaml \
  --source-run-root /data/qijunrong/06-RL/offline-rl/exp/RLC/runs \
  --output-root /data/qijunrong/06-RL/offline-rl/exp/RLC/diagnostics \
  --checkpoint last --gpus 0,1 --episodes-per-task 20 \
  --evaluation-seed 18018 --eval-temperature 0 --dry-run

# 用户确认后才执行 M18-D evaluation-only matrix。
"$RLC_PYTHON" tools/m18_cross_k_eval.py \
  --study experiments/M18_puzzle_recurrent_compute_scaling/study.yaml \
  --source-run-root /data/qijunrong/06-RL/offline-rl/exp/RLC/runs \
  --output-root /data/qijunrong/06-RL/offline-rl/exp/RLC/diagnostics \
  --checkpoint last --gpus 0,1 --episodes-per-task 20 \
  --evaluation-seed 18018 --eval-temperature 0 --execute
```

## 10. 已完成的 preparation gates

| Gate | 实际结果 |
|---|---|
| M18 config/runtime/doctor/cross-K restore/aggregation/lifecycle tests | PASS，6 tests（含 K1 checkpoint → K8 五任务 evaluate-only smoke）。 |
| M17 modular framework + real Puzzle smoke | PASS，包含 FF/SS parity；纳入下方 31-test group。 |
| M15 structured + real Puzzle smoke | PASS；纳入 31-test group。 |
| M16A/M16B/M16C Study regressions | PASS；纳入 31-test group。 |
| 上述 environment/Study group | PASS，31 tests。 |
| computation discovery（含 vector SingleState/TwoState） | PASS，53 tests。 |
| M13 canonical、M14 base algorithm/GCIQL、provenance、checkpoint lifecycle | PASS，31 tests。 |
| `compileall -q impls tools tests` | PASS。 |
| `tools/m18_doctor.py` | PASS；四个输出路径均不存在。 |
| `scripts/run_study.sh --dry-run` | PASS；planned=4、completed=0、remaining=4，恰好输出 K1/K2/K4/K8 四个 jobs。 |

测试运行使用 `JAX_PLATFORMS=cpu`；需要 environment 的 cases 使用 `MUJOCO_GL=egl PYOPENGL_PLATFORM=egl`。这是当前 Terra 的无 X11 display 要求，不是 M18 failure。

唯一实现修复是 M17 resolved-config materialization 的兼容性缺陷：`_normalize_structured_compute_defaults()` 曾对 `ml_collections.ConfigDict` 调用不存在的 `.setdefault()`。该错误会使 Study YAML 的 structured SingleState 在训练前解析失败；现已替换为语义等价的显式 membership 判断。它不改变任何 M18 科学变量或 M17 recurrent equation，且上述全套回归已验证。

当前 `/data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M18` 不存在，因此没有正式 M18 training artifact 被创建。所有 Git 操作仍留给用户手动完成。
