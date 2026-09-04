# M19A 实现交付与手动启动说明

## 结论

M19A 已完成代码、配置、doctor、分析器、真实 Puzzle tiny smoke 和回归准备；唯一 formal run `M19A-4x4-E001` **尚未启动**，也没有创建其正式 run artifact。M16B 的两个历史锚点已通过 doctor 的实物校验。

## 起始版本与审计边界

prompt 给出的远端预期基线是 `01b615d699e500a8f9ac0a256780b62709ba3156`（`9-1 RLC M18-D: add final closed-loop and cross-critic diagnostics`）。根据用户的永久要求，Codex 没有执行任何 Git 命令（包括只读查询），因此不能把该值宣称为本地实际起始 HEAD。请在手动 review 时用 Git 核验并在 commit/worktree 中记录实际 commit。

编码前已实际审计：

- `PuzzleTokenAdapter → StructuredComputationBody → computation core → MeanContextReadout → algorithm head` 是当前 Puzzle production path；
- `MLPMixerBlock` 含 token branch 与 channel branch，各有 residual，内部没有 block-level LayerNorm；
- `MLPMixerStack(L)` 是 L 个 untied blocks；
- adapter 已拥有 button projection、index embedding、robot/context projection，readout 已拥有 mean pooling 与 fusion；
- factory 在改动前仅接受 structured `mlp_mixer`；
- M16B 4×4 B000/S002 的真实 artifacts 均存在且 `completed`。

## 改动文件

| 类别 | 文件 |
|---|---|
| 新 block | `impls/computation/blocks/entity_mlp.py` |
| export | `impls/computation/blocks/__init__.py`、`impls/computation/__init__.py` |
| factory/validation/runtime | `impls/computation/factory.py`、`impls/computation/slots.py`、`impls/main.py` |
| accounting | `impls/computation/accounting.py` |
| study/config | `experiments/M19A_puzzle_entity_factorization_isolation/study.yaml`、`configs/M19A-4x4-E001.yaml` |
| doctor/analyzer | `tools/m19a_doctor.py`、`tools/analyze_m19a.py` |
| tests | `tests/computation/test_entity_mlp.py`、`tests/integration/test_m19a_entity_factorization.py`、`tests/integration/test_m19a_puzzle_real_smoke.py` |
| 文档 | `docs/9-4/M19A_puzzle_entity_factorization_isolation.md`、本文件 |

## 精确实现语义

`EntityMLPBlock` 为：

```text
H_next = H + Dense2(GELU(Dense1(H)))
```

Dense 仅作用于最后一维，所有 tokens 共享参数；L=2 stack 中两个 blocks 不共享。没有 token transpose/einsum、token Dense、`tm_weights`、attention、token convolution、block LayerNorm、final activation 或 recurrent state。

canonical vector MLP 没有被拿来充当 token block，因为它并不等于 Mixer 的 channel branch。`MLPMixerBlock` 本身未改动，旧 Mixer parameter tree 与正常路径保持不变；`PuzzleTokenAdapter`、critic 的 action-only-context 语义、`MeanContextReadout`、GCIQL loss/Polyak/optimizer/evaluation 均未改变。

factory 按 `block` 严格分流：原 `mlp_mixer` 路径保持原逻辑；`entity_mlp` 只能是 `puzzle_tokens + feedforward + direct + mean_context`。normalization 会在 resolved runtime config 显式记录：

```text
block_type = entity_mlp
block_depth_L = 2
channel_hidden_dim = 256
token_interaction = false
```

alpha 同时写入 study fixed design、config factor、`agent_overrides.alpha` 与 resolved runtime config，doctor 已确认值精确为 `1.0`。

## M16B anchor 核验

| Anchor | 路径 | commit | status | final@1M |
|---|---|---|---|---:|
| B000 Flat | `/data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M16B/M16B-4x4-B000__4x4_flat_alpha1p0/puzzle-4x4-play-v0/seed_000` | `1eb3ac0f7ef40773bad5f0015d4fe4f490d4de6b` | completed | 0.260000 |
| S002 Mixer L2 | `/data/qijunrong/06-RL/offline-rl/exp/RLC/runs/M16B/M16B-4x4-S002__4x4_mixer_l2_alpha1p0/puzzle-4x4-play-v0/seed_000` | `1eb3ac0f7ef40773bad5f0015d4fe4f490d4de6b` | completed | 0.470000 |

doctor 已检查 env、seed、dataset identity/path、alpha、GCIQL、resolved agent fields、1M training/evaluation protocol、B000 Flat、S002 Mixer L2（`D=128,H_channel=256,H_token=64,L=2,index=true,mean,tm=none`）和 final@1M row。兼容性结论是 `ANCHOR REUSE: PASS`，且 `cross_commit_anchor_reuse=true`。M19A source commit 在正式前必须由用户手动提供给 doctor；当前未由 Codex 猜测或查询。

## 参数、MAC 与 depth

| 口径 | Entity L2 | Mixer L2 | Mixer − Entity |
|---|---:|---:|---:|
| actor/value computation block params | 131,840 | 136,096 | 4,256 |
| actor/value computation block Dense MAC | 2,097,152 | 2,621,440 | 524,288 |
| actor/value structured body depth | 6 | 10 | 4 |
| GCIQL actor+value+critic total params | 1,095,688 | 1,112,712 | 17,024 |
| GCIQL actor+value+critic total Dense MAC | 9,003,264 | 11,100,416 | 2,097,152 |

这些是实际 parameter tree/accounting 验证值，不是 budget matching。critic ensemble 的物理参数/MAC 计数相对 actor/value 翻倍。

## 验证结果

- EntityMLP shape、[B,T,D] canonical contract、unbatched structured boundary：通过；
- pooling 前 perturbation：改变一个 token 后，所有其他 token output 不变：通过；
- JAX Jacobian off-diagonal：零（tolerance 内）：通过；
- Mixer positive control：构造确定性非退化 token branch 后 cross-token derivative 非零：通过；
- zero-token-branch Mixer channel transplant：forward 与 channel gradients 一致：通过；
- token sharing / L=2 untied / 无 token-axis parameter：通过；
- GCIQL actor/value/critic integration、finite loss/gradient、参数变化、target critic Polyak update、action sampling：通过；
- 真实 Puzzle-4×4：正式 E001 architecture、真实 batch、2 updates、`actor/value/critic` loss finite、临时 checkpoint save/restore、1 episode eval：通过；这是 tiny smoke，不是科学结果；
- M19A doctor：通过；
- M19A generic dry-run：通过。

已记录的测试退出结果：

| 测试批次 | 结果 |
|---|---|
| `tests.computation.test_entity_mlp` | 7 passed |
| `tests.integration.test_m19a_entity_factorization` | 4 passed |
| `tests.integration.test_m19a_puzzle_real_smoke` | 1 passed |
| foundation / SingleState / TwoState / M12B / EntityMLP computation | 39 passed |
| M15 structured + M16A/B study | 15 passed |
| M17 modular structured computation | 7 passed |
| HIQL/MLP parity | 21 passed |
| M18 recurrent scaling | 6 passed |
| M18-D diagnostics | 12 passed |
| M15 + M17 real lifecycle smoke | 2 passed |
| canonical agents + M14 + provenance + checkpoint/management/sweep/launcher | 50 passed |
| `python -m compileall -q impls tools tests` | passed |

运行环境没有可用 CUDA device 时 JAX 给出 plugin warning 并回退 CPU；全部 smoke/regression 仍以明确退出码 0 完成。这不构成 formal training 或性能测量。

## Doctor / dry-run 输出摘要

```text
M19A PREFLIGHT: PASS
new formal runs = 1; historical anchors = 2; alpha=1.0; env=puzzle-4x4-play-v0; seed=0
ANCHOR REUSE: PASS
Entity accounting: params=1095688 MACs=9003264
Output path (must remain absent): .../runs/M19A/M19A-4x4-E001__4x4_entity_mlp_l2_alpha1p0/puzzle-4x4-play-v0/seed_000
Formal training was not started.
```

```text
total=1 planned=1 completed=0 failed=0 running=0 retained=0 remaining=1
[PLANNED] M19A-4x4-E001 ENTITY_TOKEN_ENTITY_MLP_L2 ... seed=0
```

## 用户手动 Git、preflight 与正式启动命令

以下命令仅供用户手动执行；Codex 没有执行其中任一 Git 或 `--execute` 操作。

### 1. 在主线 review 并提交

```bash
cd /home/eai/Research/RLC
git status
git diff --stat
git diff --check

git add impls/computation/blocks/entity_mlp.py \
  impls/computation/blocks/__init__.py \
  impls/computation/__init__.py \
  impls/computation/factory.py \
  impls/computation/slots.py \
  impls/computation/accounting.py \
  impls/main.py \
  experiments/M19A_puzzle_entity_factorization_isolation \
  tools/m19a_doctor.py tools/analyze_m19a.py \
  tests/computation/test_entity_mlp.py \
  tests/integration/test_m19a_entity_factorization.py \
  tests/integration/test_m19a_puzzle_real_smoke.py \
  docs/9-4
git commit -m "M19A: add Puzzle entity-wise MLP isolation control"
git push origin main
```

### 2. 创建 detached frozen M19A worktree

```bash
cd /home/eai/Research/RLC
M19A_SOURCE_COMMIT="$(git rev-parse HEAD)"
git worktree add --detach /home/eai/Research/RLC-m19a-exp "$M19A_SOURCE_COMMIT"
cd /home/eai/Research/RLC-m19a-exp

export RLC_PYTHON=/home/eai/Tools/miniforge3/envs/brain_nav/bin/python
export OGBENCH_DATASET_DIR=/data/qijunrong/06-RL/offline-rl/data/raw_ogbench
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export RLC_SOURCE_COMMIT="$M19A_SOURCE_COMMIT"
```

### 3. Doctor（不启动训练）

```bash
"$RLC_PYTHON" tools/m19a_doctor.py \
  --study experiments/M19A_puzzle_entity_factorization_isolation/study.yaml \
  --dataset-root "$OGBENCH_DATASET_DIR" \
  --run-root /data/qijunrong/06-RL/offline-rl/exp/RLC/runs \
  --gpus 0 \
  --m19a-source-commit "$RLC_SOURCE_COMMIT" \
  --require-m19a-source-commit
```

### 4. Formal launcher dry-run（不启动训练）

```bash
bash scripts/run_study.sh \
  --study experiments/M19A_puzzle_entity_factorization_isolation/study.yaml \
  --configs M19A-4x4-E001 \
  --gpus 0 \
  --run-root /data/qijunrong/06-RL/offline-rl/exp/RLC/runs \
  --dataset-root "$OGBENCH_DATASET_DIR" \
  --train-steps 1000000 --batch-size 1024 --log-interval 5000 \
  --eval-interval 100000 --eval-tasks all --eval-episodes 20 \
  --save-interval 100000 --eval-temperature 0.0 \
  --save-best-checkpoint --save-last-checkpoint \
  --dry-run
```

注意：当前 `run_study.sh` 对显式 `--video-episodes 0` 判为无效；故命令刻意省略它，`impls.main` 的默认值仍为 `video_episodes=0`，与 study protocol 一致。

### 5. Formal execute（仅在 doctor/dry-run 都通过后，由用户手动执行）

```bash
bash scripts/run_study.sh \
  --study experiments/M19A_puzzle_entity_factorization_isolation/study.yaml \
  --configs M19A-4x4-E001 \
  --gpus 0 \
  --run-root /data/qijunrong/06-RL/offline-rl/exp/RLC/runs \
  --dataset-root "$OGBENCH_DATASET_DIR" \
  --train-steps 1000000 --batch-size 1024 --log-interval 5000 \
  --eval-interval 100000 --eval-tasks all --eval-episodes 20 \
  --save-interval 100000 --eval-temperature 0.0 \
  --save-best-checkpoint --save-last-checkpoint \
  --execute
```

该用户发起的 launcher 会对 frozen worktree 做自己的 Git clean/HEAD preflight；不要由 Codex 代替执行。

### 6. 训练完成后的三组分析

```bash
"$RLC_PYTHON" tools/analyze_m19a.py \
  --study experiments/M19A_puzzle_entity_factorization_isolation/study.yaml \
  --run-root /data/qijunrong/06-RL/offline-rl/exp/RLC/runs \
  --output-dir docs/9-4/M19A_results
```

分析器固定输出顺序为 Flat → EntityMLP → Mixer，并写 primary table、task-level final success、两个 descriptive deltas 与可用的 wall-clock/throughput。E001 缺失、partial 或 provenance invalid 时，它不会用其他实验替代，也不会给出三组正式结论。

## Deferred work

- parameter-matched EntityMLP；
- MAC-matched EntityMLP；
- additional seeds；
- other Puzzle sizes；
- EntityMLP alpha sweep；
- alternative readout；
- recurrent EntityMLP 或任何 M18 intervention。

这些必须作为后续独立设计，不能根据 E001 的中途表现临时加入。
