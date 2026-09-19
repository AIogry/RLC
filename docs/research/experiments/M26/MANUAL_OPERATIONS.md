# M26 人工 Git、冻结、GPU smoke 与启动操作

2026-09-19，基于本机实查，不是一键脚本。下面所有Git写操作、GPU smoke和正式训练
**由用户按阶段执行**；agent没有执行它们。配置及CPU验证完成，但尚未formal-ready。
不要整份复制执行；特别是8组/10组/补2组只能选一套，不同时开两个scheduler。

## A/B. main/M24 与能力发布已经完成

本次只读实查及 `ls-remote`：

- main干净，HEAD与远端main均为 `38fccf0ff8e034723e85a2016026244f425f0309`。
- M26入口干净，能力HEAD与远端分支均为 `f0484f425cfd4092d1b95c7bf3e1348ad00ea0ff`。
- 分支 `m26-goal-conditioning-diagnostics`；共同Git目录 `/home/eai/Research/RLC/.git`。
- main没有此前两份untracked说明。本轮配置写入后只有M26树变脏。
- 因此直接从C开始：**不要重复提交能力或把本轮配置混进能力提交**。

在Bash会话中设置；zsh用户先运行 `bash`。变量不修改共享环境或系统安装。

```bash
M26_MAIN=/home/eai/Research/RLC
M26_DEV=/home/eai/Research/RLC-M26
M26_BRANCH=m26-goal-conditioning-diagnostics
M26_PY=/home/eai/Tools/miniforge3/envs/brain_nav/bin/python
M26_STUDY=experiments/M26_puzzle_goal_coordinate_diagnostics/study.yaml
M26_DATA=/data/qijunrong/06-RL/offline-rl/data/raw_ogbench
M26_RUNS=/data/qijunrong/06-RL/offline-rl/exp/RLC/runs

git -C "$M26_MAIN" status --short --branch --untracked-files=all
git -C "$M26_DEV" status --short --branch --untracked-files=all
git -C "$M26_MAIN" remote get-url origin
git -C "$M26_DEV" log -2 --format='%H %s'
git -C "$M26_MAIN" ls-remote --heads origin refs/heads/main refs/heads/m26-goal-conditioning-diagnostics
```

远端URL应为 `git@github.com:AIogry/RLC.git`。`ls-remote`不是fetch，不更新本地origin缓存。
如果需要重新做A的发布检查，下面由用户执行；已同步时没有必要制造新commit：

```bash
(
  set -euo pipefail
  [[ "$(git -C "$M26_MAIN" branch --show-current)" == main ]]
  [[ -z "$(git -C "$M26_MAIN" status --porcelain --untracked-files=all)" ]]
  git -C "$M26_MAIN" fetch origin
  git -C "$M26_MAIN" rev-list --left-right --count origin/main...main
  git -C "$M26_MAIN" merge-base --is-ancestor origin/main main
  git -C "$M26_MAIN" push origin refs/heads/main:refs/heads/main
)
```

B已经是独立能力commit，无需add/commit。若远端读取失败，publication未核验，先排查网络/权限，
不要改写本地历史。若出现未预期用户改动，先保留和审查，不reset/clean/stash来强行清空。

## C. 审查并提交本轮实验配置（第二个独立提交）

先审查tracked与全部新文件。原能力handoff不改，impls/ogbench/历史M24配置不应出现本轮diff。

```bash
cd "$M26_DEV"
git diff --check
git diff --stat
git diff
git ls-files --others --exclude-standard
git diff --cached --stat
git diff f0484f425cfd4092d1b95c7bf3e1348ad00ea0ff -- impls ogbench experiments/M24A_puzzle_goal_conditioning_intervention

# CPU复查；mock-device测试不是GPU smoke。
env PYTHONPATH="$M26_DEV" PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu \
  "$M26_PY" -m unittest tests.experiment.test_m26 tests.experiment.test_study_preparation_tools -v
```

本轮精确allowlist：3个tracked上下文文件和22个新增文件。不要 `git add .`。
先确认没有原先已staged的未知改动，再暂存以下文件：

```bash
(
  set -euo pipefail
  [[ "$(git -C "$M26_DEV" branch --show-current)" == "$M26_BRANCH" ]]
  git -C "$M26_DEV" diff --cached --quiet
  git -C "$M26_DEV" add -- \
    experiments/M26_puzzle_goal_coordinate_diagnostics/study.yaml \
    experiments/M26_puzzle_goal_coordinate_diagnostics/README.md \
    experiments/M26_puzzle_goal_coordinate_diagnostics/configs/M26-C001.yaml \
    experiments/M26_puzzle_goal_coordinate_diagnostics/configs/M26-C002.yaml \
    experiments/M26_puzzle_goal_coordinate_diagnostics/configs/M26-C003.yaml \
    experiments/M26_puzzle_goal_coordinate_diagnostics/configs/M26-C004.yaml \
    experiments/M26_puzzle_goal_coordinate_diagnostics/configs/M26-C005.yaml \
    experiments/M26_puzzle_goal_coordinate_diagnostics/configs/M26-C006.yaml \
    experiments/M26_puzzle_goal_coordinate_diagnostics/configs/M26-C007.yaml \
    experiments/M26_puzzle_goal_coordinate_diagnostics/configs/M26-C008.yaml \
    experiments/M26_puzzle_goal_coordinate_diagnostics/configs/M26-C009.yaml \
    experiments/M26_puzzle_goal_coordinate_diagnostics/configs/M26-C010.yaml \
    tests/experiment/test_m26.py \
    tests/experiment/test_study_preparation_tools.py \
    tools/prepare_goal_coordinate_transform.py \
    tools/audit_goal_conditioning_study.py \
    tools/audit_study_inheritance.py \
    tools/study_gpu_smoke.py \
    docs/research/experiments/M26/EXPERIMENT_DESIGN.md \
    docs/research/experiments/M26/M24_INHERITANCE_AUDIT.md \
    docs/research/experiments/M26/EXPERIMENT_HANDOFF.md \
    docs/research/experiments/M26/MANUAL_OPERATIONS.md \
    docs/research/PROJECT_STATE.md \
    docs/research/EXPERIMENT_REGISTRY.md \
    docs/research/DECISIONS.md
  git -C "$M26_DEV" diff --cached --check
  git -C "$M26_DEV" diff --cached --stat
  git -C "$M26_DEV" diff --cached
)
```

人工确认staged diff仅含这些配置/工具/测试/文档后，单独执行：

```bash
(
  set -euo pipefail
  git -C "$M26_DEV" diff --cached --check
  git -C "$M26_DEV" commit -m "experiment: configure M26 4x5 seed-0 diagnostics"
  git -C "$M26_DEV" push origin HEAD:refs/heads/m26-goal-conditioning-diagnostics
  git -C "$M26_DEV" rev-parse HEAD
  git -C "$M26_DEV" status --short --untracked-files=all
)
```

push被拒绝时停下检查分叉/保护/权限，不force、不自动rebase。

## D/E. 快进main，冻结现有RLC-M26

先人工检查没有从M26运行的训练/smoke/审计进程；这里不结束任何已有进程：

```bash
ps -eo pid,ppid,args | rg 'RLC-M26|impls.main|study_gpu_smoke|tools/sweep' || true
git -C "$M26_MAIN" worktree list --porcelain
```

两个工作树必须clean。main如果已有后续研发，先审查保留；不能快进就停止让人处理分叉。

```bash
(
  set -euo pipefail
  [[ "$(git -C "$M26_MAIN" branch --show-current)" == main ]]
  [[ "$(git -C "$M26_DEV" branch --show-current)" == "$M26_BRANCH" ]]
  [[ -z "$(git -C "$M26_MAIN" status --porcelain --untracked-files=all)" ]]
  [[ -z "$(git -C "$M26_DEV" status --porcelain --untracked-files=all)" ]]
  git -C "$M26_MAIN" fetch origin
  git -C "$M26_MAIN" merge-base --is-ancestor origin/main main
  git -C "$M26_MAIN" merge --ff-only "$M26_BRANCH"
  [[ "$(git -C "$M26_MAIN" rev-parse HEAD)" == "$(git -C "$M26_DEV" rev-parse HEAD)" ]]
  git -C "$M26_MAIN" push origin refs/heads/main:refs/heads/main
  M26_FINAL_SHA="$(git -C "$M26_DEV" rev-parse HEAD)"
  git -C "$M26_DEV" show "$M26_FINAL_SHA:$M26_STUDY" >/dev/null
  git -C "$M26_DEV" switch --detach "$M26_FINAL_SHA"
  [[ -z "$(git -C "$M26_DEV" branch --show-current)" ]]
  [[ -z "$(git -C "$M26_DEV" status --porcelain --untracked-files=all)" ]]
  printf 'Frozen source: %s\n' "$M26_FINAL_SHA"
)
```

若人工采用真正merge commit而非快进，必须重验合并树，以该实际最终提交冻结；
不能套用上述HEAD相等检查后擅自reset。无需另建worktree或删除开发分支。

紧接上一步，在同一Bash中记录这次固定提交；之后main发展也不能重新选择“最新main”：

```bash
M26_TRAIN_SHA="$(git -C "$M26_DEV" rev-parse HEAD)"
M26_ENGINEERING=/data/qijunrong/06-RL/offline-rl/exp/RLC/engineering/M26
mkdir -p "$M26_ENGINEERING"
M26_RELEASE="$(mktemp -d "$M26_ENGINEERING/release_${M26_TRAIN_SHA:0:12}.XXXXXX")"
printf '%s\n' "$M26_TRAIN_SHA" > "$M26_RELEASE/train_source_sha.txt"
git -C "$M26_DEV" show "$M26_TRAIN_SHA:$M26_STUDY" > "$M26_RELEASE/study_snapshot.yaml"
printf 'Keep this release directory: %s\n' "$M26_RELEASE"

# 可选：将本次CPU证据归档到外部release，避免/tmp被清理；不提交进Git。
cp -a -- /tmp/rlc-m26-experiment.qO8B5C "$M26_RELEASE/cpu_configuration_evidence"

# 可选且只在尚未locked时执行；锁不防源码编辑。
# git -C "$M26_MAIN" worktree lock --reason 'M26 frozen experiment source' "$M26_DEV"
```

工程目录在本次审查时尚不存在，以上是用户创建动作，不是已完成的资源gate。
后续新终端先重设A中的路径变量，并把 `M26_RELEASE` 指向刚才打印的实际目录，再读取
`M26_TRAIN_SHA="$(<"$M26_RELEASE/train_source_sha.txt")"`。不填历史M24/capability SHA。

## F1. 人工单/双进程GPU1 smoke

启动前重查。2026-09-19只读快照：GPU1是RTX4090 24564MiB，UUID
`GPU-8c1feab3-0fcf-396d-1a2e-f12af187f066`，当时18MiB且无计算进程；这不是未来占用保证。
主存当时约46GiB available，磁盘约3.3TiB可用；每个worker独立持有数据。

```bash
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
free -h
df -h "$M26_RUNS" "$M26_ENGINEERING"
```

GPU1出现未知计算进程就先协调，不kill、不共享、不改用GPU0、不缩batch。
工具默认仅打印计划；下面先核对CLI与单/双计划，不创建smoke目录：

```bash
cd "$M26_DEV"
M26_SMOKE_ARGS=(
  --study "$M26_STUDY" --gpu 1 --batch-size 1024 --steps 200 --warmup 20
  --eval-episodes 1 --dataset-root "$M26_DATA" --expected-source-sha "$M26_TRAIN_SHA"
)
env -u JAX_PLATFORMS -u CUDA_VISIBLE_DEVICES PYTHONPATH="$M26_DEV" PYTHONDONTWRITEBYTECODE=1 \
  "$M26_PY" tools/study_gpu_smoke.py --help
env -u JAX_PLATFORMS -u CUDA_VISIBLE_DEVICES PYTHONPATH="$M26_DEV" PYTHONDONTWRITEBYTECODE=1 \
  "$M26_PY" tools/study_gpu_smoke.py "${M26_SMOKE_ARGS[@]}" \
  --configs M26-C008 --workers 1 --output "$M26_RELEASE/gpu_single"
env -u JAX_PLATFORMS -u CUDA_VISIBLE_DEVICES PYTHONPATH="$M26_DEV" PYTHONDONTWRITEBYTECODE=1 \
  "$M26_PY" tools/study_gpu_smoke.py "${M26_SMOKE_ARGS[@]}" \
  --configs M26-C007,M26-C008 --workers 2 --output "$M26_RELEASE/gpu_dual"
```

单进程由用户显式执行，成功再执行双进程：

```bash
(
  set -euo pipefail
  cd "$M26_DEV"
  env -u JAX_PLATFORMS -u CUDA_VISIBLE_DEVICES PYTHONPATH="$M26_DEV" PYTHONDONTWRITEBYTECODE=1 \
    XLA_PYTHON_CLIENT_PREALLOCATE=false "$M26_PY" tools/study_gpu_smoke.py "${M26_SMOKE_ARGS[@]}" \
    --configs M26-C008 --workers 1 --output "$M26_RELEASE/gpu_single" --execute
  env -u JAX_PLATFORMS -u CUDA_VISIBLE_DEVICES PYTHONPATH="$M26_DEV" PYTHONDONTWRITEBYTECODE=1 \
    XLA_PYTHON_CLIENT_PREALLOCATE=false "$M26_PY" tools/study_gpu_smoke.py "${M26_SMOKE_ARGS[@]}" \
    --configs M26-C007,M26-C008 --workers 2 --output "$M26_RELEASE/gpu_dual" --execute
)
```

每个进程完整生产网络、batch1024、200updates，其中前20warmup；不平分batch。
私有子进程按物理GPU1的UUID绑定，强制JAX CUDA backend，PREALLOCATE=false。
真实train/val采样、一次save/fresh restore、生产 `_evaluate_tasks` 全5tasks各1episode。
报告包含source/config/data hash、PID、设备、loss、退出码、主存、分阶段NVML抽样峰值。
编译/训练/评估有双进程同步入口，监测未知占用，建议安全线90%；失败保留输出。
工具只可能结束本次创建的子进程，绝不结束外部进程。

覆盖边界：这是新工具的**待人工GPU验收**入口，当前只做CPU/mock设备控制流和CLI检查。
抽样峰值是下界，不保证捕获瞬时尖峰；每task1episode不等于正式20episode的全生命周期峰值。
报告fail、OOM、NaN、缺阶段telemetry、超90%或未知占用均不得放行，也不为过gate缩科学配置。
源码修复必须回开发树形成新提交、重新冻结/验证，不原地改已开始smoke的M26。
失败重试使用新工程output路径，保留旧路径；相应更新下方preflight所引用的报告位置。

## F2. 初始8组 / 全部10组 / 补2组：只选择一套

```bash
# 默认：8个新增诊断。
M26_CONFIGS=M26-C003,M26-C004,M26-C005,M26-C006,M26-C007,M26-C008,M26-C009,M26-C010
# 替代：用户选择一次做完整10组时，只替换上一行，不开第二个scheduler。
# M26_CONFIGS=M26-C001,M26-C002,M26-C003,M26-C004,M26-C005,M26-C006,M26-C007,M26-C008,M26-C009,M26-C010
# 后续仅补匹配参照；不能将M24结果标为完成。
# M26_CONFIGS=M26-C001,M26-C002
M26_ATTEMPT=0
M26_RUN_ARGS=(
  --study "$M26_STUDY" --configs "$M26_CONFIGS" --gpus 1 --jobs-per-gpu 2
  --run-root "$M26_RUNS" --run-attempt "$M26_ATTEMPT" --dataset-root "$M26_DATA"
  --train-steps 1000000 --batch-size 1024 --log-interval 5000 --eval-interval 100000
  --eval-tasks all --eval-episodes 20 --eval-temperature 0.0 --save-interval 100000
  --save-best-checkpoint --save-last-checkpoint
)
```

更换 `M26_CONFIGS/M26_ATTEMPT` 后重新构建数组。没有launcher `--seed`，由Study `[0]`提供。
不传 `--video-episodes 0`（wrapper拒绝0），底层default已核验为0；省略gaussian flag保持None。
`--run-root`是runs父目录，不再追加/M26。

冻结之后再次dry-run；CPU只用于非执行解析，formal块会显式切到CUDA：

```bash
(
  set -euo pipefail
  [[ "$M26_TRAIN_SHA" =~ ^[0-9a-f]{40}$ ]]
  [[ "$(git -C "$M26_DEV" rev-parse HEAD)" == "$M26_TRAIN_SHA" ]]
  [[ -z "$(git -C "$M26_DEV" branch --show-current)" ]]
  [[ -z "$(git -C "$M26_DEV" status --porcelain --untracked-files=all)" ]]
  cd "$M26_DEV"
  env -u CUDA_VISIBLE_DEVICES PYTHONPATH="$M26_DEV" PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu \
    RLC_PYTHON="$M26_PY" RLC_SOURCE_COMMIT="$M26_TRAIN_SHA" XLA_PYTHON_CLIENT_PREALLOCATE=false \
    bash scripts/run_study.sh "${M26_RUN_ARGS[@]}" --dry-run
)
```

初始应为planned8/completed0/remaining8；完整10或补2对应10/2。dry-run跳过Git/GPU/目录gate，
不是允许直接训练。下面用户执行真正的只读preflight并把证据写在仓库外；任一断言失败就停下。

```bash
(
  set -euo pipefail
  cd "$M26_DEV"
  env PYTHONPATH="$M26_DEV" PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu \
    "$M26_PY" - "$M26_STUDY" "$M26_RUNS" "$M26_CONFIGS" "$M26_ATTEMPT" \
    "$M26_TRAIN_SHA" "$M26_RELEASE" "$M26_DATA" <<'PY'
import datetime, json, sys
from pathlib import Path
from tools.sweep import _jobs
from tools.study_gpu_smoke import read_git, gpu_snapshot, host_snapshot
from tools.audit_goal_conditioning_study import dataset_metadata, resolved_configurations
from impls.experiment.management import config_fingerprint
study_path, run_root, ids, attempt, sha, release, data_root = sys.argv[1:]
release = Path(release)
assert sha == (release/'train_source_sha.txt').read_text().strip()
assert read_git('rev-parse','HEAD') == sha
assert not read_git('branch','--show-current')
assert not read_git('status','--porcelain','--untracked-files=all')
study, configs = resolved_configurations(study_path)
hashes = {c.config_id: config_fingerprint(cfg) for c,cfg in configs}
gpu = gpu_snapshot(1)
assert not gpu['processes'], gpu
assert gpu['used_mib'] / gpu['total_mib'] < 0.9
datasets = [dataset_metadata(Path(data_root)/f'puzzle-4x5-play-v0{s}.npz') for s in ('','-val')]
smokes = {}
for name, expected_ids in [('gpu_single',['M26-C008']),('gpu_dual',['M26-C007','M26-C008'])]:
    manifest = json.loads((release/name/'manifest.json').read_text())
    report = json.loads((release/name/'report.json').read_text())
    assert report['status'] == 'pass' and report['source_sha'] == sha
    assert manifest['source_sha'] == sha and manifest['gpu_snapshot']['uuid'] == gpu['uuid']
    assert manifest['batch_size_per_worker'] == 1024 and manifest['updates_per_worker'] == 200
    assert [c['config_id'] for c in manifest['configs']] == expected_ids
    assert all(hashes[c['config_id']] == c['config_sha256'] for c in manifest['configs'])
    assert [d['sha256'] for d in datasets] == [d['sha256'] for d in manifest['datasets']]
    smokes[name] = str(release/name/'report.json')
jobs = _jobs(study_path,run_root,include_configs=set(ids.split(',')),run_attempt=int(attempt))
assert len(jobs) == len(ids.split(','))
for job in jobs:
    assert job['seed'] == 0 and job['environment'] == 'puzzle-4x5-play-v0'
    assert job['status'] == 'planned'
    assert not job['run_dir'].exists() and not job['run_dir'].is_symlink(), str(job['run_dir'])
record = dict(source_sha=sha, source_worktree=str(Path.cwd()), clean_detached=True,
              study_sha256=config_fingerprint(study.data), protocol=study.data['protocol'],
              config_hashes={c:hashes[c] for c in ids.split(',')}, run_attempt=int(attempt),
              run_paths=[str(j['run_dir']) for j in jobs], datasets=datasets,
              gpu=gpu, host=host_snapshot(), smoke_reports=smokes)
stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
path = release/f'formal_preflight_{stamp}.json'
with path.open('x') as file:
    json.dump(record,file,indent=2,sort_keys=True)
print('Preflight passed at this instant:',path)
print('\n'.join(record['run_paths']))
PY
)
```

保留上面的preflight记录，不向待提交源码回写未来TRAIN_SHA。短时间内仍需重查GPU占用；
preflight不是预约GPU。首批路径必须不存在，补C01/C02只检查选中路径，不要求整个M26目录空。
任意已有失败/部分/orphan目录都要保留，不删除来让gate通过。

全部通过后，在自己的tmux终端执行一次；正式进程强制CUDA防止继承CPU测试设置：

```bash
(
  set -euo pipefail
  [[ "$(git -C "$M26_DEV" rev-parse HEAD)" == "$M26_TRAIN_SHA" ]]
  [[ -z "$(git -C "$M26_DEV" branch --show-current)" ]]
  [[ -z "$(git -C "$M26_DEV" status --porcelain --untracked-files=all)" ]]
  cd "$M26_DEV"
  env PYTHONPATH="$M26_DEV" JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 \
    "$M26_PY" -c 'from tools.study_gpu_smoke import gpu_snapshot; s=gpu_snapshot(1); print(s); assert not s["processes"]'
  env -u JAX_PLATFORMS -u CUDA_VISIBLE_DEVICES \
    PYTHONPATH="$M26_DEV" PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cuda \
    RLC_PYTHON="$M26_PY" RLC_SOURCE_COMMIT="$M26_TRAIN_SHA" XLA_PYTHON_CLIENT_PREALLOCATE=false \
    bash scripts/run_study.sh "${M26_RUN_ARGS[@]}" --execute
)
```

没有CPU fallback；launcher给child设置 `CUDA_VISIBLE_DEVICES=1` 和
`RLC_ASSIGNED_PHYSICAL_GPU=1`。child逻辑GPU0是正常映射，核验物理UUID及runtime_metadata，
不是看到device0就改卡。此调度器管理两slot，不能再同时运行另一套8/10组命令。

## 监控、checkpoint与失败重试

读取GPU和选中run状态（不启动）：

```bash
nvidia-smi -i 1
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
cd "$M26_DEV"
env PYTHONPATH="$M26_DEV" PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu \
  "$M26_PY" tools/sweep.py --study "$M26_STUDY" --configs "$M26_CONFIGS" \
  --run-root "$M26_RUNS" --run-attempt "$M26_ATTEMPT" --summary-only

# C003 attempt0的精确路径示例；其他路径由dry-run/preflight逐个打印。
M26_ONE_RUN="$M26_RUNS/M26/M26-C003__4x5_residual_distance/puzzle-4x5-play-v0/seed_000"
tail -n 5 "$M26_ONE_RUN/train.csv"
tail -n 5 "$M26_ONE_RUN/eval.csv"
```

周期checkpoint为 `checkpoints/params_<step>.pkl`；best/last的真实路径以
`checkpoints/index.json` 为准。last仅成功完成后写出，运行中缺last是正常的；结束后核验：

```bash
env PYTHONPATH="$M26_DEV" PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu \
  "$M26_PY" - "$M26_ONE_RUN" <<'PY'
import json,sys
from pathlib import Path
from impls.utils.checkpointing import resolve_checkpoint
root=Path(sys.argv[1])
metadata=json.loads((root/'runtime_metadata.json').read_text())
print('status/source/devices:', metadata.get('status'), metadata.get('git_commit'), metadata.get('jax_device_descriptions'))
assert metadata.get('jax_backend') == 'gpu'
for role in ('best','last'):
    record=resolve_checkpoint(root,role)
    print(role,record['checkpoint_step'],record['checkpoint_path'],record['checkpoint_sha256'])
    if role=='last':
        assert record['checkpoint_step']==1000000 and metadata['status']=='completed'
PY
```

失败保留attempt0全部metadata/log/partial checkpoints，先诊断。经人工决定重试时仅将
`M26_CONFIGS`改为失败ID，`M26_ATTEMPT`改为未用正整数（如1），重建数组、dry-run和preflight，
产生 `seed_000__attempt_001`。当前launcher不是checkpoint-resume；新attempt是独立从头训练，
不把历史checkpoint隐式恢复，也不为一个失败run重跑全部8/10组。

首批8组完成仍缺C01/C02匹配参照。训练期间只在main或新的研发feature工作树改代码，
不能对冻结的RLC-M26执行编辑/pull/merge/切分支；不能升级共享Python、驱动或修改数据。
