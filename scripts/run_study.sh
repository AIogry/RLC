#!/usr/bin/env bash
# Generic formal Study launcher for RLC.
# Scientific factors stay in experiments/<study>/; this script only validates
# the frozen environment and forwards a common protocol to tools/sweep.py.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${RLC_PYTHON:-python}"

STUDY=""
CONFIGS=""
EXCLUDE_CONFIGS=""
GPUS=""
RUN_ROOT=""
DATASET_ROOT="${OGBENCH_DATASET_DIR:-}"
TRAIN_STEPS=""
BATCH_SIZE=""
LOG_INTERVAL=""
EVAL_INTERVAL=""
EVAL_TASKS=""
EVAL_EPISODES=""
SAVE_INTERVAL=""
EVAL_TEMPERATURE=""
EVAL_GAUSSIAN=""
VIDEO_EPISODES=""
MODE=""

usage() {
    cat <<'EOF'
Usage:
  bash scripts/run_study.sh --study experiments/<study>/study.yaml \
    [--configs ID1,ID2,... | --exclude-configs ID1,ID2,...] \
    --gpus 0,1 --run-root /data/.../RLC/runs \
    --dataset-root /data/.../ogbench \
    --train-steps N --batch-size N --log-interval N --eval-interval N \
    --eval-tasks N|all --eval-episodes N --save-interval N \
    --eval-temperature FLOAT [--eval-gaussian FLOAT] [--video-episodes N] \
    (--dry-run | --execute)

The formal launcher refuses a dirty Git worktree, validates all common
protocol fields explicitly, and never prompts interactively.
EOF
}

die() {
    echo "run_study.sh: $*" >&2
    exit 2
}

need_value() {
    [[ $# -ge 2 && -n "${2:-}" ]] || die "missing value for $1"
}

positive_int() {
    [[ "${1:-}" =~ ^[1-9][0-9]*$ ]]
}

while (($# > 0)); do
    case "$1" in
        --study) need_value "$@"; STUDY="$2"; shift 2 ;;
        --configs) need_value "$@"; CONFIGS="$2"; shift 2 ;;
        --exclude-configs) need_value "$@"; EXCLUDE_CONFIGS="$2"; shift 2 ;;
        --gpus) need_value "$@"; GPUS="$2"; shift 2 ;;
        --run-root) need_value "$@"; RUN_ROOT="$2"; shift 2 ;;
        --dataset-root) need_value "$@"; DATASET_ROOT="$2"; shift 2 ;;
        --train-steps) need_value "$@"; TRAIN_STEPS="$2"; shift 2 ;;
        --batch-size) need_value "$@"; BATCH_SIZE="$2"; shift 2 ;;
        --log-interval) need_value "$@"; LOG_INTERVAL="$2"; shift 2 ;;
        --eval-interval) need_value "$@"; EVAL_INTERVAL="$2"; shift 2 ;;
        --eval-tasks) need_value "$@"; EVAL_TASKS="$2"; shift 2 ;;
        --eval-episodes) need_value "$@"; EVAL_EPISODES="$2"; shift 2 ;;
        --save-interval) need_value "$@"; SAVE_INTERVAL="$2"; shift 2 ;;
        --eval-temperature) need_value "$@"; EVAL_TEMPERATURE="$2"; shift 2 ;;
        --eval-gaussian) need_value "$@"; EVAL_GAUSSIAN="$2"; shift 2 ;;
        --video-episodes) need_value "$@"; VIDEO_EPISODES="$2"; shift 2 ;;
        --dry-run|--execute)
            [[ -z "$MODE" ]] || die '--dry-run and --execute are mutually exclusive'
            MODE="$1"
            shift
            ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown option $1; scientific factors must be declared in the Study" ;;
    esac
done

[[ -n "$STUDY" ]] || die '--study is required'
[[ -z "$CONFIGS" || -z "$EXCLUDE_CONFIGS" ]] || die '--configs and --exclude-configs are mutually exclusive'
[[ -n "$GPUS" ]] || die '--gpus must be non-empty'
[[ -n "$RUN_ROOT" ]] || die '--run-root is required for formal execution'
[[ -n "$DATASET_ROOT" ]] || die '--dataset-root is required (or set OGBENCH_DATASET_DIR)'
positive_int "$TRAIN_STEPS" || die '--train-steps must be a positive integer'
positive_int "$BATCH_SIZE" || die '--batch-size must be a positive integer'
positive_int "$LOG_INTERVAL" || die '--log-interval must be a positive integer'
positive_int "$EVAL_INTERVAL" || die '--eval-interval must be a positive integer'
positive_int "$EVAL_EPISODES" || die '--eval-episodes must be a positive integer'
positive_int "$SAVE_INTERVAL" || die '--save-interval must be a positive integer'
if [[ "$EVAL_TASKS" != all && "$EVAL_TASKS" != none ]]; then
    positive_int "$EVAL_TASKS" || die '--eval-tasks must be a positive integer, all, or none'
fi
[[ -n "$EVAL_TEMPERATURE" ]] || die '--eval-temperature is required'
[[ -n "$MODE" ]] || die 'exactly one of --dry-run or --execute is required'

cd "$REPO_ROOT"
if [[ "$MODE" == --execute ]]; then
    CALLER_TOPLEVEL="$(git rev-parse --show-toplevel 2>/dev/null || true)"
    [[ "$CALLER_TOPLEVEL" == "$REPO_ROOT" ]] || die 'current directory is not the RLC Git worktree'
    GIT_TOPLEVEL="$(git rev-parse --show-toplevel 2>/dev/null)" || die 'not inside a Git worktree'
    [[ "$GIT_TOPLEVEL" == "$REPO_ROOT" ]] || die "launcher root is not the Git worktree: $GIT_TOPLEVEL"
    GIT_STATUS="$(git status --porcelain --untracked-files=all)"
    if [[ -n "$GIT_STATUS" ]]; then
        echo 'Formal execution refused: Git worktree is dirty.' >&2
        echo "$GIT_STATUS" >&2
        exit 2
    fi
else
    echo "Git preflight: skipped for --dry-run (set RLC_SOURCE_COMMIT after manual review)"
fi

STUDY_PATH="$STUDY"
case "$STUDY_PATH" in
    /*) ;;
    *) STUDY_PATH="$REPO_ROOT/$STUDY_PATH" ;;
esac
[[ -f "$STUDY_PATH" ]] || die "Study file is not accessible: $STUDY_PATH"

case "$RUN_ROOT" in
    /*) ;;
    *) RUN_ROOT="$REPO_ROOT/$RUN_ROOT" ;;
esac
if [[ "$MODE" == --execute ]]; then
    mkdir -p "$RUN_ROOT" || die "cannot create run root: $RUN_ROOT"
    [[ -d "$RUN_ROOT" && -w "$RUN_ROOT" ]] || die "run root is not writable: $RUN_ROOT"
else
    echo "Run-root preflight: skipped for --dry-run (no run directory will be created)"
fi
[[ -d "$DATASET_ROOT" && -r "$DATASET_ROOT" ]] || die "dataset root is not readable: $DATASET_ROOT"

GPU_CSV="$(printf '%s' "$GPUS" | tr -d '[:space:]')"
[[ -n "$GPU_CSV" ]] || die '--gpus must contain at least one GPU ID'
if [[ "$MODE" == --execute ]]; then
    command -v nvidia-smi >/dev/null 2>&1 || die 'nvidia-smi is required for formal GPU execution'
    VISIBLE_GPUS="$(nvidia-smi --query-gpu=index --format=csv,noheader | awk 'NF { if (seen++) printf ","; printf "%s", $1 }')" \
        || die 'unable to query visible GPUs with nvidia-smi'
    IFS=',' read -r -a GPU_IDS <<< "$GPU_CSV"
    for gpu in "${GPU_IDS[@]}"; do
        [[ "$gpu" =~ ^[0-9]+$ ]] || die "invalid physical GPU ID: $gpu"
        case ",$VISIBLE_GPUS," in
            *,"$gpu",*) ;;
            *) die "requested GPU $gpu is not visible; visible GPUs: $VISIBLE_GPUS" ;;
        esac
    done
else
    echo "GPU preflight: skipped for --dry-run (requested GPUs: $GPU_CSV)"
fi

export OGBENCH_DATASET_DIR="$DATASET_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
# This allocation policy is uniform across all workers and is not a numerical
# XLA flag or a model/scientific definition.
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

SUMMARY_ARGS=(
    --study "$STUDY_PATH"
    --gpus "$GPU_CSV"
    --run-root "$RUN_ROOT"
    --dataset-root "$DATASET_ROOT"
    --summary-only
)
if [[ "$MODE" == --dry-run && "$STUDY" == *M11B* ]]; then
    SUMMARY_ARGS+=(--allow-missing-dataset)
fi
if [[ -n "$CONFIGS" ]]; then
    SUMMARY_ARGS+=(--configs "$CONFIGS")
fi
if [[ -n "$EXCLUDE_CONFIGS" ]]; then
    SUMMARY_ARGS+=(--exclude-configs "$EXCLUDE_CONFIGS")
fi
SUMMARY="$("$PYTHON_BIN" "$REPO_ROOT/tools/sweep.py" "${SUMMARY_ARGS[@]}")" \
    || die 'Study validation/dry-run failed'

field() {
    printf '%s\n' "$SUMMARY" | sed -n "s/.* $1=\([0-9][0-9]*\).*/\1/p"
}

echo 'Formal RLC Study execution'
if [[ "$MODE" == --execute ]]; then
    echo "Git commit: $(git rev-parse HEAD)"
else
    echo "Git commit: ${RLC_SOURCE_COMMIT:-<manual-user-supplied>}"
fi
echo "Study: $STUDY_PATH"
echo "Run root: $RUN_ROOT"
echo "Dataset root: $DATASET_ROOT"
echo "GPUs: $GPU_CSV"
echo "Config filter: ${CONFIGS:-${EXCLUDE_CONFIGS:+exclude:$EXCLUDE_CONFIGS}}"
echo "Planned runs: $(field planned)"
echo "Completed runs: $(field completed)"
echo "Remaining runs: $(field remaining)"
echo "XLA_PYTHON_CLIENT_PREALLOCATE: $XLA_PYTHON_CLIENT_PREALLOCATE"
echo "Protocol: train_steps=$TRAIN_STEPS batch_size=$BATCH_SIZE log_interval=$LOG_INTERVAL eval_interval=$EVAL_INTERVAL eval_tasks=$EVAL_TASKS eval_episodes=$EVAL_EPISODES save_interval=$SAVE_INTERVAL eval_temperature=$EVAL_TEMPERATURE eval_gaussian=${EVAL_GAUSSIAN:-None}"
echo "Mode: $MODE"
echo 'No interactive confirmation is required.'

SWEEP_ARGS=(
    --study "$STUDY_PATH"
    --gpus "$GPU_CSV"
    --run-root "$RUN_ROOT"
    --dataset-root "$DATASET_ROOT"
    "--train_steps=$TRAIN_STEPS"
    "--batch_size=$BATCH_SIZE"
    "--log_interval=$LOG_INTERVAL"
    "--eval_interval=$EVAL_INTERVAL"
    "--eval_tasks=$EVAL_TASKS"
    "--eval_episodes=$EVAL_EPISODES"
    "--save_interval=$SAVE_INTERVAL"
    "--eval_temperature=$EVAL_TEMPERATURE"
)
if [[ -n "$CONFIGS" ]]; then
    SWEEP_ARGS+=(--configs "$CONFIGS")
fi
if [[ -n "$EXCLUDE_CONFIGS" ]]; then
    SWEEP_ARGS+=(--exclude-configs "$EXCLUDE_CONFIGS")
fi
if [[ -n "$EVAL_GAUSSIAN" ]]; then
    SWEEP_ARGS+=("--eval_gaussian=$EVAL_GAUSSIAN")
fi
if [[ -n "$VIDEO_EPISODES" ]]; then
    positive_int "$VIDEO_EPISODES" || die '--video-episodes must be a positive integer'
    SWEEP_ARGS+=("--video_episodes=$VIDEO_EPISODES")
fi
if [[ "$MODE" == --dry-run && "$STUDY" == *M11B* ]]; then
    SWEEP_ARGS+=(--allow-missing-dataset)
fi
if [[ "$MODE" == --dry-run ]]; then
    SWEEP_ARGS+=(--dry-run)
else
    SWEEP_ARGS+=(--execute)
fi

exec "$PYTHON_BIN" "$REPO_ROOT/tools/sweep.py" "${SWEEP_ARGS[@]}"
