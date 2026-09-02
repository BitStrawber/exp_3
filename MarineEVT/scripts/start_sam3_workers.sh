#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_FILE="${MARINEEVT_ENV_FILE:-${HOME}/xcx/configs/marineevt.env}"

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Configuration not found: $CONFIG_FILE" >&2
    echo "Copy deploy/marineevt.env.example there and edit its paths." >&2
    exit 1
fi

# shellcheck disable=SC1090
source "$CONFIG_FILE"
: "${SAM3_CHECKPOINT_PATH:?SAM3_CHECKPOINT_PATH is required}"
: "${SAM3_BPE_PATH:?SAM3_BPE_PATH is required}"
: "${SAM3_ALLOWED_DATA_ROOT:?SAM3_ALLOWED_DATA_ROOT is required}"

SAM3_NUM_GPUS="${SAM3_NUM_GPUS:-8}"
SAM3_BASE_PORT="${SAM3_BASE_PORT:-8111}"
SAM3_BIND_HOST="${SAM3_BIND_HOST:-127.0.0.1}"
LOG_ROOT="${LOG_ROOT:-${HOME}/xcx/logs}"
RUN_ROOT="${RUN_ROOT:-${HOME}/xcx/run}"

if [[ ! -f "$SAM3_CHECKPOINT_PATH" ]]; then
    echo "Checkpoint not found: $SAM3_CHECKPOINT_PATH" >&2
    exit 1
fi
if [[ ! -f "$SAM3_BPE_PATH" ]]; then
    echo "BPE vocabulary not found: $SAM3_BPE_PATH" >&2
    exit 1
fi

available_gpus="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
if (( available_gpus < SAM3_NUM_GPUS )); then
    echo "Requested $SAM3_NUM_GPUS GPUs, but nvidia-smi reports $available_gpus." >&2
    exit 1
fi

mkdir -p "$LOG_ROOT/sam3" "$RUN_ROOT/sam3" "$SAM3_ALLOWED_DATA_ROOT"
cd "$PROJECT_ROOT"

for ((gpu=0; gpu<SAM3_NUM_GPUS; gpu++)); do
    port=$((SAM3_BASE_PORT + gpu))
    pid_file="$RUN_ROOT/sam3/gpu_${gpu}.pid"
    if [[ -f "$pid_file" ]]; then
        old_pid="$(cat "$pid_file")"
        if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
            echo "GPU $gpu worker is already running as PID $old_pid" >&2
            continue
        fi
    fi

    nohup env \
        CUDA_VISIBLE_DEVICES="$gpu" \
        SAM3_PHYSICAL_GPU="$gpu" \
        SAM3_CHECKPOINT_PATH="$SAM3_CHECKPOINT_PATH" \
        SAM3_BPE_PATH="$SAM3_BPE_PATH" \
        SAM3_ALLOWED_DATA_ROOT="$SAM3_ALLOWED_DATA_ROOT" \
        SAM3_DEVICE="${SAM3_DEVICE:-cuda}" \
        SAM3_BASE_CONFIDENCE="${SAM3_BASE_CONFIDENCE:-0.05}" \
        python -m uvicorn sam_server:app \
            --host "$SAM3_BIND_HOST" \
            --port "$port" \
            --workers 1 \
        > "$LOG_ROOT/sam3/gpu_${gpu}.log" 2>&1 &

    worker_pid=$!
    echo "$worker_pid" > "$pid_file"
    echo "Started GPU $gpu -> $SAM3_BIND_HOST:$port (PID $worker_pid)"
done

echo "Model loading can take several minutes. Run scripts/check_sam3_workers.sh until all workers are ready."
