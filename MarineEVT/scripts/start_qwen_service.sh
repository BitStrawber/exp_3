#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_FILE="${MARINEEVT_ENV_FILE:-${HOME}/xcx/configs/marineevt.env}"

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Configuration not found: $CONFIG_FILE" >&2
    exit 1
fi

# shellcheck disable=SC1090
source "$CONFIG_FILE"
: "${QWEN_MODEL_PATH:?QWEN_MODEL_PATH is required}"
: "${QWEN_ALLOWED_DATA_ROOT:?QWEN_ALLOWED_DATA_ROOT is required}"

QWEN_GPU_ID="${QWEN_GPU_ID:-0}"
QWEN_BIND_HOST="${QWEN_BIND_HOST:-127.0.0.1}"
QWEN_PORT="${QWEN_PORT:-8100}"
LOG_ROOT="${LOG_ROOT:-${HOME}/xcx/logs}"
RUN_ROOT="${RUN_ROOT:-${HOME}/xcx/run}"
PID_FILE="$RUN_ROOT/qwen/qwen.pid"

if [[ ! -d "$QWEN_MODEL_PATH" ]] || [[ ! -f "$QWEN_MODEL_PATH/config.json" ]]; then
    echo "Qwen model directory is incomplete: $QWEN_MODEL_PATH" >&2
    exit 1
fi
available_gpus="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
if [[ ! "$QWEN_GPU_ID" =~ ^[0-9]+$ ]] || (( QWEN_GPU_ID >= available_gpus )); then
    echo "Invalid QWEN_GPU_ID '$QWEN_GPU_ID'; nvidia-smi reports $available_gpus GPU(s)." >&2
    exit 1
fi

mkdir -p "$LOG_ROOT/qwen" "$RUN_ROOT/qwen" "$QWEN_ALLOWED_DATA_ROOT"
if [[ -f "$PID_FILE" ]]; then
    old_pid="$(cat "$PID_FILE")"
    if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
        echo "Qwen service is already running as PID $old_pid"
        exit 0
    fi
fi

cd "$PROJECT_ROOT"
nohup env \
    CUDA_VISIBLE_DEVICES="$QWEN_GPU_ID" \
    QWEN_PHYSICAL_GPU="$QWEN_GPU_ID" \
    QWEN_MODEL_PATH="$QWEN_MODEL_PATH" \
    QWEN_ALLOWED_DATA_ROOT="$QWEN_ALLOWED_DATA_ROOT" \
    QWEN_DEVICE="${QWEN_DEVICE:-cuda:0}" \
    QWEN_DTYPE="${QWEN_DTYPE:-float16}" \
    QWEN_MIN_PIXELS="${QWEN_MIN_PIXELS:-100352}" \
    QWEN_MAX_PIXELS="${QWEN_MAX_PIXELS:-200704}" \
    QWEN_MAX_NEW_TOKENS="${QWEN_MAX_NEW_TOKENS:-512}" \
    HF_HOME="${HF_HOME:-${HOME}/xcx/models/huggingface}" \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    python -m uvicorn qwen_server:app \
        --host "$QWEN_BIND_HOST" \
        --port "$QWEN_PORT" \
        --workers 1 \
    > "$LOG_ROOT/qwen/qwen.log" 2>&1 &

service_pid=$!
echo "$service_pid" > "$PID_FILE"
echo "Started Qwen3-VL on physical GPU $QWEN_GPU_ID -> $QWEN_BIND_HOST:$QWEN_PORT (PID $service_pid)"
echo "Model loading can take several minutes. Run scripts/check_qwen_service.sh until it is ready."
