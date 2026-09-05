#!/usr/bin/env bash
set -euo pipefail

CONFIG_FILE="${MARINEEVT_ENV_FILE:-${HOME}/xcx/configs/marineevt.env}"
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Configuration not found: $CONFIG_FILE" >&2
    exit 1
fi
# shellcheck disable=SC1090
source "$CONFIG_FILE"

SAM3_NUM_GPUS="${SAM3_NUM_GPUS:-8}"
SAM3_GPU_IDS="${SAM3_GPU_IDS:-}"
SAM3_BASE_PORT="${SAM3_BASE_PORT:-8111}"
SAM3_BIND_HOST="${SAM3_BIND_HOST:-127.0.0.1}"
failed=0

if [[ -n "$SAM3_GPU_IDS" ]]; then
    IFS=',' read -r -a gpu_ids <<< "$SAM3_GPU_IDS"
else
    gpu_ids=()
    for ((gpu=0; gpu<SAM3_NUM_GPUS; gpu++)); do gpu_ids+=("$gpu"); done
fi

for slot in "${!gpu_ids[@]}"; do
    gpu="${gpu_ids[$slot]}"
    port=$((SAM3_BASE_PORT + slot))
    if response="$(curl --fail --silent --show-error --max-time 10 "http://${SAM3_BIND_HOST}:${port}/health")"; then
        echo "GPU $gpu port $port READY $response"
    else
        echo "GPU $gpu port $port NOT_READY"
        failed=$((failed + 1))
    fi
done

if (( failed > 0 )); then
    echo "$failed worker(s) are not ready. Check logs under ${LOG_ROOT:-${HOME}/xcx/logs}/sam3/." >&2
    exit 1
fi
