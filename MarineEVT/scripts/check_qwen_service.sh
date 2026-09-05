#!/usr/bin/env bash
set -euo pipefail

CONFIG_FILE="${MARINEEVT_ENV_FILE:-${HOME}/xcx/configs/marineevt.env}"
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Configuration not found: $CONFIG_FILE" >&2
    exit 1
fi
# shellcheck disable=SC1090
source "$CONFIG_FILE"

QWEN_BIND_HOST="${QWEN_BIND_HOST:-127.0.0.1}"
QWEN_PORT="${QWEN_PORT:-8100}"
if curl -fsS --max-time 5 "http://${QWEN_BIND_HOST}:${QWEN_PORT}/health"; then
    echo
    echo "Qwen3-VL port $QWEN_PORT READY"
else
    echo "Qwen3-VL port $QWEN_PORT NOT_READY" >&2
    exit 1
fi
