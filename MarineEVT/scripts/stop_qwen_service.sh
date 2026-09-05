#!/usr/bin/env bash
set -euo pipefail

CONFIG_FILE="${MARINEEVT_ENV_FILE:-${HOME}/xcx/configs/marineevt.env}"
if [[ -f "$CONFIG_FILE" ]]; then
    # shellcheck disable=SC1090
    source "$CONFIG_FILE"
fi
RUN_ROOT="${RUN_ROOT:-${HOME}/xcx/run}"
PID_FILE="$RUN_ROOT/qwen/qwen.pid"

if [[ ! -f "$PID_FILE" ]]; then
    echo "No Qwen PID file found: $PID_FILE"
    exit 0
fi

pid="$(cat "$PID_FILE")"
if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
    kill "$pid"
    for _ in {1..30}; do
        if ! kill -0 "$pid" 2>/dev/null; then
            break
        fi
        sleep 1
    done
    if kill -0 "$pid" 2>/dev/null; then
        echo "Qwen service did not stop within 30 seconds; PID $pid is still running." >&2
        exit 1
    fi
    echo "Stopped Qwen service PID $pid"
else
    echo "Qwen service PID is not running: $pid"
fi
rm -f "$PID_FILE"
