#!/usr/bin/env bash
set -euo pipefail

CONFIG_FILE="${MARINEEVT_ENV_FILE:-${HOME}/xcx/configs/marineevt.env}"
if [[ -f "$CONFIG_FILE" ]]; then
    # shellcheck disable=SC1090
    source "$CONFIG_FILE"
fi
RUN_ROOT="${RUN_ROOT:-${HOME}/xcx/run}"
PID_DIR="$RUN_ROOT/sam3"

if [[ ! -d "$PID_DIR" ]]; then
    echo "No SAM3 PID directory found: $PID_DIR"
    exit 0
fi

for pid_file in "$PID_DIR"/gpu_*.pid; do
    [[ -e "$pid_file" ]] || continue
    pid="$(cat "$pid_file")"
    if [[ ! "$pid" =~ ^[0-9]+$ ]]; then
        echo "Ignoring invalid PID file: $pid_file" >&2
        continue
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
        echo "Worker already stopped: PID $pid"
        continue
    fi
    command_line="$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)"
    if [[ "$command_line" != *"uvicorn"* || "$command_line" != *"sam_server:app"* ]]; then
        echo "Refusing to stop PID $pid because it is not a MarineEVT SAM3 worker: $command_line" >&2
        continue
    fi
    kill "$pid"
    echo "Stopped PID $pid"
done
