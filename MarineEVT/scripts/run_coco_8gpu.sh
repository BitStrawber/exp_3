#!/usr/bin/env bash
set -euo pipefail

if (( $# < 3 )); then
    echo "Usage: $0 INPUT OUTPUT CATEGORIES_JSON [extra generate_coco_dataset.py arguments...]" >&2
    exit 2
fi

INPUT="$1"
OUTPUT="$2"
CATEGORIES_FILE="$3"
shift 3

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_FILE="${MARINEEVT_ENV_FILE:-${HOME}/xcx/configs/marineevt.env}"
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Configuration not found: $CONFIG_FILE" >&2
    exit 1
fi
# shellcheck disable=SC1090
source "$CONFIG_FILE"

SAM3_NUM_GPUS="${SAM3_NUM_GPUS:-8}"
SAM3_BASE_PORT="${SAM3_BASE_PORT:-8111}"
SAM3_BIND_HOST="${SAM3_BIND_HOST:-127.0.0.1}"
urls=""
for ((gpu=0; gpu<SAM3_NUM_GPUS; gpu++)); do
    port=$((SAM3_BASE_PORT + gpu))
    endpoint="http://${SAM3_BIND_HOST}:${port}/v1/detect"
    urls="${urls:+${urls},}${endpoint}"
done

cd "$PROJECT_ROOT"
exec python scripts/generate_coco_dataset.py \
    --input "$INPUT" \
    --output "$OUTPUT" \
    --categories-file "$CATEGORIES_FILE" \
    --sam-urls "$urls" \
    --workers "$SAM3_NUM_GPUS" \
    "$@"
