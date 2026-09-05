#!/usr/bin/env bash
set -euo pipefail

if (( $# < 2 )); then
    echo "Usage: $0 VIDEO_OR_INPUT_DIR OUTPUT_DIR [extra pipeline arguments...]" >&2
    exit 2
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_FILE="${MARINEEVT_ENV_FILE:-${HOME}/xcx/configs/marineevt.env}"
INPUT="$1"
OUTPUT="$2"
shift 2

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Configuration not found: $CONFIG_FILE" >&2
    exit 1
fi
# shellcheck disable=SC1090
source "$CONFIG_FILE"

QWEN_BIND_HOST="${QWEN_BIND_HOST:-127.0.0.1}"
QWEN_PORT="${QWEN_PORT:-8100}"
SAM3_BASE_PORT="${SAM3_BASE_PORT:-8111}"
SAM3_GPU_IDS="${SAM3_GPU_IDS:-1,2,3,4,5,6,7}"
IFS=',' read -r -a sam_gpu_ids <<< "$SAM3_GPU_IDS"

sam_urls=""
for slot in "${!sam_gpu_ids[@]}"; do
    port=$((SAM3_BASE_PORT + slot))
    endpoint="http://127.0.0.1:${port}/v1/detect"
    sam_urls="${sam_urls:+${sam_urls},}${endpoint}"
done

cd "$PROJECT_ROOT"
CUDA_VISIBLE_DEVICES="" python scripts/generate_evt_label_dataset.py \
    --input "$INPUT" \
    --output "$OUTPUT" \
    --categories-file scripts/evt_label_categories.sfishtrack.json \
    --planner qwen \
    --allow-planner-fallback \
    --qwen-review \
    --require-vlm-accept \
    --qwen-url "http://${QWEN_BIND_HOST}:${QWEN_PORT}/v1/generate-json" \
    --qwen-timeout 600 \
    --planner-coarse-frames 12 \
    --planner-max-segments 12 \
    --sample-every-seconds 1.0 \
    --sam-urls "$sam_urls" \
    --workers "${#sam_gpu_ids[@]}" \
    --include-masks \
    --exclude-review-frames \
    --splits 0.8,0.1,0.1 \
    --limit 0 \
    --max-frames-per-video 0 \
    --review-overlays 1000 \
    "$@"
