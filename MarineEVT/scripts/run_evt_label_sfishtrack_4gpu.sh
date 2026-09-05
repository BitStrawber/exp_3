#!/usr/bin/env bash
set -euo pipefail

if (( $# < 2 )); then
    echo "Usage: $0 VIDEO_OR_INPUT_DIR OUTPUT_DIR [QWEN_MODEL] [extra pipeline arguments...]" >&2
    exit 2
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INPUT="$1"
OUTPUT="$2"
QWEN_MODEL="${3:-Qwen/Qwen3-VL-8B-Instruct}"
if (( $# >= 3 )); then shift 3; else shift 2; fi

cd "$PROJECT_ROOT"

# The server exposes physical GPU 4 to this process as logical cuda:0.
# SAM3 workers are separate processes on physical GPUs 5, 6 and 7.
CUDA_VISIBLE_DEVICES=4 python scripts/generate_evt_label_dataset.py \
    --input "$INPUT" \
    --output "$OUTPUT" \
    --categories-file scripts/evt_label_categories.sfishtrack.json \
    --planner qwen \
    --allow-planner-fallback \
    --qwen-review \
    --require-vlm-accept \
    --qwen-model "$QWEN_MODEL" \
    --qwen-device cuda:0 \
    --qwen-dtype float16 \
    --qwen-max-new-tokens 512 \
    --qwen-min-pixels 200704 \
    --qwen-max-pixels 401408 \
    --planner-coarse-frames 8 \
    --planner-max-segments 4 \
    --sam-urls "http://127.0.0.1:8111/v1/detect,http://127.0.0.1:8112/v1/detect,http://127.0.0.1:8113/v1/detect" \
    --workers 3 \
    --include-masks \
    --exclude-review-frames \
    --splits 1,0,0 \
    --limit 40 \
    --review-overlays 40 \
    "$@"
