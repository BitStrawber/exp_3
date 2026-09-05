#!/usr/bin/env bash
set -euo pipefail

if (( $# < 2 )); then
    echo "Usage: $0 DEEPFISH_IMAGES OUTPUT_DIR [QWEN_MODEL] [extra pipeline arguments...]" >&2
    exit 2
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INPUT_DIR="$1"
OUTPUT_DIR="$2"
QWEN_MODEL="${3:-Qwen/Qwen3-VL-8B-Instruct}"
if (( $# >= 3 )); then shift 3; else shift 2; fi

cd "$PROJECT_ROOT"
python scripts/generate_evt_label_dataset.py \
    --input "$INPUT_DIR" \
    --output "$OUTPUT_DIR" \
    --categories-file scripts/evt_label_categories.deepfish.json \
    --planner qwen \
    --qwen-review \
    --require-vlm-accept \
    --qwen-model "$QWEN_MODEL" \
    --qwen-device cuda:0 \
    --sam-urls "http://127.0.0.1:8111/v1/detect,http://127.0.0.1:8112/v1/detect,http://127.0.0.1:8113/v1/detect,http://127.0.0.1:8114/v1/detect,http://127.0.0.1:8115/v1/detect,http://127.0.0.1:8116/v1/detect,http://127.0.0.1:8117/v1/detect" \
    --workers 7 \
    --include-masks \
    --exclude-review-frames \
    "$@"
