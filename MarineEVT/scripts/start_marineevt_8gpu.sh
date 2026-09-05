#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

bash scripts/start_qwen_service.sh
bash scripts/start_sam3_workers.sh

echo
echo "Services are starting. Check readiness with:"
echo "  bash scripts/check_qwen_service.sh"
echo "  bash scripts/check_sam3_workers.sh"
