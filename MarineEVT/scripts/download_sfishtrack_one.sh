#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DESTINATION="${1:-${HOME}/xcx/data/sfishtrack_smoke}"
VIDEO_NAME="${2:-}"
DOWNLOAD_ROOT="${SFISHTRACK_DOWNLOAD_ROOT:-${HOME}/xcx/downloads/sfishtrack}"
ARCHIVE="${DOWNLOAD_ROOT}/SFISHTRACK.zip"
FILE_ID="1yYWr5aEAJ2lMfLAqWHOHy5HkG7AIh-_"

if ! command -v gdown >/dev/null 2>&1; then
    echo "gdown is missing. Activate marineevt-label and run: python -m pip install 'gdown>=5.2,<6'" >&2
    exit 1
fi

mkdir -p "$DOWNLOAD_ROOT" "$(dirname "$DESTINATION")"
echo "Downloading the official SFISHTRACK archive (about 20 GB) with resume support..."
gdown --continue --id "$FILE_ID" --output "$ARCHIVE"

echo "Extracting only one video and its matching annotation..."
args=(--source "$ARCHIVE" --output "$DESTINATION" --force)
if [[ -n "$VIDEO_NAME" ]]; then
    args+=(--video "$VIDEO_NAME")
fi
python "$PROJECT_ROOT/scripts/prepare_sfishtrack_one.py" "${args[@]}"

echo
echo "Subset ready: $DESTINATION"
echo "The full archive is retained at: $ARCHIVE"
echo "After confirming the subset, reclaim space with: rm -- '$ARCHIVE'"
