#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DESTINATION="${1:-${HOME}/xcx/data/sfishtrack_smoke}"
VIDEO_NAME="${2:-}"
DOWNLOAD_ROOT="${SFISHTRACK_DOWNLOAD_ROOT:-${HOME}/xcx/downloads/sfishtrack}"
ARCHIVE="${DOWNLOAD_ROOT}/SFISHTRACK.zip"
FILE_ID="1yYWr5aEAJ2lMfLAqWHOHy5HkG7AIh-_"
ZENODO_URL="https://zenodo.org/api/records/20617668/files/SFISHTRACK.zip/content"
EXPECTED_SIZE="27487705333"
EXPECTED_MD5="9ec025a052104729b9d83df18db9e58b"

verify_archive() {
    local candidate="$1"
    [[ -f "$candidate" ]] || return 1
    local actual_size
    actual_size="$(stat -c '%s' "$candidate")"
    if [[ "$actual_size" != "$EXPECTED_SIZE" ]]; then
        echo "Archive is incomplete: $actual_size / $EXPECTED_SIZE bytes" >&2
        return 1
    fi
    echo "$EXPECTED_MD5  $candidate" | md5sum --check --status
}

quarantine_non_resumable_archive() {
    local candidate="$1"
    [[ -f "$candidate" ]] || return 0
    local actual_size magic quarantine
    actual_size="$(stat -c '%s' "$candidate")"
    magic="$(head -c 4 "$candidate" | od -An -tx1 | tr -d '[:space:]')"
    if [[ "$actual_size" -ge "$EXPECTED_SIZE" || "$magic" != "504b0304" ]]; then
        quarantine="${candidate}.invalid.$(date +%Y%m%d%H%M%S)"
        mv -- "$candidate" "$quarantine"
        echo "Moved a non-resumable or invalid previous download to: $quarantine" >&2
    else
        echo "Resuming existing ZIP prefix: $actual_size / $EXPECTED_SIZE bytes"
    fi
}

mkdir -p "$DOWNLOAD_ROOT" "$(dirname "$DESTINATION")"

if verify_archive "$ARCHIVE"; then
    echo "Using verified archive: $ARCHIVE"
else
    quarantine_non_resumable_archive "$ARCHIVE"
    echo "Downloading the official SFISHTRACK archive (25.6 GiB) from Zenodo with resume support..."
    if ! command -v curl >/dev/null 2>&1; then
        echo "curl is required for the primary Zenodo download." >&2
        exit 1
    fi
    if ! curl --location --fail --show-error \
        --retry 5 --retry-delay 5 --retry-all-errors \
        --continue-at - --output "$ARCHIVE" "$ZENODO_URL"; then
        echo "Zenodo download failed; trying the official Google Drive mirror with the active Python..." >&2
        if ! python -c 'import gdown' >/dev/null 2>&1; then
            echo "gdown is missing from $(command -v python). Run: python -m pip install 'gdown>=5.2,<6'" >&2
            exit 1
        fi
        python -m gdown --continue "$FILE_ID" --output "$ARCHIVE"
    fi
    if ! verify_archive "$ARCHIVE"; then
        echo "Downloaded archive failed size or MD5 verification." >&2
        echo "Expected: $EXPECTED_SIZE bytes, MD5 $EXPECTED_MD5" >&2
        echo "Keep the partial file for resume; rerun the same command." >&2
        exit 1
    fi
fi

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
