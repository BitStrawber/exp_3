#!/usr/bin/env bash
set -euo pipefail

MIRROR_URL="${SAM3_GITCODE_REPO:-https://gitcode.com/hf_mirrors/facebook/sam3.git}"
MODEL_DIR="${1:-${SAM3_MODEL_DIR:-${HOME}/xcx/models/sam3}}"
CHECKPOINT_PATH="${MODEL_DIR}/sam3.pt"
EXPECTED_SHA256="9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e"
EXPECTED_SIZE="3450062241"

for command_name in git sha256sum stat; do
    if ! command -v "$command_name" >/dev/null 2>&1; then
        echo "Required command not found: $command_name" >&2
        exit 1
    fi
done

if ! git lfs version >/dev/null 2>&1; then
    echo "Git LFS is required because sam3.pt is stored as an LFS object." >&2
    echo "Ubuntu/Debian: sudo apt-get update && sudo apt-get install -y git-lfs" >&2
    exit 1
fi

verify_checkpoint() {
    local actual_size actual_sha256

    [[ -f "$CHECKPOINT_PATH" ]] || return 1
    actual_size="$(stat -c '%s' "$CHECKPOINT_PATH")"
    [[ "$actual_size" == "$EXPECTED_SIZE" ]] || return 1

    actual_sha256="$(sha256sum "$CHECKPOINT_PATH" | awk '{print $1}')"
    [[ "$actual_sha256" == "$EXPECTED_SHA256" ]]
}

if verify_checkpoint; then
    echo "SAM3 checkpoint is already complete and verified: $CHECKPOINT_PATH"
    echo "SHA-256: $EXPECTED_SHA256"
    exit 0
fi

if [[ -e "$MODEL_DIR" && ! -d "$MODEL_DIR" ]]; then
    echo "Refusing to overwrite an existing non-directory path: $MODEL_DIR" >&2
    exit 1
fi

if [[ -d "$MODEL_DIR" && ! -d "$MODEL_DIR/.git" ]] && \
    find "$MODEL_DIR" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
    echo "Refusing to overwrite an existing non-empty directory: $MODEL_DIR" >&2
    echo "Move that directory aside or select another destination." >&2
    exit 1
fi

if [[ ! -d "$MODEL_DIR/.git" ]]; then
    mkdir -p "$(dirname "$MODEL_DIR")"
    echo "Cloning GitCode metadata without downloading every large model file..."
    GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 "$MIRROR_URL" "$MODEL_DIR"
else
    configured_remote="$(git -C "$MODEL_DIR" remote get-url origin 2>/dev/null || true)"
    if [[ "$configured_remote" != "$MIRROR_URL" ]]; then
        echo "Refusing to use a repository with an unexpected origin:" >&2
        echo "  directory: $MODEL_DIR" >&2
        echo "  expected:  $MIRROR_URL" >&2
        echo "  actual:    ${configured_remote:-<missing>}" >&2
        exit 1
    fi
fi

git -C "$MODEL_DIR" lfs install --local

# Download only the native checkpoint used by MarineEVT. The mirror also has a
# similarly sized model.safetensors file, which this deployment does not need.
echo "Downloading sam3.pt from $MIRROR_URL ..."
git -C "$MODEL_DIR" lfs pull --include="sam3.pt" --exclude=""

if [[ ! -f "$MODEL_DIR/config.json" ]]; then
    echo "Missing config.json after cloning the mirror." >&2
    exit 1
fi

if ! verify_checkpoint; then
    actual_size="$(stat -c '%s' "$CHECKPOINT_PATH" 2>/dev/null || echo missing)"
    actual_sha256="$(sha256sum "$CHECKPOINT_PATH" 2>/dev/null | awk '{print $1}' || true)"
    echo "SAM3 checkpoint verification failed." >&2
    echo "Expected size:   $EXPECTED_SIZE" >&2
    echo "Actual size:     $actual_size" >&2
    echo "Expected SHA256: $EXPECTED_SHA256" >&2
    echo "Actual SHA256:   ${actual_sha256:-unavailable}" >&2
    exit 1
fi

echo
echo "SAM3 checkpoint downloaded and verified."
echo "Path:      $CHECKPOINT_PATH"
echo "Size:      $EXPECTED_SIZE bytes"
echo "SHA-256:   $EXPECTED_SHA256"
echo
echo "Configure MarineEVT with:"
printf "export SAM3_CHECKPOINT_PATH=%q\n" "$CHECKPOINT_PATH"
