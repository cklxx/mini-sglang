#!/usr/bin/env bash
# Helper to run Z-Image with a local GGUF transformer. Downloads the base pipeline
# if missing, then launches z_image_mps.py with sensible defaults for MPS.

set -euo pipefail

GGUF_PATH=${1:-model/gguf/z_image_turbo-Q8_0.gguf}
PROMPT=${PROMPT:-"Cyberpunk night market, neon haze"}
MODEL_ID=${MODEL_ID:-"Tongyi-MAI/Z-Image-Turbo"}
MODEL_DIR=${Z_IMAGE_MODEL_DIR:-"$PWD/model/z-image-base"}
HF_HOME=${HF_HOME:-"$HOME/.cache/hf-zimage"}

STEPS=${STEPS:-8}
HEIGHT=${HEIGHT:-768}
WIDTH=${WIDTH:-768}
DEVICE=${DEVICE:-mps}
OUTPUT=${OUTPUT:-out.png}

export HF_HOME
export Z_IMAGE_MODEL_DIR="$MODEL_DIR"

if [[ ! -f "$GGUF_PATH" ]]; then
  echo "GGUF file not found: $GGUF_PATH" >&2
  echo "Usage: $0 /path/to/z_image_turbo-Q8_0.gguf" >&2
  exit 1
fi

python3 - "$MODEL_ID" "$MODEL_DIR" <<'PY'
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

model_id = sys.argv[1]
model_dir = Path(sys.argv[2])
model_dir.mkdir(parents=True, exist_ok=True)

if any(model_dir.iterdir()):
    print(f"Base pipeline already present at {model_dir}")
else:
    print(f"Downloading base pipeline {model_id} into {model_dir} ...")
    snapshot_download(
        repo_id=model_id,
        local_dir=str(model_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    print("Base pipeline ready.")
PY

PYTORCH_MPS_HIGH_WATERMARK_RATIO=${PYTORCH_MPS_HIGH_WATERMARK_RATIO:-0.0} \
python3 z_image_mps.py \
  --gguf "$GGUF_PATH" \
  --model "$MODEL_ID" \
  --model-dir "$MODEL_DIR" \
  --device "$DEVICE" \
  --steps "$STEPS" \
  --height "$HEIGHT" \
  --width "$WIDTH" \
  -p "$PROMPT" \
  --output "$OUTPUT"
