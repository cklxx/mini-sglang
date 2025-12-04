#!/usr/bin/env bash

# Build/install sgl_kernel for the local CUDA architecture (e.g. SM86 on A40).

set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

SGL_KERNEL_VERSION="${SGL_KERNEL_VERSION:-0.3.18.post2}"
ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-}"
LOCAL_SRC="${SGL_KERNEL_SRC:-${ROOT_DIR}/third_party/sgl-kernel}"

if [[ -z "${ARCH_LIST}" ]]; then
    ARCH_LIST="$(python - <<'PY' || true
import torch
if not torch.cuda.is_available():
    raise SystemExit(0)
caps = {f"{major}.{minor}" for major, minor in (
    torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())
)}
print(";".join(sorted(caps)))
PY
)"
fi

if [[ -z "${ARCH_LIST}" ]]; then
    cat <<'EOF' >&2
No CUDA arch detected. Set TORCH_CUDA_ARCH_LIST manually, e.g.:
  A40 (SM86): 8.6
  A100 (SM80): 8.0
  H100 (SM90): 9.0
  B100 (SM100): 10.0
Then re-run this script.
EOF
    exit 1
fi

echo "Installing sgl_kernel==${SGL_KERNEL_VERSION} for TORCH_CUDA_ARCH_LIST=${ARCH_LIST}"
echo "Removing any existing sgl_kernel wheel to force a local rebuild"
python -m pip uninstall -y sgl_kernel >/dev/null 2>&1 || true
python -m pip install --upgrade pip setuptools wheel ninja packaging
export TORCH_CUDA_ARCH_LIST="${ARCH_LIST}"
export MAX_JOBS="${MAX_JOBS:-"$(command -v nproc >/dev/null 2>&1 && nproc || sysctl -n hw.ncpu)"}"
if [[ -d "${LOCAL_SRC}" ]]; then
    echo "Found local sgl_kernel source at ${LOCAL_SRC}; building from source"
    python -m pip install --no-build-isolation "${LOCAL_SRC}"
else
    echo "No local sgl_kernel source; falling back to PyPI build"
    python -m pip install --no-binary=sgl-kernel "sgl_kernel==${SGL_KERNEL_VERSION}"
fi
