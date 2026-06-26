#!/bin/bash
# Task10 compile script for CUDA GEMM.

set -e

cd "$(dirname "$0")"
mkdir -p bin

if [ -n "${NVCC:-}" ]; then
    NVCC_BIN="$NVCC"
elif command -v nvcc > /dev/null 2>&1; then
    NVCC_BIN="$(command -v nvcc)"
elif [ -x /usr/local/cuda/bin/nvcc ]; then
    NVCC_BIN="/usr/local/cuda/bin/nvcc"
else
    NVCC_BIN=""
    for candidate in /usr/local/cuda-*/bin/nvcc; do
        if [ -x "$candidate" ]; then
            NVCC_BIN="$candidate"
            break
        fi
    done
fi

if [ -z "$NVCC_BIN" ]; then
    echo "Error: nvcc not found. Please load/install the CUDA Toolkit first."
    echo ""
    echo "Try one of these, depending on your environment:"
    echo "  module avail cuda && module load cuda"
    echo "  conda install -c nvidia cuda-nvcc cuda-cudart-dev cuda-libraries-dev"
    echo "  export PATH=/usr/local/cuda/bin:\$PATH"
    echo "  NVCC=/path/to/nvcc bash compile.sh"
    echo ""
    echo "You can also check:"
    echo "  nvidia-smi"
    echo "  conda list | grep cuda"
    echo "  ls /usr/local/cuda*/bin/nvcc"
    exit 1
fi

CUDA_ARCH=${CUDA_ARCH:-sm_86}
NVCC_FLAGS=(-O2 -std=c++14 -arch="$CUDA_ARCH" -Xcompiler -Wall)

echo "========================================"
echo "  Task10 - CUDA GEMM Build"
echo "========================================"
echo "CUDA_ARCH=$CUDA_ARCH"
echo "NVCC=$NVCC_BIN"
echo ""

echo "[1/1] Building cuda_gemm..."
"$NVCC_BIN" "${NVCC_FLAGS[@]}" -o bin/cuda_gemm src/cuda_gemm.cu
echo "    OK: bin/cuda_gemm"

echo ""
echo "Usage:"
echo "  ./bin/cuda_gemm [m] [n] [k] [block_x] [block_y] [naive|tiled|both] [repeat] [tile_k]"
echo ""
