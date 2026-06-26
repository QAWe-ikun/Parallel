#!/bin/bash
# Task11 compile script for CUDA convolution.

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
    echo "Try: module load cuda, export PATH=/usr/local/cuda/bin:\$PATH,"
    echo "or NVCC=/path/to/nvcc bash compile.sh"
    exit 1
fi

CUDA_ARCH=${CUDA_ARCH:-sm_86}
NVCC_FLAGS=(-O2 -std=c++14 -arch="$CUDA_ARCH" -Xcompiler -Wall)

SCRIPT_CUDA_ROOT="$(cd "$(dirname "$NVCC_BIN")/.." && pwd)"

find_first_file_dir() {
    local filename="$1"
    shift
    for dir in "$@"; do
        if [ -n "$dir" ] && [ -f "$dir/$filename" ]; then
            echo "$dir"
            return 0
        fi
    done
    return 1
}

find_first_lib_dir() {
    local _filename="$1"
    shift
    for dir in "$@"; do
        if [ -n "$dir" ] && { [ -f "$dir/libcudnn.so" ] || [ -f "$dir/libcudnn_static.a" ]; }; then
            echo "$dir"
            return 0
        fi
    done
    return 1
}

PY_CUDNN_ROOTS=()
if [ -n "${CONDA_PREFIX:-}" ]; then
    for dir in "$CONDA_PREFIX"/lib/python*/site-packages/nvidia/cudnn; do
        if [ -d "$dir" ]; then
            PY_CUDNN_ROOTS+=("$dir")
        fi
    done
fi

CUDNN_INCLUDE_DIR_FOUND="$(find_first_file_dir cudnn.h \
    "${CUDNN_INCLUDE_DIR:-}" \
    "${CONDA_PREFIX:-}/include" \
    "${CONDA_PREFIX:-}/targets/x86_64-linux/include" \
    "${PY_CUDNN_ROOTS[@]/%//include}" \
    "$SCRIPT_CUDA_ROOT/include" \
    "$SCRIPT_CUDA_ROOT/targets/x86_64-linux/include" \
    /usr/local/cuda/include \
    /usr/include \
    /usr/local/include \
    /usr/local/cuda-*/include || true)"

CUDNN_LIB_DIR_FOUND="$(find_first_lib_dir libcudnn.so \
    "${CUDNN_LIB_DIR:-}" \
    "${CONDA_PREFIX:-}/lib" \
    "${CONDA_PREFIX:-}/targets/x86_64-linux/lib" \
    "${PY_CUDNN_ROOTS[@]/%//lib}" \
    "$SCRIPT_CUDA_ROOT/lib" \
    "$SCRIPT_CUDA_ROOT/lib64" \
    "$SCRIPT_CUDA_ROOT/targets/x86_64-linux/lib" \
    /usr/local/cuda/lib64 \
    /usr/lib/x86_64-linux-gnu \
    /usr/local/cuda-*/lib64 || true)"

if [ -z "$CUDNN_INCLUDE_DIR_FOUND" ]; then
    echo "Error: cudnn.h not found. cuDNN is required for task11."
    echo ""
    echo "Checked common locations including:"
    echo "  CONDA_PREFIX/include: ${CONDA_PREFIX:-<unset>}/include"
    echo "  CONDA targets include: ${CONDA_PREFIX:-<unset>}/targets/x86_64-linux/include"
    echo "  Python nvidia/cudnn package under: ${CONDA_PREFIX:-<unset>}/lib/python*/site-packages/nvidia/cudnn/include"
    echo "  NVCC root include:    $SCRIPT_CUDA_ROOT/include"
    echo "  /usr/local/cuda/include"
    echo ""
    echo "If cuDNN is installed elsewhere, run:"
    echo "  CUDNN_INCLUDE_DIR=/path/to/include CUDNN_LIB_DIR=/path/to/lib bash compile.sh"
    echo ""
    echo "For a conda CUDA environment, install cuDNN headers/libraries into the env first, for example:"
    echo "  conda install -c conda-forge cudnn"
    echo "or if you use NVIDIA Python wheels:"
    echo "  pip install nvidia-cudnn-cu12"
    exit 1
fi

if [ -z "$CUDNN_LIB_DIR_FOUND" ]; then
    echo "Error: libcudnn not found. cuDNN is required for task11."
    echo ""
    echo "If cuDNN is installed elsewhere, run:"
    echo "  CUDNN_INCLUDE_DIR=/path/to/include CUDNN_LIB_DIR=/path/to/lib bash compile.sh"
    exit 1
fi

echo "========================================"
echo "  Task11 - CUDA Conv2D Build"
echo "========================================"
echo "CUDA_ARCH=$CUDA_ARCH"
echo "NVCC=$NVCC_BIN"
echo "CUDNN_INCLUDE_DIR=$CUDNN_INCLUDE_DIR_FOUND"
echo "CUDNN_LIB_DIR=$CUDNN_LIB_DIR_FOUND"
echo ""

echo "cuDNN support: required"

echo "[1/1] Building cuda_conv2d..."
"$NVCC_BIN" "${NVCC_FLAGS[@]}" -I"$CUDNN_INCLUDE_DIR_FOUND" -L"$CUDNN_LIB_DIR_FOUND" \
    -Xlinker -rpath -Xlinker "$CUDNN_LIB_DIR_FOUND" \
    -o bin/cuda_conv2d src/cuda_conv2d.cu -lcudnn
echo "    OK: bin/cuda_conv2d"
echo ""
echo "Usage:"
echo "  ./bin/cuda_conv2d [input_size] [kernel_size] [stride] [padding] [direct|im2col|cudnn|all] [repeat] [filters]"
