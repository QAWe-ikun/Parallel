#!/bin/bash
# Task9 compile script for CUDA Hello World and matrix transpose.

set -e

cd "$(dirname "$0")"
mkdir -p bin

if ! command -v nvcc > /dev/null 2>&1; then
    echo "Error: nvcc not found. Please load/install the CUDA Toolkit first."
    echo "On EasyHPC, select a CUDA-capable environment before running this script."
    exit 1
fi

CUDA_ARCH=${CUDA_ARCH:-sm_86}
NVCC_FLAGS=(-O2 -std=c++14 -arch="$CUDA_ARCH" -Xcompiler -Wall)

echo "========================================"
echo "  Task9 - CUDA Build"
echo "========================================"
echo "CUDA_ARCH=$CUDA_ARCH"
echo ""

echo "[1/2] Building cuda_hello_world..."
nvcc "${NVCC_FLAGS[@]}" -o bin/cuda_hello_world src/cuda_hello_world.cu
echo "    OK: bin/cuda_hello_world"

echo "[2/2] Building cuda_matrix_transpose..."
nvcc "${NVCC_FLAGS[@]}" -o bin/cuda_matrix_transpose src/cuda_matrix_transpose.cu
echo "    OK: bin/cuda_matrix_transpose"

echo ""
echo "Usage:"
echo "  ./bin/cuda_hello_world <num_blocks> <block_dim_x> <block_dim_y>"
echo "  ./bin/cuda_matrix_transpose [n] [block_x] [block_y] [naive|tiled|both] [repeat]"
echo ""
