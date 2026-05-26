#!/bin/bash
# Task7 - Build Script
# 编译 MPI FFT 和串行 FFT

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "========================================"
echo "  Task7 - Build Script"
echo "========================================"

mkdir -p "$SCRIPT_DIR/bin"

# 检查 MPI 编译器
if ! command -v mpicc &> /dev/null; then
    echo "Error: mpicc not found. Please install MPI."
    exit 1
fi

# 1. 编译串行 FFT (C++ 版本)
echo "[1/2] Building fft_serial..."
g++ -O2 -o "$SCRIPT_DIR/bin/fft_serial" "$SCRIPT_DIR/docs/fft_serial.cpp" -lm
echo "    OK: bin/fft_serial"

# 2. 编译 MPI FFT (C 版本)
echo "[2/2] Building fft_mpi..."
mpicc -O2 -o "$SCRIPT_DIR/bin/fft_mpi" "$SCRIPT_DIR/src/fft_mpi.c" -lm
echo "    OK: bin/fft_mpi"

echo ""
echo "========================================"
echo "  Build Complete!"
echo "========================================"
echo ""
echo "Executables:"
echo "  - bin/fft_serial   (Serial FFT)"
echo "  - bin/fft_mpi      (MPI parallel FFT)"
echo ""
echo "Usage:"
echo "  ./bin/fft_serial"
echo "  mpirun -np 4 ./bin/fft_mpi"
echo ""
