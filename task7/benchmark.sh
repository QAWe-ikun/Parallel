#!/bin/bash
# Task7 - MPI FFT 性能测试脚本
# 测试不同进程数下的 MPI FFT 性能

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SERIAL_EXE="$SCRIPT_DIR/bin/fft_serial"
MPI_EXE="$SCRIPT_DIR/bin/fft_mpi"

MPIRUN="mpirun --oversubscribe"
PROCS=(1 2 4 8)

echo ""
echo "============================================"
echo "  Task7 - MPI FFT Performance Benchmark"
echo "============================================"
echo ""

mkdir -p "$SCRIPT_DIR/results"

# 串行版本
echo "[1] 运行串行 FFT..."
"$SERIAL_EXE" > "$SCRIPT_DIR/results/serial_output.txt" 2>&1
echo "  完成"
echo ""

# MPI 版本
echo "[2] 运行 MPI 并行 FFT..."
for p in "${PROCS[@]}"; do
    echo "  --- $p processes ---"
    $MPIRUN -np $p "$MPI_EXE" > "$SCRIPT_DIR/results/mpi_p${p}_output.txt" 2>&1
    grep -E "^\s+[0-9]+" "$SCRIPT_DIR/results/mpi_p${p}_output.txt" | head -10
    echo ""
done

echo "============================================"
echo "  结果摘要"
echo "============================================"
echo ""

# 提取各版本的时间
echo "串行版本:"
grep -E "^\s+[0-9]+" "$SCRIPT_DIR/results/serial_output.txt" | tail -5
echo ""

for p in "${PROCS[@]}"; do
    echo "$p 进程 MPI:"
    grep -E "^\s+[0-9]+" "$SCRIPT_DIR/results/mpi_p${p}_output.txt" | tail -5
    echo ""
done

echo "Benchmark complete."
echo ""
