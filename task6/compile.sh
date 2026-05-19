#!/bin/bash
# Task6 Heated Plate - Build Script (Linux / WSL)
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "========================================"
echo "  Task6 Heated Plate - Build Script"
echo "========================================"

mkdir -p "$SCRIPT_DIR/bin" "$SCRIPT_DIR/lib"

# 1. Compile parallel_for_pool dynamic library (thread pool version)
echo "[1/3] Building libparallel_for_pool.so..."
gcc -shared -fPIC -O2 -o "$SCRIPT_DIR/lib/libparallel_for_pool.so" "$SCRIPT_DIR/src/parallel_for_pool.c" -lpthread
echo "    OK: lib/libparallel_for_pool.so"

# 2. Compile OpenMP reference
echo "[2/3] Building heated_plate_openmp..."
gcc -fopenmp -O3 -march=native -ftree-vectorize -funroll-loops -o "$SCRIPT_DIR/bin/heated_plate_openmp" "$SCRIPT_DIR/src/heated_plate_openmp.c" -lm
echo "    OK: bin/heated_plate_openmp"

# 3. Compile Pthreads version (uses thread pool)
echo "[3/3] Building heated_plate_pthreads..."
gcc -O3 -march=native -ftree-vectorize -funroll-loops -I"$SCRIPT_DIR/src" -o "$SCRIPT_DIR/bin/heated_plate_pthreads" "$SCRIPT_DIR/src/heated_plate_pthreads.c" -L"$SCRIPT_DIR/lib" -lparallel_for_pool -lpthread -lm -Wl,-rpath,'$ORIGIN/../lib'
echo "    OK: bin/heated_plate_pthreads"

echo ""
echo "========================================"
echo "  Build Complete!"
echo "========================================"
echo ""
echo "Executables:"
echo "  - bin/heated_plate_openmp   (OpenMP reference)"
echo "  - bin/heated_plate_pthreads (Pthreads thread pool)"
echo ""
echo "Library:"
echo "  - lib/libparallel_for_pool.so"
echo ""
echo "Usage:"
echo "  ./bin/heated_plate_openmp [num_threads]"
echo "  ./bin/heated_plate_pthreads [num_threads] [schedule:0=static,1=dynamic,2=guided] [chunk_size]"
