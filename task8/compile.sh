#!/bin/bash
# Task8 compile script

set -e

cd "$(dirname "$0")"
mkdir -p bin

echo "Compiling OpenMP multi-source shortest path program..."
gcc -std=c99 -O2 -Wall -Wextra -fopenmp -o bin/mssp_omp src/mssp_omp.c -lm

echo "Done: task8/bin/mssp_omp"
echo ""
echo "Usage:"
echo "  ./bin/mssp_omp <graph.csv> <queries.csv> <output.csv> <threads> [repeat]"
echo "  ./bin/mssp_omp --generate <graph.csv> <queries.csv> <num_queries> [seed]"
