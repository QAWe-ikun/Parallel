#!/bin/bash
# Task9 CUDA matrix transpose benchmark.

set -e

cd "$(dirname "$0")"

if [ ! -f bin/cuda_matrix_transpose ]; then
    bash compile.sh
fi

mkdir -p results

if [ -z "${SIZES:-}" ]; then
    SIZES=(512 1024 2048)
else
    read -r -a SIZES <<< "$SIZES"
fi

if [ -z "${BLOCKS:-}" ]; then
    BLOCKS=("8x8" "16x16" "32x8" "32x16")
else
    read -r -a BLOCKS <<< "$BLOCKS"
fi

if [ -z "${VARIANTS:-}" ]; then
    VARIANTS=(naive tiled)
else
    read -r -a VARIANTS <<< "$VARIANTS"
fi

REPEAT=${REPEAT:-20}
RESULT_CSV="results/transpose_performance.csv"

echo "variant,n,block_x,block_y,repeat,avg_time_ms,bandwidth_gbps,max_error,status" > "$RESULT_CSV"

for n in "${SIZES[@]}"; do
    for block in "${BLOCKS[@]}"; do
        bx="${block%x*}"
        by="${block#*x}"

        for variant in "${VARIANTS[@]}"; do
            if [ "$variant" = "tiled" ] && [ "$by" -gt "$bx" ]; then
                continue
            fi

            log_file="results/${variant}_n${n}_b${bx}x${by}.log"
            ./bin/cuda_matrix_transpose "$n" "$bx" "$by" "$variant" "$REPEAT" > "$log_file"

            avg_time=$(grep '^avg_time_ms=' "$log_file" | cut -d= -f2)
            bandwidth=$(grep '^bandwidth_gbps=' "$log_file" | cut -d= -f2)
            max_error=$(grep '^max_error=' "$log_file" | cut -d= -f2)
            status=$(grep '^status=' "$log_file" | cut -d= -f2)

            echo "$variant,$n,$bx,$by,$REPEAT,$avg_time,$bandwidth,$max_error,$status" >> "$RESULT_CSV"
            echo "$variant n=$n block=${bx}x${by} avg=${avg_time}ms bandwidth=${bandwidth}GB/s status=$status"
        done
    done
done

echo "Benchmark complete: $RESULT_CSV"
