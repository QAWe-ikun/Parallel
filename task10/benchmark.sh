#!/bin/bash
# Task10 CUDA GEMM benchmark.

set -e

cd "$(dirname "$0")"

if [ ! -f bin/cuda_gemm ]; then
    bash compile.sh
fi

mkdir -p results

if [ -z "${SIZES:-}" ]; then
    SIZES=("512x512x512" "1024x1024x1024" "2048x2048x2048" "512x1024x512" "1024x512x2048")
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

REPEAT=${REPEAT:-10}
TILE_K=${TILE_K:-32}
VERIFY=${VERIFY:-sample}
SAMPLES=${SAMPLES:-2048}
RESULT_CSV="results/gemm_performance.csv"

echo "variant,m,n,k,block_x,block_y,tile_k,cols_per_thread,repeat,verify,checked,avg_time_ms,gflops,max_error,status" > "$RESULT_CSV"

for size in "${SIZES[@]}"; do
    m=$(echo "$size" | cut -dx -f1)
    n=$(echo "$size" | cut -dx -f2)
    k=$(echo "$size" | cut -dx -f3)

    for block in "${BLOCKS[@]}"; do
        bx="${block%x*}"
        by="${block#*x}"

        for variant in "${VARIANTS[@]}"; do
            if [ "$variant" = "both" ]; then
                echo "benchmark.sh expects VARIANTS to contain naive and/or tiled, not both."
                echo "Use ./bin/cuda_gemm ... both ... for a single combined run."
                exit 1
            fi

            log_file="results/${variant}_m${m}_n${n}_k${k}_b${bx}x${by}.log"
            ./bin/cuda_gemm "$m" "$n" "$k" "$bx" "$by" "$variant" "$REPEAT" "$TILE_K" \
                --verify "$VERIFY" --samples "$SAMPLES" > "$log_file"

            tile_k=$(grep '^tile_k=' "$log_file" | cut -d= -f2)
            cols_per_thread=$(grep '^cols_per_thread=' "$log_file" | cut -d= -f2)
            verify=$(grep '^verify=' "$log_file" | cut -d= -f2)
            checked=$(grep '^checked=' "$log_file" | cut -d= -f2)
            avg_time=$(grep '^avg_time_ms=' "$log_file" | cut -d= -f2)
            gflops=$(grep '^gflops=' "$log_file" | cut -d= -f2)
            max_error=$(grep '^max_error=' "$log_file" | cut -d= -f2)
            status=$(grep '^status=' "$log_file" | cut -d= -f2)

            echo "$variant,$m,$n,$k,$bx,$by,$tile_k,$cols_per_thread,$REPEAT,$verify,$checked,$avg_time,$gflops,$max_error,$status" >> "$RESULT_CSV"
            echo "$variant m=$m n=$n k=$k block=${bx}x${by} avg=${avg_time}ms gflops=${gflops} status=$status"
        done
    done
done

echo "Benchmark complete: $RESULT_CSV"
