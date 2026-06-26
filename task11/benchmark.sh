#!/bin/bash
# Task11 CUDA convolution benchmark.

set -e

cd "$(dirname "$0")"

if [ ! -f bin/cuda_conv2d ]; then
    bash compile.sh
fi

mkdir -p results

if [ -z "${SIZES:-}" ]; then
    SIZES=(256 512 1024)
else
    read -r -a SIZES <<< "$SIZES"
fi

if [ -z "${STRIDES:-}" ]; then
    STRIDES=(1 2 3)
else
    read -r -a STRIDES <<< "$STRIDES"
fi

if [ -z "${BLOCKS:-}" ]; then
    BLOCKS=("8x8" "16x16" "32x8")
else
    read -r -a BLOCKS <<< "$BLOCKS"
fi

if [ -z "${VARIANTS:-}" ]; then
    VARIANTS=(direct im2col cudnn)
else
    read -r -a VARIANTS <<< "$VARIANTS"
fi

KERNEL_SIZE=${KERNEL_SIZE:-3}
PADDING=${PADDING:-1}
FILTERS=${FILTERS:-16}
REPEAT=${REPEAT:-100}
TILE_K=${TILE_K:-32}
VERIFY=${VERIFY:-sample}
SAMPLES=${SAMPLES:-2048}
RESULT_CSV="results/conv_performance.csv"

echo "variant,input_size,kernel_size,channels,filters,stride,padding,out_h,out_w,block_x,block_y,tile_k,repeat,verify,checked,avg_time_ms,gops,max_error,status" > "$RESULT_CSV"

for size in "${SIZES[@]}"; do
    for stride in "${STRIDES[@]}"; do
        for block in "${BLOCKS[@]}"; do
            bx="${block%x*}"
            by="${block#*x}"
            for variant in "${VARIANTS[@]}"; do
                if [ "$variant" = "all" ]; then
                    echo "benchmark.sh expects explicit variants, not all."
                    exit 1
                fi

                log_file="results/${variant}_n${size}_k${KERNEL_SIZE}_s${stride}_p${PADDING}_f${FILTERS}_b${bx}x${by}.log"
                ./bin/cuda_conv2d "$size" "$KERNEL_SIZE" "$stride" "$PADDING" "$variant" "$REPEAT" "$FILTERS" \
                    --block "$block" --tile-k "$TILE_K" --verify "$VERIFY" --samples "$SAMPLES" > "$log_file"

                status=$(grep '^status=' "$log_file" | cut -d= -f2)
                channels=$(grep '^channels=' "$log_file" | cut -d= -f2)
                filters=$(grep '^filters=' "$log_file" | cut -d= -f2)
                out_h=$(grep '^out_h=' "$log_file" | cut -d= -f2)
                out_w=$(grep '^out_w=' "$log_file" | cut -d= -f2)
                tile_k=$(grep '^tile_k=' "$log_file" | cut -d= -f2)
                verify=$(grep '^verify=' "$log_file" | cut -d= -f2)
                checked=$(grep '^checked=' "$log_file" | cut -d= -f2)
                avg_time=$(grep '^avg_time_ms=' "$log_file" | cut -d= -f2)
                gops=$(grep '^gops=' "$log_file" | cut -d= -f2)
                max_error=$(grep '^max_error=' "$log_file" | cut -d= -f2)

                echo "$variant,$size,$KERNEL_SIZE,$channels,$filters,$stride,$PADDING,$out_h,$out_w,$bx,$by,$tile_k,$REPEAT,$verify,$checked,$avg_time,$gops,$max_error,$status" >> "$RESULT_CSV"
                echo "$variant n=$size stride=$stride block=$block avg=${avg_time}ms gops=${gops} status=$status"
            done
        done
    done
done

echo "Benchmark complete: $RESULT_CSV"
