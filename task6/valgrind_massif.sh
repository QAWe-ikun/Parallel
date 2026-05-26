#!/bin/bash
# Task6 - Valgrind Massif 内存分析脚本
# 采集 heated_plate_pthreads (Pthreads 版本) 的内存消耗数据

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXE="$SCRIPT_DIR/bin/heated_plate_pthreads"
MASSIF_DIR="$SCRIPT_DIR/massif_output"

THREADS=(1 2 4 8)
SIZES=(250 500 1000)

mkdir -p "$MASSIF_DIR"

echo "============================================"
echo "  Task6 - Valgrind Massif 内存分析"
echo "  程序: heated_plate_pthreads (Pthreads 版本)"
echo "============================================"
echo ""

if ! command -v valgrind &> /dev/null; then
    echo "Error: valgrind not found. Please install valgrind."
    exit 1
fi

for t in "${THREADS[@]}"; do
    for s in "${SIZES[@]}"; do
        echo "Running: threads=$t, size=${s}x${s}..."
        
        OUTPUT_FILE="$MASSIF_DIR/massif_t${t}_s${s}.out"
        LOG_FILE="$MASSIF_DIR/massif_t${t}_s${s}.log"
        
        valgrind --tool=massif --stacks=yes --pages-as-heap=yes \
                 --massif-out-file="$OUTPUT_FILE" \
                 "$EXE" $t $s 2>"$LOG_FILE"
        
        echo "  Output: $OUTPUT_FILE"
        echo "  Log: $LOG_FILE"
        echo ""
    done
done

echo "============================================"
echo "  分析结果摘要"
echo "============================================"
echo ""

for t in "${THREADS[@]}"; do
    for s in "${SIZES[@]}"; do
        OUTPUT_FILE="$MASSIF_DIR/massif_t${t}_s${s}.out"
        LOG_FILE="$MASSIF_DIR/massif_t${t}_s${s}.log"
        
        if [ -f "$OUTPUT_FILE" ]; then
            echo "=== 线程=$t, 网格=${s}x${s} ==="
            # 从 massif 输出中提取峰值内存
            PEAK=$(grep -oP 'mem_heap_B=\K[0-9]+' "$OUTPUT_FILE" 2>/dev/null | sort -n | tail -1)
            if [ -n "$PEAK" ]; then
                PEAK_MB=$(echo "scale=2; $PEAK / 1024 / 1024" | bc)
                echo "  峰值堆内存: ${PEAK_MB} MB"
            fi
            
            # 使用 ms_print 获取更详细信息
            if command -v ms_print &> /dev/null; then
                echo "  详细分析: ms_print $OUTPUT_FILE"
            fi
            echo ""
        fi
    done
done

echo "完整分析请使用: ms_print $MASSIF_DIR/massif_t<T>_s<S>.out"
echo ""
