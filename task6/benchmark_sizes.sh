#!/bin/bash
# Task6 - Pthreads 版本不同问题规模和线程数的性能测试脚本
# 对比加速比，不使用 Valgrind（运行速度快）

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXE="$SCRIPT_DIR/bin/heated_plate_pthreads"
export LD_LIBRARY_PATH="$SCRIPT_DIR/lib:$LD_LIBRARY_PATH"

SIZES=(250 500 1000)
THREADS=(1 2 4 8)
SCHEDULE=0 # 0=Static

# 关联数组存储时间
declare -A TIMES

extract_time() {
    echo "$1" | grep -oP 'Wallclock time\s*=\s*\K[\d.]+'
}

echo ""
echo "============================================"
echo "  Task6 - Pthreads 性能基准测试"
echo "  网格规模: 250, 500, 1000"
echo "  线程数: 1, 2, 4, 8"
echo "  调度策略: Static"
echo "============================================"
echo ""

for s in "${SIZES[@]}"; do
    echo "----------------------------------------"
    echo "  网格规模: ${s}x${s}"
    echo "----------------------------------------"

    for t in "${THREADS[@]}"; do
        output=$("$EXE" $t $s $SCHEDULE 2>&1)
        wallclock=$(extract_time "$output")
        TIMES["${s}_${t}"]="$wallclock"
        printf "  T=%d: %s s\n" "$t" "$wallclock"
    done
    echo ""
done

# ====== 输出汇总表格 ======

echo "============================================"
echo "  Wallclock 时间 (秒)"
echo "============================================"
echo ""

printf "%-15s" "网格规模"
for t in "${THREADS[@]}"; do
    printf "| T=%-10s" "$t"
done
echo ""
printf "%-15s" "----------------"
for t in "${THREADS[@]}"; do printf "|------------"; done
echo ""

for s in "${SIZES[@]}"; do
    printf "%-15s" "${s}x${s}"
    for t in "${THREADS[@]}"; do
        printf "| %-10s" "${TIMES[${s}_${t}]}"
    done
    echo ""
done

# ====== 输出加速比表格 ======

echo ""
echo "============================================"
echo "  加速比 (相对于同规模 T=1)"
echo "============================================"
echo ""

printf "%-15s" "网格规模"
for t in "${THREADS[@]}"; do
    printf "| T=%-10s" "$t"
done
echo ""
printf "%-15s" "----------------"
for t in "${THREADS[@]}"; do printf "|------------"; done
echo ""

for s in "${SIZES[@]}"; do
    t1="${TIMES[${s}_1]}"
    printf "%-15s" "${s}x${s}"
    for t in "${THREADS[@]}"; do
        tn="${TIMES[${s}_${t}]}"
        if [ -n "$tn" ] && [ "$tn" != "0.0" ]; then
            speedup=$(echo "scale=2; $t1 / $tn" | bc)
            printf "| %-10sx" "$speedup"
        else
            printf "| %-10s" "N/A"
        fi
    done
    echo ""
done

echo ""
echo "基准测试完成。"
echo ""