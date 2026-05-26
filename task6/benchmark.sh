#!/bin/bash
# Task6 Heated Plate - Benchmark Script (Linux / WSL)
# 测试不同线程数、调度策略的并行性能，并与 OpenMP 版本对比
# 使用 parallel_for_pool（线程池版本）

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OPENMP_EXE="$SCRIPT_DIR/bin/heated_plate_openmp"
PTHREADS_EXE="$SCRIPT_DIR/bin/heated_plate_pthreads"
export LD_LIBRARY_PATH="$SCRIPT_DIR/lib:$LD_LIBRARY_PATH"

THREADS=(1 2 4 8)
SCHEDULES=( "0 static" "1 dynamic" "2 guided" )

declare -A OPENMP_TIME
declare -A PTHREADS_TIME

extract_wallclock() {
    echo "$1" | grep -oP 'Wallclock time\s*=\s*\K[\d.]+'
}

echo ""
echo "============================================"
echo "  Task6 Heated Plate - 性能基准测试"
echo "  网格: 500x500,  收敛精度: 0.001"
echo "  Pthreads: parallel_for_pool (线程池版本)"
echo "============================================"
echo ""

# Step 1: OpenMP 参考版本
echo "[Step 1] 运行 OpenMP 参考版本..."
for t in "${THREADS[@]}"; do
    export OMP_NUM_THREADS=$t
    output=$("$OPENMP_EXE" $t 2>&1)
    OPENMP_TIME[$t]=$(extract_wallclock "$output")
    printf "  OpenMP T=%2d: %s s\n" "$t" "${OPENMP_TIME[$t]}"
done

# Step 2: Pthreads —— 按调度策略分组，每组跑完所有线程
echo ""
echo "[Step 2] 运行 Pthreads parallel_for_pool..."
for sched_entry in "${SCHEDULES[@]}"; do
    sched_id=$(echo "$sched_entry" | cut -d' ' -f1)
    sched_name=$(echo "$sched_entry" | cut -d' ' -f2)
    printf "\n  --- %s ---\n" "$sched_name"
    for t in "${THREADS[@]}"; do
        output=$("$PTHREADS_EXE" $t 500 $sched_id 1 2>&1)
        PTHREADS_TIME["${t}_${sched_name}"]=$(extract_wallclock "$output")
        printf "  T=%2d: %s s\n" "$t" "${PTHREADS_TIME[${t}_${sched_name}]}"
    done
done

# ====== 输出表格 ======

# --- Wallclock 时间表 ---
echo ""
echo "============================================"
echo "  Wallclock 时间 (秒)"
echo "============================================"
echo ""

printf "%-18s" "方法"
for t in "${THREADS[@]}"; do printf "| T=%-5s" "$t"; done
echo ""
printf "%-18s" "------------------"
for t in "${THREADS[@]}"; do printf "|------"; done
echo ""

printf "%-18s" "OpenMP"
for t in "${THREADS[@]}"; do
    printf "| %-6s" "${OPENMP_TIME[$t]}"
done
echo ""

for sched_entry in "${SCHEDULES[@]}"; do
    sched_name=$(echo "$sched_entry" | cut -d' ' -f2)
    printf "%-18s" "Pthreads $sched_name"
    for t in "${THREADS[@]}"; do
        printf "| %-6s" "${PTHREADS_TIME[${t}_${sched_name}]}"
    done
    echo ""
done

# --- 加速比表（以 OpenMP 1T 为基准） ---
echo ""
echo "============================================"
echo "  加速比 (相对于 OpenMP 1T 基线)"
echo "============================================"
echo ""

BASE="${OPENMP_TIME[1]}"
if [ "$BASE" = "0.0" ] || [ -z "$BASE" ]; then
    BASE="${PTHREADS_TIME[1_static]}"
fi

printf "%-18s" "方法"
for t in "${THREADS[@]}"; do printf "| T=%-5s" "$t"; done
echo ""
printf "%-18s" "------------------"
for t in "${THREADS[@]}"; do printf "|------"; done
echo ""

printf "%-18s" "OpenMP"
for t in "${THREADS[@]}"; do
    time="${OPENMP_TIME[$t]}"
    if [ "$time" != "0.0" ] && [ -n "$time" ]; then
        speedup=$(echo "scale=2; $BASE / $time" | bc)
        printf "| %-6sx" "$speedup"
    else
        printf "| N/A   "
    fi
done
echo ""

for sched_entry in "${SCHEDULES[@]}"; do
    sched_name=$(echo "$sched_entry" | cut -d' ' -f2)
    printf "%-18s" "Pthreads $sched_name"
    for t in "${THREADS[@]}"; do
        time="${PTHREADS_TIME[${t}_${sched_name}]}"
        if [ "$time" != "0.0" ] && [ -n "$time" ]; then
            speedup=$(echo "scale=2; $BASE / $time" | bc)
            printf "| %-6sx" "$speedup"
        else
            printf "| N/A   "
        fi
    done
    echo ""
done

# --- 并行效率表 ---
echo ""
echo "============================================"
echo "  并行效率 (加速比 / 线程数 %)"
echo "============================================"
echo ""

printf "%-18s" "方法"
for t in "${THREADS[@]}"; do
    if [ "$t" -gt 1 ]; then printf "| T=%-5s" "$t"; fi
done
echo ""
printf "%-18s" "------------------"
for t in "${THREADS[@]}"; do
    if [ "$t" -gt 1 ]; then printf "|------"; fi
done
echo ""

for sched_entry in "${SCHEDULES[@]}"; do
    sched_name=$(echo "$sched_entry" | cut -d' ' -f2)
    printf "%-18s" "Pthreads $sched_name"
    for t in "${THREADS[@]}"; do
        if [ "$t" -eq 1 ]; then continue; fi
        time="${PTHREADS_TIME[${t}_${sched_name}]}"
        if [ "$time" != "0.0" ] && [ -n "$time" ]; then
            speedup=$(echo "scale=4; $BASE / $time" | bc)
            efficiency=$(echo "scale=2; $speedup * 100 / $t" | bc)
            efficiency=$(echo "$efficiency" | sed 's/^\./0./')
            printf "| %-6s%%" "$efficiency"
        else
            printf "| N/A   "
        fi
    done
    echo ""
done

echo ""
echo "基准测试完成。"
echo ""
