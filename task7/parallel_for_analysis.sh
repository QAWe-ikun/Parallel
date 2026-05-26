#!/bin/bash
# Task7 - parallel_for 并行应用分析
# a) 不同问题规模 N、线程数和调度策略的性能分析
# b) Valgrind massif 内存分析

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
EXE="$PROJECT_DIR/task6/bin/heated_plate_pthreads"

echo "========================================"
echo "  parallel_for 并行应用分析"
echo "========================================"

mkdir -p "$SCRIPT_DIR/results"

# 检查可执行文件
if [ ! -f "$EXE" ]; then
    echo "Error: $EXE not found."
    echo "Please build task6 first: bash task6/compile.sh"
    exit 1
fi

# a) 性能分析：不同 N、线程数和调度策略
echo ""
echo "[1/2] 性能分析：不同问题规模 N、线程数和调度策略"
echo ""

THREADS=(1 2 4 8)
SCALES=(64 128 256 512)
SCHEDULES=(0 1 2)  # 0=STATIC, 1=DYNAMIC, 2=GUIDED
SCHEDULE_NAMES=("STATIC" "DYNAMIC" "GUIDED")

# 测试每种调度策略
for sched_idx in "${!SCHEDULES[@]}"; do
    SCHED="${SCHEDULES[$sched_idx]}"
    SCHED_NAME="${SCHEDULE_NAMES[$sched_idx]}"

    echo "========================================"
    echo "  调度策略: $SCHED_NAME (schedule=$SCHED)"
    echo "========================================"
    echo ""

    RESULTS_FILE="$SCRIPT_DIR/results/performance_${SCHED_NAME}.csv"
    echo "N,Threads,Time(s),Speedup,Efficiency" > "$RESULTS_FILE"

    for N in "${SCALES[@]}"; do
        echo "  测试 N=$N..."

        # 获取串行基准时间
        T1=$("$EXE" 1 "$N" "$SCHED" 1 2>&1 | grep "Wallclock time" | awk '{print $4}')

        if [ -z "$T1" ] || [ "$T1" = "0" ]; then
            echo "    警告: N=$N 串行运行失败，跳过"
            continue
        fi

        for T in "${THREADS[@]}"; do
            if [ "$N" -lt "$T" ]; then
                echo "$N,$T,0,0,0" >> "$RESULTS_FILE"
                continue
            fi

            TIME=$("$EXE" "$T" "$N" "$SCHED" 1 2>&1 | grep "Wallclock time" | awk '{print $4}')

            if [ -z "$TIME" ] || [ "$TIME" = "0" ]; then
                echo "$N,$T,0,0,0" >> "$RESULTS_FILE"
                continue
            fi

            SPEEDUP=$(echo "scale=4; $T1 / $TIME" | bc)
            EFFICIENCY=$(echo "scale=4; $SPEEDUP / $T * 100" | bc)

            echo "$N,$T,$TIME,$SPEEDUP,$EFFICIENCY" >> "$RESULTS_FILE"
        done
    done

    echo ""
    echo "性能测试结果已保存到: $RESULTS_FILE"
    echo ""

    # 生成表格格式输出
    echo "----------------------------------------"
    printf "%-8s" "N"
    for T in "${THREADS[@]}"; do
        printf "%-14s" "T=$T"
    done
    echo ""
    echo "----------------------------------------"

    for N in "${SCALES[@]}"; do
        printf "%-8d" "$N"
        for T in "${THREADS[@]}"; do
            if [ "$N" -lt "$T" ]; then
                printf "%-14s" "-"
            else
                TIME=$(grep "^$N,$T," "$RESULTS_FILE" | cut -d',' -f3)
                SPEEDUP=$(grep "^$N,$T," "$RESULTS_FILE" | cut -d',' -f4)
                if [ -n "$TIME" ] && [ "$TIME" != "0" ]; then
                    printf "%-14s" "${TIME}s(${SPEEDUP}x)"
                else
                    printf "%-14s" "-"
                fi
            fi
        done
        echo ""
    done
    echo ""
done

# b) Valgrind massif 内存分析
echo ""
echo "[2/2] Valgrind massif 内存分析"
echo ""

ANALYSIS_N=256
ANALYSIS_T=4

echo "  分析配置: N=$ANALYSIS_N, Threads=$ANALYSIS_T, Schedule=STATIC"
echo "  运行 Valgrind massif..."

MASSIF_OUT="$SCRIPT_DIR/results/massif.out.$$"

# 检查 valgrind 是否安装
if ! command -v valgrind &> /dev/null; then
    echo "  警告: valgrind 未安装，跳过内存分析"
    echo "  安装: sudo apt install valgrind"
else
    # 运行 Valgrind massif
    valgrind --tool=massif \
        --stacks=yes \
        --detailed-freq=1 \
        --massif-out-file="$MASSIF_OUT" \
        "$EXE" "$ANALYSIS_T" "$ANALYSIS_N" 0 1 > /dev/null 2>&1

    echo "  Massif 输出已保存到: $MASSIF_OUT"

    # 使用 ms_print 生成可读报告
    MS_PRINT_OUT="$SCRIPT_DIR/results/massif_report.txt"
    ms_print "$MASSIF_OUT" > "$MS_PRINT_OUT" 2>&1

    echo "  可读报告已保存到: $MS_PRINT_OUT"
    echo ""

    # 提取关键内存信息
    echo "========================================"
    echo "  内存分析摘要"
    echo "========================================"
    echo ""

    # 峰值内存
    PEAK_MEM=$(grep -A 5 "Peak" "$MS_PRINT_OUT" | grep "Memory" | head -1 | awk '{print $2}')
    echo "峰值内存: $PEAK_MEM"

    # 详细内存分布
    echo ""
    echo "内存分布详情（前 20 行）:"
    grep -A 20 "Detailed snapshots" "$MS_PRINT_OUT" | head -25
fi

echo ""
echo "========================================"
echo "  分析完成"
echo "========================================"
echo ""
echo "结果文件:"
echo "  - STATIC 调度: results/performance_STATIC.csv"
echo "  - DYNAMIC 调度: results/performance_DYNAMIC.csv"
echo "  - GUIDED 调度: results/performance_GUIDED.csv"
if command -v valgrind &> /dev/null; then
    echo "  - Massif 原始输出: results/massif.out.$$"
    echo "  - Massif 可读报告: results/massif_report.txt"
fi
echo ""
