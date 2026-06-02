#!/bin/bash
# Task8 performance benchmark for Linux / WSL.

set -e

cd "$(dirname "$0")"

if [ ! -f bin/mssp_omp ]; then
    bash compile.sh
fi

mkdir -p results

THREADS=(1 2 4 8 16)
QUERY_COUNT=${QUERY_COUNT:-10000}
REPEAT=${REPEAT:-5}
RESULT_CSV="results/performance.csv"

echo "dataset,vertices,edges,avg_degree,queries,unique_sources,threads,repeat,total_time,avg_time,speedup,efficiency" > "$RESULT_CSV"

for graph in data/updated_mouse.csv data/updated_flower.csv; do
    dataset=$(basename "$graph" .csv)
    queries="results/${dataset}_queries.csv"
    ./bin/mssp_omp --generate "$graph" "$queries" "$QUERY_COUNT" 2026 > "results/${dataset}_generate.txt"

    baseline=""
    for t in "${THREADS[@]}"; do
        output_csv="results/${dataset}_threads_${t}.csv"
        log_file="results/${dataset}_threads_${t}.log"
        ./bin/mssp_omp "$graph" "$queries" "$output_csv" "$t" "$REPEAT" > "$log_file"

        vertices=$(grep '^vertices=' "$log_file" | cut -d= -f2)
        edges=$(grep '^edges=' "$log_file" | cut -d= -f2)
        avg_degree=$(grep '^avg_degree=' "$log_file" | cut -d= -f2)
        query_total=$(grep '^queries=' "$log_file" | cut -d= -f2)
        unique_sources=$(grep '^unique_sources=' "$log_file" | cut -d= -f2)
        total_time=$(grep '^time_seconds=' "$log_file" | cut -d= -f2)
        avg_time=$(grep '^avg_time_seconds=' "$log_file" | cut -d= -f2)

        if [ "$t" = "1" ]; then
            baseline="$avg_time"
        fi

        speedup=$(awk -v b="$baseline" -v a="$avg_time" 'BEGIN { if (a > 0) printf "%.4f", b / a; else printf "0.0000" }')
        efficiency=$(awk -v s="$speedup" -v t="$t" 'BEGIN { printf "%.4f", s / t }')

        echo "$dataset,$vertices,$edges,$avg_degree,$query_total,$unique_sources,$t,$REPEAT,$total_time,$avg_time,$speedup,$efficiency" >> "$RESULT_CSV"
        echo "$dataset threads=$t avg_time=${avg_time}s speedup=${speedup}"
    done
done

echo "Benchmark complete: $RESULT_CSV"
