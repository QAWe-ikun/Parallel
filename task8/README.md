# Task8: OpenMP 多源最短路径搜索

## 实验内容

使用 OpenMP 在无向加权图上实现多源最短路径搜索。程序读取邻接表 CSV 和测试点对 CSV，对测试文件中的所有 `(source, target)` 查询计算最短路径距离，并将距离作为第三列输出。

## 并行策略

本实现采用“按唯一源点并行”的方式：

1. 读入邻接表，将输入中的边全部视为无向边。
2. 将原始顶点 ID 压缩为连续下标，并构建 CSR 邻接表。
3. 读入测试点对，将查询按 `source` 分组。
4. 对每个不同源点运行一次 Dijkstra，OpenMP 并行分发这些源点。
5. 将每个查询对应的 `target` 距离写回输出 CSV。

这种方式避免同一个源点在测试文件中出现多次时重复运行 Dijkstra，比逐查询并行更适合“多源”最短路径。

## 文件结构

```text
task8/
├── src/
│   └── mssp_omp.c              # OpenMP 多源最短路径实现
├── data/
│   ├── updated_mouse.csv       # 图数据
│   └── updated_flower.csv      # 图数据
├── results/                    # benchmark 输出目录
├── report/
│   └── report.md               # 实验报告
├── compile.sh                  # Linux / WSL 编译脚本
├── benchmark.sh                # Linux / WSL 性能测试脚本
└── README.md
```

## 编译

Linux / WSL:

```bash
cd task8
bash compile.sh
```

## 运行

生成随机测试点对文件：

```bash
./bin/mssp_omp --generate data/updated_mouse.csv results/mouse_queries.csv 10000 2026
```

计算最短路径并输出 CSV：

```bash
./bin/mssp_omp data/updated_mouse.csv results/mouse_queries.csv results/mouse_output.csv 8 5
```

也可以直接使用仓库中的小型示例测试文件：

```bash
./bin/mssp_omp data/updated_mouse.csv data/sample_mouse_queries.csv results/sample_mouse_output.csv 4
./bin/mssp_omp data/updated_flower.csv data/sample_flower_queries.csv results/sample_flower_output.csv 4
```

参数含义：

```text
mssp_omp <graph.csv> <queries.csv> <output.csv> <threads> [repeat]
```

- `graph.csv`: 邻接表文件，格式为 `source,target,distance`
- `queries.csv`: 测试文件，格式为 `source,target`
- `output.csv`: 输出文件，格式为 `source,target,distance`
- `threads`: OpenMP 线程数，实验中建议设置为 `1,2,4,8,16`
- `repeat`: 可选重复次数，用于降低短任务计时误差，默认 `1`

## 性能测试

Linux / WSL:

```bash
cd task8
bash benchmark.sh
```

脚本默认对 `updated_mouse.csv` 和 `updated_flower.csv` 分别生成 10000 条随机查询，并测试 `1,2,4,8,16` 个线程。结果写入：

```text
task8/results/performance.csv
```

可通过环境变量调整测试规模：

```bash
QUERY_COUNT=50000 REPEAT=3 bash benchmark.sh
```

## 输出说明

程序标准输出会打印计时和图统计信息：

```text
vertices=525
edges=14691
avg_degree=55.9657
queries=10000
unique_sources=525
threads=8
repeat=5
time_seconds=...
avg_time_seconds=...
```

若测试点对中存在图中不存在的顶点，或两点不连通，输出距离为 `inf`。

## 数据规模

当前仓库中的两个图数据：

| 数据 | 节点数 | 边数 | 平均度数 |
|---|---:|---:|---:|
| updated_mouse | 525 | 14691 | 55.97 |
| updated_flower | 930 | 13521 | 29.08 |

`updated_mouse` 节点更少但平均度更高，单次 Dijkstra 会扫描更密集的邻接边；`updated_flower` 节点更多但平均度更低，源点数量更大时可提供更多并行任务。
