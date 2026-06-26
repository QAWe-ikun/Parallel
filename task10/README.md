# Task10: CUDA 并行通用矩阵乘法

本实验根据 `docs/10-CUDA矩阵乘法.docx` 完成 CUDA 通用矩阵乘法：

```text
A(m x n) * B(n x k) = C(m x k)
```

程序随机生成矩阵 `A` 和 `B`，使用 CUDA 计算矩阵 `C`，输出 kernel 平均耗时、GFLOPS、校验误差和矩阵校验和，并支持将 `A/B/C` 导出到文件。

## 文件结构

```text
task10/
├── src/
│   └── cuda_gemm.cu
├── results/
├── report/
├── compile.sh
├── benchmark.sh
└── README.md
```

## 编译

Linux / EasyHPC CUDA 环境：

```bash
cd task10
bash compile.sh
```

默认 CUDA 架构为 `sm_86`。如果运行环境不同，可以指定：

```bash
CUDA_ARCH=sm_89 bash compile.sh
```

## 运行

```bash
./bin/cuda_gemm 1024 1024 1024 16 16 both 20 32
```

参数含义：

```text
cuda_gemm [m] [n] [k] [block_x] [block_y] [variant] [repeat] [tile_k] [options]
```

- `m n k`: 矩阵规模，实验要求范围为 `[128, 2048]`
- `block_x`: 线程块 x 维度；`naive` 每个 block 覆盖 `block_x` 列，`tiled` 每个线程计算 4 个连续列，因此每个 block 覆盖 `4 * block_x` 列
- `block_y`: 每个线程块负责的输出行数
- `variant`: `naive`、`tiled` 或 `both`
- `repeat`: 计时重复次数
- `tile_k`: tiled 版本沿归约维度 `n` 的分块宽度
- `--verify none|sample|full`: 校验模式，默认 `sample`
- `--samples <count>`: 抽样校验的元素个数
- `--print`: 当 `m,n,k <= 16` 时打印矩阵
- `--dump <csv>`: 将完整 `A/B/C` 写入文件
- `--seed <value>`: 设置随机种子

示例：

```bash
./bin/cuda_gemm 512 512 512 16 16 naive 20 16
./bin/cuda_gemm 2048 2048 2048 32 8 tiled 20 32
./bin/cuda_gemm 128 128 128 16 16 both 5 16 --verify full --dump results/matrix_dump.csv
```

输出字段包括：

```text
variant=...
m=...
n=...
k=...
block_x=...
block_y=...
tile_k=...
avg_time_ms=...
gflops=...
max_error=...
status=PASS
```

## 性能测试

```bash
bash benchmark.sh
```

默认测试：

- 矩阵规模：`512x512x512`、`1024x1024x1024`、`2048x2048x2048`、`512x1024x512`、`1024x512x2048`
- 线程块：`8x8`、`16x16`、`32x8`、`32x16`
- 访存方式：`naive` 与 `tiled`
- `tile_k=32`
- `repeat=10`

结果写入：

```text
results/gemm_performance.csv
```

可以通过环境变量调整：

```bash
SIZES="512x512x512 1024x1024x1024" BLOCKS="16x16 32x8" VARIANTS="tiled" REPEAT=20 TILE_K=32 bash benchmark.sh
```

## 当前实测结果

当前 `results/gemm_performance.csv` 已在 `NVIDIA GeForce RTX 4090` 上完成优化后测试，所有配置均通过抽样校验，`status=PASS`。

| 矩阵规模 `m x n x k` | 最快 naive | 最快 tiled | 结论 |
|---|---:|---:|---|
| `512x512x512` | 4504.19 GFLOPS (`32x16`) | 4599.02 GFLOPS (`16x16`) | tiled 小幅领先 |
| `1024x1024x1024` | 5092.65 GFLOPS (`32x16`) | 6326.25 GFLOPS (`16x16`) | tiled 提升明显 |
| `2048x2048x2048` | 5157.78 GFLOPS (`32x8`) | 6870.84 GFLOPS (`16x16`) | tiled 约 1.33x |
| `512x1024x512` | 4753.96 GFLOPS (`32x16`) | 4796.78 GFLOPS (`16x16`) | tiled 略优 |
| `1024x512x2048` | 5101.32 GFLOPS (`32x16`) | 6543.38 GFLOPS (`16x16`) | tiled 约 1.28x |

当前源码中的 `tiled` 版本已经升级为寄存器分块版本：每个线程计算 4 个连续输出列，shared memory 中的 `A` tile 会被更多输出列复用。优化后 `16x16` tiled 在多数规模下成为最快配置。

## 实现要点

- `naive` 版本采用二维 grid/block，每个线程计算 `C` 的一个元素，直接从 global memory 读取 `A` 的一行和 `B` 的一列。
- `tiled` 版本沿公共维度 `n` 分块，将 `A` 和 `B` 的 tile 放入 shared memory，并让每个线程计算 4 个连续输出列，减少 global memory 重复读取和同步开销摊销。
- 数据划分采用二维输出块：`blockIdx.x` 对应 `C` 的列块，`blockIdx.y` 对应 `C` 的行块。
- 不同线程块大小会影响 occupancy、访存合并、shared memory 使用量和边界线程比例。
- `tile_k` 会影响 shared memory 复用程度和同步次数，通常可以从 `16`、`32`、`64` 中比较。
