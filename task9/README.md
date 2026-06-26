# Task9: CUDA Hello World 与矩阵转置

本实验根据 `docs/9-CUDA矩阵转置.docx` 完成两个 CUDA 入门任务：

1. CUDA Hello World：输入 `n m k`，启动 `n` 个线程块，每个线程块为 `m x k` 个线程，设备端每个线程输出自己的块编号和二维线程编号，主线程输出 host 端问候。
2. CUDA 矩阵转置：输入矩阵规模 `n`，随机生成 `n x n` 矩阵 `A`，在 GPU 上计算转置矩阵 `AT`，输出计时、带宽、校验误差，并支持导出矩阵。

## 文件结构

```text
task9/
├── src/
│   ├── cuda_hello_world.cu
│   └── cuda_matrix_transpose.cu
├── results/
├── report/
├── compile.sh
├── benchmark.sh
└── README.md
```

## 编译

Linux / EasyHPC CUDA 环境：

```bash
cd task9
bash compile.sh
```

文档中给出的 RTX 3090 属于 Ampere 架构，脚本默认使用 `sm_86`。如果运行环境不同，可以指定：

```bash
CUDA_ARCH=sm_75 bash compile.sh
```

## CUDA Hello World

```bash
./bin/cuda_hello_world 2 4 3
```

参数含义：

```text
cuda_hello_world <num_blocks> <block_dim_x> <block_dim_y>
```

三个参数均按实验要求限制在 `[1, 32]`。线程输出顺序通常没有稳定规律，因为不同线程块和线程的调度由 GPU 硬件决定，`printf` 的实际刷新顺序也不等同于逻辑线程编号顺序。

## CUDA 矩阵转置

```bash
./bin/cuda_matrix_transpose 1024 16 16 both 20
```

参数含义：

```text
cuda_matrix_transpose [n] [block_x] [block_y] [variant] [repeat] [options]
```

- `n`: 矩阵规模，实验建议范围为 `[512, 2048]`
- `block_x`, `block_y`: 线程块维度
- `variant`: `naive`、`tiled` 或 `both`
- `repeat`: 计时重复次数，输出平均 kernel 时间
- `--print`: 当 `n <= 16` 时打印矩阵 `A` 和 `AT`
- `--dump <csv>`: 将完整矩阵 `A` 和 `AT` 写入文件
- `--seed <value>`: 设置随机种子

示例：

```bash
./bin/cuda_matrix_transpose 512 16 16 naive 20
./bin/cuda_matrix_transpose 2048 32 8 tiled 50
./bin/cuda_matrix_transpose 512 16 16 tiled 20 --dump results/matrix_dump.csv
```

输出字段包括：

```text
variant=...
n=...
block_x=...
block_y=...
repeat=...
avg_time_ms=...
bandwidth_gbps=...
max_error=...
status=PASS
```

程序会在 CPU 端计算参考转置，并用最大绝对误差校验 GPU 结果。

## 性能测试

```bash
bash benchmark.sh
```

默认测试：

- 矩阵规模：`512 1024 2048`
- 线程块：`8x8 16x16 32x8 32x16`
- 访存方式：`naive` 与 `tiled`
- 重复次数：`20`

结果写入：

```text
results/transpose_performance.csv
```

可以通过环境变量调整：

```bash
SIZES="512 1024" BLOCKS="16x16 32x8" VARIANTS="tiled" REPEAT=50 bash benchmark.sh
```

## 当前实测结果

当前 `results/transpose_performance.csv` 已在 `NVIDIA GeForce RTX 4090` 上完成一组测试，所有配置均通过 CPU 参考结果校验，`max_error=0`。

| 规模 | 最快 naive | 最快 tiled | 说明 |
|---:|---:|---:|---|
| 512 | 0.009165 ms (`32x8`) | 0.030157 ms (`32x16`) | 小规模下测量波动和调度开销占比较高 |
| 1024 | 0.013158 ms (`16x16`) | 0.033280 ms (`32x8`) | naive 部分配置受缓存/调度影响较明显 |
| 2048 | 0.061030 ms (`8x8`) | 0.014234 ms (`32x16`) | tiled shared-memory 版本优势明显 |

报告要点位于 `report/report.md`，可直接结合 CSV 完成实验报告模板。

## 实现要点

- `naive` 版本采用二维 grid/block，每个线程处理一个矩阵元素，读取连续但写入跨行，转置写回存在非合并访存。
- `tiled` 版本用 shared memory 缓存一个方形 tile，并额外加 1 列 padding 减少 shared memory bank conflict；读写都更接近合并访存。
- 数据划分采用二维块划分：`blockIdx.x/blockIdx.y` 定位 tile 或矩阵块，`threadIdx.x/threadIdx.y` 定位块内元素。
- 不同块大小会影响 occupancy、访存合并、shared memory 使用量和边界线程比例，通常 `16x16`、`32x8`、`32x16` 是较适合比较的配置。
