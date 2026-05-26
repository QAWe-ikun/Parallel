# Task7: MPI 并行 FFT 与 parallel_for 并行应用分析

## 实验内容

### Part 1: MPI 并行 FFT

将串行快速傅里叶变换（FFT）使用 MPI 进行并行化：

- **并行策略**: 在 `step()` 函数的 j 循环上做数据分配
- **通信机制**: `MPI_Allreduce(SUM)` 合并各进程的部分结果
- **内存布局**: 单个 `local_y` 缓冲区解决 c/d 视图重叠问题

### Part 2: parallel_for 并行应用分析

对 task6 的 heated_plate_pthreads 应用进行：
- 不同问题规模（N = 64, 128, 256, 512）和线程数（T = 1, 2, 4, 8）的性能测试
- STATIC / DYNAMIC / GUIDED 三种调度策略对比
- Valgrind massif 内存消耗分析（含 --stacks=yes 栈内存）

## 文件结构

```
task7/
├── src/
│   └── fft_mpi.c                    ← MPI 并行 FFT 实现
├── docs/
│   ├── fft_serial.cpp               ← 串行 FFT 参考代码（C++）
│   └── fft_openmp.c.txt             ← OpenMP FFT 参考代码
├── results/                         ← 测试结果（.gitignore 排除）
│   ├── performance_STATIC.csv
│   ├── performance_DYNAMIC.csv
│   ├── performance_GUIDED.csv
│   ├── massif_report.txt
│   └── mpi_p{1,2,4,8}_output.txt
├── report/
│   └── report.md                    ← 实验报告
├── compile.sh                       ← 编译脚本
├── benchmark.sh                     ← MPI FFT 性能测试脚本
├── parallel_for_analysis.sh         ← parallel_for 性能分析脚本
└── README.md
```

## 编译

```bash
# 需要 MPI 环境（OpenMPI）
sudo apt install libopenmpi-dev openmpi-bin

# 编译
bash task7/compile.sh
```

## 运行

```bash
# 串行 FFT
./task7/bin/fft_serial

# MPI 并行 FFT
mpirun --oversubscribe -np 4 ./task7/bin/fft_mpi

# MPI FFT 性能测试（1/2/4/8 进程）
bash task7/benchmark.sh

# parallel_for 性能分析（三种调度 + Valgrind massif）
bash task7/parallel_for_analysis.sh
```

## 关键实验结论

| 项目 | 结论 |
|------|------|
| MPI FFT | 通信量与计算量同阶 O(N log N)，单机 MPI 无法加速 |
| STATIC 调度 | 均匀负载下最优，N=512 T=8 达到 5.03x 加速比 |
| DYNAMIC/GUIDED | atomic_fetch_add 额外开销约 1.6s，比 STATIC 慢 1.5-1.7x |
| 问题规模 | 并行加速只在 N≥256 时出现，N 越大加速比越好 |
| 内存消耗 | 峰值 ~1 MB（256×256 网格），运行期间恒定不变 |
