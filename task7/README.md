# Task7: MPI并行FFT与性能分析

## 实验内容

### 1. MPI并行FFT
将串行快速傅里叶变换（FFT）代码使用 MPI 进行并行化。

**文件结构**：
```
task7/
├── src/
│   └── fft_mpi.c              ← MPI 并行 FFT 实现
├── docs/
│   ├── fft_serial.cpp         ← 串行 FFT 参考代码
│   └── fft_openmp.c.txt       ← OpenMP FFT 参考代码
├── bin/
│   ├── fft_serial             ← 串行可执行文件（编译后生成）
│   └── fft_mpi                ← MPI 并行可执行文件（编译后生成）
├── report/
│   └── 实验报告.md             ← 实验报告
├── compile.sh                 ← 编译脚本
├── benchmark.sh               ← 性能测试脚本
└── README.md
```

### 2. parallel_for 性能分析
对 Lab6 的 heated_plate 应用进行：
- 不同问题规模（N）和并行规模（线程数）的性能测试
- Valgrind massif 内存消耗分析

## 编译

```bash
# 需要 MPI 环境
bash task7/compile.sh
```

## 运行

```bash
# 串行 FFT
./task7/bin/fft_serial

# MPI 并行 FFT（4 进程）
mpirun -np 4 ./task7/bin/fft_mpi

# MPI 并行 FFT（8 进程）
mpirun -np 8 ./task7/bin/fft_mpi
```

## Valgrind Massif 内存分析

```bash
# 分析 task6 heated_plate_openmp
bash task6/valgrind_massif.sh

# 查看结果
ms_print task6/massif_output/massif_t4_s500.out
```

## 实验报告

详见 `report/实验报告.md`
