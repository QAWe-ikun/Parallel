# Task6: Heated Plate — 基于 Pthreads 线程池 parallel_for 的并行应用

## 问题描述

稳态热传导模拟：500×500 规则网格上的 Jacobi 迭代

$$W_{i,j}^{t+1} = \frac{1}{4}(W_{i-1,j}^t + W_{i+1,j}^t + W_{i,j-1}^t + W_{i,j+1}^t)$$

**边界条件：** 上边界 = 0°C，其余三边 = 100°C  
**收敛条件：** 相邻两次迭代的最大差值 ≤ 0.001

## 文件结构

```
task6/
├── src/
│   ├── heated_plate_pthreads.c     ← Pthreads 线程池版本（主程序）
│   ├── heated_plate_openmp.c       ← OpenMP 参考实现（对比基准）
│   ├── parallel_for_pool.h         ← 线程池版 parallel_for API 头文件
│   └── parallel_for_pool.c         ← 线程池版 parallel_for 实现
├── lib/
│   └── libparallel_for_pool.so     ← 动态链接库（编译后生成）
├── bin/
│   ├── heated_plate_pthreads       ← Pthreads 可执行文件
│   └── heated_plate_openmp         ← OpenMP 可执行文件
├── report/
│   └── 实验报告.md                  ← 实验报告
├── compile.bat                     ← Windows 编译脚本
├── compile.sh                      ← Linux/WSL 编译脚本
├── benchmark.ps1                   ← Windows 性能测试脚本
├── benchmark.sh                    ← Linux/WSL 性能测试脚本
└── README.md
```

## 编译

### Linux / WSL

```bash
chmod +x compile.sh
./compile.sh
```

## 运行

```bash
# OpenMP 参考实现
./bin/heated_plate_openmp [num_threads]

# Pthreads parallel_for_pool（线程池版本）
./bin/heated_plate_pthreads [num_threads] [schedule] [chunk_size]
```

**参数说明：**
- `num_threads`: 线程数，默认 4
- `schedule`: 调度策略，0=static（默认），1=dynamic，2=guided
- `chunk_size`: 块大小，默认 1

## 性能测试

```bash
# Linux/WSL
./benchmark.sh
```

## 核心改进：线程池

与 task5 的 `parallel_for` 不同，`parallel_for_pool` 使用**线程池模式**：

| 特性 | task5 parallel_for | task6 parallel_for_pool |
|---|---|---|
| 线程管理 | 每次调用创建/销毁 | 首次创建，后续复用 |
| 同步机制 | pthread_join | 条件变量 (condvar) |
| 适合场景 | 大粒度（调用次数少） | 小粒度（频繁调用） |
| heated_plate 4T | ~11s（无加速） | ~2s（5x 加速） |

## 改造要点

将 OpenMP 的 `#pragma omp parallel for` 替换为 `parallel_for_advanced()` 调用：

| OpenMP 构造 | Pthreads parallel_for_pool 替换 |
|---|---|
| `#pragma omp parallel for` | `parallel_for_advanced(start, end, 1, functor, &args, &config)` |
| `reduction(+:var)` | 手动局部求和 + 串行归约 |
| `reduction(max:var)` | `local_diffs` 数组 + `for` 循环取最大值 |

程序结束前需调用 `parallel_for_pool_destroy()` 清理线程池资源。
