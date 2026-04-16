# Task2 - MPI 集合通信实现并行矩阵乘法

使用 MPI 集合通信（MPI_Bcast, MPI_Scatterv, MPI_Gatherv）实现并行通用矩阵乘法 C = A × B，并尝试不同的数据/任务划分方式。

## 目录结构

```
task2/
├── src/
│   ├── mpi_collective_mat_mul.c      # MPI 集合通信矩阵乘法源码（使用MPI_Type_create_struct）
│   ├── mpi_col_distrib_mat_mul.c     # MPI 列划分矩阵乘法源码（对比不同任务划分方式）
│   └── serial_mat_mul.c              # 串行矩阵乘法（从 task1 复制）
├── bin/
│   ├── mpi_collective_mat_mul.exe    # 集合通信版编译产物
│   ├── mpi_col_distrib_mat_mul.exe   # 列划分版编译产物
│   └── serial_mat_mul.exe            # 串行版编译产物
├── obj/
│   └── *.obj                         # 中间目标文件
├── compile.bat                       # 编译脚本
└── benchmark.ps1                     # 性能测试脚本
```

## 编译方法

```cmd
cd task2
compile.bat
```

使用 MSVC 编译器 + MS-MPI SDK 编译。

## 运行方法

### 命令行参数

```cmd
mpiexec -n <进程数> bin\mpi_collective_mat_mul.exe <m> <n> <k>
mpiexec -n <进程数> bin\mpi_collective_mat_mul.exe <size>  # m=n=k=size
mpiexec -n <进程数> bin\mpi_col_distrib_mat_mul.exe <m> <n> <k>  # 列划分版本
mpiexec -n <进程数> bin\mpi_col_distrib_mat_mul.exe <size>       # 列划分版本
```

### 示例

```cmd
# 2 个进程，128×128×128 矩阵，使用行划分
mpiexec.exe -n 2 bin\mpi_collective_mat_mul.exe 128

# 2 个进程，128×128×128 矩阵，使用列划分（对比实验）
mpiexec.exe -n 2 bin\mpi_col_distrib_mat_mul.exe 128

# 16 个进程，2048×2048×2048 矩阵
mpiexec.exe -n 16 bin\mpi_collective_mat_mul.exe 2048
```

### 自动性能测试

运行 `benchmark.ps1` 会自动测试所有进程数（1,2,4,8,16）和矩阵规模（128,256,512,1024,2048）的组合：

```powershell
& .\benchmark.ps1
```

## 架构设计

### 集合通信模型

使用 MPI 集合通信函数实现高效的并行计算：

- **Master (Rank 0)**: 生成随机矩阵 A 和 B，使用 `MPI_Bcast` 广播 B 给所有进程，使用 `MPI_Scatterv` 分发 A 的行块，使用 `MPI_Gatherv` 收集各进程的 C 子块并拼接为完整结果
- **Workers (Rank 1~P-1)**: 接收 B 和 A 子块，计算 C_sub = A_sub × B，将结果通过 `MPI_Gatherv` 发送回 Master

### 不同数据划分方式

1. **行划分 (Row-wise Distribution)** - mpi_collective_mat_mul.exe:
   - 将矩阵 A 按行划分给不同进程
   - 每个进程获得完整的 B 矩阵
   - 计算 A_sub × B 得到 C 的对应行

2. **列划分 (Column-wise Distribution)** - mpi_col_distrib_mat_mul.exe:
   - 将矩阵 B 按列划分给不同进程  
   - 每个进程获得完整的 A 矩阵
   - 计算 A × B_sub 得到 C 的对应列

### 通信流程

#### 行划分版本：
```
Master (Rank 0)                          Workers (Rank 1~P-1)
════════════════════                     ════════════════════
生成 A[m×n], B[n×k]                      同步等待
  │ Bcast(B) ───────────────────────────→ │ Bcast(B)
  │ Scatterv(A_sub) ────────────────────→ │ Scatterv(A_sub)
  │                                      │ 计算 C_sub = A_sub × B
  │ ←───────────────────────────────────── │ Gatherv(C_sub)
  │ Gatherv(C_sub) 收集结果
拼接完整 C[m×k]
输出结果、验证正确性、分析性能
```

#### 列划分版本：
```
Master (Rank 0)                          Workers (Rank 1~P-1)
════════════════════                     ════════════════════
生成 A[m×n], B[n×k]                      同步等待
  │ Bcast(A) ───────────────────────────→ │ Bcast(A)
  │ Scatterv(B_sub) ────────────────────→ │ Scatterv(B_sub)
  │                                      │ 计算 C_sub = A × B_sub
  │ ←───────────────────────────────────── │ Gatherv(C_sub)
  │ Gatherv(C_sub) 收集结果
拼接完整 C[m×k]
输出结果、验证正确性、分析性能
```

### 使用 MPI_Type_create_struct 聚合参数

- 使用 `MPI_Type_create_struct` 将矩阵维度 (m, n, k)、块大小 (rows, cols) 和计算时间聚合为单个结构体进行通信
- 减少了通信次数，提高了通信效率
- 代码更简洁，易于维护

### 性能特点

| 特性 | 行划分版本 | 列划分版本 |
|------|-----------|-----------|
| 通信模式 | Broadcast A + Scatterv B | Broadcast A + Scatterv B (列划分) |
| 内存使用 | 每进程存储完整 B 矩阵 | 每进程存储完整 A 矩阵 |
| 扩展性 | 适合 A >> B 场景 | 适合 B >> A 场景 |
| 通信效率 | 高效广播 B 矩阵 | 高效分发 B 列块 |

详细分析见 [report/实验报告.md](./report/实验报告.md)