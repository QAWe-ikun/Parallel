# Task1 - MPI 点对点通信实现并行矩阵乘法

使用 MPI 点对点通信（MPI_Send / MPI_Recv）实现并行通用矩阵乘法 C = A × B。

## 目录结构

```
task1/
├── src/
│   └── mpi_matrix_mul.c    # MPI 点对点通信矩阵乘法源码
├── bin/
│   └── mpi_matrix_mul.exe  # 编译产物
├── compile.bat              # 编译脚本
└── benchmark.ps1            # 性能测试脚本（自动生成完整表格）
```

## 编译方法

```cmd
cd task1
compile.bat
```

使用 MSVC 编译器 + MS-MPI SDK 编译。

## 运行方法

### 命令行参数

```cmd
mpiexec -n <进程数> bin\mpi_matrix_mul.exe <m> <n> <k>
mpiexec -n <进程数> bin\mpi_matrix_mul.exe <size>  # m=n=k=size
```

### 示例

```cmd
# 2 个进程，128×128×128 矩阵
mpiexec.exe -n 2 bin\mpi_matrix_mul.exe 128

# 16 个进程，2048×2048×2048 矩阵
mpiexec.exe -n 16 bin\mpi_matrix_mul.exe 2048
```

### 自动性能测试

运行 `benchmark.ps1` 会自动测试所有进程数（2,4,8,16）和矩阵规模（128,256,512,1024,2048）的组合：

```powershell
& .\benchmark.ps1
```

## 架构设计

### Master-Worker 模型

- **Rank 0 (Master)**: 生成随机矩阵 A 和 B，将 B 完整发送给每个 worker，将 A 按行分块发送给各 worker，收集各 worker 返回的 C 子块并拼接为完整结果
- **Rank 1~P-1 (Worker)**: 接收 B 和 A 子块，计算 C_sub = A_sub × B，将结果发送回 Master

### 通信流程

```
Master (Rank 0)                          Workers (Rank 1~P-1)
════════════════════                     ════════════════════
生成 A[m×n], B[n×k]                      等待接收数据
  │ Isend(header) ────────────────────────→ │
  │ Isend(B) ──────────────────────────────→ │
  │ Isend(A_sub) ──────────────────────────→ │
  │   Waitall                              │  Recv(header), Recv(B), Recv(A_sub)
  │                                        │  计算 C_sub = A_sub × B
  │ ←─────────────────────────────────────── │ Send(compute_time)
  │ ←─────────────────────────────────────── │ Send(C_sub)
  │ Recv(C_sub) 依次收集
拼接完整 C[m×k]
输出结果、验证正确性、分析性能
```

### 性能特点

| 矩阵规模      | 最佳进程数 | 加速比    | 说明            |
| --------- | ----- | ------ | ------------- |
| 128×128   | 16    | 12.07x | 通信开销低，扩展性好    |
| 256×256   | 16    | 12.88x | 同上            |
| 512×512   | 8     | 5.69x  | 8 进程达峰值       |
| 1024×1024 | 4     | 2.68x  | 超过 4 进程加速不再明显 |
| 2048×2048 | 2     | 1.74x  | 大矩阵通信成为瓶颈     |

详细分析见 [report/实验报告.md](./report/实验报告.md)
