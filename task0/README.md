# 矩阵乘法性能优化实现

## 目录结构

```
task0/
├── src/
│   ├── matrix.py              # Python 参考实现 (基准)
│   ├── matrix_mul_all_mkl.cpp # 全部版本对比 (含 Strassen, MKL)
│   ├── matrix_mul_ikj_opt.cpp # 循环顺序优化 (-O3)
│   └── compile.bat            # 编译脚本
├── bin/
│   ├── matrix_all_mkl.exe     # 全部版本对比
│   └── matrix_ikj_opt.exe     # 循环顺序优化 (-O3)
├── obj/
│   └── matrix_all_mkl.obj
└── README.md
```

## 编译方法

```batch
cd .\task0
compile.bat
```

## 运行方法

### 参数说明
- **M**: 矩阵 A 的行数 (512-2048)
- **K**: 矩阵 A 的列数/B 的行数 (512-2048)
- **N**: 矩阵 B 的列数 (512-2048)

### 命令行

```batch
cd .\task0

# Python 基准版本
python src\matrix.py [M] [K] [N]

# 全部版本对比 (含 Strassen, MKL)
bin\matrix_all_mkl.exe [version] [M] [K] [N]

# 循环顺序优化版本 (-O3 -march=native)
bin\matrix_ikj_opt.exe [M] [K] [N]
```

### version 参数 (matrix_all_mkl.exe)

| 值 | 版本 |
|----|------|
| 0 | ijk 基础版本 |
| 1 | ikj 循环顺序优化 |
| 2 | Strassen 分治算法 |
| 3 | Intel MKL 版本 |
| 4 | 全部版本对比 (默认) |

### 运行示例

```batch
# Python 基准 (1024x1024x1024)
python src\matrix.py 1024

# 全部版本对比
bin\matrix_all_mkl.exe 4 1024

# 循环顺序优化版本
bin\matrix_ikj_opt.exe 1024

# 单独运行某个版本
bin\matrix_all_mkl.exe 0 1024    # ijk 基础版
bin\matrix_all_mkl.exe 1 1024    # ikj 循环优化
bin\matrix_all_mkl.exe 2 1024    # Strassen 算法
bin\matrix_all_mkl.exe 3 1024    # Intel MKL
```

## 性能对比 (1024³)

| 版本 | 编译选项 | 时间 | 加速比 (vs Python) | 相对加速比 |
|------|---------|------|-------------------|-----------|
| **Python 基准** | 解释执行 | **414.997 s** | **1x** | **1x** |
| ijk 基础版 | -Od | 5.09 s | **81.5x** | 81.5x |
| ikj 循环优化 | -Od | 2.76 s | **150x** | 1.84x |
| Strassen 算法 | -Od | 2.19 s | **189x** | 1.26x |
| ikj 循环优化 | -O3 -march=native | 0.15 s | **2767x** | 18.4x |
| Intel MKL | -O2 | 0.014 s | **29643x** | 10.7x |

### 优化效果分析

```
Python (解释型)
    │
    ▼ 编译为 C++ (-Od)
ijk 基础版 ─────────────────────────────  81.5x
    │
    ▼ 循环顺序优化 (ikj)
ikj 循环优化 (-Od) ─────────────────────  150x (1.84x 提升)
    │
    ▼ Strassen 分治算法
Strassen 算法 (-Od) ────────────────────  189x (1.26x 提升)
    │
    ▼ 编译优化 (-O3 -march=native)
ikj 循环优化 (-O3) ─────────────────────  2767x (18.4x 提升)
    │
    ▼ 使用 MKL 库
Intel MKL (-Od) ────────────────────────  29643x (10.7x 提升)
```

## 优化技术说明

### 1. 编译优化 (C++ vs Python)
- **Python**: 解释执行，动态类型，GIL 锁
- **C++**: 编译为机器码，静态类型，无 GIL 开销
- **提升**: ~80x

### 2. 循环顺序优化 (ikj)
- **原理**: 改变循环顺序 i→j→k 为 i→k→j
- **效果**: 缓存 A[i][k]，连续访问 C[i][j] 和 B[k][j]
- **提升**: 1.84x

### 3. Strassen 分治算法
- **原理**: 将 2x2 分块矩阵乘法从 8 次乘法减少到 7 次
- **时间复杂度**: O(n^log₂7) ≈ O(n^2.81)，优于传统 O(n³)
- **提升**: 1.26x (vs ikj)

### 4. 编译优化 (-O3 -march=native)
- **向量化 (AVX2)**: 一次处理 4 个 double
- **循环展开**: 减少循环控制开销
- **寄存器优化**: 减少内存访问
- **提升**: ~18x

### 5. Intel MKL
- **SIMD 指令**: AVX2/AVX-512
- **分块算法**: 多级缓存优化
- **多线程**: 并行计算
- **提升**: ~30000x (vs Python)

## 结论

**从 Python 到优化后的 C++，性能提升近 30000 倍！**

| 优化阶段 | 累计加速比 |
|---------|-----------|
| Python 基准 | 1x |
| → C++ 编译 | 81x |
| → 循环优化 (ikj) | 150x |
| → Strassen 算法 | 189x |
| → 编译优化 (-O3) | 2767x |
| → Intel MKL | 29643x |
