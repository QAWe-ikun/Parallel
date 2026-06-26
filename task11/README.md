# Task11: CUDA 卷积计算

本实验根据 `docs/11-CUDA卷积计算.docx` 实现 CNN 风格 2D 卷积。这里的卷积不翻转 filter，不考虑 bias，输入通道数固定为 `3`。

程序支持三种实现：

- `direct`: 直接滑窗卷积。
- `im2col`: 先将输入展开为列矩阵，再调用 tiled GEMM。
- `cudnn`: 使用 cuDNN forward convolution。WSL 环境需要安装并可链接 `cudnn.h/libcudnn`。

## 编译

```bash
cd task11
bash compile.sh
```

编译脚本默认链接 `-lcudnn`，并会自动查找 `$CONDA_PREFIX/include`、`$CONDA_PREFIX/targets/x86_64-linux/include`、以及 `site-packages/nvidia/cudnn/{include,lib}` 等常见 conda/pip 安装位置。如果 cuDNN 安装在其他路径，可以手动指定：

```bash
CUDNN_INCLUDE_DIR=/path/to/include CUDNN_LIB_DIR=/path/to/lib bash compile.sh
```

## 运行

```bash
./bin/cuda_conv2d 512 3 1 1 all 10 16
```

参数含义：

```text
cuda_conv2d [input_size] [kernel_size] [stride] [padding] [variant] [repeat] [filters] [options]
```

- `input_size`: 输入高宽，输入为 `3 x input_size x input_size`
- `kernel_size`: filter 高宽，实验要求常用 `3`
- `stride`: 步幅，实验要求比较 `1,2,3`
- `padding`: 零填充大小，`kernel=3` 时常用 `1`
- `variant`: `direct`、`im2col`、`cudnn` 或 `all`
- `repeat`: 计时重复次数
- `filters`: 输出 filter 个数，默认 `1`；benchmark 默认 `16`
- `--block <x>x<y>`: CUDA block 大小
- `--tile-k <value>`: im2col GEMM 的归约维度 tile
- `--verify none|sample|full`: 校验模式
- `--dump <csv>`: 导出 input/filter/output

示例：

```bash
./bin/cuda_conv2d 256 3 1 1 direct 10 16 --block 16x16
./bin/cuda_conv2d 512 3 2 1 im2col 10 16 --block 16x16
./bin/cuda_conv2d 128 3 1 1 all 3 2 --verify full --dump results/small_dump.csv
```

## 性能测试

```bash
bash benchmark.sh
```

默认测试：

- 输入规模：`256 512 1024`
- stride：`1 2 3`
- block：`8x8 16x16 32x8`
- variants：`direct im2col cudnn`
- filters：`16`
- repeat：`100`，用于降低微秒级 kernel 的计时量化误差

结果写入：

```text
results/conv_performance.csv
```

## 当前实测结果

当前 `results/conv_performance.csv` 已在 `NVIDIA GeForce RTX 4090` 上完成测试，`direct`、`im2col`、`cudnn` 全部配置均为 `PASS`。

| 输入规模 | stride | 最快 direct | 最快 im2col | 最快 cuDNN | 最快实现 |
|---:|---:|---:|---:|---:|---|
| 256 | 1 | 2524.93 GOPS (`32x8`) | 2102.51 GOPS (`16x16`) | 1562.72 GOPS (`32x8`) | direct |
| 256 | 2 | 1665.54 GOPS (`16x16`) | 1047.27 GOPS (`16x16/32x8`) | 661.44 GOPS (`16x16`) | direct |
| 256 | 3 | 990.54 GOPS (`32x8`) | 452.20 GOPS (`8x8`) | 313.59 GOPS (`16x16`) | direct |
| 512 | 1 | 3038.24 GOPS (`32x8`) | 2652.09 GOPS (`16x16`) | 5642.45 GOPS (`32x8`) | cuDNN |
| 512 | 2 | 2596.82 GOPS (`32x8`) | 2103.01 GOPS (`16x16`) | 2479.64 GOPS (`16x16`) | direct |
| 512 | 3 | 1495.28 GOPS (`16x16`) | 1495.28 GOPS (`8x8`) | 1068.06 GOPS (`32x8`) | direct / im2col |
| 1024 | 1 | 3323.58 GOPS (`16x16`) | 2736.71 GOPS (`16x16`) | 5090.82 GOPS (`8x8`) | cuDNN |
| 1024 | 2 | 3128.49 GOPS (`32x8`) | 2658.46 GOPS (`16x16`) | 4766.90 GOPS (`16x16`) | cuDNN |
| 1024 | 3 | 1753.00 GOPS (`32x8`) | 2259.45 GOPS (`8x8`) | 3036.57 GOPS (`8x8`) | cuDNN |

总体趋势：小规模输入下 direct 的滑窗实现更轻量，避免了 im2col 展开和 cuDNN 调度开销；输入增大后，cuDNN 的优化 kernel 开始占优。stride 增大时输出元素减少，im2col 展开开销相对更敏感，但在 `1024, stride=3` 这类配置下仍能超过 direct。

## 实现要点

- direct 版本中每个线程计算一个输出元素 `(filter, y, x)`。
- im2col 版本先生成大小为 `(3*K*K) x (OH*OW)` 的列矩阵，再计算 `filter_matrix(F x 3K^2) * columns(3K^2 x OH*OW)`。
- im2col 的 GEMM 使用 task10 中优化过的“每线程 4 个连续输出列”的 tiled kernel。
- 输出尺寸为 `(input_size + 2*padding - kernel_size) / stride + 1`。
- cuDNN 版本使用 NCHW tensor/filter descriptor 和 `CUDNN_CROSS_CORRELATION`，与 CNN 卷积定义一致。
