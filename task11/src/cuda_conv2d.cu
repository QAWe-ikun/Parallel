/*
 * Task11: CUDA 2D CNN-style convolution.
 *
 * Input:  C x H x W, where C is fixed to 3.
 * Filter: F x C x K x K.
 * Output: F x OH x OW.
 *
 * Variants:
 *   - direct: sliding-window CUDA convolution
 *   - im2col: im2col transform + tiled GEMM
 *   - cudnn: cuDNN forward convolution
 */

#include <cuda_runtime.h>
#include <cudnn.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CUDA_CHECK(call)                                                          \
    do {                                                                         \
        cudaError_t err__ = (call);                                               \
        if (err__ != cudaSuccess) {                                               \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__,          \
                         __LINE__, cudaGetErrorString(err__));                   \
            std::exit(EXIT_FAILURE);                                              \
        }                                                                        \
    } while (0)

#define CUDNN_CHECK(call)                                                         \
    do {                                                                         \
        cudnnStatus_t status__ = (call);                                          \
        if (status__ != CUDNN_STATUS_SUCCESS) {                                   \
            std::fprintf(stderr, "cuDNN error at %s:%d: %s\n", __FILE__,         \
                         __LINE__, cudnnGetErrorString(status__));               \
            std::exit(EXIT_FAILURE);                                              \
        }                                                                        \
    } while (0)

#define CHANNELS 3
#define GEMM_COLS_PER_THREAD 4

enum class Variant {
    Direct,
    Im2col,
    Cudnn,
    All,
};

enum class VerifyMode {
    None,
    Sample,
    Full,
};

struct Options {
    int input_size = 512;
    int kernel_size = 3;
    int stride = 1;
    int padding = 1;
    int filters = 1;
    int block_x = 16;
    int block_y = 16;
    int tile_k = 32;
    int repeat = 10;
    int samples = 2048;
    unsigned int seed = 2026U;
    Variant variant = Variant::All;
    VerifyMode verify = VerifyMode::Sample;
    bool print_tensor = false;
    const char *dump_path = nullptr;
};

struct Shape {
    int h;
    int w;
    int out_h;
    int out_w;
    int im2col_rows;
    int im2col_cols;
};

__global__ void conv_direct_kernel(const float *input, const float *filter,
                                   float *output, int h, int w, int out_h,
                                   int out_w, int kernel_size, int stride,
                                   int padding, int filters) {
    int ox = blockIdx.x * blockDim.x + threadIdx.x;
    int oy = blockIdx.y * blockDim.y + threadIdx.y;
    int f = blockIdx.z;

    if (f >= filters || oy >= out_h || ox >= out_w) {
        return;
    }

    float sum = 0.0f;
    for (int c = 0; c < CHANNELS; c++) {
        for (int ky = 0; ky < kernel_size; ky++) {
            int iy = oy * stride + ky - padding;
            if (iy < 0 || iy >= h) {
                continue;
            }
            for (int kx = 0; kx < kernel_size; kx++) {
                int ix = ox * stride + kx - padding;
                if (ix < 0 || ix >= w) {
                    continue;
                }
                int input_idx = (c * h + iy) * w + ix;
                int filter_idx =
                    ((f * CHANNELS + c) * kernel_size + ky) * kernel_size + kx;
                sum += input[input_idx] * filter[filter_idx];
            }
        }
    }

    output[(f * out_h + oy) * out_w + ox] = sum;
}

__global__ void im2col_kernel(const float *input, float *columns, int h, int w,
                              int out_h, int out_w, int kernel_size, int stride,
                              int padding) {
    int rows = CHANNELS * kernel_size * kernel_size;
    int cols = out_h * out_w;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;

    if (idx >= total) {
        return;
    }

    int row = idx / cols;
    int col = idx - row * cols;
    int ox = col % out_w;
    int oy = col / out_w;

    int kernel_area = kernel_size * kernel_size;
    int c = row / kernel_area;
    int rem = row - c * kernel_area;
    int ky = rem / kernel_size;
    int kx = rem - ky * kernel_size;

    int iy = oy * stride + ky - padding;
    int ix = ox * stride + kx - padding;

    float value = 0.0f;
    if (iy >= 0 && iy < h && ix >= 0 && ix < w) {
        value = input[(c * h + iy) * w + ix];
    }
    columns[idx] = value;
}

__global__ void gemm_tiled_cols4_kernel(const float *a, const float *b, float *c,
                                        int m, int n, int k, int tile_k) {
    extern __shared__ float shared[];

    int output_cols = blockDim.x * GEMM_COLS_PER_THREAD;
    float *tile_a = shared;
    float *tile_b = tile_a + blockDim.y * tile_k;

    int col0 = blockIdx.x * output_cols + threadIdx.x * GEMM_COLS_PER_THREAD;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int threads_per_block = blockDim.x * blockDim.y;
    int linear_tid = threadIdx.y * blockDim.x + threadIdx.x;
    float sum[GEMM_COLS_PER_THREAD] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int base = 0; base < n; base += tile_k) {
        int a_elements = blockDim.y * tile_k;
        for (int idx = linear_tid; idx < a_elements; idx += threads_per_block) {
            int local_row = idx / tile_k;
            int local_col = idx - local_row * tile_k;
            int global_row = blockIdx.y * blockDim.y + local_row;
            int global_col = base + local_col;
            tile_a[idx] = (global_row < m && global_col < n)
                              ? a[global_row * n + global_col]
                              : 0.0f;
        }

        int b_elements = tile_k * output_cols;
        for (int idx = linear_tid; idx < b_elements; idx += threads_per_block) {
            int local_row = idx / output_cols;
            int local_col = idx - local_row * output_cols;
            int global_row = base + local_row;
            int global_col = blockIdx.x * output_cols + local_col;
            tile_b[idx] = (global_row < n && global_col < k)
                              ? b[global_row * k + global_col]
                              : 0.0f;
        }

        __syncthreads();

        for (int p = 0; p < tile_k; p++) {
            float a_value = tile_a[threadIdx.y * tile_k + p];
#pragma unroll
            for (int i = 0; i < GEMM_COLS_PER_THREAD; i++) {
                sum[i] += a_value *
                          tile_b[p * output_cols +
                                 threadIdx.x * GEMM_COLS_PER_THREAD + i];
            }
        }

        __syncthreads();
    }

    if (row < m) {
#pragma unroll
        for (int i = 0; i < GEMM_COLS_PER_THREAD; i++) {
            int col = col0 + i;
            if (col < k) {
                c[row * k + col] = sum[i];
            }
        }
    }
}

static void print_usage(const char *program) {
    std::fprintf(stderr,
                 "Usage:\n"
                 "  %s [input_size] [kernel_size] [stride] [padding] [variant] "
                 "[repeat] [filters] [options]\n\n"
                 "Arguments:\n"
                 "  input_size   H and W of the input image, e.g. 512\n"
                 "  kernel_size  K of the K x K filter, assignment default is 3\n"
                 "  stride       convolution stride, commonly 1, 2, or 3\n"
                 "  padding      zero padding size\n"
                 "  variant      direct | im2col | cudnn | all\n"
                 "  repeat       timed repetitions\n"
                 "  filters      number of output filters, default 1\n\n"
                 "Options:\n"
                 "  --block <x>x<y>            CUDA block size, default 16x16\n"
                 "  --tile-k <value>           GEMM reduction tile, default 32\n"
                 "  --verify none|sample|full  default: sample\n"
                 "  --samples <count>          sample count for verification\n"
                 "  --print                    print tensors when dimensions are small\n"
                 "  --dump <csv>               write input/filter/output tensors\n"
                 "  --seed <value>             random seed, default 2026\n\n"
                 "Examples:\n"
                 "  %s 512 3 1 1 all 10 16\n"
                 "  %s 1024 3 2 1 im2col 20 32 --block 16x16\n",
                 program, program, program);
}

static Variant parse_variant(const char *value) {
    if (std::strcmp(value, "direct") == 0) {
        return Variant::Direct;
    }
    if (std::strcmp(value, "im2col") == 0) {
        return Variant::Im2col;
    }
    if (std::strcmp(value, "cudnn") == 0) {
        return Variant::Cudnn;
    }
    if (std::strcmp(value, "all") == 0) {
        return Variant::All;
    }
    std::fprintf(stderr, "Unknown variant: %s\n", value);
    std::exit(EXIT_FAILURE);
}

static const char *variant_name(Variant variant) {
    switch (variant) {
    case Variant::Direct:
        return "direct";
    case Variant::Im2col:
        return "im2col";
    case Variant::Cudnn:
        return "cudnn";
    case Variant::All:
        return "all";
    }
    return "unknown";
}

static VerifyMode parse_verify_mode(const char *value) {
    if (std::strcmp(value, "none") == 0) {
        return VerifyMode::None;
    }
    if (std::strcmp(value, "sample") == 0) {
        return VerifyMode::Sample;
    }
    if (std::strcmp(value, "full") == 0) {
        return VerifyMode::Full;
    }
    std::fprintf(stderr, "Unknown verify mode: %s\n", value);
    std::exit(EXIT_FAILURE);
}

static const char *verify_name(VerifyMode mode) {
    switch (mode) {
    case VerifyMode::None:
        return "none";
    case VerifyMode::Sample:
        return "sample";
    case VerifyMode::Full:
        return "full";
    }
    return "unknown";
}

static void parse_block(const char *value, int *block_x, int *block_y) {
    int x = 0;
    int y = 0;
    if (std::sscanf(value, "%dx%d", &x, &y) != 2 || x <= 0 || y <= 0) {
        std::fprintf(stderr, "Invalid block size: %s\n", value);
        std::exit(EXIT_FAILURE);
    }
    *block_x = x;
    *block_y = y;
}

static Options parse_options(int argc, char **argv) {
    Options options;

    if (argc > 1 && (std::strcmp(argv[1], "-h") == 0 ||
                     std::strcmp(argv[1], "--help") == 0)) {
        print_usage(argv[0]);
        std::exit(EXIT_SUCCESS);
    }

    if (argc > 1 && argv[1][0] != '-') {
        options.input_size = std::atoi(argv[1]);
    }
    if (argc > 2 && argv[2][0] != '-') {
        options.kernel_size = std::atoi(argv[2]);
    }
    if (argc > 3 && argv[3][0] != '-') {
        options.stride = std::atoi(argv[3]);
    }
    if (argc > 4 && argv[4][0] != '-') {
        options.padding = std::atoi(argv[4]);
    }
    if (argc > 5 && argv[5][0] != '-') {
        options.variant = parse_variant(argv[5]);
    }
    if (argc > 6 && argv[6][0] != '-') {
        options.repeat = std::atoi(argv[6]);
    }
    if (argc > 7 && argv[7][0] != '-') {
        options.filters = std::atoi(argv[7]);
    }

    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--block") == 0) {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "Error: --block requires a value.\n");
                std::exit(EXIT_FAILURE);
            }
            parse_block(argv[++i], &options.block_x, &options.block_y);
        } else if (std::strcmp(argv[i], "--tile-k") == 0) {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "Error: --tile-k requires a value.\n");
                std::exit(EXIT_FAILURE);
            }
            options.tile_k = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--verify") == 0) {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "Error: --verify requires a mode.\n");
                std::exit(EXIT_FAILURE);
            }
            options.verify = parse_verify_mode(argv[++i]);
        } else if (std::strcmp(argv[i], "--samples") == 0) {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "Error: --samples requires a count.\n");
                std::exit(EXIT_FAILURE);
            }
            options.samples = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--print") == 0) {
            options.print_tensor = true;
        } else if (std::strcmp(argv[i], "--dump") == 0) {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "Error: --dump requires a path.\n");
                std::exit(EXIT_FAILURE);
            }
            options.dump_path = argv[++i];
        } else if (std::strcmp(argv[i], "--seed") == 0) {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "Error: --seed requires a value.\n");
                std::exit(EXIT_FAILURE);
            }
            options.seed =
                static_cast<unsigned int>(std::strtoul(argv[++i], nullptr, 10));
        }
    }

    return options;
}

static Shape make_shape(const Options &options) {
    Shape shape;
    shape.h = options.input_size;
    shape.w = options.input_size;
    shape.out_h =
        (options.input_size + 2 * options.padding - options.kernel_size) /
            options.stride +
        1;
    shape.out_w = shape.out_h;
    shape.im2col_rows = CHANNELS * options.kernel_size * options.kernel_size;
    shape.im2col_cols = shape.out_h * shape.out_w;
    return shape;
}

static void validate_options(const Options &options, const Shape &shape) {
    if (options.input_size <= 0 || options.kernel_size <= 0 ||
        options.stride <= 0 || options.padding < 0 || options.filters <= 0) {
        std::fprintf(stderr,
                     "Error: input_size, kernel_size, stride, and filters must "
                     "be positive; padding must be non-negative.\n");
        std::exit(EXIT_FAILURE);
    }
    if (options.input_size < 32 || options.input_size > 4096) {
        std::fprintf(stderr,
                     "Warning: assignment recommends input size in ranges such "
                     "as 32..512 or 256..4096; got %d.\n",
                     options.input_size);
    }
    if (shape.out_h <= 0 || shape.out_w <= 0) {
        std::fprintf(stderr,
                     "Error: invalid output size. Increase padding or reduce "
                     "kernel/stride.\n");
        std::exit(EXIT_FAILURE);
    }
    if (options.block_x <= 0 || options.block_y <= 0 ||
        options.block_x * options.block_y > 1024) {
        std::fprintf(stderr, "Error: block dimensions must be positive and <= 1024 threads.\n");
        std::exit(EXIT_FAILURE);
    }
    if (options.tile_k <= 0 || options.tile_k > 128) {
        std::fprintf(stderr, "Error: tile_k must be in [1, 128].\n");
        std::exit(EXIT_FAILURE);
    }
    if (options.repeat <= 0 || options.samples <= 0) {
        std::fprintf(stderr, "Error: repeat and samples must be positive.\n");
        std::exit(EXIT_FAILURE);
    }
    size_t shared_bytes =
        (static_cast<size_t>(options.block_y) * options.tile_k +
         static_cast<size_t>(options.tile_k) * options.block_x *
             GEMM_COLS_PER_THREAD) *
        sizeof(float);
    if ((options.variant == Variant::Im2col || options.variant == Variant::All) &&
        shared_bytes > 48U * 1024U) {
        std::fprintf(stderr,
                     "Error: im2col GEMM shared memory usage is %zu bytes, "
                     "above 48 KiB. Reduce block_x or tile_k.\n",
                     shared_bytes);
        std::exit(EXIT_FAILURE);
    }
}

static void fill_tensor(std::vector<float> &values, unsigned int seed) {
    std::srand(seed);
    for (float &value : values) {
        value = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX) -
                0.5f;
    }
}

static float cpu_conv_element(const std::vector<float> &input,
                              const std::vector<float> &filter,
                              const Options &options, const Shape &shape,
                              int f, int oy, int ox) {
    double sum = 0.0;
    for (int c = 0; c < CHANNELS; c++) {
        for (int ky = 0; ky < options.kernel_size; ky++) {
            int iy = oy * options.stride + ky - options.padding;
            if (iy < 0 || iy >= shape.h) {
                continue;
            }
            for (int kx = 0; kx < options.kernel_size; kx++) {
                int ix = ox * options.stride + kx - options.padding;
                if (ix < 0 || ix >= shape.w) {
                    continue;
                }
                int input_idx = (c * shape.h + iy) * shape.w + ix;
                int filter_idx =
                    ((f * CHANNELS + c) * options.kernel_size + ky) *
                        options.kernel_size +
                    kx;
                sum += static_cast<double>(input[input_idx]) *
                       static_cast<double>(filter[filter_idx]);
            }
        }
    }
    return static_cast<float>(sum);
}

static double verify_result(const std::vector<float> &input,
                            const std::vector<float> &filter,
                            const std::vector<float> &output,
                            const Options &options, const Shape &shape,
                            int *checked_count) {
    if (options.verify == VerifyMode::None) {
        *checked_count = 0;
        return 0.0;
    }

    double max_error = 0.0;
    int checked = 0;
    if (options.verify == VerifyMode::Full) {
        for (int f = 0; f < options.filters; f++) {
            for (int oy = 0; oy < shape.out_h; oy++) {
                for (int ox = 0; ox < shape.out_w; ox++) {
                    float expected =
                        cpu_conv_element(input, filter, options, shape, f, oy, ox);
                    int idx = (f * shape.out_h + oy) * shape.out_w + ox;
                    double diff = std::fabs(static_cast<double>(expected) -
                                            static_cast<double>(output[idx]));
                    max_error = std::max(max_error, diff);
                    checked++;
                }
            }
        }
    } else {
        unsigned int state = options.seed ^ 0x85ebca6bU;
        for (int i = 0; i < options.samples; i++) {
            state = state * 1664525U + 1013904223U;
            int f = static_cast<int>(state % static_cast<unsigned int>(options.filters));
            state = state * 1664525U + 1013904223U;
            int oy = static_cast<int>(state % static_cast<unsigned int>(shape.out_h));
            state = state * 1664525U + 1013904223U;
            int ox = static_cast<int>(state % static_cast<unsigned int>(shape.out_w));
            float expected =
                cpu_conv_element(input, filter, options, shape, f, oy, ox);
            int idx = (f * shape.out_h + oy) * shape.out_w + ox;
            double diff = std::fabs(static_cast<double>(expected) -
                                    static_cast<double>(output[idx]));
            max_error = std::max(max_error, diff);
            checked++;
        }
    }

    *checked_count = checked;
    return max_error;
}

static double checksum(const std::vector<float> &values) {
    double sum = 0.0;
    for (float value : values) {
        sum += value;
    }
    return sum;
}

static void print_tensor_3d(const char *name, const std::vector<float> &values,
                            int depth, int height, int width) {
    std::printf("%s:\n", name);
    for (int d = 0; d < depth; d++) {
        std::printf("channel/filter %d\n", d);
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                std::printf("%9.4f%s", values[(d * height + y) * width + x],
                            x + 1 == width ? "\n" : " ");
            }
        }
    }
}

static void dump_tensor(FILE *fp, const char *name,
                        const std::vector<float> &values, int depth, int height,
                        int width) {
    std::fprintf(fp, "tensor,%s,depth,%d,height,%d,width,%d\n", name, depth,
                 height, width);
    for (int d = 0; d < depth; d++) {
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                std::fprintf(fp, "%.7g%s", values[(d * height + y) * width + x],
                             x + 1 == width ? "\n" : ",");
            }
        }
    }
}

static void dump_all(const char *path, const std::vector<float> &input,
                     const std::vector<float> &filter,
                     const std::vector<float> &output, const Options &options,
                     const Shape &shape) {
    FILE *fp = std::fopen(path, "w");
    if (!fp) {
        std::fprintf(stderr, "Cannot open dump file: %s\n", path);
        std::exit(EXIT_FAILURE);
    }
    dump_tensor(fp, "input", input, CHANNELS, shape.h, shape.w);
    dump_tensor(fp, "filter", filter, options.filters * CHANNELS,
                options.kernel_size, options.kernel_size);
    dump_tensor(fp, "output", output, options.filters, shape.out_h, shape.out_w);
    std::fclose(fp);
}

static float run_direct(const float *d_input, const float *d_filter, float *d_output,
                        const Options &options, const Shape &shape) {
    dim3 block(options.block_x, options.block_y);
    dim3 grid((shape.out_w + block.x - 1) / block.x,
              (shape.out_h + block.y - 1) / block.y, options.filters);

    conv_direct_kernel<<<grid, block>>>(d_input, d_filter, d_output, shape.h,
                                        shape.w, shape.out_h, shape.out_w,
                                        options.kernel_size, options.stride,
                                        options.padding, options.filters);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < options.repeat; i++) {
        conv_direct_kernel<<<grid, block>>>(d_input, d_filter, d_output, shape.h,
                                            shape.w, shape.out_h, shape.out_w,
                                            options.kernel_size, options.stride,
                                            options.padding, options.filters);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return elapsed_ms / static_cast<float>(options.repeat);
}

static float run_im2col(const float *d_input, const float *d_filter,
                        float *d_columns, float *d_output,
                        const Options &options, const Shape &shape) {
    int im2col_total = shape.im2col_rows * shape.im2col_cols;
    int im2col_threads = 256;
    int im2col_blocks = (im2col_total + im2col_threads - 1) / im2col_threads;

    dim3 block(options.block_x, options.block_y);
    int output_cols = options.block_x * GEMM_COLS_PER_THREAD;
    dim3 grid((shape.im2col_cols + output_cols - 1) / output_cols,
              (options.filters + block.y - 1) / block.y);
    size_t shared_bytes =
        (static_cast<size_t>(options.block_y) * options.tile_k +
         static_cast<size_t>(options.tile_k) * output_cols) *
        sizeof(float);

    im2col_kernel<<<im2col_blocks, im2col_threads>>>(
        d_input, d_columns, shape.h, shape.w, shape.out_h, shape.out_w,
        options.kernel_size, options.stride, options.padding);
    gemm_tiled_cols4_kernel<<<grid, block, shared_bytes>>>(
        d_filter, d_columns, d_output, options.filters, shape.im2col_rows,
        shape.im2col_cols, options.tile_k);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < options.repeat; i++) {
        im2col_kernel<<<im2col_blocks, im2col_threads>>>(
            d_input, d_columns, shape.h, shape.w, shape.out_h, shape.out_w,
            options.kernel_size, options.stride, options.padding);
        gemm_tiled_cols4_kernel<<<grid, block, shared_bytes>>>(
            d_filter, d_columns, d_output, options.filters, shape.im2col_rows,
            shape.im2col_cols, options.tile_k);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return elapsed_ms / static_cast<float>(options.repeat);
}

static float run_cudnn(const float *d_input, const float *d_filter, float *d_output,
                       const Options &options, const Shape &shape) {
    cudnnHandle_t handle;
    cudnnTensorDescriptor_t input_desc;
    cudnnFilterDescriptor_t filter_desc;
    cudnnConvolutionDescriptor_t conv_desc;
    cudnnTensorDescriptor_t output_desc;

    CUDNN_CHECK(cudnnCreate(&handle));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&input_desc));
    CUDNN_CHECK(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&output_desc));

    CUDNN_CHECK(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW,
                                           CUDNN_DATA_FLOAT, 1, CHANNELS,
                                           shape.h, shape.w));
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT,
                                           CUDNN_TENSOR_NCHW, options.filters,
                                           CHANNELS, options.kernel_size,
                                           options.kernel_size));
    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(
        conv_desc, options.padding, options.padding, options.stride, options.stride,
        1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW,
                                           CUDNN_DATA_FLOAT, 1, options.filters,
                                           shape.out_h, shape.out_w));

    cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
    size_t workspace_bytes = 0;
    CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(
        handle, input_desc, filter_desc, conv_desc, output_desc, algo,
        &workspace_bytes));

    void *workspace = nullptr;
    if (workspace_bytes > 0) {
        CUDA_CHECK(cudaMalloc(&workspace, workspace_bytes));
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;
    CUDNN_CHECK(cudnnConvolutionForward(handle, &alpha, input_desc, d_input,
                                        filter_desc, d_filter, conv_desc, algo,
                                        workspace, workspace_bytes, &beta,
                                        output_desc, d_output));
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < options.repeat; i++) {
        CUDNN_CHECK(cudnnConvolutionForward(handle, &alpha, input_desc, d_input,
                                            filter_desc, d_filter, conv_desc, algo,
                                            workspace, workspace_bytes, &beta,
                                            output_desc, d_output));
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    if (workspace) {
        CUDA_CHECK(cudaFree(workspace));
    }
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(output_desc));
    CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(conv_desc));
    CUDNN_CHECK(cudnnDestroyFilterDescriptor(filter_desc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(input_desc));
    CUDNN_CHECK(cudnnDestroy(handle));

    return elapsed_ms / static_cast<float>(options.repeat);
}

static void run_and_report(Variant variant, const std::vector<float> &input,
                           const std::vector<float> &filter,
                           std::vector<float> &output, const float *d_input,
                           const float *d_filter, float *d_columns,
                           float *d_output, const Options &options,
                           const Shape &shape) {
    CUDA_CHECK(cudaMemset(d_output, 0,
                          output.size() * sizeof(float)));

    float avg_ms = 0.0f;
    if (variant == Variant::Direct) {
        avg_ms = run_direct(d_input, d_filter, d_output, options, shape);
    } else if (variant == Variant::Im2col) {
        avg_ms = run_im2col(d_input, d_filter, d_columns, d_output, options, shape);
    } else if (variant == Variant::Cudnn) {
        avg_ms = run_cudnn(d_input, d_filter, d_output, options, shape);
    }

    CUDA_CHECK(cudaMemcpy(output.data(), d_output, output.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    int checked = 0;
    double max_error =
        verify_result(input, filter, output, options, shape, &checked);
    double ops = 2.0 * static_cast<double>(options.filters) *
                 static_cast<double>(shape.out_h) *
                 static_cast<double>(shape.out_w) *
                 static_cast<double>(shape.im2col_rows);
    double gops = ops / (static_cast<double>(avg_ms) * 1.0e-3) / 1.0e9;

    std::printf("variant=%s\n", variant_name(variant));
    std::printf("input_size=%d\n", options.input_size);
    std::printf("kernel_size=%d\n", options.kernel_size);
    std::printf("channels=%d\n", CHANNELS);
    std::printf("filters=%d\n", options.filters);
    std::printf("stride=%d\n", options.stride);
    std::printf("padding=%d\n", options.padding);
    std::printf("out_h=%d\n", shape.out_h);
    std::printf("out_w=%d\n", shape.out_w);
    std::printf("block_x=%d\n", options.block_x);
    std::printf("block_y=%d\n", options.block_y);
    std::printf("tile_k=%d\n", variant == Variant::Im2col ? options.tile_k : 0);
    std::printf("repeat=%d\n", options.repeat);
    std::printf("verify=%s\n", verify_name(options.verify));
    std::printf("checked=%d\n", checked);
    std::printf("avg_time_ms=%.9f\n", avg_ms);
    std::printf("gops=%.6f\n", gops);
    std::printf("max_error=%.8g\n", max_error);
    std::printf("checksum_input=%.10f\n", checksum(input));
    std::printf("checksum_filter=%.10f\n", checksum(filter));
    std::printf("checksum_output=%.10f\n", checksum(output));
    std::printf("status=%s\n\n", max_error < 1.0e-3 ? "PASS" : "FAIL");
}

int main(int argc, char **argv) {
    Options options = parse_options(argc, argv);
    Shape shape = make_shape(options);
    validate_options(options, shape);

    size_t input_count =
        static_cast<size_t>(CHANNELS) * shape.h * shape.w;
    size_t filter_count =
        static_cast<size_t>(options.filters) * CHANNELS * options.kernel_size *
        options.kernel_size;
    size_t output_count =
        static_cast<size_t>(options.filters) * shape.out_h * shape.out_w;
    size_t im2col_count =
        static_cast<size_t>(shape.im2col_rows) * shape.im2col_cols;

    std::vector<float> input(input_count);
    std::vector<float> filter(filter_count);
    std::vector<float> output(output_count);
    fill_tensor(input, options.seed);
    fill_tensor(filter, options.seed + 1U);

    float *d_input = nullptr;
    float *d_filter = nullptr;
    float *d_output = nullptr;
    float *d_columns = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input, input_count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_filter, filter_count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, output_count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_columns, im2col_count * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_input, input.data(), input_count * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_filter, filter.data(), filter_count * sizeof(float),
                          cudaMemcpyHostToDevice));

    int device = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    std::printf("device=%s\n", prop.name);
    std::printf("input_elements=%zu\n", input_count);
    std::printf("filter_elements=%zu\n", filter_count);
    std::printf("output_elements=%zu\n", output_count);
    std::printf("im2col_elements=%zu\n\n", im2col_count);

    if (options.variant == Variant::Direct || options.variant == Variant::All) {
        run_and_report(Variant::Direct, input, filter, output, d_input, d_filter,
                       d_columns, d_output, options, shape);
    }
    if (options.variant == Variant::Im2col || options.variant == Variant::All) {
        run_and_report(Variant::Im2col, input, filter, output, d_input, d_filter,
                       d_columns, d_output, options, shape);
    }
    if (options.variant == Variant::Cudnn || options.variant == Variant::All) {
        run_and_report(Variant::Cudnn, input, filter, output, d_input, d_filter,
                       d_columns, d_output, options, shape);
    }

    if (options.print_tensor) {
        if (shape.h <= 8 && shape.w <= 8 && shape.out_h <= 8 && shape.out_w <= 8) {
            print_tensor_3d("input", input, CHANNELS, shape.h, shape.w);
            print_tensor_3d("filter", filter, options.filters * CHANNELS,
                            options.kernel_size, options.kernel_size);
            print_tensor_3d("output", output, options.filters, shape.out_h,
                            shape.out_w);
        } else {
            std::printf("--print is limited to small tensors; use --dump <file>.\n");
        }
    }

    if (options.dump_path) {
        dump_all(options.dump_path, input, filter, output, options, shape);
        std::printf("dump=%s\n", options.dump_path);
    }

    CUDA_CHECK(cudaFree(d_columns));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_filter));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaDeviceReset());
    return EXIT_SUCCESS;
}
