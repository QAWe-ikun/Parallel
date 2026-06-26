/*
 * Task10: CUDA general matrix multiplication.
 *
 * Computes C = A * B where:
 *   A is m x n, B is n x k, C is m x k.
 *
 * Variants:
 *   - naive: one output element per thread, direct global-memory loads
 *   - tiled: four output columns per thread, shared-memory tiles over n
 */

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
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

#define TILED_COLS_PER_THREAD 4

enum class Variant {
    Naive,
    Tiled,
    Both,
};

enum class VerifyMode {
    None,
    Sample,
    Full,
};

struct Options {
    int m = 1024;
    int n = 1024;
    int k = 1024;
    int block_x = 16;
    int block_y = 16;
    int tile_k = 16;
    int repeat = 10;
    int samples = 4096;
    unsigned int seed = 2026U;
    Variant variant = Variant::Both;
    VerifyMode verify = VerifyMode::Sample;
    bool print_matrix = false;
    const char *dump_path = nullptr;
};

__global__ void gemm_naive_kernel(const float *a, const float *b, float *c,
                                  int m, int n, int k) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < m && col < k) {
        float sum = 0.0f;
        for (int p = 0; p < n; p++) {
            sum += a[row * n + p] * b[p * k + col];
        }
        c[row * k + col] = sum;
    }
}

__global__ void gemm_tiled_kernel(const float *a, const float *b, float *c,
                                  int m, int n, int k, int tile_k) {
    extern __shared__ float shared[];

    int output_cols = blockDim.x * TILED_COLS_PER_THREAD;
    float *tile_a = shared;
    float *tile_b = tile_a + blockDim.y * tile_k;

    int col0 = blockIdx.x * output_cols +
               threadIdx.x * TILED_COLS_PER_THREAD;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int threads_per_block = blockDim.x * blockDim.y;
    int linear_tid = threadIdx.y * blockDim.x + threadIdx.x;
    float sum[TILED_COLS_PER_THREAD] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int base = 0; base < n; base += tile_k) {
        int a_elements = blockDim.y * tile_k;
        for (int idx = linear_tid; idx < a_elements; idx += threads_per_block) {
            int local_row = idx / tile_k;
            int local_col = idx % tile_k;
            int global_row = blockIdx.y * blockDim.y + local_row;
            int global_col = base + local_col;
            tile_a[idx] = (global_row < m && global_col < n)
                              ? a[global_row * n + global_col]
                              : 0.0f;
        }

        int b_elements = tile_k * output_cols;
        for (int idx = linear_tid; idx < b_elements; idx += threads_per_block) {
            int local_row = idx / output_cols;
            int local_col = idx % output_cols;
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
            for (int i = 0; i < TILED_COLS_PER_THREAD; i++) {
                sum[i] += a_value *
                          tile_b[p * output_cols +
                                 threadIdx.x * TILED_COLS_PER_THREAD + i];
            }
        }

        __syncthreads();
    }

    if (row < m) {
#pragma unroll
        for (int i = 0; i < TILED_COLS_PER_THREAD; i++) {
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
                 "  %s [m] [n] [k] [block_x] [block_y] [variant] [repeat] "
                 "[tile_k] [options]\n\n"
                 "Arguments:\n"
                 "  m n k      A is m x n, B is n x k, C is m x k; each is "
                 "recommended in [128, 2048]\n"
                 "  block_x    thread block x dimension; tiled covers 4x this "
                 "many output columns\n"
                 "  block_y    output rows handled by a CUDA block\n"
                 "  variant    naive | tiled | both\n"
                 "  repeat     timed kernel repetitions\n"
                 "  tile_k     reduction tile width for tiled variant\n\n"
                 "Options:\n"
                 "  --verify none|sample|full   default: sample\n"
                 "  --samples <count>           sample count for verification\n"
                 "  --print                     print matrices when all dimensions "
                 "are <= 16\n"
                 "  --dump <csv>                write A, B, and C to a text file\n"
                 "  --seed <value>              random seed, default 2026\n\n"
                 "Examples:\n"
                 "  %s 1024 1024 1024 16 16 both 20\n"
                 "  %s 2048 2048 2048 32 8 tiled 50 32\n",
                 program, program, program);
}

static Variant parse_variant(const char *value) {
    if (std::strcmp(value, "naive") == 0) {
        return Variant::Naive;
    }
    if (std::strcmp(value, "tiled") == 0) {
        return Variant::Tiled;
    }
    if (std::strcmp(value, "both") == 0) {
        return Variant::Both;
    }
    std::fprintf(stderr, "Unknown variant: %s\n", value);
    std::exit(EXIT_FAILURE);
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

static const char *variant_name(Variant variant) {
    switch (variant) {
    case Variant::Naive:
        return "naive";
    case Variant::Tiled:
        return "tiled";
    case Variant::Both:
        return "both";
    }
    return "unknown";
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

static Options parse_options(int argc, char **argv) {
    Options options;

    if (argc > 1 && (std::strcmp(argv[1], "-h") == 0 ||
                     std::strcmp(argv[1], "--help") == 0)) {
        print_usage(argv[0]);
        std::exit(EXIT_SUCCESS);
    }

    if (argc > 1 && argv[1][0] != '-') {
        options.m = std::atoi(argv[1]);
    }
    if (argc > 2 && argv[2][0] != '-') {
        options.n = std::atoi(argv[2]);
    }
    if (argc > 3 && argv[3][0] != '-') {
        options.k = std::atoi(argv[3]);
    }
    if (argc > 4 && argv[4][0] != '-') {
        options.block_x = std::atoi(argv[4]);
    }
    if (argc > 5 && argv[5][0] != '-') {
        options.block_y = std::atoi(argv[5]);
    }
    if (argc > 6 && argv[6][0] != '-') {
        options.variant = parse_variant(argv[6]);
    }
    if (argc > 7 && argv[7][0] != '-') {
        options.repeat = std::atoi(argv[7]);
    }
    if (argc > 8 && argv[8][0] != '-') {
        options.tile_k = std::atoi(argv[8]);
    } else {
        options.tile_k = options.block_x;
    }

    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--verify") == 0) {
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
            options.print_matrix = true;
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

static void validate_options(const Options &options) {
    if (options.m <= 0 || options.n <= 0 || options.k <= 0) {
        std::fprintf(stderr, "Error: m, n, and k must be positive.\n");
        std::exit(EXIT_FAILURE);
    }
    if (options.m < 128 || options.m > 2048 || options.n < 128 ||
        options.n > 2048 || options.k < 128 || options.k > 2048) {
        std::fprintf(stderr,
                     "Warning: assignment recommends m, n, and k in [128, "
                     "2048]; got %d, %d, %d.\n",
                     options.m, options.n, options.k);
    }
    if (options.block_x <= 0 || options.block_y <= 0) {
        std::fprintf(stderr, "Error: block dimensions must be positive.\n");
        std::exit(EXIT_FAILURE);
    }
    if (options.block_x * options.block_y > 1024) {
        std::fprintf(stderr, "Error: block_x * block_y must be <= 1024.\n");
        std::exit(EXIT_FAILURE);
    }
    if (options.tile_k <= 0 || options.tile_k > 128) {
        std::fprintf(stderr, "Error: tile_k must be in [1, 128].\n");
        std::exit(EXIT_FAILURE);
    }
    size_t tiled_shared_bytes =
        (static_cast<size_t>(options.block_y) * options.tile_k +
         static_cast<size_t>(options.tile_k) * options.block_x *
             TILED_COLS_PER_THREAD) *
        sizeof(float);
    if ((options.variant == Variant::Tiled || options.variant == Variant::Both) &&
        tiled_shared_bytes > 48U * 1024U) {
        std::fprintf(stderr,
                     "Error: tiled shared memory usage is %zu bytes, above "
                     "the portable 48 KiB limit. Reduce tile_k or block_x.\n",
                     tiled_shared_bytes);
        std::exit(EXIT_FAILURE);
    }
    if (options.repeat <= 0) {
        std::fprintf(stderr, "Error: repeat must be positive.\n");
        std::exit(EXIT_FAILURE);
    }
    if (options.samples <= 0) {
        std::fprintf(stderr, "Error: samples must be positive.\n");
        std::exit(EXIT_FAILURE);
    }
}

static void fill_matrix(std::vector<float> &matrix, unsigned int seed) {
    std::srand(seed);
    for (float &value : matrix) {
        value = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX) -
                0.5f;
    }
}

static float dot_element(const std::vector<float> &a, const std::vector<float> &b,
                         int m, int n, int k, int row, int col) {
    (void)m;
    double sum = 0.0;
    for (int p = 0; p < n; p++) {
        sum += static_cast<double>(a[row * n + p]) *
               static_cast<double>(b[p * k + col]);
    }
    return static_cast<float>(sum);
}

static double verify_result(const std::vector<float> &a, const std::vector<float> &b,
                            const std::vector<float> &c, const Options &options,
                            int *checked_count) {
    if (options.verify == VerifyMode::None) {
        *checked_count = 0;
        return 0.0;
    }

    double max_error = 0.0;
    int checked = 0;

    if (options.verify == VerifyMode::Full) {
        for (int row = 0; row < options.m; row++) {
            for (int col = 0; col < options.k; col++) {
                float expected =
                    dot_element(a, b, options.m, options.n, options.k, row, col);
                double diff = std::fabs(static_cast<double>(expected) -
                                        static_cast<double>(c[row * options.k + col]));
                max_error = std::max(max_error, diff);
                checked++;
            }
        }
    } else {
        unsigned int state = options.seed ^ 0x9e3779b9U;
        for (int i = 0; i < options.samples; i++) {
            state = state * 1664525U + 1013904223U;
            int row = static_cast<int>(state % static_cast<unsigned int>(options.m));
            state = state * 1664525U + 1013904223U;
            int col = static_cast<int>(state % static_cast<unsigned int>(options.k));
            float expected =
                dot_element(a, b, options.m, options.n, options.k, row, col);
            double diff = std::fabs(static_cast<double>(expected) -
                                    static_cast<double>(c[row * options.k + col]));
            max_error = std::max(max_error, diff);
            checked++;
        }
    }

    *checked_count = checked;
    return max_error;
}

static double checksum(const std::vector<float> &matrix) {
    double sum = 0.0;
    for (float value : matrix) {
        sum += value;
    }
    return sum;
}

static void print_matrix(const char *name, const std::vector<float> &matrix,
                         int rows, int cols) {
    std::printf("%s:\n", name);
    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            std::printf("%9.4f%s", matrix[row * cols + col],
                        col + 1 == cols ? "\n" : " ");
        }
    }
}

static void dump_one_matrix(FILE *fp, const char *name,
                            const std::vector<float> &matrix, int rows, int cols) {
    std::fprintf(fp, "matrix,%s,rows,%d,cols,%d\n", name, rows, cols);
    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            std::fprintf(fp, "%.7g%s", matrix[row * cols + col],
                         col + 1 == cols ? "\n" : ",");
        }
    }
}

static void dump_matrices(const char *path, const std::vector<float> &a,
                          const std::vector<float> &b,
                          const std::vector<float> &c, const Options &options) {
    FILE *fp = std::fopen(path, "w");
    if (!fp) {
        std::fprintf(stderr, "Cannot open dump file: %s\n", path);
        std::exit(EXIT_FAILURE);
    }

    dump_one_matrix(fp, "A", a, options.m, options.n);
    dump_one_matrix(fp, "B", b, options.n, options.k);
    dump_one_matrix(fp, "C", c, options.m, options.k);
    std::fclose(fp);
}

static float run_kernel(Variant variant, const float *d_a, const float *d_b,
                        float *d_c, const Options &options) {
    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    dim3 block(options.block_x, options.block_y);

    if (variant == Variant::Naive) {
        dim3 grid((options.k + block.x - 1) / block.x,
                  (options.m + block.y - 1) / block.y);
        gemm_naive_kernel<<<grid, block>>>(d_a, d_b, d_c, options.m, options.n,
                                           options.k);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < options.repeat; i++) {
            gemm_naive_kernel<<<grid, block>>>(d_a, d_b, d_c, options.m,
                                               options.n, options.k);
        }
        CUDA_CHECK(cudaEventRecord(stop));
    } else {
        int tiled_output_cols = options.block_x * TILED_COLS_PER_THREAD;
        dim3 grid((options.k + tiled_output_cols - 1) / tiled_output_cols,
                  (options.m + block.y - 1) / block.y);
        size_t shared_bytes =
            (static_cast<size_t>(options.block_y) * options.tile_k +
             static_cast<size_t>(options.tile_k) * tiled_output_cols) *
            sizeof(float);

        gemm_tiled_kernel<<<grid, block, shared_bytes>>>(
            d_a, d_b, d_c, options.m, options.n, options.k, options.tile_k);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < options.repeat; i++) {
            gemm_tiled_kernel<<<grid, block, shared_bytes>>>(
                d_a, d_b, d_c, options.m, options.n, options.k, options.tile_k);
        }
        CUDA_CHECK(cudaEventRecord(stop));
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return elapsed_ms / static_cast<float>(options.repeat);
}

static void run_variant(Variant variant, const std::vector<float> &a,
                        const std::vector<float> &b, std::vector<float> &c,
                        const float *d_a, const float *d_b, float *d_c,
                        const Options &options) {
    size_t c_bytes = c.size() * sizeof(float);
    CUDA_CHECK(cudaMemset(d_c, 0, c_bytes));

    float avg_ms = run_kernel(variant, d_a, d_b, d_c, options);
    CUDA_CHECK(cudaMemcpy(c.data(), d_c, c_bytes, cudaMemcpyDeviceToHost));

    int checked_count = 0;
    double max_error = verify_result(a, b, c, options, &checked_count);
    double ops = 2.0 * static_cast<double>(options.m) *
                 static_cast<double>(options.n) *
                 static_cast<double>(options.k);
    double gflops = ops / (static_cast<double>(avg_ms) * 1.0e-3) / 1.0e9;

    std::printf("variant=%s\n", variant_name(variant));
    std::printf("m=%d\n", options.m);
    std::printf("n=%d\n", options.n);
    std::printf("k=%d\n", options.k);
    std::printf("block_x=%d\n", options.block_x);
    std::printf("block_y=%d\n", options.block_y);
    std::printf("tile_k=%d\n", variant == Variant::Tiled ? options.tile_k : 0);
    std::printf("cols_per_thread=%d\n",
                variant == Variant::Tiled ? TILED_COLS_PER_THREAD : 1);
    std::printf("repeat=%d\n", options.repeat);
    std::printf("verify=%s\n", verify_name(options.verify));
    std::printf("checked=%d\n", checked_count);
    std::printf("avg_time_ms=%.6f\n", avg_ms);
    std::printf("gflops=%.6f\n", gflops);
    std::printf("max_error=%.8g\n", max_error);
    std::printf("checksum_A=%.10f\n", checksum(a));
    std::printf("checksum_B=%.10f\n", checksum(b));
    std::printf("checksum_C=%.10f\n", checksum(c));
    std::printf("status=%s\n\n", max_error < 1.0e-2 ? "PASS" : "FAIL");
}

int main(int argc, char **argv) {
    Options options = parse_options(argc, argv);
    validate_options(options);

    size_t a_count =
        static_cast<size_t>(options.m) * static_cast<size_t>(options.n);
    size_t b_count =
        static_cast<size_t>(options.n) * static_cast<size_t>(options.k);
    size_t c_count =
        static_cast<size_t>(options.m) * static_cast<size_t>(options.k);

    std::vector<float> a(a_count);
    std::vector<float> b(b_count);
    std::vector<float> c(c_count);

    fill_matrix(a, options.seed);
    fill_matrix(b, options.seed + 1U);

    float *d_a = nullptr;
    float *d_b = nullptr;
    float *d_c = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, a_count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, b_count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c, c_count * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_a, a.data(), a_count * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b.data(), b_count * sizeof(float),
                          cudaMemcpyHostToDevice));

    int device = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    std::printf("device=%s\n", prop.name);
    std::printf("a_elements=%zu\n", a_count);
    std::printf("b_elements=%zu\n", b_count);
    std::printf("c_elements=%zu\n\n", c_count);

    if (options.variant == Variant::Naive || options.variant == Variant::Both) {
        run_variant(Variant::Naive, a, b, c, d_a, d_b, d_c, options);
    }
    if (options.variant == Variant::Tiled || options.variant == Variant::Both) {
        run_variant(Variant::Tiled, a, b, c, d_a, d_b, d_c, options);
    }

    if (options.print_matrix) {
        if (options.m <= 16 && options.n <= 16 && options.k <= 16) {
            print_matrix("A", a, options.m, options.n);
            print_matrix("B", b, options.n, options.k);
            print_matrix("C", c, options.m, options.k);
        } else {
            std::printf("--print is limited to m,n,k <= 16; use --dump <file> "
                        "for full output.\n");
        }
    }

    if (options.dump_path) {
        dump_matrices(options.dump_path, a, b, c, options);
        std::printf("dump=%s\n", options.dump_path);
    }

    CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaDeviceReset());
    return EXIT_SUCCESS;
}
