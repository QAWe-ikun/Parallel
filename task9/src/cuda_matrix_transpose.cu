/*
 * Task9 Part 2: CUDA matrix transpose.
 *
 * Implements two kernels:
 *   - naive: direct global-memory transpose, coalesced reads and strided writes
 *   - tiled: shared-memory tiled transpose with padding to reduce bank conflicts
 *
 * The program also computes a CPU reference result for correctness checking.
 */

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
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

enum class Variant {
    Naive,
    Tiled,
    Both,
};

struct Options {
    int n = 1024;
    int block_x = 16;
    int block_y = 16;
    int repeat = 10;
    Variant variant = Variant::Both;
    bool print_matrix = false;
    const char *dump_path = nullptr;
    unsigned int seed = 2026U;
};

__global__ void transpose_naive_kernel(const float *input, float *output, int n) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < n && col < n) {
        output[col * n + row] = input[row * n + col];
    }
}

__global__ void transpose_tiled_kernel(const float *input, float *output, int n) {
    extern __shared__ float tile[];

    int tile_dim = blockDim.x;
    int block_rows = blockDim.y;
    int padded_width = tile_dim + 1;

    int x = blockIdx.x * tile_dim + threadIdx.x;
    int y = blockIdx.y * tile_dim + threadIdx.y;

    for (int j = 0; j < tile_dim; j += block_rows) {
        int yy = y + j;
        if (x < n && yy < n && threadIdx.y + j < tile_dim) {
            tile[(threadIdx.y + j) * padded_width + threadIdx.x] =
                input[yy * n + x];
        }
    }

    __syncthreads();

    x = blockIdx.y * tile_dim + threadIdx.x;
    y = blockIdx.x * tile_dim + threadIdx.y;

    for (int j = 0; j < tile_dim; j += block_rows) {
        int yy = y + j;
        if (x < n && yy < n && threadIdx.y + j < tile_dim) {
            output[yy * n + x] =
                tile[threadIdx.x * padded_width + threadIdx.y + j];
        }
    }
}

static void print_usage(const char *program) {
    std::fprintf(stderr,
                 "Usage:\n"
                 "  %s [n] [block_x] [block_y] [variant] [repeat] [options]\n\n"
                 "Arguments:\n"
                 "  n          matrix size, assignment range [512, 2048]\n"
                 "  block_x    x dimension for naive, tile width for tiled\n"
                 "  block_y    y dimension for naive, rows per tile for tiled\n"
                 "  variant    naive | tiled | both\n"
                 "  repeat     timed kernel repetitions\n\n"
                 "Options:\n"
                 "  --print          print A and AT when n <= 16\n"
                 "  --dump <csv>     write A and AT to a CSV-like text file\n"
                 "  --seed <value>   random seed, default 2026\n\n"
                 "Examples:\n"
                 "  %s 1024 16 16 both 20\n"
                 "  %s 2048 32 8 tiled 50\n",
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

static Options parse_options(int argc, char **argv) {
    Options options;

    if (argc > 1 && (std::strcmp(argv[1], "-h") == 0 ||
                     std::strcmp(argv[1], "--help") == 0)) {
        print_usage(argv[0]);
        std::exit(EXIT_SUCCESS);
    }

    if (argc > 1 && argv[1][0] != '-') {
        options.n = std::atoi(argv[1]);
    }
    if (argc > 2 && argv[2][0] != '-') {
        options.block_x = std::atoi(argv[2]);
    }
    if (argc > 3 && argv[3][0] != '-') {
        options.block_y = std::atoi(argv[3]);
    }
    if (argc > 4 && argv[4][0] != '-') {
        options.variant = parse_variant(argv[4]);
    }
    if (argc > 5 && argv[5][0] != '-') {
        options.repeat = std::atoi(argv[5]);
    }

    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--print") == 0) {
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
            options.seed = static_cast<unsigned int>(std::strtoul(argv[++i], nullptr, 10));
        }
    }

    return options;
}

static void validate_options(const Options &options) {
    if (options.n <= 0) {
        std::fprintf(stderr, "Error: n must be positive.\n");
        std::exit(EXIT_FAILURE);
    }
    if (options.n < 512 || options.n > 2048) {
        std::fprintf(stderr,
                     "Warning: assignment recommends n in [512, 2048]; got %d.\n",
                     options.n);
    }
    if (options.block_x <= 0 || options.block_y <= 0) {
        std::fprintf(stderr, "Error: block dimensions must be positive.\n");
        std::exit(EXIT_FAILURE);
    }
    if (options.block_x * options.block_y > 1024) {
        std::fprintf(stderr, "Error: block_x * block_y must be <= 1024.\n");
        std::exit(EXIT_FAILURE);
    }
    if (options.repeat <= 0) {
        std::fprintf(stderr, "Error: repeat must be positive.\n");
        std::exit(EXIT_FAILURE);
    }
    if ((options.variant == Variant::Tiled || options.variant == Variant::Both) &&
        options.block_y > options.block_x) {
        std::fprintf(stderr,
                     "Error: tiled transpose requires block_y <= block_x. "
                     "Use values such as 16x16, 32x8, or 32x16.\n");
        std::exit(EXIT_FAILURE);
    }
}

static void fill_matrix(std::vector<float> &matrix, unsigned int seed) {
    std::srand(seed);
    for (float &value : matrix) {
        value = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
    }
}

static void cpu_transpose(const std::vector<float> &input, std::vector<float> &output, int n) {
    for (int row = 0; row < n; row++) {
        for (int col = 0; col < n; col++) {
            output[col * n + row] = input[row * n + col];
        }
    }
}

static double max_abs_diff(const std::vector<float> &a, const std::vector<float> &b) {
    double diff = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        diff = std::max(diff, std::fabs(static_cast<double>(a[i]) - b[i]));
    }
    return diff;
}

static double checksum(const std::vector<float> &matrix) {
    double sum = 0.0;
    for (float value : matrix) {
        sum += value;
    }
    return sum;
}

static void print_matrix(const char *name, const std::vector<float> &matrix, int n) {
    std::printf("%s:\n", name);
    for (int row = 0; row < n; row++) {
        for (int col = 0; col < n; col++) {
            std::printf("%8.4f%s", matrix[row * n + col],
                        col + 1 == n ? "\n" : " ");
        }
    }
}

static void dump_matrices(const char *path, const std::vector<float> &input,
                          const std::vector<float> &output, int n) {
    FILE *fp = std::fopen(path, "w");
    if (!fp) {
        std::fprintf(stderr, "Cannot open dump file: %s\n", path);
        std::exit(EXIT_FAILURE);
    }

    std::fprintf(fp, "matrix,A,n,%d\n", n);
    for (int row = 0; row < n; row++) {
        for (int col = 0; col < n; col++) {
            std::fprintf(fp, "%.7g%s", input[row * n + col],
                         col + 1 == n ? "\n" : ",");
        }
    }

    std::fprintf(fp, "matrix,AT,n,%d\n", n);
    for (int row = 0; row < n; row++) {
        for (int col = 0; col < n; col++) {
            std::fprintf(fp, "%.7g%s", output[row * n + col],
                         col + 1 == n ? "\n" : ",");
        }
    }

    std::fclose(fp);
}

static float run_kernel(Variant variant, const float *d_input, float *d_output,
                        int n, int block_x, int block_y, int repeat) {
    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    if (variant == Variant::Naive) {
        dim3 block(block_x, block_y);
        dim3 grid((n + block.x - 1) / block.x, (n + block.y - 1) / block.y);
        transpose_naive_kernel<<<grid, block>>>(d_input, d_output, n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < repeat; i++) {
            transpose_naive_kernel<<<grid, block>>>(d_input, d_output, n);
        }
        CUDA_CHECK(cudaEventRecord(stop));
    } else {
        dim3 block(block_x, block_y);
        dim3 grid((n + block_x - 1) / block_x, (n + block_x - 1) / block_x);
        size_t shared_bytes =
            static_cast<size_t>(block_x) * static_cast<size_t>(block_x + 1) *
            sizeof(float);
        transpose_tiled_kernel<<<grid, block, shared_bytes>>>(d_input, d_output, n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < repeat; i++) {
            transpose_tiled_kernel<<<grid, block, shared_bytes>>>(d_input, d_output, n);
        }
        CUDA_CHECK(cudaEventRecord(stop));
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return elapsed_ms / static_cast<float>(repeat);
}

static void run_variant(Variant variant, const std::vector<float> &input,
                        const std::vector<float> &reference, std::vector<float> &output,
                        float *d_input, float *d_output, const Options &options) {
    size_t bytes = input.size() * sizeof(float);
    CUDA_CHECK(cudaMemset(d_output, 0, bytes));

    float avg_ms = run_kernel(variant, d_input, d_output, options.n, options.block_x,
                              options.block_y, options.repeat);
    CUDA_CHECK(cudaMemcpy(output.data(), d_output, bytes, cudaMemcpyDeviceToHost));

    double max_error = max_abs_diff(output, reference);
    double moved_bytes = 2.0 * static_cast<double>(bytes);
    double bandwidth_gbps = moved_bytes / (static_cast<double>(avg_ms) * 1.0e-3) / 1.0e9;

    std::printf("variant=%s\n", variant_name(variant));
    std::printf("n=%d\n", options.n);
    std::printf("block_x=%d\n", options.block_x);
    std::printf("block_y=%d\n", options.block_y);
    std::printf("repeat=%d\n", options.repeat);
    std::printf("avg_time_ms=%.6f\n", avg_ms);
    std::printf("bandwidth_gbps=%.6f\n", bandwidth_gbps);
    std::printf("max_error=%.8g\n", max_error);
    std::printf("checksum_A=%.10f\n", checksum(input));
    std::printf("checksum_AT=%.10f\n", checksum(output));
    std::printf("status=%s\n\n", max_error < 1.0e-6 ? "PASS" : "FAIL");
}

int main(int argc, char **argv) {
    Options options = parse_options(argc, argv);
    validate_options(options);

    size_t element_count =
        static_cast<size_t>(options.n) * static_cast<size_t>(options.n);
    size_t bytes = element_count * sizeof(float);

    std::vector<float> input(element_count);
    std::vector<float> reference(element_count);
    std::vector<float> output(element_count);

    fill_matrix(input, options.seed);
    cpu_transpose(input, reference, options.n);

    float *d_input = nullptr;
    float *d_output = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMalloc(&d_output, bytes));
    CUDA_CHECK(cudaMemcpy(d_input, input.data(), bytes, cudaMemcpyHostToDevice));

    int device = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    std::printf("device=%s\n", prop.name);
    std::printf("matrix_elements=%zu\n", element_count);
    std::printf("matrix_bytes=%zu\n\n", bytes);

    if (options.variant == Variant::Naive || options.variant == Variant::Both) {
        run_variant(Variant::Naive, input, reference, output, d_input, d_output, options);
    }
    if (options.variant == Variant::Tiled || options.variant == Variant::Both) {
        run_variant(Variant::Tiled, input, reference, output, d_input, d_output, options);
    }

    if (options.print_matrix) {
        if (options.n <= 16) {
            print_matrix("A", input, options.n);
            print_matrix("AT", output, options.n);
        } else {
            std::printf("--print is limited to n <= 16; use --dump <file> for full output.\n");
        }
    }

    if (options.dump_path) {
        dump_matrices(options.dump_path, input, output, options.n);
        std::printf("dump=%s\n", options.dump_path);
    }

    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaDeviceReset());
    return EXIT_SUCCESS;
}
