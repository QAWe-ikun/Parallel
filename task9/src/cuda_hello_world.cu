/*
 * Task9 Part 1: CUDA Hello World.
 *
 * Input:
 *   n m k
 *
 * n thread blocks are launched.  Each block has m x k threads and every
 * device thread prints its block id, 2-D thread id, and a Hello World line.
 */

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call)                                                          \
    do {                                                                         \
        cudaError_t err__ = (call);                                               \
        if (err__ != cudaSuccess) {                                               \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__,          \
                         __LINE__, cudaGetErrorString(err__));                   \
            return EXIT_FAILURE;                                                  \
        }                                                                        \
    } while (0)

__global__ void hello_kernel() {
    printf("Hello World from Thread (%d, %d) in Block %d!\n", threadIdx.x,
           threadIdx.y, blockIdx.x);
}

static void print_usage(const char *program) {
    std::fprintf(stderr,
                 "Usage:\n"
                 "  %s <num_blocks> <block_dim_x> <block_dim_y>\n\n"
                 "Constraints from the assignment: all three values are in "
                 "[1, 32].\n",
                 program);
}

int main(int argc, char **argv) {
    if (argc != 4) {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    int n = std::atoi(argv[1]);
    int m = std::atoi(argv[2]);
    int k = std::atoi(argv[3]);

    if (n < 1 || n > 32 || m < 1 || m > 32 || k < 1 || k > 32) {
        std::fprintf(stderr, "Error: n, m, and k must all be in [1, 32].\n");
        return EXIT_FAILURE;
    }

    if (m * k > 1024) {
        std::fprintf(stderr,
                     "Error: CUDA devices usually allow at most 1024 threads "
                     "per block; got %d x %d = %d.\n",
                     m, k, m * k);
        return EXIT_FAILURE;
    }

    std::printf("Hello World from the host!\n");
    std::printf("Launching %d block(s), each with %d x %d thread(s).\n", n, m, k);

    dim3 grid(n);
    dim3 block(m, k);
    hello_kernel<<<grid, block>>>();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;
}
