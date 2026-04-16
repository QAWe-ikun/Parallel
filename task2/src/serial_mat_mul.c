#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* Serial-only matrix multiplication for benchmarking (1 process) */

static double *alloc_matrix(int rows, int cols) {
  double *m = (double *)malloc((size_t)rows * cols * sizeof(double));
  if (!m) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(1);
  }
  return m;
}

static void fill_random(double *mat, int rows, int cols) {
  for (int i = 0; i < rows * cols; i++)
    mat[i] = (double)rand() / RAND_MAX;
}

static void mat_mul_serial(double *A, double *B, double *C, int m, int n,
                           int k) {
  for (int i = 0; i < m; i++)
    for (int j = 0; j < k; j++)
      C[i * k + j] = 0.0;
  for (int i = 0; i < m; i++)
    for (int p = 0; p < n; p++)
      for (int j = 0; j < k; j++)
        C[i * k + j] += A[i * n + p] * B[p * k + j];
}

int main(int argc, char **argv) {
  int m, n, k;

  if (argc >= 4) {
    m = atoi(argv[1]);
    n = atoi(argv[2]);
    k = atoi(argv[3]);
  } else if (argc == 2) {
    int size = atoi(argv[1]);
    m = n = k = size;
  } else {
    fprintf(stderr, "Usage: serial_mat_mul <m> <n> <k> or <size>\n");
    return 1;
  }

  printf("Matrix size: A(%d x %d) * B(%d x %d) = C(%d x %d)\n", m, n, n, k, m,
         k);
  printf("Processes: 1 (serial)\n");

  srand(42);
  double *A = alloc_matrix(m, n);
  double *B = alloc_matrix(n, k);
  double *C = alloc_matrix(m, k);
  fill_random(A, m, n);
  fill_random(B, n, k);

  // 使用更高精度的计时器
  clock_t start = clock();
  mat_mul_serial(A, B, C, m, n, k);
  clock_t end = clock();

  double elapsed = (double)(end - start) / CLOCKS_PER_SEC;

  // 如果时间太短，使用更精确的方法估算
  if (elapsed < 0.001) {
    // 重复运行多次以获得更精确的计时
    const int repetitions = 100;
    clock_t start_total = clock();
    for (int rep = 0; rep < repetitions; rep++) {
      mat_mul_serial(A, B, C, m, n, k);
    }
    clock_t end_total = clock();
    elapsed = ((double)(end_total - start_total) / CLOCKS_PER_SEC) / repetitions;
  }

  printf("Serial Time:   %.6f seconds\n", elapsed);

  free(A);
  free(B);
  free(C);
  return 0;
}