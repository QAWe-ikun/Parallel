/**
 * @file heated_plate_openmp.c
 * @brief OpenMP 参考实现：稳态热传导模拟（支持动态网格大小）
 *
 * 原始 OpenMP 实现，作为 Pthreads 版本的性能对比基准。
 * 边界条件：上边界 = 0，其余三边 = 100
 * 迭代公式：W[i][j] = (1/4) * (W[i-1][j] + W[i+1][j] + W[i][j-1] + W[i][j+1])
 *
 * 编译：
 *   gcc -O2 -fopenmp -o heated_plate_openmp.exe heated_plate_openmp.c -lm
 * 运行：
 *   .\heated_plate_openmp.exe [num_threads] [grid_size]
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

static inline double wtime(void) {
    return omp_get_wtime();
}

int main(int argc, char *argv[]) {
    double epsilon = 0.001;
    int num_threads = 4;
    int m = 500, n = 500; /* 默认网格大小 */

    if (argc > 1) num_threads = atoi(argv[1]);
    if (argc > 2) {
      m = atoi(argv[2]);
      n = m;
    } /* 支持正方形网格 M=N */

    omp_set_num_threads(num_threads);

    /* 动态分配网格 (BSS -> Heap，Valgrind 可追踪) */
    double *u = (double *)malloc(m * n * sizeof(double));
    double *w = (double *)malloc(m * n * sizeof(double));
    if (!u || !w) {
      fprintf(stderr, "Memory allocation failed for %dx%d grid.\n", m, n);
      return 1;
    }

    printf("\n");
    printf("HEATED_PLATE_OPENMP\n");
    printf("  C/OpenMP version (reference)\n");
    printf("  A program to solve for the steady state temperature distribution\n");
    printf("  over a rectangular plate.\n");
    printf("\n");
    printf("  Spatial grid of %d by %d points.\n", m, n);
    printf("  The iteration will be repeated until the change is <= %e\n", epsilon);
    printf("  Number of threads = %d\n", num_threads);
    printf("\n");

/* ----- 初始化边界条件 (u 和 w 都需要) ----- */

/* 上边界 = 0 */
#pragma omp parallel for
  for (int j = 0; j < n; j++) {
    w[0 * n + j] = 0.0;
    u[0 * n + j] = 0.0;
  }

/* 下边界 = 100 */
#pragma omp parallel for
  for (int j = 0; j < n; j++) {
    w[(m - 1) * n + j] = 100.0;
    u[(m - 1) * n + j] = 100.0;
  }

/* 左右边界 = 100 */
#pragma omp parallel for
  for (int i = 1; i < m - 1; i++) {
    w[i * n + 0] = 100.0;
    w[i * n + (n - 1)] = 100.0;
    u[i * n + 0] = 100.0;
    u[i * n + (n - 1)] = 100.0;
  }

  /* 计算边界平均值 */
  double boundary_sum = 0.0;
#pragma omp parallel for reduction(+ : boundary_sum)
  for (int j = 0; j < n; j++) {
    boundary_sum += w[0 * n + j] + w[(m - 1) * n + j];
  }
#pragma omp parallel for reduction(+ : boundary_sum)
  for (int i = 1; i < m - 1; i++) {
    boundary_sum += w[i * n + 0] + w[i * n + (n - 1)];
  }
  double boundary_avg = boundary_sum / (2.0 * (m + n - 2));

/* 内部点初始化为边界平均值 */
#pragma omp parallel for collapse(2)
  for (int i = 1; i < m - 1; i++) {
    for (int j = 1; j < n - 1; j++) {
      w[i * n + j] = boundary_avg;
      u[i * n + j] = boundary_avg;
    }
  }

  printf("  MEAN = %f\n", boundary_avg);
  printf("\n");
  printf(" Iteration  Change\n\n");

  /* ----- Jacobi 迭代 ----- */
  double change = 2.0 * epsilon;
  int it = 0;
  int next_print = 1; /* 打印点: 1, 8, 64, 512... */

  double start_time = wtime();

  while (change > epsilon) {
    change = 0.0;
    it++;

#pragma omp parallel
    {
      double local_max = 0.0;

#pragma omp for collapse(2)
      for (int i = 1; i < m - 1; i++) {
        for (int j = 1; j < n - 1; j++) {
          u[i * n + j] = 0.25 * (w[(i - 1) * n + j] + w[(i + 1) * n + j] +
                                 w[i * n + (j - 1)] + w[i * n + (j + 1)]);
          double diff = fabs(u[i * n + j] - w[i * n + j]);
          if (diff > local_max)
            local_max = diff;
        }
      }

#pragma omp critical
      {
        if (local_max > change)
          change = local_max;
      }
    }

    /* 交换指针，使 u 成为下一次迭代的 w */
    double *temp = w;
    w = u;
    u = temp;

    /* 按 8 的 n 次方打印 (1, 8, 64, 512...) */
    if (it == next_print) {
      printf(" %8d  %10.6f\n", it, change);
      next_print *= 8;
    }
  }

  double end_time = wtime();
  double wallclock = end_time - start_time;

  printf("\n");
  printf(" %8d  %10.6f\n", it, change);
  printf("\n");
  printf("  Error tolerance achieved.\n");
  printf("  Wallclock time = %f\n", wallclock);

  printf("\n");
  printf("HEATED_PLATE_OPENMP:\n");
  printf("  Normal end of execution.\n");
  printf("\n");

  free(u);
  free(w);
  return 0;
}
