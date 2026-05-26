/**
 * @file heated_plate_pthreads.c
 * @brief 基于 Pthreads 线程池 parallel_for_pool 的稳态热传导模拟
 *
 * 支持动态指定网格大小，使用扁平数组实现并行化
 *
 * 编译：
 *   gcc -O3 -march=native -Isrc -o bin/heated_plate_pthreads
 * src/heated_plate_pthreads.c \ -Llib -lparallel_for_pool -lpthread -lm 运行：
 *   ./bin/heated_plate_pthreads [num_threads] [grid_size] [schedule]
 * [chunk_size]
 */

#define _POSIX_C_SOURCE 199309L
#include "parallel_for_pool.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/** @brief 迭代参数结构体 (扁平数组) */
typedef struct {
  double *u;           /* 上一次迭代的解 (1D array) */
  double *w;           /* 当前迭代的解 (1D array) */
  int m_dim, n_dim;    /* 网格维度 */
  double *local_diffs; /* 每行的局部最大差值 */
  double mean;         /* 边界平均值 */
} iter_args_t;

/* ============================================================
 *  Functors
 * ============================================================ */

/** @brief 初始化上边界行 (w[0][j] = 0) */
void *init_top_row(int idx, void *arg) {
  (void)idx;
  iter_args_t *a = (iter_args_t *)arg;
  for (int j = 0; j < a->n_dim; j++)
    a->w[j] = 0.0;
  return NULL;
}

/** @brief 初始化下边界行 (w[m-1][j] = 100) */
void *init_bottom_row(int idx, void *arg) {
  iter_args_t *a = (iter_args_t *)arg;
  int row_offset = (a->m_dim - 1) * a->n_dim;
  for (int j = 0; j < a->n_dim; j++)
    a->w[row_offset + j] = 100.0;
  return NULL;
}

/** @brief 初始化第 idx 行的左右边界 */
void *init_side_boundary(int idx, void *arg) {
  iter_args_t *a = (iter_args_t *)arg;
  int row_offset = idx * a->n_dim;
  a->w[row_offset] = 100.0;
  a->w[row_offset + (a->n_dim - 1)] = 100.0;
  return NULL;
}

/** @brief 初始化第 idx 行的内部点 */
void *init_interior_row(int idx, void *arg) {
  iter_args_t *a = (iter_args_t *)arg;
  int row_offset = idx * a->n_dim;
  for (int j = 1; j < a->n_dim - 1; j++)
    a->w[row_offset + j] = a->mean;
  return NULL;
}

/** @brief 保存第 idx 行: u[idx] = w[idx] */
void *copy_w_to_u_row(int idx, void *arg) {
  iter_args_t *a = (iter_args_t *)arg;
  int row_offset = idx * a->n_dim;
  for (int j = 0; j < a->n_dim; j++)
    a->u[row_offset + j] = a->w[row_offset + j];
  return NULL;
}

/** @brief 更新第 idx 行的内部点 (Jacobi 迭代) */
void *update_interior_row(int idx, void *arg) {
  iter_args_t *a = (iter_args_t *)arg;
  int row_offset = idx * a->n_dim;
  int prev_row = row_offset - a->n_dim;
  int next_row = row_offset + a->n_dim;

  for (int j = 1; j < a->n_dim - 1; j++) {
    a->w[row_offset + j] =
        (a->u[prev_row + j] + a->u[next_row + j] + a->u[row_offset + (j - 1)] +
         a->u[row_offset + (j + 1)]) /
        4.0;
  }
  return NULL;
}

/** @brief 计算第 idx 行的局部最大差值 |w - u| */
void *compute_local_diff_row(int idx, void *arg) {
  iter_args_t *a = (iter_args_t *)arg;
  int row_offset = idx * a->n_dim;
  double local_max = 0.0;
  for (int j = 1; j < a->n_dim - 1; j++) {
    double d = a->w[row_offset + j] - a->u[row_offset + j];
    if (d < 0)
      d = -d;
    if (d > local_max)
      local_max = d;
  }
  a->local_diffs[idx] = local_max;
  return NULL;
}

/** @brief 墙上时间 */
static inline double wtime(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

int main(int argc, char *argv[]) {
  double epsilon = 0.001;
  int num_threads = 4;
  int m = 500, n = 500; /* 默认网格 */
  schedule_type_t schedule = SCHEDULE_STATIC;
  int chunk_size = 1;

  if (argc > 1)
    num_threads = atoi(argv[1]);
  if (argc > 2) {
    m = atoi(argv[2]);
    n = m;
  } /* 支持指定网格大小 */
  if (argc > 3) {
    int s = atoi(argv[3]);
    if (s == 1)
      schedule = SCHEDULE_DYNAMIC;
    else if (s == 2)
      schedule = SCHEDULE_GUIDED;
  }
  if (argc > 4)
    chunk_size = atoi(argv[4]);

  parallel_config_t config;
  config.num_threads = num_threads;
  config.schedule = schedule;
  config.chunk_size = chunk_size;

  /* 动态分配网格 (扁平数组) */
  double *u = (double *)malloc(m * n * sizeof(double));
  double *w = (double *)malloc(m * n * sizeof(double));
  double *local_diffs = (double *)calloc(m, sizeof(double));

  if (!u || !w || !local_diffs) {
    fprintf(stderr, "内存分配失败\n");
    return 1;
  }

  iter_args_t args;
  args.u = u;
  args.w = w;
  args.m_dim = m;
  args.n_dim = n;
  args.local_diffs = local_diffs;

  printf("\n");
  printf("HEATED_PLATE_PTHREADS\n");
  printf("  C/Pthreads version using parallel_for_pool (thread pool)\n");
  printf(
      "  A program to solve for the steady state temperature distribution\n");
  printf("  over a rectangular plate.\n");
  printf("\n");
  printf("  Spatial grid of %d by %d points.\n", m, n);
  printf("  The iteration will be repeated until the change is <= %e\n",
         epsilon);
  printf("  Number of threads = %d\n", num_threads);
  printf("\n");

  /* ----- 初始化边界条件 ----- */
  parallel_for_advanced(0, 1, 1, init_top_row, &args, &config);
  parallel_for_advanced(m - 1, m, 1, init_bottom_row, &args, &config);
  parallel_for_advanced(1, m - 1, 1, init_side_boundary, &args, &config);

  /* 计算边界平均值 */
  double boundary_sum = 0.0;
  for (int j = 0; j < n; j++)
    boundary_sum += w[j]; /* 行 0 */
  for (int j = 0; j < n; j++)
    boundary_sum += w[(m - 1) * n + j]; /* 行 m-1 */
  for (int i = 1; i < m - 1; i++)
    boundary_sum += w[i * n]; /* 左边界 */
  for (int i = 1; i < m - 1; i++)
    boundary_sum += w[i * n + (n - 1)]; /* 右边界 */

  double mean = boundary_sum / (double)(2 * m + 2 * n - 4);
  printf("  MEAN = %f\n", mean);
  args.mean = mean;

  /* 初始化内部点 */
  parallel_for_advanced(1, m - 1, 1, init_interior_row, &args, &config);

  /* ----- 迭代直到收敛 ----- */
  double diff;
  int iterations = 0;
  int iterations_print = 1;

  printf("\n");
  printf(" Iteration  Change\n\n");

  double start_time = wtime();
  diff = epsilon;

  while (epsilon <= diff) {
    /* 第 1 次调用：保存旧解 */
    parallel_for_advanced(0, m, 1, copy_w_to_u_row, &args, &config);
    /* 第 2 次调用：更新内部点 */
    parallel_for_advanced(1, m - 1, 1, update_interior_row, &args, &config);
    /* 第 3 次调用：计算差值 */
    memset(local_diffs, 0, m * sizeof(double));
    parallel_for_advanced(1, m - 1, 1, compute_local_diff_row, &args, &config);

    diff = 0.0;
    for (int i = 1; i < m - 1; i++) {
      if (local_diffs[i] > diff)
        diff = local_diffs[i];
    }

    iterations++;
    /* 按 8 的 n 次方打印 */
    if (iterations == iterations_print) {
      printf("  %8d  %10.6f\n", iterations, diff);
      iterations_print *= 8;
    }
  }

  double end_time = wtime();
  double wtime_elapsed = end_time - start_time;

  printf("\n");
  printf("  %8d  %10.6f\n", iterations, diff);
  printf("\n");
  printf("  Error tolerance achieved.\n");
  printf("  Wallclock time = %f\n", wtime_elapsed);

  parallel_for_pool_destroy();
  free(u);
  free(w);
  free(local_diffs);
  return 0;
}
