#include "D:\Microsoft SDKs\MPI\Include\mpi.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/*
 * MPI 集合通信实现并行通用矩阵乘法 C = A × B
 * 使用 MPI_Bcast 广播 B 矩阵，MPI_Scatterv 分发 A 矩阵行块，MPI_Gatherv 收集结果
 * A: m×n, B: n×k, C: m×k
 * 使用 MPI_Type_create_struct 聚合进程内变量后通信
 */

// 定义结构体来聚合矩阵尺寸和其他参数
typedef struct {
    int m, n, k;
    int rows;  // 该进程需要处理的行数
    double compute_time;  // 计算时间
} MatrixParams;

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

static void print_matrix(const char *name, double *mat, int rows, int cols) {
  printf("=== %s (%d x %d) ===\n", name, rows, cols);
  int max_print = 8;
  int pr = (rows <= max_print) ? rows : max_print;
  int pc = (cols <= max_print) ? cols : max_print;
  for (int i = 0; i < pr; i++) {
    for (int j = 0; j < pc; j++)
      printf("%8.4f ", mat[i * cols + j]);
    if (pc < cols)
      printf("... ");
    printf("\n");
  }
  if (pr < rows)
    printf("...\n");
}

/* 串行矩阵乘法 C = A × B (ikj 顺序，缓存友好) */
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

/* 计算每个进程的分片信息 */
static void get_worker_info(int rank, int num_procs, int m, int n, int k,
                            int *out_start_row, int *out_rows) {
  /* 循环分配：将 m 行按进程数平均分配 */
  int rows_per_proc = m / num_procs;
  int remainder = m % num_procs;
  int rows = rows_per_proc + (rank < remainder ? 1 : 0);
  int start_row = rank * rows_per_proc + (rank < remainder ? rank : remainder);
  *out_start_row = start_row;
  *out_rows = rows;
}

int main(int argc, char **argv) {
  int m, n, k;

  /* 解析参数 */
  if (argc >= 4) {
    m = atoi(argv[1]);
    n = atoi(argv[2]);
    k = atoi(argv[3]);
    if (m < 128 || m > 2048 || n < 128 || n > 2048 || k < 128 || k > 2048) {
      fprintf(stderr, "Error: m, n, k must be in range [128, 2048]\n");
      return 1;
    }
  } else if (argc == 2) {
    int size = atoi(argv[1]);
    if (size < 128 || size > 2048) {
      fprintf(stderr, "Error: matrix size must be in range [128, 2048]\n");
      return 1;
    }
    m = n = k = size;
  } else {
    fprintf(stderr, "Usage: mpi_collective_mat_mul <m> <n> <k>\n");
    fprintf(stderr, "   or: mpi_collective_mat_mul <size>  (m=n=k=size)\n");
    return 1;
  }

  /* MPI 初始化 */
  MPI_Init(&argc, &argv);
  int rank, num_procs;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

  if (num_procs < 2) {
    if (rank == 0)
      fprintf(stderr,
              "Error: need at least 2 processes (require multiple processes for collective operations)\n");
    MPI_Finalize();
    return 1;
  }

  // 创建自定义结构体数据类型
  MPI_Datatype MatrixParamsType;
  MPI_Datatype type[4] = {MPI_INT, MPI_INT, MPI_INT, MPI_DOUBLE};
  int blocklen[4] = {1, 1, 1, 1};
  MPI_Aint disp[4];

  // 计算每个字段相对于结构体起始地址的偏移量
  MatrixParams dummy;
  MPI_Get_address(&dummy.m, &disp[0]);
  MPI_Get_address(&dummy.n, &disp[1]);
  MPI_Get_address(&dummy.k, &disp[2]);
  MPI_Get_address(&dummy.compute_time, &disp[3]);

  // 使偏移量相对于结构体开头为基准
  for (int i = 1; i < 4; i++) {
      disp[i] = disp[i] - disp[0];
  }
  disp[0] = 0;

  MPI_Type_create_struct(4, blocklen, disp, type, &MatrixParamsType);
  MPI_Type_commit(&MatrixParamsType);

  if (rank == 0) {
    /* ==================== Rank 0: Root process ==================== */
    srand(42);

    /* 生成矩阵 A 和 B */
    double *A = alloc_matrix(m, n);
    double *B = alloc_matrix(n, k);
    double *C = alloc_matrix(m, k);
    fill_random(A, m, n);
    fill_random(B, n, k);

    printf("Matrix size: A(%d x %d) * B(%d x %d) = C(%d x %d)\n", m, n, n, k, m,
           k);
    printf("Processes: %d\n", num_procs);

    /* 打印输入矩阵（小规模时） */
    if (m <= 16 && n <= 16 && k <= 16) {
      print_matrix("Matrix A", A, m, n);
      print_matrix("Matrix B", B, n, k);
    }

    /* ===== 集合通信分发数据 ===== */
    double t_comm_start = MPI_Wtime();

    // 广播 B 矩阵给所有进程
    MPI_Bcast(B, n * k, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // 使用 struct 发送矩阵参数给各进程
    MatrixParams params;
    params.m = m;
    params.n = n;
    params.k = k;

    // 广播矩阵参数（使用自定义结构体类型）
    MPI_Bcast(&params, 1, MatrixParamsType, 0, MPI_COMM_WORLD);

    // 准备 Scatterv 参数
    int *sendcounts = (int*)malloc(num_procs * sizeof(int));
    int *displs = (int*)malloc(num_procs * sizeof(int));

    for (int r = 0; r < num_procs; r++) {
      int start_row, rows;
      get_worker_info(r, num_procs, m, n, k, &start_row, &rows);
      sendcounts[r] = rows * n;
      displs[r] = start_row * n;
    }

    // 将 A 矩阵按行分块分发给各进程
    int my_start_row, my_rows;
    get_worker_info(rank, num_procs, m, n, k, &my_start_row, &my_rows);
    double *A_local = alloc_matrix(my_rows, n); // 本地接收数组，只分配自己需要的部分
    MPI_Scatterv(A, sendcounts, displs, MPI_DOUBLE,
                 A_local, sendcounts[rank], MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    double t_comm_end = MPI_Wtime();
    double comm_time = t_comm_end - t_comm_start;

    /* ===== 各进程独立计算 ===== */
    double *A_sub = A_local;
    double *B_local = B; // 本地 B 矩阵（完整）
    double *C_sub = alloc_matrix(my_rows, k);

    double t_compute_start = MPI_Wtime();
    mat_mul_serial(A_sub, B_local, C_sub, my_rows, n, k);
    double t_compute_end = MPI_Wtime();
    double compute_time = t_compute_end - t_compute_start;

    /* 更新参数结构体并发送回根进程 */
    params.compute_time = compute_time;
    params.rows = my_rows;

    /* ===== 收集结果 ===== */
    double t_gather_start = MPI_Wtime();

    // 准备 Gatherv 参数
    int *recvcounts = (int*)malloc(num_procs * sizeof(int));
    int *recvdispls = (int*)malloc(num_procs * sizeof(int));

    for (int r = 0; r < num_procs; r++) {
      int start_row, rows;
      get_worker_info(r, num_procs, m, n, k, &start_row, &rows);
      recvcounts[r] = rows * k;
      recvdispls[r] = start_row * k;
    }

    // 各进程发送自己的 C_sub 回汇总到 C 矩阵
    MPI_Gatherv(C_sub, my_rows * k, MPI_DOUBLE,
                C, recvcounts, recvdispls, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    // 收集各进程的计算时间信息
    MatrixParams *all_params = (MatrixParams*)malloc(num_procs * sizeof(MatrixParams));
    MPI_Gather(&params, 1, MatrixParamsType,
               all_params, 1, MatrixParamsType,
               0, MPI_COMM_WORLD);

    double t_gather_end = MPI_Wtime();
    double gather_time = t_gather_end - t_gather_start;
    double end_to_end_time = comm_time + compute_time + gather_time;

    printf("Comm Time:     %.6f seconds\n", comm_time);
    printf("Compute Time:  %.6f seconds (pure computation)\n", compute_time);
    printf("Gather Time:   %.6f seconds\n", gather_time);
    printf("Total Time:    %.6f seconds\n", end_to_end_time);

    /* 打印结果矩阵（小规模时） */
    if (m <= 16 && k <= 16) {
      print_matrix("Matrix C", C, m, k);
    }

    /* 验证正确性：串行计算并比较 */
    if (m <= 512 && n <= 512 && k <= 512) {
      double *C_check = alloc_matrix(m, k);
      mat_mul_serial(A, B, C_check, m, n, k);

      double max_err = 0.0;
      for (int i = 0; i < m * k; i++) {
        double err = fabs(C[i] - C_check[i]);
        if (err > max_err)
          max_err = err;
      }
      if (max_err < 1e-8)
        printf("Verification: PASSED (max error = %.2e)\n", max_err);
      else
        printf("Verification: FAILED (max error = %.2e)\n", max_err);
      free(C_check);
    }

    /* 性能分析：对比串行版本 */
    double t_serial_start = MPI_Wtime();
    {
      double *C_serial = alloc_matrix(m, k);
      mat_mul_serial(A, B, C_serial, m, n, k);
      free(C_serial);
    }
    double t_serial_end = MPI_Wtime();
    double serial_time = t_serial_end - t_serial_start;

    printf("Serial Time:   %.6f seconds\n", serial_time);
    if (end_to_end_time > 0.00001)
      printf("Speedup:       %.2fx\n", serial_time / end_to_end_time);
    else
      printf("Speedup:       N/A (too fast)\n");

    free(all_params);
    free(A);
    free(B);
    free(C);
    free(sendcounts);
    free(displs);
    free(recvcounts);
    free(recvdispls);
    free(A_local);
    free(C_sub);
  } else {
    /* ==================== 非Root进程 ==================== */
    // 接收 B 矩阵（广播）
    double *B_local = alloc_matrix(n, k);
    MPI_Bcast(B_local, n * k, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // 接收矩阵参数（使用自定义结构体类型）
    MatrixParams params;
    MPI_Bcast(&params, 1, MatrixParamsType, 0, MPI_COMM_WORLD);

    int m_local = params.m, n_local = params.n, k_local = params.k;

    // 准备 Scatterv 参数
    int *sendcounts = (int*)malloc(num_procs * sizeof(int));
    int *displs = (int*)malloc(num_procs * sizeof(int));

    for (int r = 0; r < num_procs; r++) {
      int start_row, rows;
      get_worker_info(r, num_procs, m_local, n_local, k_local, &start_row, &rows);
      sendcounts[r] = rows * n_local;
      displs[r] = start_row * n_local;
    }

    // 获取自己的分片信息
    int my_start_row, my_rows;
    get_worker_info(rank, num_procs, m_local, n_local, k_local, &my_start_row, &my_rows);

    // 接收 A 矩阵子块
    double *A_local = alloc_matrix(my_rows, n_local);
    MPI_Scatterv(NULL, sendcounts, displs, MPI_DOUBLE,  // sendbuf is NULL on non-root
                 A_local, sendcounts[rank], MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    // 计算 C_sub = A_sub × B_local
    double *C_sub = alloc_matrix(my_rows, k_local);
    double t_compute_start = MPI_Wtime();
    mat_mul_serial(A_local, B_local, C_sub, my_rows, n_local, k_local);
    double t_compute_end = MPI_Wtime();
    double compute_time = t_compute_end - t_compute_start;

    // 更新参数结构体
    params.compute_time = compute_time;
    params.rows = my_rows;

    // 准备 Gatherv 参数
    int *recvcounts = (int*)malloc(num_procs * sizeof(int));
    int *recvdispls = (int*)malloc(num_procs * sizeof(int));

    for (int r = 0; r < num_procs; r++) {
      int start_row, rows;
      get_worker_info(r, num_procs, m_local, n_local, k_local, &start_row, &rows);
      recvcounts[r] = rows * k_local;
      recvdispls[r] = start_row * k_local;
    }

    // 发送结果回 root 进程
    MPI_Gatherv(C_sub, my_rows * k_local, MPI_DOUBLE,
                NULL, recvcounts, recvdispls, MPI_DOUBLE,  // recvbuf is NULL on non-root
                0, MPI_COMM_WORLD);

    // 发送参数回根进程
    MPI_Gather(&params, 1, MatrixParamsType,
               NULL, 1, MatrixParamsType,
               0, MPI_COMM_WORLD);

    free(A_local);
    free(B_local);
    free(C_sub);
    free(sendcounts);
    free(displs);
    free(recvcounts);
    free(recvdispls);
  }

  // 清理自定义数据类型
  MPI_Type_free(&MatrixParamsType);

  MPI_Finalize();
  return 0;
}