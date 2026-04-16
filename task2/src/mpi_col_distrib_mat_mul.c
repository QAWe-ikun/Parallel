#include "D:\Microsoft SDKs\MPI\Include\mpi.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/*
 * MPI 集合通信实现并行通用矩阵乘法 C = A × B
 * 使用 Column-wise Distribution 进行数据划分（与行划分对比）
 * A: m×n, B: n×k, C: m×k
 * 使用 MPI_Type_create_struct 聚合进程内变量后通信
 */

// 定义结构体来聚合矩阵尺寸和其他参数
typedef struct {
    int m, n, k;
    int rows;  // 该进程需要处理的行数（对于A矩阵）
    int cols;  // 该进程需要处理的列数（对于C矩阵）
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

/*
 * 按列划分 C 矩阵 (m x k) -> 每个进程计算 C 的一部分列
 * 每个进程拥有完整的 A (m x n) 和部分 B (n x k/p)
 */
static void get_columnwise_distribution(int rank, int num_procs, int m, int n, int k,
                                       int *out_start_col, int *out_cols) {
  /* 循环分配：将 k 列按进程数平均分配 */
  int cols_per_proc = k / num_procs;
  int remainder = k % num_procs;
  int cols = cols_per_proc + (rank < remainder ? 1 : 0);
  int start_col = rank * cols_per_proc + (rank < remainder ? rank : remainder);

  *out_start_col = start_col;
  *out_cols = cols;
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
    fprintf(stderr, "Usage: mpi_col_distrib_mat_mul <m> <n> <k>\n");
    fprintf(stderr, "   or: mpi_col_distrib_mat_mul <size>  (m=n=k=size)\n");
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
  MPI_Datatype type[5] = {MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_DOUBLE};
  int blocklen[5] = {1, 1, 1, 1, 1};
  MPI_Aint disp[5];

  // 计算每个字段相对于结构体起始地址的偏移量
  MatrixParams dummy;
  MPI_Get_address(&dummy.m, &disp[0]);
  MPI_Get_address(&dummy.n, &disp[1]);
  MPI_Get_address(&dummy.k, &disp[2]);
  MPI_Get_address(&dummy.cols, &disp[3]);
  MPI_Get_address(&dummy.compute_time, &disp[4]);

  // 使偏移量相对于结构体开头为基准
  for (int i = 1; i < 5; i++) {
      disp[i] = disp[i] - disp[0];
  }
  disp[0] = 0;

  MPI_Type_create_struct(5, blocklen, disp, type, &MatrixParamsType);
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

    printf("Matrix size: A(%d x %d) * B(%d x %d) = C(%d x %d)\n", m, n, n, k, m, k);
    printf("Processes: %d (Column-wise distribution)\n", num_procs);

    /* 打印输入矩阵（小规模时） */
    if (m <= 16 && n <= 16 && k <= 16) {
      print_matrix("Matrix A", A, m, n);
      print_matrix("Matrix B", B, n, k);
    }

    /* ===== 集合通信分发数据 ===== */
    double t_comm_start = MPI_Wtime();

    // 广播 A 矩阵给所有进程（因为每个进程需要完整A来计算其C的部分列）
    MPI_Bcast(A, m * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // 计算每个进程的列分配
    int *start_cols = (int*)malloc(num_procs * sizeof(int));
    int *col_counts = (int*)malloc(num_procs * sizeof(int));

    for (int r = 0; r < num_procs; r++) {
        get_columnwise_distribution(r, num_procs, m, n, k, &start_cols[r], &col_counts[r]);
    }

    // 使用 Scatterv 分发 B 矩阵的列给各进程
    int *sendcounts = (int*)malloc(num_procs * sizeof(int));
    int *displs = (int*)malloc(num_procs * sizeof(int));

    for (int r = 0; r < num_procs; r++) {
        sendcounts[r] = n * col_counts[r];  // n行 * col_counts[r]列
        displs[r] = n * start_cols[r];      // 在B矩阵中的偏移
    }

    // 分发 B 矩阵的列
    int my_start_col, my_cols;
    get_columnwise_distribution(rank, num_procs, m, n, k, &my_start_col, &my_cols);
    double *B_local = alloc_matrix(n, my_cols);
    MPI_Scatterv(B, sendcounts, displs, MPI_DOUBLE,
                 B_local, sendcounts[rank], MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    // 使用 struct 发送矩阵参数给各进程
    MatrixParams params;
    params.m = m;
    params.n = n;
    params.k = k;
    params.rows = m;  // 每个进程都有完整的m行A矩阵
    params.cols = my_cols;  // 每个进程计算my_cols列C矩阵

    // 广播矩阵参数（使用自定义结构体类型）
    MPI_Bcast(&params, 1, MatrixParamsType, 0, MPI_COMM_WORLD);

    double t_comm_end = MPI_Wtime();
    double comm_time = t_comm_end - t_comm_start;

    /* ===== 各进程独立计算 C(:, start_col:finish_col) = A * B(:, start_col:finish_col) ===== */
    double *C_local = alloc_matrix(m, my_cols);
    double t_compute_start = MPI_Wtime();
    mat_mul_serial(A, B_local, C_local, m, n, my_cols);
    double t_compute_end = MPI_Wtime();
    double compute_time = t_compute_end - t_compute_start;

    /* 更新参数结构体并发送回根进程 */
    params.compute_time = compute_time;

    /* ===== 收集结果 ===== */
    double t_gather_start = MPI_Wtime();

    // 使用 Gatherv 收集各进程计算的 C 的列
    int *recvcounts = (int*)malloc(num_procs * sizeof(int));
    int *recvdispls = (int*)malloc(num_procs * sizeof(int));

    for (int r = 0; r < num_procs; r++) {
        recvcounts[r] = m * col_counts[r];  // m行 * col_counts[r]列
        recvdispls[r] = m * start_cols[r];  // 在C矩阵中的偏移
    }

    MPI_Gatherv(C_local, m * my_cols, MPI_DOUBLE,
                C, recvcounts, recvdispls, MPI_DOUBLE,
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

    free(A);
    free(B);
    free(C);
    free(start_cols);
    free(col_counts);
    free(sendcounts);
    free(displs);
    free(recvcounts);
    free(recvdispls);
    free(B_local);
    free(C_local);
  } else {
    /* ==================== 非Root进程 ==================== */
    // 接收 A 矩阵（广播）
    double *A_local = alloc_matrix(m, n);  // m, n defined from params
    MPI_Bcast(A_local, m * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // 接收 B 矩阵的列（Scatterv）
    int *sendcounts = (int*)malloc(num_procs * sizeof(int));
    int *displs = (int*)malloc(num_procs * sizeof(int));

    // 这里需要重新计算sendcounts和displs，但由于不知道全局参数，我们采用另一种方式
    // 我们先接收参数，然后根据参数重新计算

    MatrixParams params;
    MPI_Bcast(&params, 1, MatrixParamsType, 0, MPI_COMM_WORLD);

    int m_local = params.m, n_local = params.n, k_local = params.k;

    // 重新计算我的列分布
    int my_start_col, my_cols;
    get_columnwise_distribution(rank, num_procs, m_local, n_local, k_local, &my_start_col, &my_cols);

    // 计算Scatterv参数
    int *temp_start_cols = (int*)malloc(num_procs * sizeof(int));
    int *temp_col_counts = (int*)malloc(num_procs * sizeof(int));

    for (int r = 0; r < num_procs; r++) {
        get_columnwise_distribution(r, num_procs, m_local, n_local, k_local, &temp_start_cols[r], &temp_col_counts[r]);
    }

    for (int r = 0; r < num_procs; r++) {
        sendcounts[r] = n_local * temp_col_counts[r];
        displs[r] = n_local * temp_start_cols[r];
    }

    // 接收 B 矩阵的相应列
    double *B_local = alloc_matrix(n_local, my_cols);
    MPI_Scatterv(NULL, sendcounts, displs, MPI_DOUBLE,  // sendbuf is NULL on non-root
                 B_local, sendcounts[rank], MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    // 计算 C_local = A_local * B_local
    double *C_local = alloc_matrix(m_local, my_cols);
    double t_compute_start = MPI_Wtime();
    mat_mul_serial(A_local, B_local, C_local, m_local, n_local, my_cols);
    double t_compute_end = MPI_Wtime();
    double compute_time = t_compute_end - t_compute_start;

    // 更新参数结构体
    params.compute_time = compute_time;

    // 发送结果回根进程（Gatherv）
    int *recvcounts = (int*)malloc(num_procs * sizeof(int));
    int *recvdispls = (int*)malloc(num_procs * sizeof(int));

    for (int r = 0; r < num_procs; r++) {
        int r_start_col, r_cols;
        get_columnwise_distribution(r, num_procs, m_local, n_local, k_local, &r_start_col, &r_cols);
        recvcounts[r] = m_local * r_cols;
        recvdispls[r] = m_local * r_start_col;
    }

    MPI_Gatherv(C_local, m_local * my_cols, MPI_DOUBLE,
                NULL, recvcounts, recvdispls, MPI_DOUBLE,  // recvbuf is NULL on non-root
                0, MPI_COMM_WORLD);

    free(temp_start_cols);
    free(temp_col_counts);
    free(sendcounts);
    free(displs);
    free(recvcounts);
    free(recvdispls);
    free(A_local);
    free(B_local);
    free(C_local);
  }

  // 清理自定义数据类型
  MPI_Type_free(&MatrixParamsType);

  MPI_Finalize();
  return 0;
}