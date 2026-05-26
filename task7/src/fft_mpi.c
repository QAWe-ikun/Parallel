/**
 * @file fft_mpi.c
 * @brief MPI 并行快速傅里叶变换
 *
 * 并行策略：在 step() 函数的 j 循环上做数据分配（与 OpenMP 版本相同策略）
 *   - 所有进程持有完整数据
 *   - step() 中每个进程只计算 lj/P 个 j 迭代
 *   - 不同 j 写入不重叠位置，无需通信
 *
 * 编译：
 *   mpicc -O2 -o bin/fft_mpi src/fft_mpi.c -lm
 *
 * 运行：
 *   mpirun -np 4 ./bin/fft_mpi
 */

#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* 全局 MPI 信息 */
static int g_rank, g_size;
static MPI_Comm g_comm;

/* 函数声明 */
void ccopy(int n, double x[], double y[]);
void cfft2(int n, double x[], double y[], double w[], double sgn);
void cffti(int n, double w[]);
double ggl(double *seed);
void step(int n, int mj, double a[], double b[], double c[],
          double d[], double w[], double sgn);
void timestamp(void);

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &g_size);
    g_comm = MPI_COMM_WORLD;

    if (g_rank == 0) {
        timestamp();
        printf("\n");
        printf("FFT_MPI\n");
        printf("  C/MPI version\n");
        printf("  Number of processes = %d\n", g_size);
        printf("\n");
        printf("  Accuracy check:\n");
        printf("    FFT ( FFT ( X(1:N) ) ) == N * X(1:N)\n");
        printf("\n");
        printf("             N      NITS    Error         Time          Time/Call     MFLOPS\n");
        printf("\n");
    }

    double error;
    int first;
    double flops, fnm1;
    int i, icase, it, ln2;
    double mflops;
    int n;
    int nits = 10000;
    double seed = 331.0;
    double sgn;
    double *w, *x, *y, *z;

    n = 1;

    for (ln2 = 1; ln2 <= 20; ln2++) {
        n = 2 * n;

        w = (double *)malloc(n * sizeof(double));
        x = (double *)malloc(2 * n * sizeof(double));
        y = (double *)malloc(2 * n * sizeof(double));
        z = (double *)malloc(2 * n * sizeof(double));

        first = 1;

        for (icase = 0; icase < 2; icase++) {
            if (first) {
                if (g_rank == 0) {
                    for (i = 0; i < 2 * n; i += 2) {
                        double z0 = ggl(&seed);
                        double z1 = ggl(&seed);
                        x[i] = z0;
                        z[i] = z0;
                        x[i + 1] = z1;
                        z[i + 1] = z1;
                    }
                }
                MPI_Bcast(x, 2 * n, MPI_DOUBLE, 0, g_comm);
                MPI_Bcast(z, 2 * n, MPI_DOUBLE, 0, g_comm);
            } else {
                memset(x, 0, 2 * n * sizeof(double));
                memset(z, 0, 2 * n * sizeof(double));
            }

            cffti(n, w);

            if (first) {
                sgn = +1.0;
                cfft2(n, x, y, w, sgn);
                sgn = -1.0;
                cfft2(n, y, x, w, sgn);

                if (g_rank == 0) {
                    fnm1 = 1.0 / (double)n;
                    error = 0.0;
                    for (i = 0; i < 2 * n; i += 2) {
                        error += pow(z[i] - fnm1 * x[i], 2) +
                                 pow(z[i + 1] - fnm1 * x[i + 1], 2);
                    }
                    error = sqrt(fnm1 * error);
                    printf("  %12d  %8d  %12e", n, nits, error);
                }
                first = 0;
            } else {
                MPI_Barrier(g_comm);
                double ctime1 = MPI_Wtime();

                for (it = 0; it < nits; it++) {
                    sgn = +1.0;
                    cfft2(n, x, y, w, sgn);
                    sgn = -1.0;
                    cfft2(n, y, x, w, sgn);
                }

                MPI_Barrier(g_comm);
                double ctime = MPI_Wtime() - ctime1;

                flops = 2.0 * (double)nits * (5.0 * (double)n * (double)ln2);
                mflops = flops / 1.0E+06 / ctime;

                if (g_rank == 0) {
                    printf("  %12e  %12e  %12f\n",
                           ctime, ctime / (double)(2 * nits), mflops);
                }
            }
        }

        if ((ln2 % 4) == 0) {
            nits = nits / 10;
        }
        if (nits < 1) {
            nits = 1;
        }

        free(w);
        free(x);
        free(y);
        free(z);
    }

    if (g_rank == 0) {
        printf("\n");
        printf("FFT_MPI:\n");
        printf("  Normal end of execution.\n");
        printf("\n");
        timestamp();
    }

    MPI_Finalize();
    return 0;
}

/* ============================================================
 *  并行 FFT 函数
 * ============================================================ */

void ccopy(int n, double x[], double y[]) {
    for (int i = 0; i < n; i++) {
        y[i * 2 + 0] = x[i * 2 + 0];
        y[i * 2 + 1] = x[i * 2 + 1];
    }
}

void cfft2(int n, double x[], double y[], double w[], double sgn) {
    int m = (int)(log((double)n) / log(1.99));
    int mj = 1;
    int tgle = 1;

    step(n, mj, &x[0], &x[(n / 2) * 2], &y[0], &y[mj * 2], w, sgn);

    if (n == 2) return;

    for (int j = 0; j < m - 2; j++) {
        mj = mj * 2;
        if (tgle) {
            step(n, mj, &y[0], &y[(n / 2) * 2], &x[0], &x[mj * 2], w, sgn);
            tgle = 0;
        } else {
            step(n, mj, &x[0], &x[(n / 2) * 2], &y[0], &y[mj * 2], w, sgn);
            tgle = 1;
        }
    }

    if (tgle) {
        ccopy(n, y, x);
    }

    mj = n / 2;
    step(n, mj, &x[0], &x[(n / 2) * 2], &y[0], &y[mj * 2], w, sgn);
}

void cffti(int n, double w[]) {
    double arg, aw;
    int i, n2;
    const double pi = 3.141592653589793;

    n2 = n / 2;
    aw = 2.0 * pi / ((double)n);

    for (i = 0; i < n2; i++) {
        arg = aw * ((double)i);
        w[i * 2 + 0] = cos(arg);
        w[i * 2 + 1] = sin(arg);
    }
}

double ggl(double *seed) {
    double d2 = 0.2147483647e10;
    double t, value;

    t = *seed;
    t = fmod(16807.0 * t, d2);
    *seed = t;
    value = (t - 1.0) / (d2 - 1.0);

    return value;
}

/**
 * step() - 并行版本
 *
 * 与 OpenMP 版本相同策略：在 j 循环上做分配
 * 每个进程计算 j ∈ [j_start, j_end) 的 butterfly 操作
 *
 * 不同 j 写入 c/d 的位置不重叠（j 写到 [j*mj2, (j+1)*mj2)），
 * 但下一步 step() 需要完整的 c/d 作为输入，
 * 因此必须用 MPI_Allreduce(SUM) 将各进程的部分结果合并为完整数组。
 */
void step(int n, int mj, double a[], double b[], double c[],
          double d[], double w[], double sgn) {
    double ambr, ambu;
    int j, ja, jb, jc, jd, jw, k, lj, mj2;
    double wjw[2];

    mj2 = 2 * mj;
    lj = n / mj2;

    /* 分配 j 循环给各进程 */
    int j_base = lj / g_size;
    int j_rem = lj % g_size;
    int j_start = g_rank * j_base + (g_rank < j_rem ? g_rank : j_rem);
    int j_end = j_start + j_base + (g_rank < j_rem ? 1 : 0);

    if (g_size == 1) {
        /* 单进程：直接计算，无通信开销 */
        for (j = 0; j < lj; j++) {
            jw = j * mj;
            ja = jw;
            jb = ja;
            jc = j * mj2;
            jd = jc;

            wjw[0] = w[jw * 2 + 0];
            wjw[1] = w[jw * 2 + 1];
            if (sgn < 0.0) wjw[1] = -wjw[1];

            for (k = 0; k < mj; k++) {
                c[(jc + k) * 2 + 0] = a[(ja + k) * 2 + 0] + b[(jb + k) * 2 + 0];
                c[(jc + k) * 2 + 1] = a[(ja + k) * 2 + 1] + b[(jb + k) * 2 + 1];

                ambr = a[(ja + k) * 2 + 0] - b[(jb + k) * 2 + 0];
                ambu = a[(ja + k) * 2 + 1] - b[(jb + k) * 2 + 1];

                d[(jd + k) * 2 + 0] = wjw[0] * ambr - wjw[1] * ambu;
                d[(jd + k) * 2 + 1] = wjw[1] * ambr + wjw[0] * ambu;
            }
        }
    } else {
        /* 多进程：各进程计算自己的 j 区间，写入局部数组，然后 Allreduce 合并
         *
         * 为什么需要 Allreduce：
         *   cfft2() 中相邻 step() 调用的输入依赖上一步的完整输出。
         *   每个进程只计算了部分 j，c/d 只有部分位置有值，
         *   必须通过 Allreduce(SUM) 让所有进程都得到完整数组。
         *
         * 为什么用 SUM：
         *   不同 j 写入不重叠位置，非本进程位置保持 0，
         *   SUM 后每个位置恰好只有一个进程的非零贡献。
         */
        int arr_len = 2 * n;
        /* 单个临时数组，模拟 y 数组的完整布局 */
        double *local_y = (double *)calloc(arr_len, sizeof(double));
        double *tmp_y = (double *)malloc(arr_len * sizeof(double));

        /* d 相对于 c 的偏移量（以 double 为单位） */
        int d_offset = mj * 2;

        for (j = j_start; j < j_end; j++) {
            jw = j * mj;
            ja = jw;
            jb = ja;
            jc = j * mj2;
            jd = jc;

            wjw[0] = w[jw * 2 + 0];
            wjw[1] = w[jw * 2 + 1];
            if (sgn < 0.0) wjw[1] = -wjw[1];

            for (k = 0; k < mj; k++) {
                /* c 的位置：相对于 c 指针（即 y[0]） */
                local_y[(jc + k) * 2 + 0] = a[(ja + k) * 2 + 0] + b[(jb + k) * 2 + 0];
                local_y[(jc + k) * 2 + 1] = a[(ja + k) * 2 + 1] + b[(jb + k) * 2 + 1];

                ambr = a[(ja + k) * 2 + 0] - b[(jb + k) * 2 + 0];
                ambu = a[(ja + k) * 2 + 1] - b[(jb + k) * 2 + 1];

                /* d 的位置：相对于 c 指针偏移 d_offset */
                local_y[d_offset + (jd + k) * 2 + 0] = wjw[0] * ambr - wjw[1] * ambu;
                local_y[d_offset + (jd + k) * 2 + 1] = wjw[1] * ambr + wjw[0] * ambu;
            }
        }

        /* 单一 Allreduce 合并整个 y 数组 */
        MPI_Allreduce(local_y, tmp_y, arr_len, MPI_DOUBLE, MPI_SUM, g_comm);

        /* 复制回 c（即 y[0]），包含 c 和 d 的所有值 */
        memcpy(c, tmp_y, arr_len * sizeof(double));

        free(local_y);
        free(tmp_y);
    }
}

void timestamp(void) {
    time_t now;
    struct tm *tm_info;
    char time_buffer[40];

    now = time(NULL);
    tm_info = localtime(&now);

    strftime(time_buffer, 40, "%d %B %Y %I:%M:%S %p", tm_info);

    printf("%s\n", time_buffer);
}
