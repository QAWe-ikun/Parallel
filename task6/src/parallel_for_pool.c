/**
 * @file parallel_for_pool.c
 * @brief 基于 Pthreads 的线程池 parallel_for 实现（稳健版）
 *
 * 使用 Phase + Atomic Counter + Mutex/CondVar 模式。
 * 修复了死锁问题，并优化了同步开销。
 */

#define _GNU_SOURCE
#include "parallel_for_pool.h"
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


#define MAX_THREADS 64

/* ============================================================
 *  线程池全局状态
 * ============================================================ */

typedef struct {
  pthread_t thread;
  int id;
} worker_t;

typedef struct {
  worker_t workers[MAX_THREADS];
  int num_workers;

  /* 同步原语 */
  pthread_mutex_t mutex;
  pthread_cond_t work_cv; /* 唤醒工作线程 */
  pthread_cond_t done_cv; /* 唤醒主线程 */

  /* 状态标记 */
  int phase;               /* 任务代次，每次调用递增 */
  atomic_int active_count; /* 剩余活跃线程数 */
  int num_active;          /* 本轮请求的线程总数 */

  /* 任务参数 */
  void *(*functor)(int, void *);
  void *arg;
  int start, end, inc;
  int mode;            /* 0=Static, 1=Dynamic, 2=Guided */
  int chunk;           /* 动态调度块大小 */
  atomic_int dyn_next; /* 动态调度计数器 */

  int exit_flag;
} pool_t;

static pool_t g_pool = {0};

/* 每个线程记录上次处理的 phase，用于判断是否有新任务 */
static int worker_last_phase[MAX_THREADS] = {0};

/* ============================================================
 *  Worker 线程逻辑
 * ============================================================ */

static void *worker_func(void *arg) {
  worker_t *w = (worker_t *)arg;
  int my_id = w->id;

  while (1) {
    /* --- 1. 等待任务 --- */
    pthread_mutex_lock(&g_pool.mutex);

    /* 等待 phase 变化（有新任务）或退出标志 */
    while (worker_last_phase[my_id] == g_pool.phase && !g_pool.exit_flag) {
      pthread_cond_wait(&g_pool.work_cv, &g_pool.mutex);
    }

    if (g_pool.exit_flag) {
      pthread_mutex_unlock(&g_pool.mutex);
      return NULL;
    }

    /* 领取任务参数（此时持有锁，保证参数一致性） */
    worker_last_phase[my_id] = g_pool.phase;

    void *(*fn)(int, void *) = g_pool.functor;
    void *farg = g_pool.arg;
    int start = g_pool.start;
    int end = g_pool.end;
    int inc = g_pool.inc;
    int mode = g_pool.mode;
    int chunk = g_pool.chunk;
    int nt = g_pool.num_active; /* 活跃线程总数，用于静态调度划分 */

    pthread_mutex_unlock(&g_pool.mutex);

    /* --- 2. 执行工作 --- */
    if (mode == 0) {
      /* 静态调度：基于线程 ID 分配连续块 */
      int total_iters = (end - start + inc - 1) / inc;
      int base = total_iters / nt;
      int rem = total_iters % nt;

      /* 计算当前线程负责的起始索引和数量 */
      int my_start_idx = my_id * base + (my_id < rem ? my_id : rem);
      int my_count = base + (my_id < rem ? 1 : 0);

      /* 转换为实际迭代值 */
      int val = start + my_start_idx * inc;
      int val_end = val + my_count * inc;

      /* 边界保护 */
      if (val < start)
        val = start;
      if (val_end > end)
        val_end = end;

      for (; val < val_end; val += inc) {
        fn(val, farg);
      }
    } else {
      /* 动态/引导调度：原子领取 */
      int my_chunk = chunk;
      if (my_chunk <= 0)
        my_chunk = 1;

      while (1) {
        int next = atomic_fetch_add_explicit(&g_pool.dyn_next, my_chunk,
                                             memory_order_relaxed);
        if (next >= end)
          break;

        int le = next + my_chunk;
        if (le > end)
          le = end;

        for (int i = next; i < le; i += inc) {
          fn(i, farg);
        }

        /* 引导调度：逐渐减小 chunk */
        if (mode == 2 && my_chunk > inc) {
          my_chunk /= 2;
          if (my_chunk < inc)
            my_chunk = inc;
        }
      }
    }

    /* --- 3. 报告完成 --- */
    /* 原子递减活跃计数 */
    int remaining = atomic_fetch_sub_explicit(&g_pool.active_count, 1,
                                              memory_order_release) -
                    1;
    if (remaining <= 0) {
      /* 最后一个完成的线程唤醒主线程 */
      pthread_mutex_lock(&g_pool.mutex);
      pthread_cond_signal(&g_pool.done_cv);
      pthread_mutex_unlock(&g_pool.mutex);
    }
  }
  return NULL;
}

/* ============================================================
 *  公共 API
 * ============================================================ */

static int pool_init(int max_workers) {
  if (g_pool.num_workers >= max_workers && g_pool.num_workers > 0)
    return 0;

  int old_max = g_pool.num_workers;

  /* 如果还没初始化过 */
  if (old_max == 0) {
    pthread_mutex_init(&g_pool.mutex, NULL);
    pthread_cond_init(&g_pool.work_cv, NULL);
    pthread_cond_init(&g_pool.done_cv, NULL);
    g_pool.exit_flag = 0;
    g_pool.phase = 0;
    memset(worker_last_phase, 0, sizeof(worker_last_phase));
  }

  /* 创建新线程 */
  for (int i = old_max; i < max_workers; i++) {
    g_pool.workers[i].id = i;
    worker_last_phase[i] = 0; /* 新线程初始 phase 为 0 */
    if (pthread_create(&g_pool.workers[i].thread, NULL, worker_func,
                       &g_pool.workers[i]) != 0) {
      fprintf(stderr, "Thread creation failed for id %d\n", i);
      return -1;
    }
  }
  g_pool.num_workers = max_workers;
  return 0;
}

int parallel_for(int start, int end, int increment,
                 void *(*functor)(int, void *), void *arg, int num_threads) {
  parallel_config_t config = {0};
  config.num_threads = num_threads;
  config.schedule = SCHEDULE_STATIC;
  config.chunk_size = 1;
  return parallel_for_advanced(start, end, increment, functor, arg, &config);
}

int parallel_for_advanced(int start, int end, int increment,
                          void *(*functor)(int, void *), void *arg,
                          parallel_config_t *config) {
  if (config == NULL || config->num_threads <= 0)
    return -1;

  int num_threads = config->num_threads;

  /* 串行优化 */
  if (num_threads == 1) {
    for (int i = start; i < end; i += increment) {
      functor(i, arg);
    }
    return 0;
  }

  /* 确保线程池足够大 */
  int pool_size =
      (num_threads > 16) ? num_threads : 16; /* 至少预分配 16 个线程 */
  if (pool_init(pool_size) != 0)
    return -1;

  /* 限制线程数不超过总迭代次数 */
  int total_iters = (end - start + increment - 1) / increment;
  if (num_threads > total_iters)
    num_threads = total_iters;

  /* --- 分发任务 --- */
  pthread_mutex_lock(&g_pool.mutex);

  g_pool.functor = functor;
  g_pool.arg = arg;
  g_pool.start = start;
  g_pool.end = end;
  g_pool.inc = increment;
  g_pool.mode = config->schedule;
  g_pool.chunk = (config->chunk_size > 0) ? config->chunk_size : 1;
  g_pool.num_active = num_threads; /* 用于静态调度计算 */

  /* 重置动态调度计数器 */
  if (config->schedule != SCHEDULE_STATIC) {
    atomic_store(&g_pool.dyn_next, start);
  }

  /* 设置活跃计数 */
  atomic_store(&g_pool.active_count, num_threads);

  /* 推进 phase 并广播 */
  g_pool.phase++;
  pthread_cond_broadcast(&g_pool.work_cv);

  /* 等待所有线程完成 */
  while (atomic_load_explicit(&g_pool.active_count, memory_order_acquire) > 0) {
    pthread_cond_wait(&g_pool.done_cv, &g_pool.mutex);
  }

  pthread_mutex_unlock(&g_pool.mutex);

  return 0;
}

void parallel_for_pool_destroy(void) {
  if (g_pool.num_workers == 0)
    return;

  pthread_mutex_lock(&g_pool.mutex);
  g_pool.exit_flag = 1;
  g_pool.phase++; /* 确保所有等待的线程醒来 */
  pthread_cond_broadcast(&g_pool.work_cv);
  pthread_mutex_unlock(&g_pool.mutex);

  for (int i = 0; i < g_pool.num_workers; i++) {
    pthread_join(g_pool.workers[i].thread, NULL);
  }

  pthread_mutex_destroy(&g_pool.mutex);
  pthread_cond_destroy(&g_pool.work_cv);
  pthread_cond_destroy(&g_pool.done_cv);
  g_pool.num_workers = 0;
}
