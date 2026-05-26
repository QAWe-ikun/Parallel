/**
 * @file parallel_for_pool.c
 * @brief 基于 Pthreads 的线程池 parallel_for 实现（Futex 自旋版）
 *
 * 同步策略：用户态自旋 + Linux futex 混合等待
 *   - Worker：先自旋 SPIN_COUNT 次（纯用户态）→ 超时才 futex_wait
 *   - 主线程：直接 futex_wait（避免忙等竞态）
 *
 * 对标 OpenMP (libgomp) 的 GOMP_SPINCOUNT 机制
 */

#define _GNU_SOURCE
#include "parallel_for_pool.h"
#include <linux/futex.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <unistd.h>

#define MAX_THREADS 64
#define SPIN_COUNT 300000 /* 与 libgomp 默认 GOMP_SPINCOUNT 相同 */

/* ============================================================
 *  Futex 辅助函数
 * ============================================================ */

static inline void futex_wait(volatile int *addr, int val) {
  syscall(SYS_futex, addr, FUTEX_WAIT, val, NULL, NULL, 0);
}

static inline void futex_wake(volatile int *addr, int count) {
  syscall(SYS_futex, addr, FUTEX_WAKE, count, NULL, NULL, 0);
}

static inline void cpu_pause(void) { __asm__ __volatile__("pause"); }

/* ============================================================
 *  线程池全局状态
 * ============================================================ */

typedef struct {
  pthread_t thread;
  int id;
  volatile int signal; /* 主线程设置=1 唤醒 worker */
} worker_t;

typedef struct {
  worker_t workers[MAX_THREADS];
  int num_workers;

  /* 任务参数（只在主线程写、worker 读，通过 signal 的 release/acquire 保证可见性） */
  void *(*functor)(int, void *);
  void *arg;
  int start, end, inc;
  int mode;  /* 0=Static, 1=Dynamic, 2=Guided */
  int chunk;
  int num_active;

  atomic_int dyn_next;     /* 动态调度计数器 */
  atomic_int active_count; /* 剩余活跃线程数 */
  volatile int done_signal; /* 最后一个 worker 设置=1 唤醒主线程 */

  int exit_flag;
} pool_t;

static pool_t g_pool = {0};

/* ============================================================
 *  Worker 线程逻辑
 * ============================================================ */

static void *worker_func(void *arg) {
  worker_t *w = (worker_t *)arg;

  while (1) {
    /* --- 1. 等待任务（自旋 + futex 混合等待） --- */
    for (;;) {
      /* 先自旋（纯用户态，纳秒级） */
      for (int i = 0; i < SPIN_COUNT; i++) {
        if (__atomic_load_n(&w->signal, __ATOMIC_ACQUIRE))
          goto got_signal;
        if (__atomic_load_n(&g_pool.exit_flag, __ATOMIC_RELAXED))
          return NULL;
        cpu_pause();
      }
      /* 自旋超时，进内核等待 */
      if (__atomic_load_n(&w->signal, __ATOMIC_ACQUIRE))
        goto got_signal;
      if (__atomic_load_n(&g_pool.exit_flag, __ATOMIC_RELAXED))
        return NULL;
      futex_wait(&w->signal, 0);
    }
  got_signal:

    /* 消费信号 */
    __atomic_store_n(&w->signal, 0, __ATOMIC_RELEASE);

    /* --- 2. 读取任务参数（acquire 保证可见性） --- */
    void *(*fn)(int, void *) = g_pool.functor;
    void *farg = g_pool.arg;
    int start = g_pool.start;
    int end = g_pool.end;
    int inc = g_pool.inc;
    int mode = g_pool.mode;
    int chunk = g_pool.chunk;
    int nt = g_pool.num_active;

    /* --- 3. 执行工作 --- */
    if (mode == 0) {
      /* 静态调度：按 ID 分配连续块 */
      int total_iters = (end - start + inc - 1) / inc;
      int base = total_iters / nt;
      int rem = total_iters % nt;

      int my_start_idx = w->id * base + (w->id < rem ? w->id : rem);
      int my_count = base + (w->id < rem ? 1 : 0);

      int val = start + my_start_idx * inc;
      int val_end = val + my_count * inc;

      if (val < start) val = start;
      if (val_end > end) val_end = end;

      for (; val < val_end; val += inc) {
        fn(val, farg);
      }
    } else {
      /* 动态/引导调度：原子领取 */
      int my_chunk = chunk;
      if (my_chunk <= 0) my_chunk = 1;

      while (1) {
        int next = atomic_fetch_add_explicit(&g_pool.dyn_next, my_chunk,
                                             memory_order_relaxed);
        if (next >= end) break;

        int le = next + my_chunk;
        if (le > end) le = end;

        for (int i = next; i < le; i += inc) {
          fn(i, farg);
        }

        if (mode == 2 && my_chunk > inc) {
          my_chunk /= 2;
          if (my_chunk < inc) my_chunk = inc;
        }
      }
    }

    /* --- 4. 报告完成 --- */
    int remaining = atomic_fetch_sub_explicit(&g_pool.active_count, 1,
                                              memory_order_acq_rel) - 1;
    if (remaining == 0) {
      /* 最后一个完成的 worker 唤醒主线程 */
      __atomic_store_n(&g_pool.done_signal, 1, __ATOMIC_RELEASE);
      futex_wake(&g_pool.done_signal, 1);
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

  if (old_max == 0) {
    g_pool.exit_flag = 0;
    g_pool.done_signal = 0;
  }

  for (int i = old_max; i < max_workers; i++) {
    g_pool.workers[i].id = i;
    g_pool.workers[i].signal = 0;
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

  /* 动态扩展线程池 */
  if (g_pool.num_workers < num_threads) {
    if (pool_init(num_threads) != 0)
      return -1;
  }

  /* 限制线程数 */
  int total_iters = (end - start + increment - 1) / increment;
  if (num_threads > total_iters)
    num_threads = total_iters;

  /* --- 设置任务参数（在 signal 之前写，release 保证可见性） --- */
  g_pool.functor = functor;
  g_pool.arg = arg;
  g_pool.start = start;
  g_pool.end = end;
  g_pool.inc = increment;
  g_pool.mode = config->schedule;
  g_pool.chunk = (config->chunk_size > 0) ? config->chunk_size : 1;
  g_pool.num_active = num_threads;

  if (config->schedule != SCHEDULE_STATIC) {
    atomic_store(&g_pool.dyn_next, start);
  }

  atomic_store_explicit(&g_pool.active_count, num_threads, memory_order_release);

  /* 重置完成信号 */
  __atomic_store_n(&g_pool.done_signal, 0, __ATOMIC_RELEASE);

  /* 唤醒需要的 worker */
  for (int t = 0; t < num_threads; t++) {
    __atomic_store_n(&g_pool.workers[t].signal, 1, __ATOMIC_RELEASE);
    futex_wake(&g_pool.workers[t].signal, 1);
  }

  /* --- 等待完成（直接 futex_wait，不自旋，避免忙等竞态） --- */
  while (!__atomic_load_n(&g_pool.done_signal, __ATOMIC_ACQUIRE)) {
    futex_wait(&g_pool.done_signal, 0);
  }

  return 0;
}

void parallel_for_pool_destroy(void) {
  if (g_pool.num_workers == 0)
    return;

  __atomic_store_n(&g_pool.exit_flag, 1, __ATOMIC_RELEASE);

  for (int i = 0; i < g_pool.num_workers; i++) {
    __atomic_store_n(&g_pool.workers[i].signal, 1, __ATOMIC_RELEASE);
    futex_wake(&g_pool.workers[i].signal, 1);
  }

  for (int i = 0; i < g_pool.num_workers; i++) {
    pthread_join(g_pool.workers[i].thread, NULL);
  }

  g_pool.num_workers = 0;
}
