#ifndef _WIN32
#define _POSIX_C_SOURCE 199309L
#endif

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "packer/packer.h"
#include "kernel/kernel.h"

#ifdef _WIN32
#include <windows.h>
#else
#include <time.h>
#endif

#define ROWS        14336
#define COLS        4096
#define REPS        10
#define DEF_THREADS 1

static float*    g_W;
static float*    g_x;
static float*    g_y;
static block_q8* g_xq;
static uint8_t*  g_Wq8;
static uint8_t*  g_Wq4;

#ifdef _WIN32
static double now_sec(void) {
    LARGE_INTEGER f, t;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&t);
    return (double)t.QuadPart / (double)f.QuadPart;
}
#else
static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}
#endif


static void run_f32(void) {
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < ROWS; r++) {
        float s = 0.0f;
        const float* row = g_W + (size_t)r * COLS;
        for (int c = 0; c < COLS; c++) s += row[c] * g_x[c];
        g_y[r] = s;
    }
}

static void run_q8(void) {
    quantize_row_q8(g_x, g_xq, COLS);
    matmul_q8(g_Wq8, g_xq, g_y, ROWS, COLS);
}

static void run_q4(void) {
    quantize_row_q8(g_x, g_xq, COLS);
    matmul_q4(g_Wq4, g_xq, g_y, ROWS, COLS);
}

static double bench(void (*fn)(void)) {
    fn();
    double best = 1e30;
    for (int i = 0; i < REPS; i++) {
        double t0 = now_sec();
        fn();
        double t = now_sec() - t0;
        if (t < best) best = t;
    }
    return best;
}

static double checksum(void) {
    double s = 0;
    for (int r = 0; r < ROWS; r++) s += g_y[r];
    return s;
}

static void report(const char* name, size_t bytes, double sec, double base_sec) {
    printf("%-4s %8.2f ms   %6.2f GB/s   %5.2fx vs f32   (checksum %.3f)\n",
           name, sec * 1e3, (double)bytes / sec / 1e9, base_sec / sec, checksum());
}

int main(int argc, char** argv) {
    int threads = (argc > 1) ? atoi(argv[1]) : DEF_THREADS;
    if (threads < 1) threads = DEF_THREADS;
    omp_set_num_threads(threads);

    size_t n  = (size_t)ROWS * COLS;
    int    nb = COLS / QK;

    g_W  = malloc(n * sizeof(float));
    g_x  = malloc(COLS * sizeof(float));
    g_y  = malloc(ROWS * sizeof(float));
    g_xq = malloc((size_t)nb * sizeof(block_q8));

    size_t bytes_q8 = packed_matrix_bytes(ROWS, COLS, '8');
    size_t bytes_q4 = packed_matrix_bytes(ROWS, COLS, '4');
    g_Wq8 = malloc(bytes_q8);
    g_Wq4 = malloc(bytes_q4);

    if (!g_W || !g_x || !g_y || !g_xq || !g_Wq8 || !g_Wq4) {
        fprintf(stderr, "out of memory\n");
        return 1;
    }

    for (size_t i = 0; i < n; i++)  g_W[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
    for (int i = 0; i < COLS; i++)  g_x[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;

    pack_matrix(g_W, ROWS, COLS, g_Wq8, '8');
    pack_matrix(g_W, ROWS, COLS, g_Wq4, '4');

    printf("FFN layer simulation (%d x %d), best of %d runs, %d threads\n\n",
           ROWS, COLS, REPS, threads);

    double t_f32 = bench(run_f32);
    report("f32", n * sizeof(float), t_f32, t_f32);

    double t_q8 = bench(run_q8);
    report("q8", bytes_q8, t_q8, t_f32);

    double t_q4 = bench(run_q4);
    report("q4", bytes_q4, t_q4, t_f32);

    free(g_W); free(g_x); free(g_y); free(g_xq); free(g_Wq8); free(g_Wq4);
    return 0;
}