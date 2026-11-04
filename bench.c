#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include "packer/packer.h"
#include "kernel/kernel.h"

#define ROWS 14336
#define COLS 4096
#define REPS 10
#define NUM_THREADS 8

static float*     g_W;
static float*     g_x;
static float*     g_y;
static block_q8*  g_xq;
static uint8_t*   g_Wq8;
static uint8_t*   g_Wq4;
static size_t     g_rb8, g_rb4;

#ifdef _WIN32
#include <windows.h>
static double now_sec(void) {
    LARGE_INTEGER f, t;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&t);
    return (double)t.QuadPart / (double)f.QuadPart;
}
#else
#define _POSIX_C_SOURCE 199309L
#include <time.h>
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
    int nb = COLS / QK;
    quantize_row_q8(g_x, g_xq, COLS);

    #pragma omp parallel for schedule(static)
    for (int r = 0; r < ROWS; r++)
        g_y[r] = dot_q8_q8((const block_q8*)(g_Wq8 + (size_t)r * g_rb8), g_xq, nb);
}

static void run_q4(void) {
    int nb = COLS / QK;
    quantize_row_q8(g_x, g_xq, COLS);

    #pragma omp parallel for schedule(static)
    for (int r = 0; r < ROWS; r++)
        g_y[r] = dot_q4_q8((const block_q4*)(g_Wq4 + (size_t)r * g_rb4), g_xq, nb);
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
    printf("%-5s %8.2f ms   %6.2f GB/s   %5.2fx faster as f32   (checksum %.3f)\n",
           name, sec * 1e3, (double)bytes / sec / 1e9, base_sec / sec, checksum());
}

int main(void) {
    omp_set_num_threads(NUM_THREADS);

    size_t n = (size_t)ROWS * COLS;
    int nb = COLS / QK;

    g_W  = malloc(n * sizeof(float));
    g_x  = malloc(COLS * sizeof(float));
    g_y  = malloc(ROWS * sizeof(float));
    g_xq = malloc(nb * sizeof(block_q8));

    for (size_t i = 0; i < n; i++)    g_W[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
    for (int i = 0; i < COLS; i++)    g_x[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;

    g_rb8 = packed_row_bytes(COLS, '8');
    g_rb4 = packed_row_bytes(COLS, '4');
    g_Wq8 = malloc(packed_matrix_bytes(ROWS, COLS, '8'));
    g_Wq4 = malloc(packed_matrix_bytes(ROWS, COLS, '4'));
    pack_matrix(g_W, ROWS, COLS, g_Wq8, '8');
    pack_matrix(g_W, ROWS, COLS, g_Wq4, '4');

    printf("Qwen3 FFN Layer Simulation (%d x %d), best of %d runs, %d threads\n\n", ROWS, COLS, REPS, NUM_THREADS);

    double t_f32 = bench(run_f32);
    report("f32", n * sizeof(float), t_f32, t_f32);

    double t_q8 = bench(run_q8);
    report("q8", packed_matrix_bytes(ROWS, COLS, '8'), t_q8, t_f32);

    double t_q4 = bench(run_q4);
    report("q4", packed_matrix_bytes(ROWS, COLS, '4'), t_q4, t_f32);

    free(g_W); free(g_x); free(g_y); free(g_xq); free(g_Wq8); free(g_Wq4);
    return 0;
}