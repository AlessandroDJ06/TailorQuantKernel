#ifndef _WIN32
#define _POSIX_C_SOURCE 199309L
#endif

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include "../packer/packer.h"
#include "../kernel/kernel.h"

#ifdef _WIN32
#include <windows.h>
#else
#include <time.h>
#endif

#define REPS           20
#define ROTATE_BYTES   ((size_t)128 << 20)
#define CHUNK_ROWS     512
#define QUANT_ITERS    2000
#define MAX_THREAD_CFG 8
#define MAX_UNIQUE     64
#define MAX_USES       8
#define MAX_FILTERS    16
#define N_FMT          2

typedef struct {
    const char *name;
    int layers;
    int hidden;
    int inter;
    int n_q_heads;
    int n_kv_heads;
    int head_dim;
    int vocab;
} ModelConfig;

static const ModelConfig MODELS[] = {
    {"qwen3-0.6b",  28, 1024,  3072, 16, 8, 128, 151936},
    {"qwen3-1.7b",  28, 2048,  6144, 16, 8, 128, 151936},
    {"qwen3-4b",    36, 2560,  9728, 32, 8, 128, 151936},
    {"qwen3-8b",    36, 4096, 12288, 32, 8, 128, 151936},
    {"llama3.2-1b", 16, 2048,  8192, 32, 8,  64, 128256},
    {"llama3.2-3b", 28, 3072,  8192, 24, 8, 128, 128256},
};
#define N_MODELS ((int)(sizeof MODELS / sizeof MODELS[0]))

typedef struct { int shape; int per_layer; int per_token; } MatUse;
typedef struct { int quant; int per_layer; int per_token; } QuantUse;

typedef struct {
    const ModelConfig *cfg;
    MatUse   mats[MAX_USES];
    int      n_mats;
    QuantUse quants[MAX_USES];
    int      n_quants;
} ModelPlan;

typedef struct { int rows; int cols; } MatShape;

typedef struct {
    uint8_t *data;
    size_t   bytes;
    size_t   stride;
    int      copies;
} PackedSet;

typedef struct {
    double rel_rms;
    double rel_max;
} ErrStats;

typedef struct {
    double   sec[MAX_THREAD_CFG][N_FMT];
    ErrStats err[N_FMT];
    size_t   bytes[N_FMT];
} ShapeResult;

typedef struct {
    int         threads[MAX_THREAD_CFG];
    int         n_threads;
    int         verbose;
    const char *filters[MAX_FILTERS];
    int         n_filters;
} Options;

static const int   FMT_ID[N_FMT]   = {TQ_FORMAT_Q8, TQ_FORMAT_Q4};
static const char  PACK_ID[N_FMT]  = {'8', '4'};
static const char *FMT_NAME[N_FMT] = {"q8", "q4"};

static ModelPlan   g_plans[N_MODELS];
static int         g_n_plans;
static MatShape    g_shapes[MAX_UNIQUE];
static ShapeResult g_results[MAX_UNIQUE];
static int         g_n_shapes;
static int         g_quant_cols[MAX_UNIQUE];
static double      g_quant_sec[MAX_UNIQUE];
static int         g_n_quant;
static uint64_t    g_rng = 88172645463325252ull;

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

static float uniform_open(void) {
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 7;
    g_rng ^= g_rng << 17;
    return (float)((g_rng >> 40) + 1) / 16777217.0f;
}

static float normal_sample(void) {
    float u1 = uniform_open();
    float u2 = uniform_open();
    return sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);
}

static void fill_normal(float *x, size_t n) {
    for (size_t i = 0; i < n; i++) x[i] = normal_sample();
}

static int register_shape(int rows, int cols) {
    for (int i = 0; i < g_n_shapes; i++) {
        if (g_shapes[i].rows == rows && g_shapes[i].cols == cols) return i;
    }
    g_shapes[g_n_shapes].rows = rows;
    g_shapes[g_n_shapes].cols = cols;
    return g_n_shapes++;
}

static int register_quant(int cols) {
    for (int i = 0; i < g_n_quant; i++) {
        if (g_quant_cols[i] == cols) return i;
    }
    g_quant_cols[g_n_quant] = cols;
    return g_n_quant++;
}

static void add_mat(ModelPlan *plan, int rows, int cols, int per_layer, int per_token) {
    MatUse use = { register_shape(rows, cols), per_layer, per_token };
    plan->mats[plan->n_mats++] = use;
}

static void add_quant(ModelPlan *plan, int cols, int per_layer, int per_token) {
    QuantUse use = { register_quant(cols), per_layer, per_token };
    plan->quants[plan->n_quants++] = use;
}

static int model_is_valid(const ModelConfig *m) {
    const int q_dim = m->n_q_heads * m->head_dim;
    return m->hidden % QK == 0 && m->inter % QK == 0 && q_dim % QK == 0 && m->head_dim % QK == 0;
}

static void build_plan(ModelPlan *plan, const ModelConfig *m) {
    const int q_dim  = m->n_q_heads * m->head_dim;
    const int kv_dim = m->n_kv_heads * m->head_dim;

    plan->cfg = m;
    add_mat(plan, q_dim,   m->hidden, 1, 0);
    add_mat(plan, kv_dim,  m->hidden, 2, 0);
    add_mat(plan, m->hidden, q_dim,   1, 0);
    add_mat(plan, m->inter,  m->hidden, 2, 0);
    add_mat(plan, m->hidden, m->inter,  1, 0);
    add_mat(plan, m->vocab,  m->hidden, 0, 1);

    add_quant(plan, m->hidden, 2, 1);
    add_quant(plan, q_dim,     1, 0);
    add_quant(plan, m->inter,  1, 0);
}

static int model_selected(const ModelConfig *m, const Options *opt) {
    if (opt->n_filters == 0) return 1;
    for (int i = 0; i < opt->n_filters; i++) {
        if (strstr(m->name, opt->filters[i])) return 1;
    }
    return 0;
}

static int build_plans(const Options *opt) {
    for (int i = 0; i < N_MODELS; i++) {
        if (!model_selected(&MODELS[i], opt)) continue;
        if (!model_is_valid(&MODELS[i])) {
            fprintf(stderr, "skipping %s: a dimension is not divisible by %d\n", MODELS[i].name, QK);
            continue;
        }
        build_plan(&g_plans[g_n_plans++], &MODELS[i]);
    }
    return g_n_plans;
}

static int make_set(PackedSet *set, const float *chunk, int chunk_rows, int rows, int cols, char fmt) {
    const size_t chunk_bytes = packed_matrix_bytes(chunk_rows, cols, fmt);
    if (chunk_bytes == 0 || chunk_bytes % (size_t)chunk_rows != 0) return -2;

    const size_t row_bytes = chunk_bytes / (size_t)chunk_rows;
    set->bytes = packed_matrix_bytes(rows, cols, fmt);
    if (set->bytes != row_bytes * (size_t)rows) return -3;

    set->stride = (set->bytes + 63) & ~(size_t)63;
    set->copies = (int)((ROTATE_BYTES + set->bytes - 1) / set->bytes);
    set->data   = malloc(set->stride * (size_t)set->copies);
    if (!set->data) return -1;

    pack_matrix(chunk, chunk_rows, cols, set->data, fmt);
    for (size_t off = chunk_bytes; off < set->bytes; off += chunk_bytes) {
        size_t n = set->bytes - off < chunk_bytes ? set->bytes - off : chunk_bytes;
        memcpy(set->data + off, set->data, n);
    }
    for (int i = 1; i < set->copies; i++) {
        memcpy(set->data + (size_t)i * set->stride, set->data, set->bytes);
    }
    return 0;
}

static void free_set(PackedSet *set) {
    free(set->data);
    set->data = NULL;
}

static ErrStats measure_error(MatmulFn fn, const uint8_t *packed, const float *chunk, const float *x,
                              const block_q8 *xq, float *y, int chunk_rows, int rows, int cols) {
    static double ref[CHUNK_ROWS];

    fn(packed, xq, y, rows, cols);

    double sum_ref2 = 0.0;
    for (int r = 0; r < chunk_rows; r++) {
        const float *row = chunk + (size_t)r * cols;
        double s = 0.0;
        for (int c = 0; c < cols; c++) s += (double)row[c] * (double)x[c];
        ref[r] = s;
        sum_ref2 += s * s;
    }

    const double rms = sqrt(sum_ref2 / chunk_rows);
    double sum_err2 = 0.0, max_err = 0.0;
    for (int r = 0; r < chunk_rows; r++) {
        double e = fabs((double)y[r] - ref[r]);
        sum_err2 += e * e;
        if (e > max_err) max_err = e;
    }

    ErrStats stats = { sqrt(sum_err2 / chunk_rows) / rms, max_err / rms };
    return stats;
}

static double bench_matmul(MatmulFn fn, const PackedSet *set, const block_q8 *xq,
                           float *y, int rows, int cols) {
    double best = 1e30;
    for (int i = 0; i < REPS + 1; i++) {
        const uint8_t *w = set->data + (size_t)(i % set->copies) * set->stride;
        double t0 = now_sec();
        fn(w, xq, y, rows, cols);
        double t = now_sec() - t0;
        if (i > 0 && t < best) best = t;
    }
    return best;
}

static double bench_quantize(int cols) {
    float    *x  = malloc((size_t)cols * sizeof(float));
    block_q8 *xq = malloc((size_t)(cols / QK) * sizeof(block_q8));
    if (!x || !xq) {
        free(x);
        free(xq);
        return -1.0;
    }

    fill_normal(x, (size_t)cols);
    quantize_row_q8(x, xq, cols);

    double t0 = now_sec();
    for (int i = 0; i < QUANT_ITERS; i++) quantize_row_q8(x, xq, cols);
    double sec = (now_sec() - t0) / QUANT_ITERS;

    free(x);
    free(xq);
    return sec;
}

static int run_shape(const MatShape *s, const Options *opt, ShapeResult *res) {
    const int chunk_rows = s->rows < CHUNK_ROWS ? s->rows : CHUNK_ROWS;
    float    *chunk = malloc((size_t)chunk_rows * s->cols * sizeof(float));
    float    *x     = malloc((size_t)s->cols * sizeof(float));
    float    *y     = malloc((size_t)s->rows * sizeof(float));
    block_q8 *xq    = malloc((size_t)(s->cols / QK) * sizeof(block_q8));
    PackedSet sets[N_FMT] = {{0}};
    int status = -1;

    if (!chunk || !x || !y || !xq) goto done;

    fill_normal(chunk, (size_t)chunk_rows * s->cols);
    fill_normal(x, (size_t)s->cols);
    quantize_row_q8(x, xq, s->cols);

    for (int f = 0; f < N_FMT; f++) {
        if (make_set(&sets[f], chunk, chunk_rows, s->rows, s->cols, PACK_ID[f]) != 0) {
            fprintf(stderr, "packing %s failed for %d x %d\n", FMT_NAME[f], s->rows, s->cols);
            goto done;
        }
        res->bytes[f] = sets[f].bytes;
        res->err[f]   = measure_error(get_matmul(FMT_ID[f]), sets[f].data, chunk, x, xq, y,
                                      chunk_rows, s->rows, s->cols);
    }

    for (int t = 0; t < opt->n_threads; t++) {
        omp_set_num_threads(opt->threads[t]);
        for (int f = 0; f < N_FMT; f++) {
            res->sec[t][f] = bench_matmul(get_matmul(FMT_ID[f]), &sets[f], xq, y, s->rows, s->cols);
        }
    }
    status = 0;

done:
    for (int f = 0; f < N_FMT; f++) free_set(&sets[f]);
    free(chunk);
    free(x);
    free(y);
    free(xq);
    return status;
}

static double token_seconds(const ModelPlan *plan, int thread_idx, int fmt) {
    double layer = 0.0, token = 0.0;
    for (int i = 0; i < plan->n_mats; i++) {
        double sec = g_results[plan->mats[i].shape].sec[thread_idx][fmt];
        layer += plan->mats[i].per_layer * sec;
        token += plan->mats[i].per_token * sec;
    }
    for (int i = 0; i < plan->n_quants; i++) {
        double sec = g_quant_sec[plan->quants[i].quant];
        layer += plan->quants[i].per_layer * sec;
        token += plan->quants[i].per_token * sec;
    }
    return plan->cfg->layers * layer + token;
}

static double model_gigabytes(const ModelPlan *plan, int fmt) {
    double layer = 0.0, token = 0.0;
    for (int i = 0; i < plan->n_mats; i++) {
        double bytes = (double)g_results[plan->mats[i].shape].bytes[fmt];
        layer += plan->mats[i].per_layer * bytes;
        token += plan->mats[i].per_token * bytes;
    }
    return (plan->cfg->layers * layer + token) / 1e9;
}

static void print_accuracy(void) {
    printf("accuracy vs double reference (%d rows per shape, relative to rms of the reference)\n", CHUNK_ROWS);
    printf("%14s | %9s %9s | %9s %9s\n", "rows x cols", "q8 rms", "q8 max", "q4 rms", "q4 max");
    for (int s = 0; s < g_n_shapes; s++) {
        const ShapeResult *r = &g_results[s];
        printf("%7d x %-5d | %8.3f%% %8.3f%% | %8.3f%% %8.3f%%\n",
               g_shapes[s].rows, g_shapes[s].cols,
               r->err[0].rel_rms * 100.0, r->err[0].rel_max * 100.0,
               r->err[1].rel_rms * 100.0, r->err[1].rel_max * 100.0);
    }
    printf("\n");
}

static void print_quantize(void) {
    printf("quantize_row_q8 (single thread)\n");
    for (int q = 0; q < g_n_quant; q++) {
        printf("  cols %5d: %7.2f us\n", g_quant_cols[q], g_quant_sec[q] * 1e6);
    }
    printf("\n");
}

static void print_threads(const Options *opt, int thread_idx) {
    const int threads = opt->threads[thread_idx];
    printf("%d thread%s (best of %d, weights rotated over >= %d MB)\n",
           threads, threads == 1 ? "" : "s", REPS, (int)(ROTATE_BYTES >> 20));
    printf("%14s | %9s %7s | %9s %7s\n", "rows x cols", "q8 ms", "GB/s", "q4 ms", "GB/s");

    for (int s = 0; s < g_n_shapes; s++) {
        const ShapeResult *r = &g_results[s];
        printf("%7d x %-5d |", g_shapes[s].rows, g_shapes[s].cols);
        for (int f = 0; f < N_FMT; f++) {
            double sec = r->sec[thread_idx][f];
            printf(" %9.3f %7.1f %s", sec * 1e3, (double)r->bytes[f] / sec / 1e9, f == 0 ? "|" : "");
        }
        printf("\n");
    }
    printf("\n");
}

static void print_model_table(const Options *opt, int fmt) {
    printf("%s: estimated decode tokens/s (upper bound: matmuls and input quantisation only)\n", FMT_NAME[fmt]);
    printf("%-12s %8s |", "model", "size GB");
    for (int t = 0; t < opt->n_threads; t++) printf(" %5dthr", opt->threads[t]);
    printf("\n");

    for (int p = 0; p < g_n_plans; p++) {
        printf("%-12s %8.2f |", g_plans[p].cfg->name, model_gigabytes(&g_plans[p], fmt));
        for (int t = 0; t < opt->n_threads; t++) {
            printf(" %8.1f", 1.0 / token_seconds(&g_plans[p], t, fmt));
        }
        printf("\n");
    }
    printf("\n");
}

static int is_number(const char *s) {
    if (*s == '\0') return 0;
    for (; *s; s++) {
        if (*s < '0' || *s > '9') return 0;
    }
    return 1;
}

static void parse_options(int argc, char **argv, Options *opt) {
    static const int defaults[] = {1, 2, 4, 8};
    memset(opt, 0, sizeof *opt);

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-v") == 0) {
            opt->verbose = 1;
        } else if (is_number(argv[i])) {
            int value = atoi(argv[i]);
            if (value >= 1 && opt->n_threads < MAX_THREAD_CFG) opt->threads[opt->n_threads++] = value;
        } else if (opt->n_filters < MAX_FILTERS) {
            opt->filters[opt->n_filters++] = argv[i];
        }
    }

    if (opt->n_threads > 0) return;
    opt->n_threads = (int)(sizeof defaults / sizeof defaults[0]);
    memcpy(opt->threads, defaults, sizeof defaults);
}

static int measure_all(const Options *opt) {
    for (int q = 0; q < g_n_quant; q++) {
        g_quant_sec[q] = bench_quantize(g_quant_cols[q]);
        if (g_quant_sec[q] >= 0.0) continue;
        fprintf(stderr, "out of memory\n");
        return -1;
    }

    for (int s = 0; s < g_n_shapes; s++) {
        fprintf(stderr, "measuring %d x %d (%d/%d)...\n", g_shapes[s].rows, g_shapes[s].cols, s + 1, g_n_shapes);
        if (run_shape(&g_shapes[s], opt, &g_results[s]) == 0) continue;
        fprintf(stderr, "failed on %d x %d\n", g_shapes[s].rows, g_shapes[s].cols);
        return -1;
    }
    return 0;
}

int main(int argc, char **argv) {
    Options opt;
    parse_options(argc, argv, &opt);

#ifndef __OPTIMIZE__
    printf("WARNING: compiled without optimization (Debug build), numbers are meaningless\n");
#endif
    printf("kernel: %s\n\n", kernel_isa_name());

    for (int f = 0; f < N_FMT; f++) {
        if (get_matmul(FMT_ID[f])) continue;
        fprintf(stderr, "no matmul available for %s\n", FMT_NAME[f]);
        return 1;
    }

    if (build_plans(&opt) == 0) {
        fprintf(stderr, "no model matches the given names\n");
        return 1;
    }
    if (measure_all(&opt) != 0) return 1;

    if (opt.verbose) {
        print_accuracy();
        print_quantize();
        for (int t = 0; t < opt.n_threads; t++) print_threads(&opt, t);
    }
    for (int f = 0; f < N_FMT; f++) print_model_table(&opt, f);
    return 0;
}