#ifndef _WIN32
#define _POSIX_C_SOURCE 199309L
#endif

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#include "model/model_ops.h"
#include "packer/packer.h"
#include "kernel/kernel.h"

#ifdef _WIN32
#include <windows.h>
#else
#include <time.h>
#endif

#define N_LAYERS       36
#define HIDDEN         2560
#define INTER          9728
#define N_Q_HEADS      32
#define N_KV_HEADS     8
#define HEAD_DIM       128
#define VOCAB          151936
#define Q_DIM          (N_Q_HEADS * HEAD_DIM)
#define KV_DIM         (N_KV_HEADS * HEAD_DIM)

#define CHUNK_ROWS     512
#define WEIGHT_SCALE   0.02f
#define WARMUP_TOKENS  1
#define TIMED_TOKENS   6
#define MAX_THREAD_CFG 8
#define N_POSITIONS    4

static const int POSITIONS[N_POSITIONS] = {0, 256, 1024, 4096};

typedef struct {
    tq_format  fmt;
    tq_kv_type kv;
    int        threads[MAX_THREAD_CFG];
    int        n_threads;
} Options;

static uint64_t g_rng = 88172645463325252ull;
static float    g_ones_hidden[HIDDEN];
static float    g_ones_head[HEAD_DIM];
static double   g_sec[MAX_THREAD_CFG][N_POSITIONS];

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

static char pack_id(tq_format fmt) {
    return fmt == TQ_FORMAT_Q8 ? '8' : '4';
}

static size_t matrix_bytes(const tq_matrix *m) {
    return packed_matrix_bytes(m->rows, m->cols, pack_id(m->format));
}

static int make_matrix(tq_matrix *m, int rows, int cols, tq_format fmt) {
    const char   id         = pack_id(fmt);
    const int    chunk_rows = rows < CHUNK_ROWS ? rows : CHUNK_ROWS;
    const size_t chunk_n    = (size_t)chunk_rows * cols;
    const size_t chunk_size = packed_matrix_bytes(chunk_rows, cols, id);
    const size_t total      = packed_matrix_bytes(rows, cols, id);

    float   *chunk = malloc(chunk_n * sizeof(float));
    uint8_t *data  = malloc(total);
    if (!chunk || !data) {
        free(chunk);
        free(data);
        return -1;
    }

    for (size_t i = 0; i < chunk_n; i++) chunk[i] = normal_sample() * WEIGHT_SCALE;

    int rc = pack_matrix(chunk, chunk_rows, cols, data, id);
    free(chunk);
    if (rc != 0) {
        free(data);
        return -1;
    }

    for (size_t off = chunk_size; off < total; off += chunk_size) {
        size_t n = total - off < chunk_size ? total - off : chunk_size;
        memcpy(data + off, data, n);
    }

    m->data   = data;
    m->format = fmt;
    m->rows   = rows;
    m->cols   = cols;
    return 0;
}

static void free_matrix(tq_matrix *m) {
    free((void *)m->data);
    m->data = NULL;
}

static int build_layer(tq_layer_weights *lw, tq_format fmt) {
    lw->input_norm  = g_ones_hidden;
    lw->post_norm   = g_ones_hidden;
    lw->attn.q_norm = g_ones_head;
    lw->attn.k_norm = g_ones_head;

    if (make_matrix(&lw->attn.wq,   Q_DIM,  HIDDEN, fmt) != 0) return -1;
    if (make_matrix(&lw->attn.wk,   KV_DIM, HIDDEN, fmt) != 0) return -1;
    if (make_matrix(&lw->attn.wv,   KV_DIM, HIDDEN, fmt) != 0) return -1;
    if (make_matrix(&lw->attn.wo,   HIDDEN, Q_DIM,  fmt) != 0) return -1;
    if (make_matrix(&lw->mlp.wgate, INTER,  HIDDEN, fmt) != 0) return -1;
    if (make_matrix(&lw->mlp.wup,   INTER,  HIDDEN, fmt) != 0) return -1;
    if (make_matrix(&lw->mlp.wdown, HIDDEN, INTER,  fmt) != 0) return -1;
    return 0;
}

static void free_layer(tq_layer_weights *lw) {
    free_matrix(&lw->attn.wq);
    free_matrix(&lw->attn.wk);
    free_matrix(&lw->attn.wv);
    free_matrix(&lw->attn.wo);
    free_matrix(&lw->mlp.wgate);
    free_matrix(&lw->mlp.wup);
    free_matrix(&lw->mlp.wdown);
}

static void free_model(tq_model *m) {
    if (m->w.layers) {
        for (int l = 0; l < m->cfg.n_layers; l++) free_layer(&m->w.layers[l]);
    }
    free(m->w.layers);
    free_matrix(&m->w.embed);
    memset(m, 0, sizeof *m);
}

static int build_model(tq_model *m, tq_format fmt) {
    memset(m, 0, sizeof *m);

    for (int i = 0; i < HIDDEN; i++)   g_ones_hidden[i] = 1.0f;
    for (int i = 0; i < HEAD_DIM; i++) g_ones_head[i]   = 1.0f;

    m->cfg.attn.n_q_heads   = N_Q_HEADS;
    m->cfg.attn.n_kv_heads  = N_KV_HEADS;
    m->cfg.attn.head_dim    = HEAD_DIM;
    m->cfg.attn.hidden_size = HIDDEN;
    m->cfg.attn.rope_theta  = 1000000.0f;
    m->cfg.attn.rms_eps     = 1e-6f;
    m->cfg.n_layers         = N_LAYERS;
    m->cfg.intermediate_size = INTER;
    m->cfg.vocab_size       = VOCAB;

    m->w.final_norm = g_ones_hidden;
    m->w.layers     = calloc(N_LAYERS, sizeof(tq_layer_weights));
    if (!m->w.layers) return -1;

    if (make_matrix(&m->w.embed, VOCAB, HIDDEN, fmt) != 0) {
        free_model(m);
        return -1;
    }
    for (int l = 0; l < N_LAYERS; l++) {
        if (build_layer(&m->w.layers[l], fmt) != 0) {
            free_model(m);
            return -1;
        }
    }
    return 0;
}

static double model_gigabytes(const tq_model *m) {
    const tq_layer_weights *lw = &m->w.layers[0];
    size_t layer = matrix_bytes(&lw->attn.wq) + matrix_bytes(&lw->attn.wk) +
                   matrix_bytes(&lw->attn.wv) + matrix_bytes(&lw->attn.wo) +
                   matrix_bytes(&lw->mlp.wgate) + matrix_bytes(&lw->mlp.wup) +
                   matrix_bytes(&lw->mlp.wdown);
    return ((double)layer * N_LAYERS + (double)matrix_bytes(&m->w.embed)) / 1e9;
}

static void zero_caches(tq_ctx *ctx) {
    const size_t slots = (size_t)N_KV_HEADS * ctx->max_tokens;
    for (int l = 0; l < N_LAYERS; l++) {
        tq_kv_cache *c = &ctx->caches[l];
        if (c->type == TQ_KV_F32) {
            memset(c->k_f32, 0, slots * HEAD_DIM * sizeof(float));
            memset(c->v_f32, 0, slots * HEAD_DIM * sizeof(float));
        } else {
            memset(c->k_q8, 0, slots * (HEAD_DIM / QK) * sizeof(block_q8));
            memset(c->v_q8, 0, slots * (HEAD_DIM / QK) * sizeof(block_q8));
        }
    }
}

static int run_tokens(tq_ctx *ctx, int count) {
    for (int i = 0; i < count; i++) {
        int token = (int)(((long long)ctx->pos * 7919 + 13) % VOCAB);
        if (tq_forward_token(ctx, token) != 0) return -1;
    }
    return 0;
}

static int logits_finite(const tq_ctx *ctx) {
    for (int i = 0; i < VOCAB; i++) {
        if (!isfinite(ctx->logits[i])) return 0;
    }
    return 1;
}

static double time_position(tq_ctx *ctx, int pos) {
    ctx->pos = pos;
    if (run_tokens(ctx, WARMUP_TOKENS) != 0) return -1.0;

    double t0 = now_sec();
    if (run_tokens(ctx, TIMED_TOKENS) != 0) return -1.0;
    return (now_sec() - t0) / TIMED_TOKENS;
}

static int measure_all(tq_ctx *ctx, const Options *opt) {
    for (int t = 0; t < opt->n_threads; t++) {
        omp_set_num_threads(opt->threads[t]);
        for (int p = 0; p < N_POSITIONS; p++) {
            fprintf(stderr, "%d thread(s), pos %d...\n", opt->threads[t], POSITIONS[p]);
            double sec = time_position(ctx, POSITIONS[p]);
            if (sec < 0.0) return -1;
            if (!logits_finite(ctx)) fprintf(stderr, "WARNING: non-finite logits at pos %d\n", POSITIONS[p]);
            g_sec[t][p] = sec;
        }
    }
    return 0;
}

static void print_table(const Options *opt, const char *title, int per_second) {
    printf("%s\n%6s |", title, "pos");
    for (int t = 0; t < opt->n_threads; t++) printf(" %7dthr", opt->threads[t]);
    printf("\n");

    for (int p = 0; p < N_POSITIONS; p++) {
        printf("%6d |", POSITIONS[p]);
        for (int t = 0; t < opt->n_threads; t++) {
            double sec = g_sec[t][p];
            printf(" %10.2f", per_second ? 1.0 / sec : sec * 1e3);
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
    opt->fmt = TQ_FORMAT_Q4;
    opt->kv  = TQ_KV_Q8;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "q8") == 0)       opt->fmt = TQ_FORMAT_Q8;
        else if (strcmp(argv[i], "q4") == 0)  opt->fmt = TQ_FORMAT_Q4;
        else if (strcmp(argv[i], "f32") == 0) opt->kv = TQ_KV_F32;
        else if (strcmp(argv[i], "kv8") == 0) opt->kv = TQ_KV_Q8;
        else if (is_number(argv[i])) {
            int value = atoi(argv[i]);
            if (value >= 1 && opt->n_threads < MAX_THREAD_CFG) opt->threads[opt->n_threads++] = value;
        }
    }

    if (opt->n_threads > 0) return;
    opt->n_threads = (int)(sizeof defaults / sizeof defaults[0]);
    memcpy(opt->threads, defaults, sizeof defaults);
}

int main(int argc, char **argv) {
    Options opt;
    parse_options(argc, argv, &opt);

#ifndef __OPTIMIZE__
    printf("WARNING: compiled without optimization (Debug build), numbers are meaningless\n");
#endif
    printf("kernel: %s\n", kernel_isa_name());

    tq_model model;
    fprintf(stderr, "building random qwen3-4b...\n");
    if (build_model(&model, opt.fmt) != 0) {
        fprintf(stderr, "out of memory while building the model\n");
        return 1;
    }
    printf("weights: %s, %.2f GB | kv cache: %s\n\n",
           opt.fmt == TQ_FORMAT_Q8 ? "q8" : "q4", model_gigabytes(&model),
           opt.kv == TQ_KV_Q8 ? "q8" : "f32");

    const int max_tokens = POSITIONS[N_POSITIONS - 1] + WARMUP_TOKENS + TIMED_TOKENS;
    tq_ctx ctx;
    if (tq_ctx_init(&ctx, &model, opt.kv, max_tokens) != 0) {
        fprintf(stderr, "tq_ctx_init failed\n");
        free_model(&model);
        return 1;
    }
    zero_caches(&ctx);

    int rc = measure_all(&ctx, &opt);
    if (rc == 0) {
        print_table(&opt, "decode tokens/s at context position", 1);
        print_table(&opt, "ms per token at context position", 0);
    } else {
        fprintf(stderr, "forward pass failed\n");
    }

    tq_ctx_free(&ctx);
    free_model(&model);
    return rc == 0 ? 0 : 1;
}