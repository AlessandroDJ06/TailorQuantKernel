#ifndef _WIN32
#define _POSIX_C_SOURCE 199309L
#endif

#include <math.h>
#include <stdarg.h>
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

typedef struct {
    const char *name;
    int         n_layers;
    int         hidden;
    int         inter;
    int         n_q_heads;
    int         n_kv_heads;
    int         head_dim;
    int         vocab;
} ModelConfig;

static const ModelConfig MODEL_QWEN3_06B = { "qwen3-0.6b", 28, 1024, 3072, 16, 8, 128, 151936 };
static const ModelConfig MODEL_QWEN3_17B = { "qwen3-1.7b", 28, 2048, 5632, 16, 8, 128, 151936 };
static const ModelConfig MODEL_QWEN3_4B  = { "qwen3-4b",  36, 2560, 9728, 32, 8, 128, 151936 };

#define CHUNK_ROWS     512
#define WEIGHT_SCALE   0.02f
#define WARMUP_TOKENS  1
#define TIMED_TOKENS   9
#define COMP_REPS      10
#define MAX_THREAD_CFG 8
#define N_POSITIONS    6
#define ROPE_POS       100
#define DEFAULT_RESULTS "bench_forward_results.txt"
#define DEFAULT_CSV     "bench_forward_results.csv"

static const int POSITIONS[N_POSITIONS] = {0, 128, 512, 1024, 2048, 4096};

typedef struct {
    int         threads[MAX_THREAD_CFG];
    int         n_threads;
    int         want_q4, want_q8, want_kv8, want_f32;
    const ModelConfig *model_cfg;
    const char *results_path;
    const char *csv_path;
} Options;

typedef struct {
    double embed, attn, mlp, head;
} Phase;

typedef enum {
    J_WQ, J_WK, J_WV, J_WO, J_GATE, J_UP, J_DOWN, J_LM_HEAD, J_EMBED,
    J_QUANT, J_RMSNORM, J_ROPE, J_SILU, J_VEC_ADD, N_JOBS
} Job;

static const char *JOB_NAME[N_JOBS] = {
    "wq", "wk", "wv", "wo", "wgate", "wup", "wdown",
    "lm_head+quant", "embed_lookup",
    "quantize_row_q8", "rmsnorm", "rope", "silu_mul", "vec_add"
};

typedef struct {
    float    *a, *b, *out;
    block_q8 *xq_hidden, *xq_q, *xq_inter;
} Scratch;

static uint64_t g_rng = 88172645463325252ull;
static float   *g_ones_hidden = NULL;
static float   *g_ones_head   = NULL;

static double g_median[MAX_THREAD_CFG][N_POSITIONS];
static Phase  g_phase[MAX_THREAD_CFG][N_POSITIONS];
static double g_comp[MAX_THREAD_CFG][N_JOBS];

static FILE *g_out[2];
static int   g_n_out;

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

static double lap(double *mark) {
    double now = now_sec();
    double elapsed = now - *mark;
    *mark = now;
    return elapsed;
}

static void say(const char *fmt, ...) {
    for (int i = 0; i < g_n_out; i++) {
        va_list ap;
        va_start(ap, fmt);
        vfprintf(g_out[i], fmt, ap);
        va_end(ap);
    }
}

static void say_flush(void) {
    for (int i = 0; i < g_n_out; i++) fflush(g_out[i]);
}

static void open_outputs(const char *path) {
    g_out[g_n_out++] = stdout;
    FILE *f = fopen(path, "w");
    if (!f) {
        fprintf(stderr, "cannot write %s, printing to stdout only\n", path);
        return;
    }
    g_out[g_n_out++] = f;
}

static void close_outputs(void) {
    for (int i = 1; i < g_n_out; i++) fclose(g_out[i]);
}


static void export_csv(const Options *opt, const char *label) {
    FILE *f = fopen(opt->csv_path, "a");
    if (!f) {
        fprintf(stderr, "Kon CSV-bestand %s niet openen voor schrijven.\n", opt->csv_path);
        return;
    }

    fprintf(f, "Model,Configuration,Metric,Position");
    for (int t = 0; t < opt->n_threads; t++) {
        fprintf(f, ",%d_threads", opt->threads[t]);
    }
    fprintf(f, "\n");

    // 1. Tokens per seconde exporteren
    for (int p = 0; p < N_POSITIONS; p++) {
        fprintf(f, "%s,%s,tokens_per_sec,%d", opt->model_cfg->name, label, POSITIONS[p]);
        for (int t = 0; t < opt->n_threads; t++) {
            double tps = 1.0 / g_median[t][p];
            fprintf(f, ",%.2f", tps);
        }
        fprintf(f, "\n");
    }

    // 2. Milliseconden per token exporteren
    for (int p = 0; p < N_POSITIONS; p++) {
        fprintf(f, "%s,%s,ms_per_token,%d", opt->model_cfg->name, label, POSITIONS[p]);
        for (int t = 0; t < opt->n_threads; t++) {
            double ms = g_median[t][p] * 1e3;
            fprintf(f, ",%.2f", ms);
        }
        fprintf(f, "\n");
    }

    fclose(f);
}

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

static const char *fmt_name(tq_format fmt) {
    return fmt == TQ_FORMAT_Q8 ? "q8" : "q4";
}

static const char *kv_name(tq_kv_type kv) {
    return kv == TQ_KV_Q8 ? "q8" : "f32";
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

static int build_layer(tq_layer_weights *lw, const ModelConfig *cfg, tq_format fmt) {
    int q_dim = cfg->n_q_heads * cfg->head_dim;
    int kv_dim = cfg->n_kv_heads * cfg->head_dim;

    lw->input_norm  = g_ones_hidden;
    lw->post_norm   = g_ones_hidden;
    lw->attn.q_norm = g_ones_head;
    lw->attn.k_norm = g_ones_head;

    if (make_matrix(&lw->attn.wq,   q_dim,  cfg->hidden, fmt) != 0) return -1;
    if (make_matrix(&lw->attn.wk,   kv_dim, cfg->hidden, fmt) != 0) return -1;
    if (make_matrix(&lw->attn.wv,   kv_dim, cfg->hidden, fmt) != 0) return -1;
    if (make_matrix(&lw->attn.wo,   cfg->hidden, q_dim,  fmt) != 0) return -1;
    if (make_matrix(&lw->mlp.wgate, cfg->inter,  cfg->hidden, fmt) != 0) return -1;
    if (make_matrix(&lw->mlp.wup,   cfg->inter,  cfg->hidden, fmt) != 0) return -1;
    if (make_matrix(&lw->mlp.wdown, cfg->hidden, cfg->inter,  fmt) != 0) return -1;
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

static int build_model(tq_model *m, const ModelConfig *cfg, tq_format fmt) {
    memset(m, 0, sizeof *m);

    g_ones_hidden = malloc(cfg->hidden * sizeof(float));
    g_ones_head   = malloc(cfg->head_dim * sizeof(float));
    if (!g_ones_hidden || !g_ones_head) return -1;

    for (int i = 0; i < cfg->hidden; i++)   g_ones_hidden[i] = 1.0f;
    for (int i = 0; i < cfg->head_dim; i++) g_ones_head[i]   = 1.0f;

    m->cfg.attn.n_q_heads    = cfg->n_q_heads;
    m->cfg.attn.n_kv_heads   = cfg->n_kv_heads;
    m->cfg.attn.head_dim     = cfg->head_dim;
    m->cfg.attn.hidden_size  = cfg->hidden;
    m->cfg.attn.rope_theta   = 1000000.0f;
    m->cfg.attn.rms_eps      = 1e-6f;
    m->cfg.n_layers          = cfg->n_layers;
    m->cfg.intermediate_size = cfg->inter;
    m->cfg.vocab_size        = cfg->vocab;

    m->w.final_norm = g_ones_hidden;
    m->w.layers     = calloc(cfg->n_layers, sizeof(tq_layer_weights));
    if (!m->w.layers) return -1;

    if (make_matrix(&m->w.embed, cfg->vocab, cfg->hidden, fmt) != 0) {
        free_model(m);
        return -1;
    }
    for (int l = 0; l < cfg->n_layers; l++) {
        if (build_layer(&m->w.layers[l], cfg, fmt) != 0) {
            free_model(m);
            return -1;
        }
    }
    return 0;
}

static double model_megabytes(const tq_model *m) {
    const tq_layer_weights *lw = &m->w.layers[0];
    size_t layer = matrix_bytes(&lw->attn.wq) + matrix_bytes(&lw->attn.wk) +
                   matrix_bytes(&lw->attn.wv) + matrix_bytes(&lw->attn.wo) +
                   matrix_bytes(&lw->mlp.wgate) + matrix_bytes(&lw->mlp.wup) +
                   matrix_bytes(&lw->mlp.wdown);
    return ((double)layer * m->cfg.n_layers + (double)matrix_bytes(&m->w.embed)) / 1e6;
}

static void zero_caches(tq_ctx *ctx, const ModelConfig *cfg) {
    const size_t slots = (size_t)cfg->n_kv_heads * ctx->max_tokens;
    for (int l = 0; l < cfg->n_layers; l++) {
        tq_kv_cache *c = &ctx->caches[l];
        if (c->type == TQ_KV_F32) {
            memset(c->k_f32, 0, slots * cfg->head_dim * sizeof(float));
            memset(c->v_f32, 0, slots * cfg->head_dim * sizeof(float));
        } else {
            memset(c->k_q8, 0, slots * (cfg->head_dim / QK) * sizeof(block_q8));
            memset(c->v_q8, 0, slots * (cfg->head_dim / QK) * sizeof(block_q8));
        }
    }
}

static int token_for(const tq_ctx *ctx, int vocab) {
    return (int)(((long long)ctx->pos * 7919 + 13) % vocab);
}

static int run_tokens(tq_ctx *ctx, int count, int vocab) {
    for (int i = 0; i < count; i++) {
        if (tq_forward_token(ctx, token_for(ctx, vocab)) != 0) return -1;
    }
    return 0;
}

static int logits_finite(const tq_ctx *ctx, int vocab) {
    for (int i = 0; i < vocab; i++) {
        if (!isfinite(ctx->logits[i])) return 0;
    }
    return 1;
}

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a;
    double y = *(const double *)b;
    return (x > y) - (x < y);
}

static double median(double *v, int n) {
    qsort(v, (size_t)n, sizeof *v, cmp_double);
    return v[n / 2];
}

static int time_position(tq_ctx *ctx, int pos, int vocab, double *median_sec) {
    double samples[TIMED_TOKENS];

    ctx->pos = pos;
    if (run_tokens(ctx, WARMUP_TOKENS, vocab) != 0) return -1;

    for (int i = 0; i < TIMED_TOKENS; i++) {
        double t0 = now_sec();
        if (run_tokens(ctx, 1, vocab) != 0) return -1;
        samples[i] = now_sec() - t0;
    }
    *median_sec = median(samples, TIMED_TOKENS);
    return 0;
}

static int forward_phased(tq_ctx *ctx, int token, Phase *pt) {
    const tq_model       *m   = ctx->model;
    const tq_attn_config *cfg = &m->cfg.attn;
    double mark = now_sec();

    if (embed_lookup(&m->w.embed, token, ctx->x) != 0) return -1;
    pt->embed += lap(&mark);

    for (int l = 0; l < m->cfg.n_layers; l++) {
        const tq_layer_weights *lw = &m->w.layers[l];

        rmsnorm(ctx->x, lw->input_norm, ctx->xn, cfg->hidden_size, cfg->rms_eps);
        if (attention_layer(&lw->attn, cfg, &ctx->caches[l], ctx->pos,
                            ctx->xn, ctx->scores, ctx->attn_out) != 0) return -1;
        vec_add(ctx->x, ctx->attn_out, cfg->hidden_size);
        pt->attn += lap(&mark);

        rmsnorm(ctx->x, lw->post_norm, ctx->xn, cfg->hidden_size, cfg->rms_eps);
        if (mlp_layer(&lw->mlp, ctx->xn, ctx->xq, ctx->gate, ctx->up,
                      ctx->hq, ctx->mlp_out) != 0) return -1;
        vec_add(ctx->x, ctx->mlp_out, cfg->hidden_size);
        pt->mlp += lap(&mark);
    }

    rmsnorm(ctx->x, m->w.final_norm, ctx->xn, cfg->hidden_size, cfg->rms_eps);
    if (lm_head(&m->w.embed, ctx->xn, ctx->xq, ctx->logits) != 0) return -1;
    pt->head += lap(&mark);

    ctx->pos++;
    return 0;
}

static int phase_position(tq_ctx *ctx, int pos, int vocab, Phase *out) {
    Phase sum = {0};

    ctx->pos = pos;
    if (run_tokens(ctx, WARMUP_TOKENS, vocab) != 0) return -1;

    ctx->pos = pos;
    for (int i = 0; i < TIMED_TOKENS; i++) {
        if (forward_phased(ctx, token_for(ctx, vocab), &sum) != 0) return -1;
    }

    out->embed = sum.embed / TIMED_TOKENS;
    out->attn  = sum.attn  / TIMED_TOKENS;
    out->mlp   = sum.mlp   / TIMED_TOKENS;
    out->head  = sum.head  / TIMED_TOKENS;
    return 0;
}

static int measure_speed(tq_ctx *ctx, const Options *opt) {
    for (int t = 0; t < opt->n_threads; t++) {
        omp_set_num_threads(opt->threads[t]);
        for (int p = 0; p < N_POSITIONS; p++) {
            fprintf(stderr, "%d thread(s), pos %d...\n", opt->threads[t], POSITIONS[p]);

            if (time_position(ctx, POSITIONS[p], opt->model_cfg->vocab, &g_median[t][p]) != 0) return -1;
            if (!logits_finite(ctx, opt->model_cfg->vocab)) fprintf(stderr, "WARNING: non-finite logits at pos %d\n", POSITIONS[p]);

            if (phase_position(ctx, POSITIONS[p], opt->model_cfg->vocab, &g_phase[t][p]) != 0) return -1;
        }
    }
    return 0;
}

static int scratch_init(Scratch *s, const ModelConfig *cfg) {
    int q_dim = cfg->n_q_heads * cfg->head_dim;
    memset(s, 0, sizeof *s);
    s->a         = malloc(cfg->inter * sizeof(float));
    s->b         = malloc(cfg->inter * sizeof(float));
    s->out       = malloc((size_t)cfg->vocab * sizeof(float));
    s->xq_hidden = malloc((cfg->hidden / QK) * sizeof(block_q8));
    s->xq_q      = malloc((q_dim / QK) * sizeof(block_q8));
    s->xq_inter  = malloc((cfg->inter / QK) * sizeof(block_q8));

    if (!s->a || !s->b || !s->out || !s->xq_hidden || !s->xq_q || !s->xq_inter) return -1;

    for (int i = 0; i < cfg->inter; i++) {
        s->a[i] = normal_sample();
        s->b[i] = normal_sample();
    }
    quantize_row_q8(s->a, s->xq_hidden, cfg->hidden);
    quantize_row_q8(s->a, s->xq_q, q_dim);
    quantize_row_q8(s->a, s->xq_inter, cfg->inter);
    return 0;
}

static void scratch_free(Scratch *s) {
    free(s->a);
    free(s->b);
    free(s->out);
    free(s->xq_hidden);
    free(s->xq_q);
    free(s->xq_inter);
    memset(s, 0, sizeof *s);
}

static const tq_matrix *layer_matrix(const tq_layer_weights *lw, Job job) {
    switch (job) {
        case J_WQ:   return &lw->attn.wq;
        case J_WK:   return &lw->attn.wk;
        case J_WV:   return &lw->attn.wv;
        case J_WO:   return &lw->attn.wo;
        case J_GATE: return &lw->mlp.wgate;
        case J_UP:   return &lw->mlp.wup;
        case J_DOWN: return &lw->mlp.wdown;
        default:     return NULL;
    }
}

static const block_q8 *input_for(const Scratch *s, int cols, const ModelConfig *cfg) {
    if (cols == cfg->hidden) return s->xq_hidden;
    if (cols == (cfg->n_q_heads * cfg->head_dim)) return s->xq_q;
    return s->xq_inter;
}

static void run_matmul_job(const tq_model *m, const Scratch *s, const ModelConfig *cfg, Job job) {
    for (int l = 0; l < cfg->n_layers; l++) {
        const tq_matrix *w = layer_matrix(&m->w.layers[l], job);
        get_matmul(w->format)(w->data, input_for(s, w->cols, cfg), s->out, w->rows, w->cols);
    }
}

static void run_job(const tq_model *m, Scratch *s, const ModelConfig *cfg, Job job) {
    const float eps = m->cfg.attn.rms_eps;

    switch (job) {
        case J_LM_HEAD:
            lm_head(&m->w.embed, s->a, s->xq_hidden, s->out);
            break;
        case J_EMBED:
            embed_lookup(&m->w.embed, 12345, s->a);
            break;
        case J_QUANT:
            for (int l = 0; l < cfg->n_layers; l++) {
                quantize_row_q8(s->a, s->xq_hidden, cfg->hidden);
                quantize_row_q8(s->a, s->xq_hidden, cfg->hidden);
                quantize_row_q8(s->a, s->xq_q, cfg->n_q_heads * cfg->head_dim);
                quantize_row_q8(s->a, s->xq_inter, cfg->inter);
            }
            break;
        case J_RMSNORM:
            for (int l = 0; l < cfg->n_layers; l++) {
                rmsnorm(s->a, g_ones_hidden, s->b, cfg->hidden, eps);
                rmsnorm(s->a, g_ones_hidden, s->b, cfg->hidden, eps);
                for (int h = 0; h < (cfg->n_q_heads + cfg->n_kv_heads); h++) {
                    rmsnorm(s->a + h * cfg->head_dim, g_ones_head, s->b + h * cfg->head_dim, cfg->head_dim, eps);
                }
            }
            break;
        case J_ROPE:
            for (int l = 0; l < cfg->n_layers; l++) {
                for (int h = 0; h < (cfg->n_q_heads + cfg->n_kv_heads); h++) {
                    rope(s->a + h * cfg->head_dim, cfg->head_dim, ROPE_POS, m->cfg.attn.rope_theta);
                }
            }
            break;
        case J_SILU:
            for (int l = 0; l < cfg->n_layers; l++) silu_mul(s->a, s->b, s->out, cfg->inter);
            break;
        case J_VEC_ADD:
            for (int l = 0; l < cfg->n_layers; l++) {
                vec_add(s->a, s->b, cfg->hidden);
                vec_add(s->a, s->b, cfg->hidden);
            }
            break;
        default:
            run_matmul_job(m, s, cfg, job);
            break;
    }
}

static double time_job(const tq_model *m, Scratch *s, const ModelConfig *cfg, Job job) {
    double best = 1e30;
    for (int i = 0; i < COMP_REPS + 1; i++) {
        double t0 = now_sec();
        run_job(m, s, cfg, job);
        double t = now_sec() - t0;
        if (i > 0 && t < best) best = t;
    }
    return best;
}

static int measure_components(const tq_model *m, const Options *opt) {
    Scratch s;
    if (scratch_init(&s, opt->model_cfg) != 0) {
        scratch_free(&s);
        return -1;
    }

    for (int t = 0; t < opt->n_threads; t++) {
        omp_set_num_threads(opt->threads[t]);
        fprintf(stderr, "components, %d thread(s)...\n", opt->threads[t]);
        for (int j = 0; j < N_JOBS; j++) g_comp[t][j] = time_job(m, &s, opt->model_cfg, (Job)j);
    }

    scratch_free(&s);
    return 0;
}

static void print_header(const Options *opt, const char *title) {
    say("%s\n%6s |", title, "pos");
    for (int t = 0; t < opt->n_threads; t++) say(" %7dthr", opt->threads[t]);
    say("\n");
}

static void print_speed(const Options *opt, const char *label, int per_second) {
    char title[128];
    snprintf(title, sizeof title, "[%s] %s (median of %d tokens)", label,
             per_second ? "decode tokens/s at context position" : "ms per token at context position",
             TIMED_TOKENS);
    print_header(opt, title);

    for (int p = 0; p < N_POSITIONS; p++) {
        say("%6d |", POSITIONS[p]);
        for (int t = 0; t < opt->n_threads; t++) {
            double sec = g_median[t][p];
            say(" %10.2f", per_second ? 1.0 / sec : sec * 1e3);
        }
        say("\n");
    }
    say("\n");
}

static double phase_total(const Phase *p) {
    return p->embed + p->attn + p->mlp + p->head;
}

static void print_phases(const Options *opt, const char *label) {
    for (int t = 0; t < opt->n_threads; t++) {
        say("[%s] phases at %d thread(s), ms per token\n", label, opt->threads[t]);
        say("%6s | %8s %8s %8s %8s | %8s %6s\n", "pos", "embed", "attn", "mlp", "lm_head", "total", "attn%");

        for (int p = 0; p < N_POSITIONS; p++) {
            const Phase *ph = &g_phase[t][p];
            double total = phase_total(ph);
            say("%6d | %8.2f %8.2f %8.2f %8.2f | %8.2f %5.0f%%\n", POSITIONS[p],
                ph->embed * 1e3, ph->attn * 1e3, ph->mlp * 1e3, ph->head * 1e3,
                total * 1e3, 100.0 * ph->attn / total);
        }
        say("\n");
    }
}

static void print_components(const Options *opt, const char *fmt) {
    say("[%s weights] component cost per token, ms (all layers, best of %d)\n", fmt, COMP_REPS);
    say("%-16s |", "component");
    for (int t = 0; t < opt->n_threads; t++) say(" %7dthr", opt->threads[t]);
    say("\n");

    for (int j = 0; j < N_JOBS; j++) {
        say("%-16s |", JOB_NAME[j]);
        for (int t = 0; t < opt->n_threads; t++) say(" %10.3f", g_comp[t][j] * 1e3);
        say("\n");
    }

    say("%-16s |", "sum components");
    for (int t = 0; t < opt->n_threads; t++) {
        double sum = 0.0;
        for (int j = 0; j < N_JOBS; j++) sum += g_comp[t][j];
        say(" %10.3f", sum * 1e3);
    }
    say("\n%-16s |", "measured pos 0");
    for (int t = 0; t < opt->n_threads; t++) say(" %10.3f", phase_total(&g_phase[t][0]) * 1e3);
    say("\n%-16s |", "rest");
    for (int t = 0; t < opt->n_threads; t++) {
        double sum = 0.0;
        for (int j = 0; j < N_JOBS; j++) sum += g_comp[t][j];
        say(" %10.3f", (phase_total(&g_phase[t][0]) - sum) * 1e3);
    }
    say("\n\n");
}

static int run_kv(const tq_model *model, tq_format fmt, tq_kv_type kv,
                  const Options *opt, int with_components) {
    const int max_tokens = POSITIONS[N_POSITIONS - 1] + WARMUP_TOKENS + TIMED_TOKENS + 1;
    char label[64];
    snprintf(label, sizeof label, "%s weights, %s cache", fmt_name(fmt), kv_name(kv));

    tq_ctx ctx;
    if (tq_ctx_init(&ctx, model, kv, max_tokens) != 0) {
        fprintf(stderr, "tq_ctx_init failed\n");
        return -1;
    }
    zero_caches(&ctx, opt->model_cfg);

    int rc = measure_speed(&ctx, opt);
    if (rc == 0) {
        print_speed(opt, label, 1);
        print_speed(opt, label, 0);
        print_phases(opt, label);
        if (with_components) print_components(opt, fmt_name(fmt));
        export_csv(opt, label);

        say_flush();
    } else {
        fprintf(stderr, "forward pass failed\n");
    }

    tq_ctx_free(&ctx);
    return rc;
}

static int run_format(tq_format fmt, const Options *opt) {
    tq_model model;
    fprintf(stderr, "building random %s (%s)...\n", opt->model_cfg->name, fmt_name(fmt));
    if (build_model(&model, opt->model_cfg, fmt) != 0) {
        fprintf(stderr, "out of memory while building the model\n");
        return -1;
    }
    say("==== %s - %s weights, %.2f MB ====\n\n", opt->model_cfg->name, fmt_name(fmt), model_megabytes(&model));

    int rc = measure_components(&model, opt);
    int first = 1;

    if (rc == 0 && opt->want_kv8) {
        rc = run_kv(&model, fmt, TQ_KV_Q8, opt, first);
        first = 0;
    }
    if (rc == 0 && opt->want_f32) {
        rc = run_kv(&model, fmt, TQ_KV_F32, opt, first);
    }

    free_model(&model);
    free(g_ones_hidden);
    free(g_ones_head);
    g_ones_hidden = NULL;
    g_ones_head = NULL;
    return rc;
}

static int is_number(const char *s) {
    if (*s == '\0') return 0;
    for (; *s; s++) {
        if (*s < '0' || *s > '9') return 0;
    }
    return 1;
}

static void parse_options(int argc, char **argv, Options *opt) {
    static const int defaults[] = {1, 2, 4, 8, 16};
    memset(opt, 0, sizeof *opt);
    opt->results_path = DEFAULT_RESULTS;
    opt->csv_path = DEFAULT_CSV;
    opt->model_cfg = &MODEL_QWEN3_06B;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "q4") == 0)       opt->want_q4 = 1;
        else if (strcmp(argv[i], "q8") == 0)  opt->want_q8 = 1;
        else if (strcmp(argv[i], "kv8") == 0) opt->want_kv8 = 1;
        else if (strcmp(argv[i], "f32") == 0) opt->want_f32 = 1;
        else if (strcmp(argv[i], "qwen3-0.6b") == 0) opt->model_cfg = &MODEL_QWEN3_06B;
        else if (strcmp(argv[i], "qwen3-1.7b") == 0) opt->model_cfg = &MODEL_QWEN3_17B;
        else if (strcmp(argv[i], "qwen3-4b") == 0)  opt->model_cfg = &MODEL_QWEN3_4B;
        else if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) opt->results_path = argv[++i];
        else if (strcmp(argv[i], "--csv") == 0 && i + 1 < argc) opt->csv_path = argv[++i];
        else if (is_number(argv[i])) {
            int value = atoi(argv[i]);
            if (value >= 1 && opt->n_threads < MAX_THREAD_CFG) opt->threads[opt->n_threads++] = value;
        }
    }

    if (!opt->want_q4 && !opt->want_q8) opt->want_q4 = opt->want_q8 = 1;
    if (!opt->want_kv8 && !opt->want_f32) opt->want_kv8 = opt->want_f32 = 1;

    if (opt->n_threads > 0) return;
    opt->n_threads = (int)(sizeof defaults / sizeof defaults[0]);
    memcpy(opt->threads, defaults, sizeof defaults);
}

int main(int argc, char **argv) {
    Options opt;
    parse_options(argc, argv, &opt);
    open_outputs(opt.results_path);

#ifndef __OPTIMIZE__
    say("WARNING: compiled without optimization (Debug build), numbers are meaningless\n");
#endif
    say("kernel: %s\n\n", kernel_isa_name());
    const ModelConfig *models[] = {
        &MODEL_QWEN3_06B,
        &MODEL_QWEN3_17B,
        &MODEL_QWEN3_4B
    };
    int n_models = 3;

    int status = 0;
    for (int m = 0; m < n_models; m++) {
        opt.model_cfg = models[m];
        if (opt.want_q4) {
            status = run_format(TQ_FORMAT_Q4, &opt);
            if (status != 0) break;
        }
        if (opt.want_q8) {
            status = run_format(TQ_FORMAT_Q8, &opt);
            if (status != 0) break;
        }
    }

    close_outputs();
    return status == 0 ? 0 : 1;
}