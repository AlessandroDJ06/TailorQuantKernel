#ifndef TAILORQUANT_MODEL_OPS_H
#define TAILORQUANT_MODEL_OPS_H
#include "kernel/kernel.h"

#ifdef __cplusplus
extern "C" {
#endif

#include "../packer/packer.h"
#define Q8_TYPE '8'

typedef struct {
    int n_q_heads;
    int n_kv_heads;
    int head_dim;
    int hidden_size;
    float rope_theta;
    float rms_eps;
} tq_attn_config;

typedef struct {
    const void *data;
    tq_format format;
    int rows;
    int cols;
} tq_matrix;


typedef struct {
    tq_matrix wgate, wup, wdown;
} tq_mlp_weights;

typedef struct {
    tq_matrix wq, wk, wv, wo;
    const float *q_norm;
    const float *k_norm;
} tq_attn_weights;

typedef enum { TQ_KV_F32, TQ_KV_Q8 } tq_kv_type;

typedef struct {
    tq_kv_type type;
    float *k_f32;
    float *v_f32;
    block_q8 *k_q8;
    block_q8 *v_q8;
    int max_tokens;
} tq_kv_cache;

typedef struct {
    tq_attn_config attn;
    int n_layers;
    int intermediate_size;
    int vocab_size;
} tq_model_config;

typedef struct {
    const float *input_norm;
    tq_attn_weights attn;
    const float *post_norm;
    tq_mlp_weights mlp;
} tq_layer_weights;

typedef struct {
    tq_matrix embed;
    const float *final_norm;
    tq_layer_weights *layers;
} tq_model_weights;

typedef struct {
    tq_model_config cfg;
    tq_model_weights w;
} tq_model;

typedef struct {
    const tq_model *model;
    tq_kv_cache *caches;
    int pos;
    int max_tokens;

    float *x;
    float *xn;
    float *attn_out;
    float *mlp_out;
    float *gate;
    float *up;
    block_q8 *xq;
    block_q8 *hq;
    float *scores;
    float *logits;
} tq_ctx;



int  tq_kv_cache_init(tq_kv_cache *cache, tq_kv_type type,const tq_attn_config *cfg, int max_tokens);

void tq_kv_cache_free(tq_kv_cache *cache);

void silu_mul(const float *gate, const float *up, float *out, int n);

void rmsnorm(const float *x, const float *weight, float *out, int n, float eps);

void softmax(float *x, int n);

void rope(float *x, int head_dim, int pos, float theta);
void rope_table(float *cos_t, float *sin_t, int head_dim, int pos, float theta);
void rope_apply(float *x, int head_dim, const float *cos_t, const float *sin_t);

void attention_head(const float *q,
                    const float *k_cache, const float *v_cache,
                    int n, int head_dim, int stride,
                    float *scores, float *out);

void attention_head_q8(const float *q, block_q8 *q_q8,
                       const block_q8 *k_cache, const block_q8 *v_cache,
                       int n, int head_dim,
                       float *scores, float *out);

int attention_layer(const tq_attn_weights *w, const tq_attn_config *cfg,
                    tq_kv_cache *cache, int pos, const float *xn,
                    float *scores, float *out);

int tq_attn_validate(const tq_attn_weights *w, const tq_attn_config *cfg);
int tq_attn_validate_cache(const tq_kv_cache *cache);

int mlp_layer(const tq_mlp_weights *w, const float *xn, block_q8 *xq,
              float *gate, float *up, block_q8 *hq, float *out);

int embed_lookup(const tq_matrix *embed, int token, float *out);

int lm_head(const tq_matrix *embed, const float *xn, block_q8 *xq, float *logits);

int sample_greedy(const float *logits, int n);

void vec_add(float *x, const float *y, int n);

int  tq_ctx_init(tq_ctx *ctx, const tq_model *m, tq_kv_type kv, int max_tokens);

void tq_ctx_free(tq_ctx *ctx);

int  tq_forward_token(tq_ctx *ctx, int token);

/* ---- batched prefill ---- */
#define TQ_PREFILL_BATCH 16

typedef struct {
    int cap;                 /* max tokens per batch */
    float *x, *xn, *sub;     /* cap * hidden */
    float *gate, *up;        /* cap * inter */
    float *q, *attn;         /* cap * q_dim */
    float *k, *v;            /* cap * kv_dim */
    block_q8 *xq;            /* cap * max(hidden, q_dim) / QK */
    block_q8 *hq;            /* cap * inter / QK */
} tq_batch_ws;

int  tq_batch_ws_init(tq_batch_ws *ws, const tq_model *m, int cap);
void tq_batch_ws_free(tq_batch_ws *ws);

/* xn / out: n tokens, token-major (token i starts at i * hidden) */
int attention_layer_batch(const tq_attn_weights *w, const tq_attn_config *cfg,
                          tq_kv_cache *cache, int pos0, int n, const float *xn,
                          float *scores, float *out, tq_batch_ws *ws);

int mlp_layer_batch(const tq_mlp_weights *w, int n, const float *xn,
                    float *out, tq_batch_ws *ws);

/* Processes n tokens at positions ctx->pos .. ctx->pos+n-1 in one pass.
 * Logits are only computed (for the LAST token) when want_logits != 0. */
int tq_forward_batch(tq_ctx *ctx, tq_batch_ws *ws, const int *tokens, int n, int want_logits);

/* Whole prompt, in chunks of TQ_PREFILL_BATCH. Leaves logits of the last token in ctx->logits. */
int tq_prefill(tq_ctx *ctx, const int *tokens, int n);

void attn_scores_q8(const block_q8 *q, int g, const block_q8 *k, int n, int nb,
                    float scale, float *scores, size_t stride);

void attn_accum_q8(const float *w, size_t w_stride, int g, const block_q8 *v, int n, int nb,
                   float *out, size_t out_stride);

void attn_scores_q8_scalar(const block_q8 *q, int g, const block_q8 *k, int n, int nb,
                           float scale, float *scores, size_t stride);

void attn_accum_q8_scalar(const float *w, size_t w_stride, int g, const block_q8 *v, int n, int nb,
                          float *out, size_t out_stride);


#ifdef __cplusplus
}
#endif

#endif