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

int  tq_kv_cache_init(tq_kv_cache *cache, tq_kv_type type,const tq_attn_config *cfg, int max_tokens);

void tq_kv_cache_free(tq_kv_cache *cache);

void silu_mul(const float *gate, const float *up, float *out, int n);

void rmsnorm(const float *x, const float *weight, float *out, int n, float eps);

void softmax(float *x, int n);

void rope(float *x, int head_dim, int pos, float theta);

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

typedef struct {
    tq_matrix wgate, wup, wdown;
} tq_mlp_weights;

int mlp_layer(const tq_mlp_weights *w, const float *xn, block_q8 *xq,
              float *gate, float *up, block_q8 *hq, float *out);

int embed_lookup(const tq_matrix *embed, int token, float *out);

int lm_head(const tq_matrix *embed, const float *xn, block_q8 *xq, float *logits);

int sample_greedy(const float *logits, int n);

#ifdef __cplusplus
}
#endif

#endif