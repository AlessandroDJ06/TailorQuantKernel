#include <assert.h>
#include <math.h>
#include <string.h>

#include "kernel/kernel.h"
#include "model/model_ops.h"

#define ATTN_PARALLEL_MIN_TOKENS 256

int tq_attn_validate(const tq_attn_weights *w, const tq_attn_config *cfg) {
    if (cfg->n_q_heads <= 0 || cfg->n_kv_heads <= 0 || cfg->head_dim <= 0) return -8;
    if (cfg->hidden_size % QK != 0) return -1;
    if (w->wk.cols != cfg->hidden_size || w->wq.cols != cfg->hidden_size ||w->wv.cols != cfg->hidden_size || w->wo.rows != cfg->hidden_size) return -2;
    if (w->wq.rows != cfg->n_q_heads * cfg->head_dim) return -3;
    if (w->wk.rows != cfg->n_kv_heads * cfg->head_dim ||w->wv.rows != cfg->n_kv_heads * cfg->head_dim) return -4;
    if (cfg->n_q_heads % cfg->n_kv_heads != 0) return -5;
    if (cfg->head_dim % QK != 0) return -6;
    if (w->wo.cols != cfg->n_q_heads * cfg->head_dim) return -7;
    if (w->q_norm == NULL || w->k_norm == NULL) return -9;
    return 0;
}

int tq_attn_validate_cache(const tq_kv_cache *cache) {
    if (cache->max_tokens <= 0) return -1;
    switch (cache->type) {
        case TQ_KV_F32:
            if (cache->k_f32 == NULL || cache->v_f32 == NULL) return -2;
            break;
        case TQ_KV_Q8:
            if (cache->k_q8 == NULL || cache->v_q8 == NULL) return -2;
            break;
        default:
            return -3;
    }
    return 0;
}

static void project_qkv(const tq_attn_weights *w, const tq_attn_config *cfg,
                        const float *xn, float *q, float *k, float *v) {
    block_q8 xn_q8[cfg->hidden_size / QK];
    quantize_row_q8(xn, xn_q8, cfg->hidden_size);

    get_matmul(w->wq.format)(w->wq.data, xn_q8, q, w->wq.rows, w->wq.cols);
    get_matmul(w->wk.format)(w->wk.data, xn_q8, k, w->wk.rows, w->wk.cols);
    get_matmul(w->wv.format)(w->wv.data, xn_q8, v, w->wv.rows, w->wv.cols);
}

static void norm_heads(float *x, int n_heads, int head_dim, const float *norm_w, float eps) {
    for (int h = 0; h < n_heads; h++) {
        float *head_ptr = x + (size_t)h * head_dim;
        rmsnorm(head_ptr, norm_w, head_ptr, head_dim, eps); /* in place */
    }
}

static void rope_heads(float *x, int n_heads, int head_dim,
                       const float *cos_t, const float *sin_t) {
    for (int h = 0; h < n_heads; h++) {
        rope_apply(x + (size_t)h * head_dim, head_dim, cos_t, sin_t);
    }
}

static inline size_t kv_slot(const tq_kv_cache *cache, int h, int pos) {
    return (size_t)h * cache->max_tokens + pos;
}

static void store_kv(tq_kv_cache *cache, const tq_attn_config *cfg, int pos,
                     const float *k, const float *v) {
    assert(pos >= 0 && pos < cache->max_tokens);

    for (int h = 0; h < cfg->n_kv_heads; h++) {
        const float *start_k = k + (size_t)h * cfg->head_dim;
        const float *start_v = v + (size_t)h * cfg->head_dim;
        size_t slot = kv_slot(cache, h, pos);

        switch (cache->type) {
            case TQ_KV_F32: {
                size_t offset = slot * cfg->head_dim;
                memcpy(cache->k_f32 + offset, start_k, cfg->head_dim * sizeof(float));
                memcpy(cache->v_f32 + offset, start_v, cfg->head_dim * sizeof(float));
                break;
            }
            case TQ_KV_Q8: {
                size_t boff = slot * (cfg->head_dim / QK);
                quantize_row_q8(start_k, cache->k_q8 + boff, cfg->head_dim);
                quantize_row_q8(start_v, cache->v_q8 + boff, cfg->head_dim);
                break;
            }
        }
    }
}

/* scores: n_q_heads * cache->max_tokens floats, one slice per query head */
static void attend_f32(const tq_kv_cache *cache, const tq_attn_config *cfg,
                       const float *q, int pos, float *scores, float *out) {
    const int hd = cfg->head_dim;
    const int n = pos + 1;
    const int group = cfg->n_q_heads / cfg->n_kv_heads;
    const float scale = 1.0f / sqrtf((float)hd);

#pragma omp parallel for schedule(static) if (n >= ATTN_PARALLEL_MIN_TOKENS)
    for (int h = 0; h < cfg->n_q_heads; h++) {
        const float *qh = q + (size_t)h * hd;
        float *oh = out + (size_t)h * hd;
        float *sc = scores + (size_t)h * cache->max_tokens;
        const int kvh = h / group;

        for (int t = 0; t < n; t++) {
            const float *kt = cache->k_f32 + kv_slot(cache, kvh, t) * hd;
            float dot = 0.0f;
            for (int j = 0; j < hd; j++) dot += qh[j] * kt[j];
            sc[t] = dot * scale;
        }

        softmax(sc, n);

        for (int j = 0; j < hd; j++) oh[j] = 0.0f;
        for (int t = 0; t < n; t++) {
            const float wt = sc[t];
            const float *vt = cache->v_f32 + kv_slot(cache, kvh, t) * hd;
            for (int j = 0; j < hd; j++) oh[j] += wt * vt[j];
        }
    }
}

static void attend_q8(const tq_kv_cache *cache, const tq_attn_config *cfg,
                      const float *q, int pos, float *scores, float *out) {
    const int hd = cfg->head_dim;
    const int nb = hd / QK;
    const int n = pos + 1;
    const int group = cfg->n_q_heads / cfg->n_kv_heads;
    const float scale = 1.0f / sqrtf((float)hd);
    const size_t stride = (size_t)cache->max_tokens;

#pragma omp parallel for schedule(static) if (n >= ATTN_PARALLEL_MIN_TOKENS)
    for (int kvh = 0; kvh < cfg->n_kv_heads; kvh++) {
        block_q8 qq[group * nb];
        const float *qg = q + (size_t)kvh * group * hd;
        float *og = out + (size_t)kvh * group * hd;
        float *sc = scores + (size_t)kvh * group * stride;
        const size_t slot0 = kv_slot(cache, kvh, 0);

        for (int g = 0; g < group; g++) {
            quantize_row_q8(qg + (size_t)g * hd, qq + (size_t)g * nb, hd);
        }

        attn_scores_q8(qq, group, cache->k_q8 + slot0 * nb, n, nb, scale, sc, stride);
        for (int g = 0; g < group; g++) softmax(sc + (size_t)g * stride, n);
        attn_accum_q8(sc, stride, group, cache->v_q8 + slot0 * nb, n, nb, og, (size_t)hd);
    }
}

static void attend_all_heads(const tq_kv_cache *cache, const tq_attn_config *cfg,
                             const float *q, int pos, float *scores, float *out) {
    if (cache->type == TQ_KV_Q8) {
        attend_q8(cache, cfg, q, pos, scores, out);
        return;
    }
    attend_f32(cache, cfg, q, pos, scores, out);
}

static void project_output(const tq_attn_weights *w, const tq_attn_config *cfg,
                           const float *attn, float *out) {
    int in_dim = cfg->n_q_heads * cfg->head_dim;
    block_q8 attn_q8[in_dim / QK];
    quantize_row_q8(attn, attn_q8, in_dim);
    get_matmul(w->wo.format)(w->wo.data, attn_q8, out, w->wo.rows, w->wo.cols);
}

int attention_layer(const tq_attn_weights *w, const tq_attn_config *cfg,
                    tq_kv_cache *cache, int pos, const float *xn,
                    float *scores, float *out) {
    if (pos < 0 || pos >= cache->max_tokens) return -1;

    const int hd     = cfg->head_dim;
    const int q_dim  = cfg->n_q_heads * hd;
    const int kv_dim = cfg->n_kv_heads * hd;

    float q[q_dim], k[kv_dim], v[kv_dim], attn[q_dim];
    float rope_cos[hd / 2], rope_sin[hd / 2];

    project_qkv(w, cfg, xn, q, k, v);

    norm_heads(q, cfg->n_q_heads,  hd, w->q_norm, cfg->rms_eps);
    norm_heads(k, cfg->n_kv_heads, hd, w->k_norm, cfg->rms_eps);

    rope_table(rope_cos, rope_sin, hd, pos, cfg->rope_theta);
    rope_heads(q, cfg->n_q_heads,  hd, rope_cos, rope_sin);
    rope_heads(k, cfg->n_kv_heads, hd, rope_cos, rope_sin);

    store_kv(cache, cfg, pos, k, v);
    attend_all_heads(cache, cfg, q, pos, scores, attn);
    project_output(w, cfg, attn, out);
    return 0;
}

/* Batched version of attention_layer for n consecutive tokens (pos0 .. pos0+n-1).
 * The expensive parts (q/k/v and output projection) run as batched matmuls.
 * Norm, RoPE, KV store and attention stay per token, in order, so token i only
 * sees positions <= pos0+i (causal) and results match n sequential calls. */
int attention_layer_batch(const tq_attn_weights *w, const tq_attn_config *cfg,
                          tq_kv_cache *cache, int pos0, int n, const float *xn,
                          float *scores, float *out, tq_batch_ws *ws) {
    if (n <= 0 || n > ws->cap) return -1;
    if (pos0 < 0 || pos0 + n > cache->max_tokens) return -1;

    const int hd     = cfg->head_dim;
    const int q_dim  = cfg->n_q_heads * hd;
    const int kv_dim = cfg->n_kv_heads * hd;
    const int hidden = cfg->hidden_size;
    const int nbh    = hidden / QK;
    const int nbq    = q_dim / QK;

    for (int i = 0; i < n; i++)
        quantize_row_q8(xn + (size_t)i * hidden, ws->xq + (size_t)i * nbh, hidden);

    get_matmul_batch(w->wq.format)(w->wq.data, ws->xq, ws->q, w->wq.rows, w->wq.cols, n);
    get_matmul_batch(w->wk.format)(w->wk.data, ws->xq, ws->k, w->wk.rows, w->wk.cols, n);
    get_matmul_batch(w->wv.format)(w->wv.data, ws->xq, ws->v, w->wv.rows, w->wv.cols, n);

    float rope_cos[hd / 2], rope_sin[hd / 2];
    for (int i = 0; i < n; i++) {
        float *qi = ws->q + (size_t)i * q_dim;
        float *ki = ws->k + (size_t)i * kv_dim;
        float *vi = ws->v + (size_t)i * kv_dim;
        const int pos = pos0 + i;

        norm_heads(qi, cfg->n_q_heads,  hd, w->q_norm, cfg->rms_eps);
        norm_heads(ki, cfg->n_kv_heads, hd, w->k_norm, cfg->rms_eps);

        rope_table(rope_cos, rope_sin, hd, pos, cfg->rope_theta);
        rope_heads(qi, cfg->n_q_heads,  hd, rope_cos, rope_sin);
        rope_heads(ki, cfg->n_kv_heads, hd, rope_cos, rope_sin);

        store_kv(cache, cfg, pos, ki, vi);
        attend_all_heads(cache, cfg, qi, pos, scores, ws->attn + (size_t)i * q_dim);
    }

    /* output projection (ws->xq is free again: q/k/v matmuls are done) */
    for (int i = 0; i < n; i++)
        quantize_row_q8(ws->attn + (size_t)i * q_dim, ws->xq + (size_t)i * nbq, q_dim);
    get_matmul_batch(w->wo.format)(w->wo.data, ws->xq, out, w->wo.rows, w->wo.cols, n);
    return 0;
}