#include <assert.h>
#include <math.h>
#include <string.h>

#include "kernel/kernel.h"
#include "model/model_ops.h"

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

static void rope_heads(float *x, int n_heads, int head_dim, int pos, float theta) {
    for (int h = 0; h < n_heads; h++) {
        float *head_ptr = x + (size_t)h * head_dim;
        rope(head_ptr, head_dim, pos, theta);
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

static void attend_all_heads(const tq_kv_cache *cache, const tq_attn_config *cfg,
                             const float *q, int pos, float *scores, float *out) {
    const int hd = cfg->head_dim;
    const int nb = hd / QK;
    const int n = pos + 1;
    const int group = cfg->n_q_heads / cfg->n_kv_heads;
    const float scale = 1.0f / sqrtf((float)hd);

    for (int h = 0; h < cfg->n_q_heads; h++) {
        const float *qh = q + (size_t)h * hd;
        float *oh = out + (size_t)h * hd;
        const int kvh = h / group;

        for (int t = 0; t < n; t++) {
            size_t slot = kv_slot(cache, kvh, t);
            float dot = 0.0f;
            if (cache->type == TQ_KV_F32) {
                const float *kt = cache->k_f32 + slot * hd;
                for (int j = 0; j < hd; j++) dot += qh[j] * kt[j];
            } else {
                const block_q8 *kb = cache->k_q8 + slot * nb;
                for (int b = 0; b < nb; b++) {
                    float s = 0.0f;
                    for (int j = 0; j < QK; j++) s += qh[b * QK + j] * (float)kb[b].q[j];
                    dot += s * kb[b].d;
                }
            }
            scores[t] = dot * scale;
        }

        softmax(scores, n);

        for (int j = 0; j < hd; j++) oh[j] = 0.0f;
        for (int t = 0; t < n; t++) {
            float wt = scores[t];
            size_t slot = kv_slot(cache, kvh, t);
            if (cache->type == TQ_KV_F32) {
                const float *vt = cache->v_f32 + slot * hd;
                for (int j = 0; j < hd; j++) oh[j] += wt * vt[j];
            } else {
                const block_q8 *vb = cache->v_q8 + slot * nb;
                for (int b = 0; b < nb; b++) {
                    float ws = wt * vb[b].d;
                    for (int j = 0; j < QK; j++) oh[b * QK + j] += ws * (float)vb[b].q[j];
                }
            }
        }
    }
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

    project_qkv(w, cfg, xn, q, k, v);

    norm_heads(q, cfg->n_q_heads,  hd, w->q_norm, cfg->rms_eps);
    norm_heads(k, cfg->n_kv_heads, hd, w->k_norm, cfg->rms_eps);

    rope_heads(q, cfg->n_q_heads,  hd, pos, cfg->rope_theta);
    rope_heads(k, cfg->n_kv_heads, hd, pos, cfg->rope_theta);

    store_kv(cache, cfg, pos, k, v);
    attend_all_heads(cache, cfg, q, pos, scores, attn);
    project_output(w, cfg, attn, out);
    return 0;
}