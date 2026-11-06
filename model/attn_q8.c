#include <math.h>
#include <stddef.h>

#include "model/model_ops.h"

#ifdef TQ_HAVE_AVX2
void attn_scores_q8_avx2(const block_q8 *q, int g, const block_q8 *k, int n, int nb,
                         float scale, float *scores, size_t stride);
void attn_accum_q8_avx2(const float *w, size_t w_stride, int g, const block_q8 *v, int n, int nb,
                        float *out, size_t out_stride);

static int cpu_has_avx2_fma(void) {
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
}
#endif

void attn_scores_q8_scalar(const block_q8 *q, int g, const block_q8 *k, int n, int nb,
                           float scale, float *scores, size_t stride) {
    for (int t = 0; t < n; t++) {
        const block_q8 *kb = k + (size_t)t * nb;

        for (int gi = 0; gi < g; gi++) {
            const block_q8 *qb = q + (size_t)gi * nb;
            float acc = 0.0f;

            for (int b = 0; b < nb; b++) {
                int sum = 0;
                for (int j = 0; j < QK; j++) sum += (int)qb[b].q[j] * (int)kb[b].q[j];
                acc += (float)sum * (qb[b].d * kb[b].d);
            }
            scores[(size_t)gi * stride + t] = acc * scale;
        }
    }
}

void attn_accum_q8_scalar(const float *w, size_t w_stride, int g, const block_q8 *v, int n, int nb,
                          float *out, size_t out_stride) {
    const int hd = nb * QK;

    for (int gi = 0; gi < g; gi++) {
        float *o = out + (size_t)gi * out_stride;
        for (int j = 0; j < hd; j++) o[j] = 0.0f;

        for (int t = 0; t < n; t++) {
            const float wt = w[(size_t)gi * w_stride + t];
            const block_q8 *vb = v + (size_t)t * nb;

            for (int b = 0; b < nb; b++) {
                const float ws = wt * vb[b].d;
                for (int j = 0; j < QK; j++) o[b * QK + j] += ws * (float)vb[b].q[j];
            }
        }
    }
}

void attn_scores_q8(const block_q8 *q, int g, const block_q8 *k, int n, int nb,
                    float scale, float *scores, size_t stride) {
#ifdef TQ_HAVE_AVX2
    if (cpu_has_avx2_fma()) {
        attn_scores_q8_avx2(q, g, k, n, nb, scale, scores, stride);
        return;
    }
#endif
    attn_scores_q8_scalar(q, g, k, n, nb, scale, scores, stride);
}

void attn_accum_q8(const float *w, size_t w_stride, int g, const block_q8 *v, int n, int nb,
                   float *out, size_t out_stride) {
#ifdef TQ_HAVE_AVX2
    if (cpu_has_avx2_fma()) {
        attn_accum_q8_avx2(w, w_stride, g, v, n, nb, out, out_stride);
        return;
    }
#endif
    attn_accum_q8_scalar(w, w_stride, g, v, n, nb, out, out_stride);
}

void attention_head_q8(const float *q, block_q8 *q_q8,
                       const block_q8 *k_cache, const block_q8 *v_cache,
                       int n, int head_dim,
                       float *scores, float *out) {
    const int nb = head_dim / QK;
    const float scale = 1.0f / sqrtf((float)head_dim);

    quantize_row_q8(q, q_q8, head_dim);

    attn_scores_q8(q_q8, 1, k_cache, n, nb, scale, scores, (size_t)n);
    softmax(scores, n);
    attn_accum_q8(scores, (size_t)n, 1, v_cache, n, nb, out, (size_t)head_dim);
}