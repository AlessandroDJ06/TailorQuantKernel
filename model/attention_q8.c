#include "model/model_ops.h"
#include <math.h>

#include "kernel/kernel.h"



static void scale_scores(float *scores, int n, float scale) {
    for (int t = 0; t < n; t++) {
        scores[t] *= scale;
    }
}

static void out_zero(float *out, int head_dim) {
    for (int i = 0; i < head_dim; i++) {
        out[i] = 0.0f;
    }
}

static void attn_mix_v_q8(float *out, const block_q8 *v_cache, const float *scores,
                          int n, int head_dim) {
    int nb = head_dim / QK;
    for (int t = 0; t < n; t++) {
        for (int b = 0; b < nb; b++) {
            const block_q8 *blk = &v_cache[t * nb + b];
            float factor = scores[t] * blk->d;
            for (int j = 0; j < QK; j++) {
                out[b * QK + j] += factor * (float)blk->q[j];
            }
        }
    }
}

void attention_head_q8(const float *q, block_q8 *q_q8,
                       const block_q8 *k_cache, const block_q8 *v_cache,
                       int n, int head_dim,
                       float *scores, float *out) {
    MatmulFn matmul = get_matmul(TQ_FORMAT_Q8);
    float scale = 1.0f / sqrtf((float)head_dim);

    quantize_row_q8(q, q_q8, head_dim);
    matmul(k_cache, q_q8, scores, n, head_dim);
    scale_scores(scores, n, scale);
    softmax(scores, n);

    out_zero(out, head_dim);
    attn_mix_v_q8(out, v_cache, scores, n, head_dim);
}