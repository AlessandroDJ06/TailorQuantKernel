#include "model/model_ops.h"
#include <math.h>

static float dot_f32(const float *q, const float *k, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += q[i] * k[i];
    }
    return sum;
}

static void calc_scores(const float *q, const float *k_cache, float *scores, int stride, float scale, int n, int head_dim) {
    for (int i = 0; i < n; i++) {
        const float *k_i = k_cache + (i * stride);
        scores[i] = dot_f32(q, k_i, head_dim) * scale;
    }
}

static void out_zero(float *out, int head_dim) {
    for (int i = 0; i < head_dim; i++) {
        out[i] = 0;
    }
}

static void attn_mix_v(float *out,const float *v_cache,const float *scores,int n,int stride,int head_dim) {
    for (int i = 0; i < n; i++) {
        const float *v = v_cache + ( i * stride);
        float score = scores[i];
        for (int j = 0; j < head_dim; j++) {
            out[j] += score * v[j];
        }
    }
}

void attention_head(const float *q, const float *k_cache, const float *v_cache,
                    int n, int head_dim, int stride,
                    float *scores, float *out) {

    float scale = 1.0f/ sqrtf((float)head_dim);
    calc_scores(q,k_cache,scores,stride,scale,n,head_dim);
    softmax(scores,n);
    out_zero(out, head_dim);
    attn_mix_v(out,v_cache,scores,n,stride,head_dim);
}
