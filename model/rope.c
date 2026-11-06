#include "model/model_ops.h"
#include <math.h>

void rope_table(float *cos_t, float *sin_t, int head_dim, int pos, float theta) {
    float inv_head_dim = 1.0f / (float)head_dim;
    for (int i = 0; i < head_dim / 2; i++) {
        float freq = powf(theta, -2.0f * i * inv_head_dim);
        float a = pos * freq;
        cos_t[i] = cosf(a);
        sin_t[i] = sinf(a);
    }
}

void rope_apply(float *x, int head_dim, const float *cos_t, const float *sin_t) {
    int half = head_dim / 2;
    for (int i = 0; i < half; i++) {
        float x0 = x[i];
        float x1 = x[i + half];
        x[i]        = x0 * cos_t[i] - x1 * sin_t[i];
        x[i + half] = x0 * sin_t[i] + x1 * cos_t[i];
    }
}

void rope(float *x, int head_dim, int pos, float theta) {
    if (head_dim < 2) return;
    float cos_t[head_dim / 2], sin_t[head_dim / 2];
    rope_table(cos_t, sin_t, head_dim, pos, theta);
    rope_apply(x, head_dim, cos_t, sin_t);
}