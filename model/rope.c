#include "model/model_ops.h"
#include <math.h>

void rope(float *x, int head_dim, int pos, float theta) {
    float inv_head_dim = 1.0f / (float)head_dim;
    for (int i = 0; i < head_dim / 2; i++) {
        float freq = powf(theta, -2.0f * i * inv_head_dim);
        float a = pos * freq;
        float c = cosf(a);
        float s = sinf(a);
        int idx_alt = i + head_dim / 2;
        float x0 = x[i];
        float x1 = x[idx_alt];
        x[i] = x0 * c - x1 * s;
        x[idx_alt] = x0 * s + x1 * c;
    }
}