#include "model/model_ops.h"
#include <math.h>

static float rms(const float *x, int n, float eps) {
    float sum = 0.0f;

    for (int i = 0; i < n; i++) {
        sum += x[i] * x[i];
    }

    return sqrtf(eps + sum / (float)n);
}

void rmsnorm(const float *x, const float *weight, float *out, int n, float eps) {
    float inv_rms = 1.0f / rms(x, n, eps);

    for (int i = 0; i < n; i++) {
        out[i] = x[i] * inv_rms * weight[i];
    }
}