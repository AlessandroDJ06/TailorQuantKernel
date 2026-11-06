#include <math.h>
#include "model/model_ops.h"

#define SILU_PARALLEL_MIN 2048

static float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

static float silu(float x) {
    return x * sigmoid(x);
}

void silu_mul(const float *gate, const float *up, float *out, int n) {
#pragma omp parallel for schedule(static) if (n >= SILU_PARALLEL_MIN)
    for (int i = 0; i < n; i++) {
        out[i] = silu(gate[i]) * up[i];
    }
}