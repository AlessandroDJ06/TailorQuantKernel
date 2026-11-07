#include "kernel_internal.h"

static float dot_q4_q8_scalar(const block_q4* w, const block_q8* x, int nb) {
    float sum = 0.0f;
    for (int b = 0; b < nb; b++) {
        int32_t s = 0;
        for (int j = 0; j < QK / 2; j++) {
            int lo = (w[b].q[j] & 0x0F) - 8;
            int hi = (w[b].q[j] >> 4) - 8;
            s += lo * x[b].q[j] + hi * x[b].q[j + QK / 2];
        }
        sum += (float)s * w[b].d * x[b].d;
    }
    return sum;
}

void matmul_q4_scalar(const void* W, const block_q8* xq, float* y, int rows, int cols) {
    int nb = cols / QK;
    const block_q4* w = (const block_q4*)W;
#pragma omp parallel for schedule(static)
    for (int r = 0; r < rows; r++) {
        y[r] = dot_q4_q8_scalar(w + (size_t)r * nb, xq, nb);
    }
}

/* Batched fallback: one matmul per token (correct everywhere, no weight reuse). */
void matmul_q4_scalar_batch(const void* W, const block_q8* xq, float* y,
                            int rows, int cols, int n_tok) {
    int nb = cols / QK;
    for (int t = 0; t < n_tok; t++)
        matmul_q4_scalar(W, xq + (size_t)t * nb, y + (size_t)t * rows, rows, cols);
}