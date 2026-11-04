#include "kernel.h"
#include <math.h>

void quantize_row_q8(const float* x, block_q8* out, int n) {
    int nb = n / QK;

    for (int b = 0; b < nb; b++) {
        const float* xb = x + b * QK;

        float max_val = 0.0f;
        for (int j = 0; j < QK; j++) {
            float val = xb[j];
            if (val < 0) val = -val;
            if (val > max_val) max_val = val;
        }

        float scale = max_val / 127.0f;
        float inversed_scale = 0.0f;
        if (scale != 0.0f) inversed_scale = 1.0f / scale;

        out[b].d = scale;
        for (int j = 0; j < QK; j++) {
            out[b].q[j] = (int8_t)lroundf(xb[j] * inversed_scale);
        }
    }
}

float dot_q8_q8(const block_q8* w, const block_q8* x, int nb){
    float sum = 0.0f;

    for (int b = 0; b < nb ; b++) {
        int32_t s = 0;
        for (int j = 0; j < QK; j++) {
            s += w[b].q[j] * x[b].q[j];
        }
        sum += (float)s * w[b].d * x[b].d;
    }

    return sum;
}

float dot_q4_q8(const block_q4* w, const block_q8* x, int nb) {
    float sum = 0.0f;

    for (int b = 0; b < nb; b++) {
        int32_t s = 0;

        for (int j = 0; j < QK/2; j++) {
            int lo = (w[b].q[j] & 0x0F) - 8;
            int hi = (w[b].q[j] >> 4) - 8;
            s += lo * x[b].q[j] + hi * x[b].q[j + QK / 2];
        }

        sum += (float)s * w[b].d * x[b].d;
    }

    return sum;
}

void matmul_q8(const void* W, const block_q8* xq, float* y, int rows, int cols) {
    int nb = cols/QK;
    const block_q8* w = (const block_q8*)W;

    for (int r = 0; r < rows; r++) {
        y[r] = dot_q8_q8(w + (size_t)r * nb, xq, nb);
    }
}

void matmul_q4(const void* W, const block_q8* xq, float* y, int rows, int cols) {
    int nb = cols/QK;
    const block_q4* w = (const block_q4*)W;

    for (int r = 0; r < rows; r++) {
        y[r] = dot_q4_q8(w + (size_t)r * nb, xq, nb);
    }
}


