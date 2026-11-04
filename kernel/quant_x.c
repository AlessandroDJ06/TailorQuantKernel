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