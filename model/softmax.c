#include "model/model_ops.h"
#include <math.h>

void softmax(float *x, int n) {
    if (n <= 0) return;

    float max_val = x[0];
    for (int i = 1; i < n; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }

    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }

    float inv_sum = 1.0f / sum;
    for (int i = 0; i < n; i++) {
        x[i] *= inv_sum;
    }
}