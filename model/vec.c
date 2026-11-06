#include "model/model_ops.h"

void vec_add(float *x, const float *y, int n) {
    for (int i = 0; i < n; i++) {
        x[i]+= y[i];
    }
}