#include "model/model_ops.h"

int sample_greedy(const float *logits, int n) {
    if (n <= 0) return -1;
    float max = logits[0];
    int index = 0;
    for (int i = 1; i < n; i++) {
        if (logits[i] > max) {
            max = logits[i];
            index = i;
        }
    }
    return index;
}