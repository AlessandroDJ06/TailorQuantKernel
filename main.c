#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "packer/packer.h"

static float rel_err(const float* a, const float* b, size_t n) {
    double num = 0, den = 0;
    for (size_t i = 0; i < n; i++) { double d = a[i] - b[i]; num += d * d; den += (double)b[i] * b[i]; }
    return (float)sqrt(num / den);
}

int main(void) {
    int rows = 256, cols = 512;
    size_t n = (size_t)rows * cols;
    float* W  = malloc(n * sizeof(float));
    float* Wu = malloc(n * sizeof(float));
    for (size_t i = 0; i < n; i++) W[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;

    const char types[] = {'8', '4'};
    for (int t = 0; t < 2; t++) {
        void* packed = malloc(packed_matrix_bytes(rows, cols, types[t]));
        pack_matrix(W, rows, cols, packed, types[t]);
        unpack_matrix(packed, rows, cols, Wu, types[t]);
        printf("q%c: %zu bytes, relative wrong %.4f%%\n",
               types[t], packed_matrix_bytes(rows, cols, types[t]), 100.0f * rel_err(Wu, W, n));
        free(packed);
    }
    free(W); free(Wu);
    return 0;
}