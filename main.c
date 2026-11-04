#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "packer/packer.h"
#include "kernel/kernel.h"

static float frand(void) { return ((float)rand() / RAND_MAX - 0.5f) * 2.0f; }

static float rel_err(const float* a, const float* b, int n) {
    double num = 0, den = 0;
    for (int i = 0; i < n; i++) {
        double d = a[i] - b[i];
        num += d * d;
        den += (double)b[i] * b[i];
    }
    return (float)sqrt(num / den);
}

int main(void) {
    int rows = 256, cols = 512;
    int nb = cols / QK;

    float* W     = malloc((size_t)rows * cols * sizeof(float));
    float* x     = malloc(cols * sizeof(float));
    float* x_rt  = malloc(cols * sizeof(float));
    float* y_ref = malloc(rows * sizeof(float));
    float* y     = malloc(rows * sizeof(float));
    block_q8* xq = malloc(nb * sizeof(block_q8));

    for (size_t i = 0; i < (size_t)rows * cols; i++) W[i] = frand();
    for (int i = 0; i < cols; i++) x[i] = frand();

    quantize_row_q8(x, xq, cols);
    for (int b = 0; b < nb; b++)
        for (int j = 0; j < QK; j++)
            x_rt[b * QK + j] = xq[b].q[j] * xq[b].d;
    printf("test 1:  error %.4f%%   (expected ~0.4%%)\n",
           100.0f * rel_err(x_rt, x, cols));

    for (int r = 0; r < rows; r++) {
        double s = 0;
        for (int c = 0; c < cols; c++) s += (double)W[(size_t)r * cols + c] * x[c];
        y_ref[r] = (float)s;
    }

    block_q8* Wq = malloc(packed_matrix_bytes(rows, cols, '8'));
    pack_matrix(W, rows, cols, Wq, '8');

    for (int r = 0; r < rows; r++)
        y[r] = dot_q8_q8(Wq + (size_t)r * nb, xq, nb);

    printf("test 2 (dot_q8_q8):    error %.4f%%   (expected ~0.5%%)\n",
           100.0f * rel_err(y, y_ref, rows));

    printf("\n row 0: y_ref = %f, y = %f\n", y_ref[0], y[0]);

    free(W); free(x); free(x_rt); free(y_ref); free(y); free(xq); free(Wq);
    return 0;
}