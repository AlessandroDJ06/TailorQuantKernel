#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "packer/packer.h"
#include "kernel/kernel.h"
#include "kernel/kernel_internal.h"   /* only for the scalar reference functions */

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

    float* W        = malloc((size_t)rows * cols * sizeof(float));
    float* x        = malloc(cols * sizeof(float));
    float* x_rt     = malloc(cols * sizeof(float));
    float* y_ref    = malloc(rows * sizeof(float));
    float* y        = malloc(rows * sizeof(float));
    float* y_scalar = malloc(rows * sizeof(float));
    block_q8* xq    = malloc(nb * sizeof(block_q8));

    for (size_t i = 0; i < (size_t)rows * cols; i++) W[i] = frand();
    for (int i = 0; i < cols; i++) x[i] = frand();

    /* Test 1: quantize_row_q8 roundtrip */
    quantize_row_q8(x, xq, cols);
    for (int b = 0; b < nb; b++)
        for (int j = 0; j < QK; j++)
            x_rt[b * QK + j] = xq[b].q[j] * xq[b].d;
    printf("test 1 (roundtrip x):        error %.4f%%   (expected ~0.4%%)\n",
           100.0f * rel_err(x_rt, x, cols));

    /* float reference */
    for (int r = 0; r < rows; r++) {
        double s = 0;
        for (int c = 0; c < cols; c++) s += (double)W[(size_t)r * cols + c] * x[c];
        y_ref[r] = (float)s;
    }

    const char types[]     = {'8', '4'};
    const char* names[]    = {"q8", "q4"};
    const char* expected[] = {"~0.5%", "~6.5%"};
    MatmulFn scalar[]      = {matmul_q8_scalar, matmul_q4_scalar};

    for (int t = 0; t < 2; t++) {
        MatmulFn fast = get_matmul(types[t]);
        if (!fast) { fprintf(stderr, "no matmul for type %c\n", types[t]); return 1; }

        void* packed = malloc(packed_matrix_bytes(rows, cols, types[t]));
        pack_matrix(W, rows, cols, packed, types[t]);

        /* Test 2: accuracy of the dispatched kernel vs float */
        fast(packed, xq, y, rows, cols);
        printf("test 2 (matmul_%s vs float):  error %.4f%%   (expected %s)\n",
               names[t], 100.0f * rel_err(y, y_ref, rows), expected[t]);

        /* Test 3: dispatched kernel vs scalar kernel (should be ~identical) */
        scalar[t](packed, xq, y_scalar, rows, cols);
        printf("test 3 (%s fast vs scalar):   diff  %.6f%%   (expected < 0.001%%)\n",
               names[t], 100.0f * rel_err(y, y_scalar, rows));

        free(packed);
    }

    free(W); free(x); free(x_rt); free(y_ref); free(y); free(y_scalar); free(xq);
    return 0;
}