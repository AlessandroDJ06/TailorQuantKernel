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

    /* Test 1: quantize_row_q8, terugrekenen naar float */
    quantize_row_q8(x, xq, cols);
    for (int b = 0; b < nb; b++)
        for (int j = 0; j < QK; j++)
            x_rt[b * QK + j] = xq[b].q[j] * xq[b].d;
    printf("test 1 (roundtrip x):  fout %.4f%%   (verwacht ~0.4%%)\n",
           100.0f * rel_err(x_rt, x, cols));

    /* Referentie in float */
    for (int r = 0; r < rows; r++) {
        double s = 0;
        for (int c = 0; c < cols; c++) s += (double)W[(size_t)r * cols + c] * x[c];
        y_ref[r] = (float)s;
    }

    /* Test 2: q8 en q4 */
    const char types[] = {'8', '4'};
    const char* names[] = {"q8", "q4"};
    const char* expected[] = {"~0.5%", "~7%"};

    for (int t = 0; t < 2; t++) {
        char type = types[t];
        size_t rb = packed_row_bytes(cols, type);
        uint8_t* packed = malloc(packed_matrix_bytes(rows, cols, type));
        pack_matrix(W, rows, cols, packed, type);

        for (int r = 0; r < rows; r++) {
            const uint8_t* row = packed + (size_t)r * rb;
            if (type == '8') y[r] = dot_q8_q8((const block_q8*)row, xq, nb);
            else             y[r] = dot_q4_q8((const block_q4*)row, xq, nb);
        }

        printf("test 2 (dot_%s_q8):    fout %.4f%%   (verwacht %s)\n",
               names[t], 100.0f * rel_err(y, y_ref, rows), expected[t]);
        free(packed);
    }

    free(W); free(x); free(x_rt); free(y_ref); free(y); free(xq);
    return 0;
}