#include "kernel_internal.h"
#include "avx2_util.h"

static float dot_q8_q8_avx2(const block_q8* w, const block_q8* x, int nb) {
    __m256 acc = _mm256_setzero_ps();

    for (int b = 0; b < nb; b++) {
        __m256i wv = _mm256_loadu_si256((const __m256i*)w[b].q);
        __m256i xv = _mm256_loadu_si256((const __m256i*)x[b].q);

        __m256 partial = _mm256_cvtepi32_ps(dot_s8_s8_to_i32(wv, xv));
        __m256 scale   = _mm256_set1_ps(w[b].d * x[b].d);
        acc = _mm256_add_ps(acc, _mm256_mul_ps(partial, scale));
    }
    return hsum_ps(acc);
}

void matmul_q8_avx2(const void* W, const block_q8* xq, float* y, int rows, int cols) {
    int nb = cols / QK;
    const block_q8* w = (const block_q8*)W;
#pragma omp parallel for schedule(static)
    for (int r = 0; r < rows; r++) {
        y[r] = dot_q8_q8_avx2(w + (size_t)r * nb, xq, nb);
    }
}