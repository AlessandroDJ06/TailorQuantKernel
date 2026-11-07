#include "kernel_internal.h"
#include "avx2_util.h"

static inline __m256i unpack_q4_signed(const uint8_t* q) {
    const __m128i mask = _mm_set1_epi8(0x0F);
    __m128i bytes = _mm_loadu_si128((const __m128i*)q);
    __m128i lo = _mm_and_si128(bytes, mask);
    __m128i hi = _mm_and_si128(_mm_srli_epi16(bytes, 4), mask);
    __m256i unsigned_w = _mm256_set_m128i(hi, lo);
    return _mm256_sub_epi8(unsigned_w, _mm256_set1_epi8(8));
}

static float dot_q4_q8_avx2(const block_q4* w, const block_q8* x, int nb) {
    __m256 acc = _mm256_setzero_ps();

    for (int b = 0; b < nb; b++) {
        __m256i wv = unpack_q4_signed(w[b].q);
        __m256i xv = _mm256_loadu_si256((const __m256i*)x[b].q);

        __m256 partial = _mm256_cvtepi32_ps(dot_s8_s8_to_i32(wv, xv));
        __m256 scale   = _mm256_set1_ps(w[b].d * x[b].d);
        acc = _mm256_add_ps(acc, _mm256_mul_ps(partial, scale));
    }
    return hsum_ps(acc);
}

void matmul_q4_avx2(const void* W, const block_q8* xq, float* y, int rows, int cols) {
    int nb = cols / QK;
    const block_q4* w = (const block_q4*)W;
#pragma omp parallel for schedule(static)
    for (int r = 0; r < rows; r++) {
        y[r] = dot_q4_q8_avx2(w + (size_t)r * nb, xq, nb);
    }
}

DEFINE_MATMUL_BATCH(matmul_q4_avx2_batch, block_q4, unpack_q4_signed)