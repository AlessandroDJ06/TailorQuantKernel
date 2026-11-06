#include "kernel_internal.h"
#include "avx2_util.h"

#define Q4_ROWS 4

static inline __m256i unpack_q4_u(const uint8_t* q) {
    __m256i b = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)q));
    b = _mm256_srlv_epi32(b, _mm256_setr_epi32(0, 0, 0, 0, 4, 4, 4, 4));
    return _mm256_and_si256(b, _mm256_set1_epi8(0x0F));
}

static float dot_q4_q8_tail(const block_q4* w, const block_q8* x, int nb) {
    float acc = 0.0f;
    for (int b = 0; b < nb; b++) {
        int s = 0;
        for (int j = 0; j < QK / 2; j++) {
            s += ((w[b].q[j] & 0x0F) - 8) * x[b].q[j];
            s += ((w[b].q[j] >> 4)   - 8) * x[b].q[j + QK / 2];
        }
        acc += (float)s * (w[b].d * x[b].d);
    }
    return acc;
}

void matmul_q4_avx2(const void* W, const block_q8* xq, float* y, int rows, int cols) {
    const int nb = cols / QK;
    const block_q4* w = (const block_q4*)W;

    __m256i corr[nb];
    float   xd[nb];
    for (int b = 0; b < nb; b++) {
        int s = 0;
        for (int j = 0; j < QK; j++) s += xq[b].q[j];
        corr[b] = _mm256_setr_epi32(8 * s, 0, 0, 0, 0, 0, 0, 0);
        xd[b]   = xq[b].d;
    }

    const int groups = rows / Q4_ROWS;

#pragma omp parallel for schedule(static)
    for (int g = 0; g < groups; g++) {
        const int r = g * Q4_ROWS;
        const block_q4* wr[Q4_ROWS];
        __m256 acc[Q4_ROWS];
        for (int i = 0; i < Q4_ROWS; i++) {
            wr[i]  = w + (size_t)(r + i) * nb;
            acc[i] = _mm256_setzero_ps();
        }

        for (int b = 0; b < nb; b++) {
            const __m256i xv  = _mm256_loadu_si256((const __m256i*)xq[b].q);
            const __m256  xdv = _mm256_set1_ps(xd[b]);
            const __m256i cv  = corr[b];
            for (int i = 0; i < Q4_ROWS; i++) {
                __m256i p = _mm256_madd_epi16(
                    _mm256_maddubs_epi16(unpack_q4_u(wr[i][b].q), xv),
                    _mm256_set1_epi16(1));
                p = _mm256_sub_epi32(p, cv);
                __m256 sc = _mm256_mul_ps(_mm256_broadcast_ss(&wr[i][b].d), xdv);
                acc[i] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p), sc, acc[i]);
            }
        }
        for (int i = 0; i < Q4_ROWS; i++) y[r + i] = hsum_ps(acc[i]);
    }

    for (int r = groups * Q4_ROWS; r < rows; r++)
        y[r] = dot_q4_q8_tail(w + (size_t)r * nb, xq, nb);
}