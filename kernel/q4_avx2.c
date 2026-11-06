#include <string.h>

#include "kernel_internal.h"
#include "avx2_util.h"


#define Q4_ROWS 4

typedef struct {
    uint8_t lo[32];
    uint8_t hi[32];
    float   xd[2];
    int     corr[8];
} XPair;

static void prep_x_pairs(const block_q8* x, int npairs, XPair* xp) {
    for (int p = 0; p < npairs; p++) {
        const block_q8* a = &x[2 * p];
        const block_q8* b = &x[2 * p + 1];
        int sa = 0, sb = 0;
        for (int j = 0; j < QK; j++) { sa += a->q[j]; sb += b->q[j]; }

        memcpy(xp[p].lo,      a->q,      16);
        memcpy(xp[p].lo + 16, b->q,      16);
        memcpy(xp[p].hi,      a->q + 16, 16);
        memcpy(xp[p].hi + 16, b->q + 16, 16);
        xp[p].xd[0] = a->d;
        xp[p].xd[1] = b->d;
        memset(xp[p].corr, 0, sizeof xp[p].corr);
        xp[p].corr[0] = 8 * sa;
        xp[p].corr[4] = 8 * sb;
    }
}


static inline __m256i dot_pair(__m256i nib, const XPair* xp) {
    const __m256i m  = _mm256_set1_epi8(0x0F);
    const __m256i lo = _mm256_and_si256(nib, m);
    const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(nib, 4), m);
    /* per int16 maximaal 2 * 15 * 127 * 2 = 7620: geen overflow */
    const __m256i p16 = _mm256_add_epi16(
        _mm256_maddubs_epi16(lo, _mm256_loadu_si256((const __m256i*)xp->lo)),
        _mm256_maddubs_epi16(hi, _mm256_loadu_si256((const __m256i*)xp->hi)));
    const __m256i p32 = _mm256_madd_epi16(p16, _mm256_set1_epi16(1));
    return _mm256_sub_epi32(p32, _mm256_loadu_si256((const __m256i*)xp->corr));
}

static inline __m256 pair_scale(float da, float db, const XPair* xp) {
    __m128 d = _mm_insert_ps(_mm_set_ss(da), _mm_set_ss(db), 0x10);
    __m128 s = _mm_mul_ps(d, _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*)xp->xd)));
    return _mm256_permutevar8x32_ps(_mm256_castps128_ps256(s),
                                    _mm256_setr_epi32(0, 0, 0, 0, 1, 1, 1, 1));
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
    const int nb      = cols / QK;
    const int npairs  = nb / 2;
    const int odd     = nb & 1;
    const block_q4* w = (const block_q4*)W;

    XPair xp[npairs > 0 ? npairs : 1];
    prep_x_pairs(xq, npairs, xp);

    const int groups = rows / Q4_ROWS;

#pragma omp parallel for schedule(static)
    for (int g = 0; g < groups; g++) {
        const int r = g * Q4_ROWS;
        const block_q4* wr[Q4_ROWS];
        __m256 acc[Q4_ROWS];
#pragma GCC unroll 4
        for (int i = 0; i < Q4_ROWS; i++) {
            wr[i]  = w + (size_t)(r + i) * nb;
            acc[i] = _mm256_setzero_ps();
        }

        for (int p = 0; p < npairs; p++) {
#pragma GCC unroll 4
            for (int i = 0; i < Q4_ROWS; i++) {
                const block_q4* a = &wr[i][2 * p];
                const block_q4* b = a + 1;
                __m256i nib = _mm256_inserti128_si256(
                    _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*)a->q)),
                    _mm_loadu_si128((const __m128i*)b->q), 1);
                acc[i] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(dot_pair(nib, &xp[p])),
                                         pair_scale(a->d, b->d, &xp[p]), acc[i]);
            }
        }

#pragma GCC unroll 4
        for (int i = 0; i < Q4_ROWS; i++) {
            float s = hsum_ps(acc[i]);
            if (odd) s += dot_q4_q8_tail(&wr[i][nb - 1], &xq[nb - 1], 1);
            y[r + i] = s;
        }
    }

    for (int r = groups * Q4_ROWS; r < rows; r++)
        y[r] = dot_q4_q8_tail(w + (size_t)r * nb, xq, nb);
}