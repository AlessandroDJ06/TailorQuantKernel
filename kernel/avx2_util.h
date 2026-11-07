#ifndef AVX2_UTIL_H
#define AVX2_UTIL_H

#ifndef __AVX2__
#error "avx2_util.h must be compiled with -mavx2"
#endif

#include <immintrin.h>

static inline float hsum_ps(__m256 v) {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
    return _mm_cvtss_f32(s);
}

static inline __m256i dot_s8_s8_to_i32(__m256i w, __m256i x) {
    __m256i abs_w    = _mm256_abs_epi8(w);
    __m256i signed_x = _mm256_sign_epi8(x, w);
    __m256i p16      = _mm256_maddubs_epi16(abs_w, signed_x);
    return _mm256_madd_epi16(p16, _mm256_set1_epi16(1));
}

#define TOK_TILE 4

#define DEFINE_MATMUL_BATCH(NAME, WT, LOADW)                                       \
static inline __attribute__((always_inline))                                       \
void NAME##_tile(const WT* w, const block_q8* x, int nb, int xs,                   \
                 float* out, int os, const int NT) {                               \
    __m256 acc[TOK_TILE];                                                          \
    for (int t = 0; t < NT; t++) acc[t] = _mm256_setzero_ps();                     \
    for (int b = 0; b < nb; b++) {                                                 \
        __m256i wv = LOADW(w[b].q);                                                \
        float wd = w[b].d;                                                         \
        for (int t = 0; t < NT; t++) {                                             \
            const block_q8* xb = x + (size_t)t * xs + b;                           \
            __m256i xv = _mm256_loadu_si256((const __m256i*)xb->q);                \
            __m256 p = _mm256_cvtepi32_ps(dot_s8_s8_to_i32(wv, xv));               \
            acc[t] = _mm256_add_ps(acc[t],                                         \
                         _mm256_mul_ps(p, _mm256_set1_ps(wd * xb->d)));            \
        }                                                                          \
    }                                                                              \
    for (int t = 0; t < NT; t++) out[(size_t)t * os] = hsum_ps(acc[t]);            \
}                                                                                  \
void NAME(const void* W, const block_q8* xq, float* y,                             \
          int rows, int cols, int n_tok) {                                         \
    int nb = cols / QK;                                                            \
    const WT* w = (const WT*)W;                                                    \
    _Pragma("omp parallel for schedule(static)")                                   \
    for (int r = 0; r < rows; r++) {                                               \
        const WT* wr = w + (size_t)r * nb;                                         \
        int t = 0;                                                                 \
        for (; t + TOK_TILE <= n_tok; t += TOK_TILE)                               \
            NAME##_tile(wr, xq + (size_t)t * nb, nb, nb,                           \
                        y + (size_t)t * rows + r, rows, TOK_TILE);                 \
        switch (n_tok - t) {                                                       \
            case 3: NAME##_tile(wr, xq + (size_t)t * nb, nb, nb,                   \
                                y + (size_t)t * rows + r, rows, 3); break;         \
            case 2: NAME##_tile(wr, xq + (size_t)t * nb, nb, nb,                   \
                                y + (size_t)t * rows + r, rows, 2); break;         \
            case 1: NAME##_tile(wr, xq + (size_t)t * nb, nb, nb,                   \
                                y + (size_t)t * rows + r, rows, 1); break;         \
            default: break;                                                        \
        }                                                                          \
    }                                                                              \
}

#endif