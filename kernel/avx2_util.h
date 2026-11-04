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

#endif