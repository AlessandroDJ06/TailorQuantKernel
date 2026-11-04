#include "kernel.h"
#include <math.h>
#include <immintrin.h>
#include <stdint.h>

void quantize_row_q8(const float* x, block_q8* out, int n) {
    int nb = n / QK;

    for (int b = 0; b < nb; b++) {
        const float* xb = x + b * QK;

        float max_val = 0.0f;
        for (int j = 0; j < QK; j++) {
            float val = xb[j];
            if (val < 0) val = -val;
            if (val > max_val) max_val = val;
        }

        float scale = max_val / 127.0f;
        float inversed_scale = 0.0f;
        if (scale != 0.0f) inversed_scale = 1.0f / scale;

        out[b].d = scale;
        for (int j = 0; j < QK; j++) {
            out[b].q[j] = (int8_t)lroundf(xb[j] * inversed_scale);
        }
    }
}

float dot_q8_q8(const block_q8* w, const block_q8* x, int nb){
    float sum = 0.0f;

    for (int b = 0; b < nb ; b++) {
        int32_t s = 0;
        for (int j = 0; j < QK; j++) {
            s += w[b].q[j] * x[b].q[j];
        }
        sum += (float)s * w[b].d * x[b].d;
    }

    return sum;
}


void matmul_q8(const void* W, const block_q8* xq, float* y, int rows, int cols) {
    int nb = cols/QK;
    const block_q8* w = (const block_q8*)W;
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < rows; r++) {
        y[r] = dot_q8_q8(w + (size_t)r * nb, xq, nb);
    }
}


float dot_q4_q8(const block_q4* w, const block_q8* x, int nb) {
    float sum = 0.0f;

    for (int b = 0; b < nb; b++) {
        int32_t s = 0;

        for (int j = 0; j < QK/2; j++) {
            int lo = (w[b].q[j] & 0x0F) - 8;
            int hi = (w[b].q[j] >> 4) - 8;
            s += lo * x[b].q[j] + hi * x[b].q[j + QK / 2];
        }
        sum += (float)s * w[b].d * x[b].d;
    }
    return sum;
}

#ifdef __AVX2__
static inline float hsum_ps(__m256 v) {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
    return _mm_cvtss_f32(s);
}

static inline __m256i unpack_q4_signed(const uint8_t* q) {
    const __m128i mask = _mm_set1_epi8(0x0F);
    __m128i bytes = _mm_loadu_si128((const __m128i*)q);
    __m128i lo = _mm_and_si128(bytes, mask);
    __m128i hi = _mm_and_si128(_mm_srli_epi16(bytes, 4), mask);
    __m256i unsigned_w = _mm256_set_m128i(hi, lo);
    return _mm256_sub_epi8(unsigned_w, _mm256_set1_epi8(8));
}

static inline __m256i dot_s8_s8_to_i32(__m256i w, __m256i x) {
    __m256i abs_w = _mm256_abs_epi8(w);
    __m256i signed_x = _mm256_sign_epi8(x, w);
    __m256i p16 = _mm256_maddubs_epi16(abs_w, signed_x);
    return _mm256_madd_epi16(p16, _mm256_set1_epi16(1));
}

float dot_q4_q8_avx2(const block_q4* w, const block_q8* x, int nb) {
    __m256 acc = _mm256_setzero_ps();

    for (int b = 0; b < nb; b++) {
        __m256i wv = unpack_q4_signed(w[b].q);
        __m256i xv = _mm256_loadu_si256((const __m256i*)x[b].q);

        __m256 partial = _mm256_cvtepi32_ps(dot_s8_s8_to_i32(wv, xv));
        __m256 scale = _mm256_set1_ps(w[b].d * x[b].d);
        acc = _mm256_add_ps(acc, _mm256_mul_ps(partial, scale));
    }
    return hsum_ps(acc);
}
#endif

void matmul_q4_scalar(const void* W, const block_q8* xq, float* y, int rows, int cols) {
    int nb = cols / QK;
    const block_q4* w = (const block_q4*)W;
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < rows; r++) {
        y[r] = dot_q4_q8(w + (size_t)r * nb, xq, nb);
    }
}

void matmul_q4(const void* W, const block_q8* xq, float* y, int rows, int cols) {
    #ifdef __AVX2__
    int nb = cols / QK;
    const block_q4* w = (const block_q4*)W;
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < rows; r++) {
        y[r] = dot_q4_q8_avx2(w + (size_t)r * nb, xq, nb);
    }
    #else
    matmul_q4_scalar(W, xq, y, rows, cols);
    #endif
}


