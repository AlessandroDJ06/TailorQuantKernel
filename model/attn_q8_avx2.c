#include <immintrin.h>
#include <string.h>

#include "kernel/kernel.h"
#include "kernel/avx2_util.h"

_Static_assert(QK == 32, "attention kernels assume 32-element blocks");

#define ACC_TILE 4
#define V_TILE   128

void attn_scores_q8_avx2(const block_q8 *q, int g, const block_q8 *k, int n, int nb,
                         float scale, float *scores, size_t stride) {
    for (int t = 0; t < n; t++) {
        const block_q8 *kb = k + (size_t)t * nb;

        for (int gi = 0; gi < g; gi++) {
            const block_q8 *qb = q + (size_t)gi * nb;
            __m256 acc = _mm256_setzero_ps();

            for (int b = 0; b < nb; b++) {
                __m256i kv = _mm256_loadu_si256((const __m256i *)kb[b].q);
                __m256i qv = _mm256_loadu_si256((const __m256i *)qb[b].q);

                __m256 partial = _mm256_cvtepi32_ps(dot_s8_s8_to_i32(kv, qv));
                acc = _mm256_fmadd_ps(partial, _mm256_set1_ps(kb[b].d * qb[b].d), acc);
            }
            scores[gi * stride + t] = hsum_ps(acc) * scale;
        }
    }
}

static inline __attribute__((always_inline))
void accum_tile(const float *w, size_t w_stride, const block_q8 *v, int n, int nb,
                float *out, size_t out_stride, const int gn) {
    for (int t0 = 0; t0 < n; t0 += V_TILE) {
        const int t1 = n - t0 < V_TILE ? n : t0 + V_TILE;

        for (int c = 0; c < nb * 2; c++) {
            const int b   = c >> 1;
            const int off = (c & 1) * 16;
            const int col = b * QK + off;

            __m256 lo[ACC_TILE], hi[ACC_TILE];
#pragma GCC unroll 4
            for (int g = 0; g < gn; g++) {
                lo[g] = _mm256_loadu_ps(out + g * out_stride + col);
                hi[g] = _mm256_loadu_ps(out + g * out_stride + col + 8);
            }

            for (int t = t0; t < t1; t++) {
                const block_q8 *blk = v + (size_t)t * nb + b;
                const __m256 f0 = _mm256_cvtepi32_ps(
                    _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)(blk->q + off))));
                const __m256 f1 = _mm256_cvtepi32_ps(
                    _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)(blk->q + off + 8))));
                const float d = blk->d;

#pragma GCC unroll 4
                for (int g = 0; g < gn; g++) {
                    const __m256 wd = _mm256_set1_ps(w[g * w_stride + t] * d);
                    lo[g] = _mm256_fmadd_ps(wd, f0, lo[g]);
                    hi[g] = _mm256_fmadd_ps(wd, f1, hi[g]);
                }
            }

#pragma GCC unroll 4
            for (int g = 0; g < gn; g++) {
                _mm256_storeu_ps(out + g * out_stride + col, lo[g]);
                _mm256_storeu_ps(out + g * out_stride + col + 8, hi[g]);
            }
        }
    }
}

void attn_accum_q8_avx2(const float *w, size_t w_stride, int g, const block_q8 *v, int n, int nb,
                        float *out, size_t out_stride) {
    const size_t hd = (size_t)nb * QK;

    for (int g0 = 0; g0 < g; g0 += ACC_TILE) {
        const int gn = g - g0 < ACC_TILE ? g - g0 : ACC_TILE;
        const float *wg = w + (size_t)g0 * w_stride;
        float *og = out + (size_t)g0 * out_stride;

        for (int i = 0; i < gn; i++) memset(og + (size_t)i * out_stride, 0, hd * sizeof(float));

        switch (gn) {
            case 1:  accum_tile(wg, w_stride, v, n, nb, og, out_stride, 1); break;
            case 2:  accum_tile(wg, w_stride, v, n, nb, og, out_stride, 2); break;
            case 3:  accum_tile(wg, w_stride, v, n, nb, og, out_stride, 3); break;
            default: accum_tile(wg, w_stride, v, n, nb, og, out_stride, 4); break;
        }
    }
}