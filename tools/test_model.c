#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "model/model_ops.h"
#include "kernel/kernel.h"

#define MAX_N 1024
#define GUARD 12345.0f

static int is_close(double actual, double expected, double tolerance) {
    return fabs(actual - expected) <= tolerance * (1.0 + fabs(expected));
}

static int expect_close(const char *label, int index, float actual, double expected, double tolerance) {
    if (is_close(actual, expected, tolerance)) {
        return 0;
    }
    printf("  %s[%d]: expected %.6f, got %.6f\n", label, index, expected, (double)actual);
    return 1;
}

static int report(const char *name, int failures) {
    printf("%s: %s\n", name, failures == 0 ? "PASS" : "FAIL");
    return failures;
}

static float next_random(unsigned *state) {
    *state = *state * 1664525u + 1013904223u;
    return (float)(*state >> 8) / (float)(1u << 23) - 1.0f;
}

static void fill_random(float *x, int n, unsigned seed, float scale) {
    unsigned state = seed;
    for (int i = 0; i < n; i++) {
        x[i] = next_random(&state) * scale;
    }
}

static void fill_constant(float *x, int n, float value) {
    for (int i = 0; i < n; i++) {
        x[i] = value;
    }
}

static double silu_reference(double x) {
    return x / (1.0 + exp(-x));
}

static void rmsnorm_reference(const float *x, const float *weight, double *out, int n, double eps) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += (double)x[i] * (double)x[i];
    }

    double rms = sqrt(sum / n + eps);
    for (int i = 0; i < n; i++) {
        out[i] = (double)x[i] / rms * (double)weight[i];
    }
}

static int test_silu_known_values(void) {
    const float gate[] = {0.0f, 1.0f, -1.0f, 2.0f, -2.0f};
    const float expected[] = {0.0f, 0.7311f, -0.2689f, 1.7616f, -0.2384f};
    const float up = 1.0f;
    int failures = 0;

    for (int i = 0; i < 5; i++) {
        float out;
        silu_mul(&gate[i], &up, &out, 1);
        failures += expect_close("silu", i, out, expected[i], 1e-4);
    }

    return report("silu known values", failures);
}

static int test_silu_mul_against_reference(void) {
    float gate[MAX_N], up[MAX_N], out[MAX_N];
    fill_random(gate, MAX_N, 1u, 10.0f);
    fill_random(up, MAX_N, 2u, 3.0f);

    silu_mul(gate, up, out, MAX_N);

    int failures = 0;
    for (int i = 0; i < MAX_N; i++) {
        double expected = silu_reference(gate[i]) * up[i];
        failures += expect_close("silu_mul", i, out[i], expected, 1e-5);
    }

    return report("silu_mul against double reference", failures);
}

static int test_silu_extreme_inputs(void) {
    const float gate[] = {-1000.0f, -100.0f, 100.0f, 1000.0f};
    const float expected[] = {0.0f, 0.0f, 100.0f, 1000.0f};
    const float up = 1.0f;
    int failures = 0;

    for (int i = 0; i < 4; i++) {
        float out;
        silu_mul(&gate[i], &up, &out, 1);
        if (!isfinite(out)) {
            printf("  silu extreme[%d]: result is not finite\n", i);
            failures++;
            continue;
        }
        failures += expect_close("silu extreme", i, out, expected[i], 1e-5);
    }

    return report("silu extreme inputs", failures);
}

static int test_rmsnorm_known_values(void) {
    const float x[] = {1.0f, 2.0f, 3.0f, 4.0f};
    const float weight[] = {1.0f, 1.0f, 1.0f, 1.0f};
    const float expected[] = {0.3651f, 0.7303f, 1.0954f, 1.4606f};
    float out[4];

    rmsnorm(x, weight, out, 4, 0.0f);

    int failures = 0;
    for (int i = 0; i < 4; i++) {
        failures += expect_close("rmsnorm", i, out[i], expected[i], 1e-4);
    }

    return report("rmsnorm known values", failures);
}

static int test_rmsnorm_output_has_unit_rms(void) {
    float x[MAX_N], weight[MAX_N], out[MAX_N];
    fill_random(x, MAX_N, 3u, 5.0f);
    fill_constant(weight, MAX_N, 1.0f);

    rmsnorm(x, weight, out, MAX_N, 1e-6f);

    double sum = 0.0;
    for (int i = 0; i < MAX_N; i++) {
        sum += (double)out[i] * (double)out[i];
    }
    double output_rms = sqrt(sum / MAX_N);

    int failures = expect_close("output rms", 0, (float)output_rms, 1.0, 1e-4);
    return report("rmsnorm output has unit rms", failures);
}

static int test_rmsnorm_weight_scales_output(void) {
    float x[MAX_N], weight_one[MAX_N], weight_two[MAX_N], out_one[MAX_N], out_two[MAX_N];
    fill_random(x, MAX_N, 4u, 2.0f);
    fill_constant(weight_one, MAX_N, 1.0f);
    fill_constant(weight_two, MAX_N, 2.0f);

    rmsnorm(x, weight_one, out_one, MAX_N, 1e-6f);
    rmsnorm(x, weight_two, out_two, MAX_N, 1e-6f);

    int failures = 0;
    for (int i = 0; i < MAX_N; i++) {
        failures += expect_close("weight scaling", i, out_two[i], 2.0 * out_one[i], 1e-6);
    }

    return report("rmsnorm weight scales output", failures);
}

static int test_rmsnorm_in_place(void) {
    float x[MAX_N], weight[MAX_N], separate[MAX_N], in_place[MAX_N];
    fill_random(x, MAX_N, 5u, 2.0f);
    fill_random(weight, MAX_N, 6u, 1.0f);
    for (int i = 0; i < MAX_N; i++) {
        in_place[i] = x[i];
    }

    rmsnorm(x, weight, separate, MAX_N, 1e-6f);
    rmsnorm(in_place, weight, in_place, MAX_N, 1e-6f);

    int failures = 0;
    for (int i = 0; i < MAX_N; i++) {
        failures += expect_close("in place", i, in_place[i], separate[i], 1e-7);
    }

    return report("rmsnorm in place", failures);
}

static int test_rmsnorm_against_reference(void) {
    float x[MAX_N], weight[MAX_N], out[MAX_N];
    double expected[MAX_N];
    fill_random(x, MAX_N, 7u, 4.0f);
    fill_random(weight, MAX_N, 8u, 1.5f);

    rmsnorm(x, weight, out, MAX_N, 1e-6f);
    rmsnorm_reference(x, weight, expected, MAX_N, 1e-6);

    int failures = 0;
    for (int i = 0; i < MAX_N; i++) {
        failures += expect_close("rmsnorm", i, out[i], expected[i], 1e-5);
    }

    return report("rmsnorm against double reference", failures);
}

static void softmax_reference(const float *x, double *out, int n) {
    double max_val = x[0];
    for (int i = 1; i < n; i++) {
        if (x[i] > max_val) max_val = x[i];
    }

    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        out[i] = exp((double)x[i] - max_val);
        sum += out[i];
    }
    for (int i = 0; i < n; i++) {
        out[i] /= sum;
    }
}

static int test_softmax_known_values(void) {
    float x[] = {1.0f, 2.0f, 3.0f};
    const float expected[] = {0.0900f, 0.2447f, 0.6652f};

    softmax(x, 3);

    int failures = 0;
    for (int i = 0; i < 3; i++) {
        failures += expect_close("softmax", i, x[i], expected[i], 1e-3);
    }
    return report("softmax known values", failures);
}

static int test_softmax_sums_to_one(void) {
    float x[MAX_N];
    fill_random(x, MAX_N, 9u, 10.0f);

    softmax(x, MAX_N);

    double sum = 0.0;
    for (int i = 0; i < MAX_N; i++) {
        sum += x[i];
    }

    int failures = expect_close("sum", 0, (float)sum, 1.0, 1e-5);
    return report("softmax sums to one", failures);
}

static int test_softmax_large_inputs(void) {
    float large[] = {1000.0f, 1001.0f, 1002.0f};
    float small[] = {0.0f, 1.0f, 2.0f};

    softmax(large, 3);
    softmax(small, 3);

    int failures = 0;
    for (int i = 0; i < 3; i++) {
        if (!isfinite(large[i])) {
            printf("  softmax large[%d]: result is not finite\n", i);
            failures++;
            continue;
        }
        failures += expect_close("softmax shift invariance", i, large[i], small[i], 1e-6);
    }
    return report("softmax large inputs", failures);
}

static int test_softmax_constant_input(void) {
    float x[8];
    fill_constant(x, 8, 5.0f);

    softmax(x, 8);

    int failures = 0;
    for (int i = 0; i < 8; i++) {
        failures += expect_close("softmax constant", i, x[i], 1.0 / 8.0, 1e-6);
    }
    return report("softmax constant input", failures);
}

static int test_softmax_against_reference(void) {
    float x[MAX_N], original[MAX_N];
    double expected[MAX_N];
    fill_random(x, MAX_N, 10u, 20.0f);
    for (int i = 0; i < MAX_N; i++) {
        original[i] = x[i];
    }

    softmax(x, MAX_N);
    softmax_reference(original, expected, MAX_N);

    int failures = 0;
    for (int i = 0; i < MAX_N; i++) {
        failures += expect_close("softmax", i, x[i], expected[i], 1e-5);
    }
    return report("softmax against double reference", failures);
}

#define ROPE_MAX_DIM 128

static void rope_reference(const float *x, double *out, int head_dim, int pos, double theta) {
    int half = head_dim / 2;
    for (int i = 0; i < half; i++) {
        double freq = pow(theta, -2.0 * i / head_dim);
        double angle = pos * freq;
        double c = cos(angle);
        double s = sin(angle);
        out[i]        = (double)x[i] * c - (double)x[i + half] * s;
        out[i + half] = (double)x[i] * s + (double)x[i + half] * c;
    }
}

static double dot(const float *a, const float *b, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += (double)a[i] * (double)b[i];
    }
    return sum;
}

static int test_rope_known_values(void) {
    float x[] = {1.0f, 0.0f};

    rope(x, 2, 1, 10000.0f);

    int failures = 0;
    failures += expect_close("rope", 0, x[0], cos(1.0), 1e-5);
    failures += expect_close("rope", 1, x[1], sin(1.0), 1e-5);
    return report("rope known values", failures);
}

static int test_rope_position_zero_is_identity(void) {
    float x[ROPE_MAX_DIM], original[ROPE_MAX_DIM];
    fill_random(x, ROPE_MAX_DIM, 11u, 3.0f);
    for (int i = 0; i < ROPE_MAX_DIM; i++) {
        original[i] = x[i];
    }

    rope(x, ROPE_MAX_DIM, 0, 1000000.0f);

    int failures = 0;
    for (int i = 0; i < ROPE_MAX_DIM; i++) {
        failures += expect_close("rope pos 0", i, x[i], original[i], 1e-7);
    }
    return report("rope position zero is identity", failures);
}

static int test_rope_preserves_norm(void) {
    float x[ROPE_MAX_DIM];
    fill_random(x, ROPE_MAX_DIM, 12u, 3.0f);
    double before = dot(x, x, ROPE_MAX_DIM);

    rope(x, ROPE_MAX_DIM, 100, 1000000.0f);

    double after = dot(x, x, ROPE_MAX_DIM);
    int failures = expect_close("norm", 0, (float)after, before, 1e-5);
    return report("rope preserves norm", failures);
}

static int test_rope_against_reference(void) {
    const int dims[] = {8, 128};
    const int positions[] = {1, 7, 100};
    int failures = 0;

    for (int d = 0; d < 2; d++) {
        for (int p = 0; p < 3; p++) {
            float x[ROPE_MAX_DIM], original[ROPE_MAX_DIM];
            double expected[ROPE_MAX_DIM];
            fill_random(x, dims[d], 13u + (unsigned)p, 3.0f);
            for (int i = 0; i < dims[d]; i++) {
                original[i] = x[i];
            }

            rope(x, dims[d], positions[p], 1000000.0f);
            rope_reference(original, expected, dims[d], positions[p], 1000000.0);

            for (int i = 0; i < dims[d]; i++) {
                failures += expect_close("rope", i, x[i], expected[i], 1e-4);
            }
        }
    }
    return report("rope against double reference", failures);
}

static int test_rope_relative_position(void) {
    float q1[ROPE_MAX_DIM], k1[ROPE_MAX_DIM], q2[ROPE_MAX_DIM], k2[ROPE_MAX_DIM];
    fill_random(q1, ROPE_MAX_DIM, 20u, 1.0f);
    fill_random(k1, ROPE_MAX_DIM, 21u, 1.0f);
    for (int i = 0; i < ROPE_MAX_DIM; i++) {
        q2[i] = q1[i];
        k2[i] = k1[i];
    }

    rope(q1, ROPE_MAX_DIM, 5, 1000000.0f);
    rope(k1, ROPE_MAX_DIM, 3, 1000000.0f);
    rope(q2, ROPE_MAX_DIM, 12, 1000000.0f);
    rope(k2, ROPE_MAX_DIM, 10, 1000000.0f);

    int failures = expect_close("relative dot", 0, (float)dot(q1, k1, ROPE_MAX_DIM),
                                dot(q2, k2, ROPE_MAX_DIM), 1e-4);
    return report("rope depends only on relative position", failures);
}


#define ATTN_MAX_DIM    128
#define ATTN_MAX_N      128
#define ATTN_MAX_STRIDE 384
#define ATTN_GARBAGE    999.0f

static float attn_k_buf[ATTN_MAX_N * ATTN_MAX_STRIDE];
static float attn_v_buf[ATTN_MAX_N * ATTN_MAX_STRIDE];
static float attn_scores[ATTN_MAX_N];

static void attn_reset_caches(void) {
    fill_constant(attn_k_buf, ATTN_MAX_N * ATTN_MAX_STRIDE, ATTN_GARBAGE);
    fill_constant(attn_v_buf, ATTN_MAX_N * ATTN_MAX_STRIDE, ATTN_GARBAGE);
}

static void attn_fill_random(float *cache, int n, int head_dim, int stride,
                             int offset, unsigned seed, float scale) {
    unsigned state = seed;
    for (int t = 0; t < n; t++) {
        for (int i = 0; i < head_dim; i++) {
            cache[t * stride + offset + i] = next_random(&state) * scale;
        }
    }
}

static void attention_reference(const float *q, const float *k_cache, const float *v_cache,
                                int n, int head_dim, int stride, double *out) {
    double scores[ATTN_MAX_N];
    double scale = 1.0 / sqrt((double)head_dim);

    for (int t = 0; t < n; t++) {
        scores[t] = dot(q, k_cache + t * stride, head_dim) * scale;
    }

    double max_val = scores[0];
    for (int t = 1; t < n; t++) {
        if (scores[t] > max_val) max_val = scores[t];
    }

    double sum = 0.0;
    for (int t = 0; t < n; t++) {
        scores[t] = exp(scores[t] - max_val);
        sum += scores[t];
    }

    for (int i = 0; i < head_dim; i++) {
        out[i] = 0.0;
    }
    for (int t = 0; t < n; t++) {
        double weight = scores[t] / sum;
        const float *v = v_cache + t * stride;
        for (int i = 0; i < head_dim; i++) {
            out[i] += weight * (double)v[i];
        }
    }
}

static int test_attention_single_token(void) {
    const int head_dim = 8, stride = 24, n = 1;
    float q[8], out[8];
    fill_random(q, head_dim, 30u, 1.0f);

    attn_reset_caches();
    attn_fill_random(attn_k_buf, n, head_dim, stride, 0, 31u, 1.0f);
    for (int i = 0; i < head_dim; i++) {
        attn_v_buf[i] = (float)(i + 1);
    }

    attention_head(q, attn_k_buf, attn_v_buf, n, head_dim, stride, attn_scores, out);

    int failures = 0;
    for (int i = 0; i < head_dim; i++) {
        failures += expect_close("attention n=1", i, out[i], (double)(i + 1), 1e-6);
    }
    return report("attention single token returns v", failures);
}

static int test_attention_identical_keys(void) {
    const int head_dim = 4, stride = 12, n = 4;
    float q[4], key[4], out[4];
    fill_random(q, head_dim, 32u, 1.0f);
    fill_random(key, head_dim, 33u, 1.0f);

    attn_reset_caches();
    for (int t = 0; t < n; t++) {
        for (int i = 0; i < head_dim; i++) {
            attn_k_buf[t * stride + i] = key[i];
            attn_v_buf[t * stride + i] = (float)(t * 10 + i);
        }
    }

    attention_head(q, attn_k_buf, attn_v_buf, n, head_dim, stride, attn_scores, out);

    /* alle gewichten 1/4, dus out[i] = (0 + 10 + 20 + 30) / 4 + i = 15 + i */
    int failures = 0;
    for (int i = 0; i < head_dim; i++) {
        failures += expect_close("attention mean", i, out[i], 15.0 + i, 1e-5);
    }
    return report("attention identical keys gives mean of v", failures);
}

static int test_attention_dominant_key(void) {
    const int head_dim = 8, stride = 16, n = 5, hot = 2;
    float q[8], out[8];
    fill_constant(q, head_dim, 5.0f);

    attn_reset_caches();
    for (int t = 0; t < n; t++) {
        for (int i = 0; i < head_dim; i++) {
            attn_k_buf[t * stride + i] = (t == hot) ? 5.0f : -5.0f;
            attn_v_buf[t * stride + i] = (float)(100 * t + i);
        }
    }

    attention_head(q, attn_k_buf, attn_v_buf, n, head_dim, stride, attn_scores, out);

    /* score hot-token ≈ +70, de rest ≈ -70: gewicht van de rest is 0 in float */
    int failures = 0;
    for (int i = 0; i < head_dim; i++) {
        failures += expect_close("attention dominant", i, out[i], (double)(100 * hot + i), 1e-5);
    }
    return report("attention dominant key selects its v", failures);
}

static int test_attention_against_reference(void) {
    const int dims[] = {8, 128};
    const int lengths[] = {1, 7, 100};
    int failures = 0;

    for (int d = 0; d < 2; d++) {
        for (int l = 0; l < 3; l++) {
            for (int kv = 0; kv < 3; kv++) {
                int head_dim = dims[d];
                int n = lengths[l];
                int stride = 3 * head_dim;
                int offset = kv * head_dim;   /* simuleert KV-kop 0, 1 of 2 */
                unsigned seed = 40u + (unsigned)(d * 100 + l * 10 + kv);

                float q[ATTN_MAX_DIM], out[ATTN_MAX_DIM];
                double expected[ATTN_MAX_DIM];
                fill_random(q, head_dim, seed, 1.0f);

                attn_reset_caches();
                attn_fill_random(attn_k_buf, n, head_dim, stride, offset, seed + 1000u, 1.0f);
                attn_fill_random(attn_v_buf, n, head_dim, stride, offset, seed + 2000u, 1.0f);

                attention_head(q, attn_k_buf + offset, attn_v_buf + offset,
                               n, head_dim, stride, attn_scores, out);
                attention_reference(q, attn_k_buf + offset, attn_v_buf + offset,
                                    n, head_dim, stride, expected);

                for (int i = 0; i < head_dim; i++) {
                    failures += expect_close("attention", i, out[i], expected[i], 1e-4);
                }
            }
        }
    }
    return report("attention against double reference", failures);
}

static int test_attention_large_scores(void) {
    const int head_dim = 128, stride = 256, n = 16;
    float q[ATTN_MAX_DIM], out[ATTN_MAX_DIM];
    double expected[ATTN_MAX_DIM];
    fill_random(q, head_dim, 50u, 30.0f);

    attn_reset_caches();
    attn_fill_random(attn_k_buf, n, head_dim, stride, 0, 51u, 30.0f);
    attn_fill_random(attn_v_buf, n, head_dim, stride, 0, 52u, 1.0f);

    attention_head(q, attn_k_buf, attn_v_buf, n, head_dim, stride, attn_scores, out);
    attention_reference(q, attn_k_buf, attn_v_buf, n, head_dim, stride, expected);

    int failures = 0;
    for (int i = 0; i < head_dim; i++) {
        if (!isfinite(out[i])) {
            printf("  attention large[%d]: result is not finite\n", i);
            failures++;
            continue;
        }
        failures += expect_close("attention large", i, out[i], expected[i], 1e-3);
    }
    return report("attention large scores", failures);
}

#define Q8_MAX_BLOCKS (ATTN_MAX_N * (ATTN_MAX_DIM / QK))

static float     q8_k_float[ATTN_MAX_N * ATTN_MAX_DIM];
static float     q8_v_float[ATTN_MAX_N * ATTN_MAX_DIM];
static float     q8_k_deq[ATTN_MAX_N * ATTN_MAX_DIM];
static float     q8_v_deq[ATTN_MAX_N * ATTN_MAX_DIM];
static block_q8  q8_k_blocks[Q8_MAX_BLOCKS];
static block_q8  q8_v_blocks[Q8_MAX_BLOCKS];

static void q8_fill_garbage(block_q8 *blocks, int count) {
    for (int b = 0; b < count; b++) {
        blocks[b].d = 999.0f;
        for (int j = 0; j < QK; j++) {
            blocks[b].q[j] = 127;
        }
    }
}

static void quantize_rows(const float *x, block_q8 *out, int n, int head_dim) {
    int nb = head_dim / QK;
    for (int t = 0; t < n; t++) {
        quantize_row_q8(x + t * head_dim, out + t * nb, head_dim);
    }
}

static void dequantize_rows(const block_q8 *in, float *out, int n, int head_dim) {
    int nb = head_dim / QK;
    for (int t = 0; t < n; t++) {
        for (int b = 0; b < nb; b++) {
            const block_q8 *blk = &in[t * nb + b];
            for (int j = 0; j < QK; j++) {
                out[t * head_dim + b * QK + j] = blk->d * (float)blk->q[j];
            }
        }
    }
}

static int attn_q8_case(int head_dim, int n, unsigned seed, float data_scale,
                        double strict_tol, double *noise) {
    int nb = head_dim / QK;
    float q[ATTN_MAX_DIM], q_deq[ATTN_MAX_DIM];
    float out_q8[ATTN_MAX_DIM], out_deq[ATTN_MAX_DIM], out_orig[ATTN_MAX_DIM];
    block_q8 q_blocks[ATTN_MAX_DIM / QK];

    fill_random(q, head_dim, seed, data_scale);
    fill_random(q8_k_float, n * head_dim, seed + 1000u, data_scale);
    fill_random(q8_v_float, n * head_dim, seed + 2000u, 1.0f);

    q8_fill_garbage(q8_k_blocks, Q8_MAX_BLOCKS);
    q8_fill_garbage(q8_v_blocks, Q8_MAX_BLOCKS);
    quantize_rows(q8_k_float, q8_k_blocks, n, head_dim);
    quantize_rows(q8_v_float, q8_v_blocks, n, head_dim);

    attention_head_q8(q, q_blocks, q8_k_blocks, q8_v_blocks, n, head_dim, attn_scores, out_q8);
    dequantize_rows(q8_k_blocks, q8_k_deq, n, head_dim);
    dequantize_rows(q8_v_blocks, q8_v_deq, n, head_dim);
    dequantize_rows(q_blocks, q_deq, 1, head_dim);

    attention_head(q_deq, q8_k_deq, q8_v_deq, n, head_dim, head_dim, attn_scores, out_deq);
    attention_head(q, q8_k_float, q8_v_float, n, head_dim, head_dim, attn_scores, out_orig);

    (void)nb;
    int failures = 0;
    double max_out = 0.0, max_diff = 0.0;
    for (int i = 0; i < head_dim; i++) {
        if (!isfinite(out_q8[i])) {
            printf("  attention q8[%d]: result is not finite\n", i);
            failures++;
            continue;
        }
        failures += expect_close("attention q8", i, out_q8[i], out_deq[i], strict_tol);

        if (fabs((double)out_orig[i]) > max_out) max_out = fabs((double)out_orig[i]);
        double diff = fabs((double)out_q8[i] - (double)out_orig[i]);
        if (diff > max_diff) max_diff = diff;
    }
    if (noise != NULL) {
        *noise = max_diff / (max_out > 0.0 ? max_out : 1.0);
    }
    return failures;
}

static int test_attention_q8_single_token(void) {
    const int head_dim = 32;
    float q[32], k[32], v[32], v_deq[32], out[32];
    block_q8 q_blocks[1], k_blocks[1], v_blocks[1];

    fill_random(q, head_dim, 60u, 1.0f);
    fill_random(k, head_dim, 61u, 1.0f);
    fill_random(v, head_dim, 62u, 1.0f);
    quantize_rows(k, k_blocks, 1, head_dim);
    quantize_rows(v, v_blocks, 1, head_dim);

    attention_head_q8(q, q_blocks, k_blocks, v_blocks, 1, head_dim, attn_scores, out);
    dequantize_rows(v_blocks, v_deq, 1, head_dim);

    int failures = 0;
    for (int i = 0; i < head_dim; i++) {
        failures += expect_close("attention q8 n=1", i, out[i], v_deq[i], 1e-5);
    }
    return report("attention q8 single token returns dequantized v", failures);
}

static int test_attention_q8_matches_float_on_dequantized(void) {
    const int dims[] = {32, 128};
    const int lengths[] = {1, 7, 100};
    int failures = 0;

    for (int d = 0; d < 2; d++) {
        for (int l = 0; l < 3; l++) {
            unsigned seed = 70u + (unsigned)(d * 10 + l);
            failures += attn_q8_case(dims[d], lengths[l], seed, 1.0f, 1e-4, NULL);
        }
    }
    return report("attention q8 matches float on dequantized data", failures);
}

static int test_attention_q8_quantization_noise(void) {
    const int dims[] = {32, 128};
    const int lengths[] = {1, 7, 100};
    const double tolerance = 1e-2;
    double worst = 0.0;
    int failures = 0;

    for (int d = 0; d < 2; d++) {
        for (int l = 0; l < 3; l++) {
            unsigned seed = 80u + (unsigned)(d * 10 + l);
            double noise = 0.0;
            attn_q8_case(dims[d], lengths[l], seed, 1.0f, 1e-4, &noise);
            if (noise > worst) worst = noise;
        }
    }

    printf("  worst relative error vs float: %.4f%%\n", worst * 100.0);
    if (worst > tolerance) {
        printf("  noise %.6f exceeds tolerance %.6f\n", worst, tolerance);
        failures++;
    }
    return report("attention q8 quantization noise vs float", failures);
}

static int test_attention_q8_large_scores(void) {
    int failures = attn_q8_case(128, 16, 90u, 30.0f, 1e-3, NULL);
    return report("attention q8 large scores", failures);
}

#define TEST_FMT_Q8  TQ_FORMAT_Q8

#define L_HIDDEN   64
#define L_HD       32
#define L_NQ       4
#define L_NKV      2
#define L_QDIM     (L_NQ * L_HD)
#define L_KVDIM    (L_NKV * L_HD)
#define L_MAXTOK   16
#define L_TOKENS   8
#define L_EPS      1e-6f
#define L_THETA    1000000.0f
#define L_Q8_BLOCKS (L_NKV * L_MAXTOK * (L_HD / QK))

typedef struct {
    block_q8 blk[L_QDIM * (L_HIDDEN / QK)];
    float    deq[L_QDIM * L_HIDDEN];
} layer_mat;

typedef struct {
    layer_mat      wq, wk, wv, wo;
    float          qn[L_HD], kn[L_HD];
    tq_attn_weights w;
    tq_attn_config  cfg;
} layer_setup;

typedef struct {
    float k[L_NKV][L_MAXTOK][L_HD];
    float v[L_NKV][L_MAXTOK][L_HD];
} layer_ref_cache;

static layer_setup     LS;
static layer_ref_cache LREF;
static float layer_tmp[L_QDIM * L_HIDDEN];
static float layer_scores[L_MAXTOK];
static float layer_k_f32[L_NKV * L_MAXTOK * L_HD];
static float layer_v_f32[L_NKV * L_MAXTOK * L_HD];
static block_q8 layer_k_q8[L_Q8_BLOCKS];
static block_q8 layer_v_q8[L_Q8_BLOCKS];
static float layer_out_f32[L_TOKENS][L_HIDDEN];
static float layer_out_q8[L_TOKENS][L_HIDDEN];

static int expect_int(const char *label, int actual, int expected) {
    if (actual == expected) return 0;
    printf("  %s: expected %d, got %d\n", label, expected, actual);
    return 1;
}

static void layer_make_matrix(layer_mat *m, int rows, int cols, unsigned seed) {
    fill_random(layer_tmp, rows * cols, seed, 0.5f);
    quantize_rows(layer_tmp, m->blk, rows, cols);
    dequantize_rows(m->blk, m->deq, rows, cols);
}

static void layer_set_shape(layer_setup *s, int nq, int nkv, int hd) {
    s->cfg.n_q_heads  = nq;
    s->cfg.n_kv_heads = nkv;
    s->cfg.head_dim   = hd;
    s->w.wq.rows = nq * hd;
    s->w.wk.rows = nkv * hd;
    s->w.wv.rows = nkv * hd;
    s->w.wo.cols = nq * hd;
}

#define LAYER_BIND(field, mat, r, c)                          \
    do {                                                      \
        s->w.field.data   = (mat).blk;                        \
        s->w.field.rows   = (r);                              \
        s->w.field.cols   = (c);                              \
        s->w.field.format = TEST_FMT_Q8;                      \
    } while (0)

static void layer_init(layer_setup *s, unsigned seed) {
    memset(&s->cfg, 0, sizeof s->cfg);
    memset(&s->w, 0, sizeof s->w);

    layer_make_matrix(&s->wq, L_QDIM,   L_HIDDEN, seed + 1u);
    layer_make_matrix(&s->wk, L_KVDIM,  L_HIDDEN, seed + 2u);
    layer_make_matrix(&s->wv, L_KVDIM,  L_HIDDEN, seed + 3u);
    layer_make_matrix(&s->wo, L_HIDDEN, L_QDIM,   seed + 4u);

    /* norm-gewichten in [0.5, 2.0): grotere scores, dus gevoeliger voor fouten */
    fill_random(s->qn, L_HD, seed + 5u, 1.0f);
    fill_random(s->kn, L_HD, seed + 6u, 1.0f);
    for (int i = 0; i < L_HD; i++) {
        s->qn[i] = 1.25f + 0.75f * s->qn[i];
        s->kn[i] = 1.25f + 0.75f * s->kn[i];
    }

    s->cfg.hidden_size = L_HIDDEN;
    s->cfg.rms_eps     = L_EPS;
    s->cfg.rope_theta  = L_THETA;

    LAYER_BIND(wq, s->wq, L_QDIM,   L_HIDDEN);
    LAYER_BIND(wk, s->wk, L_KVDIM,  L_HIDDEN);
    LAYER_BIND(wv, s->wv, L_KVDIM,  L_HIDDEN);
    LAYER_BIND(wo, s->wo, L_HIDDEN, L_QDIM);
    s->w.q_norm = s->qn;
    s->w.k_norm = s->kn;
    layer_set_shape(s, L_NQ, L_NKV, L_HD);
}

static void layer_cache_f32(tq_kv_cache *c, float fill) {
    memset(c, 0, sizeof *c);
    c->type       = TQ_KV_F32;
    c->max_tokens = L_MAXTOK;
    c->k_f32      = layer_k_f32;
    c->v_f32      = layer_v_f32;
    fill_constant(layer_k_f32, L_NKV * L_MAXTOK * L_HD, fill);
    fill_constant(layer_v_f32, L_NKV * L_MAXTOK * L_HD, fill);
}

static void layer_cache_q8(tq_kv_cache *c) {
    memset(c, 0, sizeof *c);
    c->type       = TQ_KV_Q8;
    c->max_tokens = L_MAXTOK;
    c->k_q8       = layer_k_q8;
    c->v_q8       = layer_v_q8;
    q8_fill_garbage(layer_k_q8, L_Q8_BLOCKS);
    q8_fill_garbage(layer_v_q8, L_Q8_BLOCKS);
}

static void layer_input(int t, float *xn) {
    fill_random(xn, L_HIDDEN, 500u + (unsigned)t, 1.0f);
}

static int layer_run(layer_setup *s, tq_kv_cache *cache, float out[L_TOKENS][L_HIDDEN]) {
    int failures = 0;
    for (int t = 0; t < L_TOKENS; t++) {
        float xn[L_HIDDEN];
        layer_input(t, xn);
        int rc = attention_layer(&s->w, &s->cfg, cache, t, xn, layer_scores, out[t]);
        if (rc != 0) {
            printf("  attention_layer returned %d at pos %d\n", rc, t);
            failures++;
        }
    }
    return failures;
}


static void layer_matvec(const float *w, int rows, int cols, const float *x, double *out) {
    for (int r = 0; r < rows; r++) {
        double sum = 0.0;
        for (int c = 0; c < cols; c++) {
            sum += (double)w[r * cols + c] * (double)x[c];
        }
        out[r] = sum;
    }
}

static void layer_head_prepare(const double *in, const float *norm_w, int pos, float *out) {
    float head[L_HD], normed[L_HD];
    double nd[L_HD], rd[L_HD];
    for (int i = 0; i < L_HD; i++) head[i] = (float)in[i];
    rmsnorm_reference(head, norm_w, nd, L_HD, (double)L_EPS);
    for (int i = 0; i < L_HD; i++) normed[i] = (float)nd[i];
    rope_reference(normed, rd, L_HD, pos, (double)L_THETA);
    for (int i = 0; i < L_HD; i++) out[i] = (float)rd[i];
}


static void layer_reference(const layer_setup *s, layer_ref_cache *rc, int pos,
                            const float *xn, double *out) {
    block_q8 xb[L_HIDDEN / QK];
    float xq[L_HIDDEN];
    quantize_row_q8(xn, xb, L_HIDDEN);
    dequantize_rows(xb, xq, 1, L_HIDDEN);

    double qd[L_QDIM], kd[L_KVDIM], vd[L_KVDIM];
    layer_matvec(s->wq.deq, L_QDIM,  L_HIDDEN, xq, qd);
    layer_matvec(s->wk.deq, L_KVDIM, L_HIDDEN, xq, kd);
    layer_matvec(s->wv.deq, L_KVDIM, L_HIDDEN, xq, vd);

    float q[L_QDIM];
    for (int h = 0; h < L_NQ; h++) {
        layer_head_prepare(qd + h * L_HD, s->qn, pos, q + h * L_HD);
    }
    for (int h = 0; h < L_NKV; h++) {
        layer_head_prepare(kd + h * L_HD, s->kn, pos, rc->k[h][pos]);
        for (int i = 0; i < L_HD; i++) rc->v[h][pos][i] = (float)vd[h * L_HD + i];
    }

    float attn[L_QDIM];
    const int group = L_NQ / L_NKV;
    for (int h = 0; h < L_NQ; h++) {
        int kvh = h / group;
        double ad[L_HD];
        attention_reference(q + h * L_HD, rc->k[kvh][0], rc->v[kvh][0],
                            pos + 1, L_HD, L_HD, ad);
        for (int i = 0; i < L_HD; i++) attn[h * L_HD + i] = (float)ad[i];
    }

    block_q8 ab[L_QDIM / QK];
    float aq[L_QDIM];
    quantize_row_q8(attn, ab, L_QDIM);
    dequantize_rows(ab, aq, 1, L_QDIM);
    layer_matvec(s->wo.deq, L_HIDDEN, L_QDIM, aq, out);
}


static int test_attn_validate(void) {
    layer_setup *s = &LS;
    layer_init(s, 300u);
    int failures = 0;

    failures += expect_int("valid config", tq_attn_validate(&s->w, &s->cfg), 0);

    s->cfg.hidden_size = L_HIDDEN + 1;
    failures += expect_int("hidden % QK", tq_attn_validate(&s->w, &s->cfg), -1);
    s->cfg.hidden_size = L_HIDDEN;

    s->w.wq.cols = L_HIDDEN + QK;
    failures += expect_int("wq.cols", tq_attn_validate(&s->w, &s->cfg), -2);
    s->w.wq.cols = L_HIDDEN;

    s->w.wo.rows = L_HIDDEN + QK;
    failures += expect_int("wo.rows", tq_attn_validate(&s->w, &s->cfg), -2);
    s->w.wo.rows = L_HIDDEN;

    s->w.wq.rows = L_QDIM - QK;
    failures += expect_int("wq.rows", tq_attn_validate(&s->w, &s->cfg), -3);
    s->w.wq.rows = L_QDIM;

    s->w.wk.rows = L_KVDIM + QK;
    failures += expect_int("wk.rows", tq_attn_validate(&s->w, &s->cfg), -4);
    s->w.wk.rows = L_KVDIM;

    layer_set_shape(s, 3, 2, L_HD);
    failures += expect_int("n_q % n_kv", tq_attn_validate(&s->w, &s->cfg), -5);

    layer_set_shape(s, L_NQ, L_NKV, 16);
    failures += expect_int("head_dim % QK", tq_attn_validate(&s->w, &s->cfg), -6);

    layer_set_shape(s, L_NQ, L_NKV, L_HD);
    s->w.wo.cols = L_QDIM + QK;
    failures += expect_int("wo.cols", tq_attn_validate(&s->w, &s->cfg), -7);
    s->w.wo.cols = L_QDIM;

    layer_set_shape(s, L_NQ, 0, L_HD);
    failures += expect_int("n_kv_heads == 0", tq_attn_validate(&s->w, &s->cfg), -8);
    layer_set_shape(s, 0, L_NKV, L_HD);
    failures += expect_int("n_q_heads == 0", tq_attn_validate(&s->w, &s->cfg), -8);
    layer_set_shape(s, L_NQ, L_NKV, L_HD);

    s->w.q_norm = NULL;
    failures += expect_int("q_norm NULL", tq_attn_validate(&s->w, &s->cfg), -9);
    s->w.q_norm = s->qn;
    s->w.k_norm = NULL;
    failures += expect_int("k_norm NULL", tq_attn_validate(&s->w, &s->cfg), -9);
    s->w.k_norm = s->kn;

    failures += expect_int("valid again", tq_attn_validate(&s->w, &s->cfg), 0);
    return report("attention validate", failures);
}

static int test_attn_validate_cache(void) {
    tq_kv_cache c;
    int failures = 0;

    layer_cache_f32(&c, 0.0f);
    failures += expect_int("f32 ok", tq_attn_validate_cache(&c), 0);
    c.max_tokens = 0;
    failures += expect_int("max_tokens 0", tq_attn_validate_cache(&c), -1);
    c.max_tokens = L_MAXTOK;
    c.k_f32 = NULL;
    failures += expect_int("k_f32 NULL", tq_attn_validate_cache(&c), -2);
    c.k_f32 = layer_k_f32;
    c.v_f32 = NULL;
    failures += expect_int("v_f32 NULL", tq_attn_validate_cache(&c), -2);
    c.v_f32 = layer_v_f32;
    c.type = 99;
    failures += expect_int("unknown type", tq_attn_validate_cache(&c), -3);

    layer_cache_q8(&c);
    failures += expect_int("q8 ok", tq_attn_validate_cache(&c), 0);
    c.k_q8 = NULL;
    failures += expect_int("k_q8 NULL", tq_attn_validate_cache(&c), -2);
    c.k_q8 = layer_k_q8;
    c.v_q8 = NULL;
    failures += expect_int("v_q8 NULL", tq_attn_validate_cache(&c), -2);

    return report("attention validate cache", failures);
}

static int test_attention_layer_pos_out_of_range(void) {
    layer_setup *s = &LS;
    layer_init(s, 400u);
    tq_kv_cache cache;
    float xn[L_HIDDEN], out[L_HIDDEN];
    layer_input(0, xn);
    int failures = 0;

    layer_cache_f32(&cache, ATTN_GARBAGE);
    const int bad_pos[] = {-1, L_MAXTOK, L_MAXTOK + 5};
    for (int i = 0; i < 3; i++) {
        fill_constant(out, L_HIDDEN, 7.0f);
        int rc = attention_layer(&s->w, &s->cfg, &cache, bad_pos[i], xn, layer_scores, out);
        failures += expect_int("bad pos returns -1", rc, -1);

        for (int j = 0; j < L_HIDDEN; j++) {
            if (out[j] != 7.0f) { printf("  out written for bad pos %d\n", bad_pos[i]); failures++; break; }
        }
        for (int j = 0; j < L_NKV * L_MAXTOK * L_HD; j++) {
            if (layer_k_f32[j] != ATTN_GARBAGE || layer_v_f32[j] != ATTN_GARBAGE) {
                printf("  cache modified for bad pos %d\n", bad_pos[i]);
                failures++;
                break;
            }
        }
    }

    layer_cache_f32(&cache, 0.0f);
    int rc = attention_layer(&s->w, &s->cfg, &cache, L_MAXTOK - 1, xn, layer_scores, out);
    failures += expect_int("last valid pos", rc, 0);
    for (int j = 0; j < L_HIDDEN; j++) {
        if (!isfinite(out[j])) { printf("  out[%d] not finite\n", j); failures++; break; }
    }
    return report("attention layer pos out of range", failures);
}

static int test_attention_layer_against_reference(void) {
    layer_setup *s = &LS;
    layer_init(s, 100u);
    tq_kv_cache cache;
    layer_cache_f32(&cache, ATTN_GARBAGE);   /* ongebruikte slots blijven 999 */

    int failures = layer_run(s, &cache, layer_out_f32);

    memset(&LREF, 0, sizeof LREF);
    for (int t = 0; t < L_TOKENS; t++) {
        float xn[L_HIDDEN];
        double expected[L_HIDDEN];
        layer_input(t, xn);
        layer_reference(s, &LREF, t, xn, expected);

        for (int i = 0; i < L_HIDDEN; i++) {
            failures += expect_close("layer out", t * L_HIDDEN + i, layer_out_f32[t][i], expected[i], 2e-2);
        }
        for (int h = 0; h < L_NKV; h++) {
            for (int i = 0; i < L_HD; i++) {
                size_t idx = ((size_t)h * L_MAXTOK + t) * L_HD + i;
                failures += expect_close("cache k", (int)idx, layer_k_f32[idx], LREF.k[h][t][i], 1e-3);
                failures += expect_close("cache v", (int)idx, layer_v_f32[idx], LREF.v[h][t][i], 1e-3);
            }
        }
    }

    for (int h = 0; h < L_NKV; h++) {
        for (int t = L_TOKENS; t < L_MAXTOK; t++) {
            for (int i = 0; i < L_HD; i++) {
                size_t idx = ((size_t)h * L_MAXTOK + t) * L_HD + i;
                if (layer_k_f32[idx] != ATTN_GARBAGE || layer_v_f32[idx] != ATTN_GARBAGE) {
                    printf("  slot kvh=%d pos=%d overwritten\n", h, t);
                    failures++;
                    break;
                }
            }
        }
    }
    return report("attention layer against double reference", failures);
}

static int test_attention_layer_q8_cache_vs_f32(void) {
    layer_setup *s = &LS;
    layer_init(s, 100u);
    tq_kv_cache cf, cq;
    layer_cache_f32(&cf, 0.0f);
    layer_cache_q8(&cq);

    int failures = layer_run(s, &cf, layer_out_f32);
    failures += layer_run(s, &cq, layer_out_q8);

    double max_out = 0.0, max_diff = 0.0;
    for (int t = 0; t < L_TOKENS; t++) {
        for (int i = 0; i < L_HIDDEN; i++) {
            if (!isfinite(layer_out_q8[t][i])) {
                printf("  q8 out[%d][%d] not finite\n", t, i);
                failures++;
                continue;
            }
            double ref = fabs((double)layer_out_f32[t][i]);
            double diff = fabs((double)layer_out_q8[t][i] - (double)layer_out_f32[t][i]);
            if (ref > max_out) max_out = ref;
            if (diff > max_diff) max_diff = diff;
        }
    }

    for (int h = 0; h < L_NKV; h++) {
        for (int t = L_TOKENS; t < L_MAXTOK; t++) {
            size_t b = ((size_t)h * L_MAXTOK + t) * (L_HD / QK);
            if (layer_k_q8[b].d != 999.0f || layer_v_q8[b].d != 999.0f) {
                printf("  q8 slot kvh=%d pos=%d overwritten\n", h, t);
                failures++;
            }
        }
    }

    double noise = max_diff / (max_out > 0.0 ? max_out : 1.0);
    printf("  worst relative error q8 cache vs f32 cache: %.4f%%\n", noise * 100.0);
    if (noise > 3e-2) {
        printf("  noise %.6f exceeds tolerance 0.03\n", noise);
        failures++;
    }
    return report("attention layer q8 cache vs f32 cache", failures);
}

static uint32_t rng_state = 12345u;

static float frand(float lo, float hi) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return lo + (hi - lo) * ((float)(rng_state & 0xFFFFFF) / (float)0x1000000);
}

static tq_matrix make_matrix(const float *src, int rows, int cols, tq_format fmt) {
    char bit = (fmt == TQ_FORMAT_Q8) ? '8' : '4';
    size_t block = (fmt == TQ_FORMAT_Q8) ? sizeof(block_q8) : sizeof(block_q4);
    size_t bytes = (size_t)rows * (size_t)(cols / QK) * block;
    void *buf = malloc(bytes);
    if (pack_matrix(src, rows, cols, buf, bit) != 0) { free(buf); exit(1); }
    tq_matrix m = { buf, fmt, rows, cols };
    return m;
}

static void free_matrix(tq_matrix *m) {
    free((void *)m->data);
    m->data = NULL;
}


static int test_embed(tq_format fmt, float qmax, const char *name) {
    enum { ROWS = 4, COLS = 64 };
    float src[ROWS * COLS];
    for (int i = 0; i < ROWS * COLS; i++) src[i] = frand(-1.0f, 1.0f);

    tq_matrix m = make_matrix(src, ROWS, COLS, fmt);
    int fails = 0;
    float worst = 0.0f;

    for (int r = 0; r < ROWS; r++) {
        float buf[COLS + 4];
        for (int i = COLS; i < COLS + 4; i++) buf[i] = GUARD;

        if (embed_lookup(&m, r, buf) != 0) { fails++; continue; }

        float rowmax = 0.0f;
        for (int j = 0; j < COLS; j++) {
            float a = fabsf(src[r * COLS + j]);
            if (a > rowmax) rowmax = a;
        }
        float tol = 0.5f * rowmax / qmax * 1.01f + 1e-6f;

        for (int j = 0; j < COLS; j++) {
            float err = fabsf(buf[j] - src[r * COLS + j]);
            if (err > worst) worst = err;
            if (err > tol) fails++;                  // verkeerde waarde (bv. nibble-volgorde)
        }
        for (int i = COLS; i < COLS + 4; i++)        // buiten de rij geschreven?
            if (buf[i] != GUARD) fails++;
    }

    float tmp[COLS];
    if (embed_lookup(&m, -1, tmp) == 0) fails++;     // ongeldig token
    if (embed_lookup(&m, ROWS, tmp) == 0) fails++;   // net buiten het bereik

    free_matrix(&m);
    printf("embed_lookup %s: %s (grootste fout %.5f)\n", name, fails ? "FAIL" : "ok", worst);
    return fails;
}

// ---------- mlp_layer ----------

static int test_mlp(tq_format fmt_gu, tq_format fmt_down, double tol_rel, const char *name) {
    enum { H = 64, I = 96 };                         // beide deelbaar door 32
    static float wg[I * H], wu[I * H], wd[H * I];
    float xn[H];
    for (int i = 0; i < I * H; i++) wg[i] = frand(-0.2f, 0.2f);
    for (int i = 0; i < I * H; i++) wu[i] = frand(-0.2f, 0.2f);
    for (int i = 0; i < H * I; i++) wd[i] = frand(-0.2f, 0.2f);
    for (int i = 0; i < H; i++) xn[i] = frand(-1.0f, 1.0f);

    // referentie in double, op de originele floats
    double g[I], u[I], h[I], ref[H];
    for (int i = 0; i < I; i++) {
        g[i] = 0.0; u[i] = 0.0;
        for (int j = 0; j < H; j++) {
            g[i] += (double)wg[i * H + j] * xn[j];
            u[i] += (double)wu[i * H + j] * xn[j];
        }
        h[i] = (g[i] / (1.0 + exp(-g[i]))) * u[i];   // silu(gate) * up
    }
    for (int i = 0; i < H; i++) {
        ref[i] = 0.0;
        for (int j = 0; j < I; j++) ref[i] += (double)wd[i * I + j] * h[j];
    }

    tq_mlp_weights w;
    w.wgate = make_matrix(wg, I, H, fmt_gu);
    w.wup   = make_matrix(wu, I, H, fmt_gu);
    w.wdown = make_matrix(wd, H, I, fmt_down);

    float gate[I], up[I], out[H + 4];
    block_q8 xq[H / QK], hq[I / QK];
    for (int i = H; i < H + 4; i++) out[i] = GUARD;

    int fails = 0;
    int rc = mlp_layer(&w, xn, xq, gate, up, hq, out);
    if (rc != 0) fails++;

    double refmax = 0.0, worst = 0.0;
    for (int i = 0; i < H; i++) if (fabs(ref[i]) > refmax) refmax = fabs(ref[i]);
    for (int i = 0; i < H; i++) {
        double err = fabs((double)out[i] - ref[i]);
        if (err > worst) worst = err;
    }
    if (worst / refmax > tol_rel) fails++;
    for (int i = H; i < H + 4; i++) if (out[i] != GUARD) fails++;

    printf("mlp_layer %s: %s (fout %.2f%% van het grootste uitvoerelement)\n",
           name, fails ? "FAIL" : "ok", 100.0 * worst / refmax);

    free_matrix(&w.wgate); free_matrix(&w.wup); free_matrix(&w.wdown);
    return fails;
}

static int test_mlp_wrong_shape(void) {
    enum { H = 64, I = 96 };
    static float wg[I * H], wu[I * H], wd[H * I];
    float xn[H], gate[I], up[I], out[H];
    block_q8 xq[H / QK], hq[I / QK];
    for (int i = 0; i < I * H; i++) { wg[i] = frand(-0.2f, 0.2f); wu[i] = frand(-0.2f, 0.2f); }
    for (int i = 0; i < H * I; i++) wd[i] = frand(-0.2f, 0.2f);
    for (int i = 0; i < H; i++) xn[i] = frand(-1.0f, 1.0f);

    tq_mlp_weights w;
    w.wgate = make_matrix(wg, I, H, TQ_FORMAT_Q8);
    w.wup   = make_matrix(wu, I, H, TQ_FORMAT_Q8);
    w.wdown = make_matrix(wd, I, H, TQ_FORMAT_Q8);

    int fails = (mlp_layer(&w, xn, xq, gate, up, hq, out) == 0) ? 1 : 0;
    printf("mlp_layer verkeerde vorm: %s\n", fails ? "FAIL" : "ok");

    free_matrix(&w.wgate); free_matrix(&w.wup); free_matrix(&w.wdown);
    return fails;
}

static int test_lm_head(tq_format fmt, double tol_rel, const char *name) {
    enum { ROWS = 100, COLS = 64 };
    static float w[ROWS * COLS];
    float xn[COLS];
    float logits[ROWS + 4];
    block_q8 xq[COLS / QK];
    double ref[ROWS];

    for (int i = 0; i < ROWS * COLS; i++) w[i] = frand(-0.5f, 0.5f);
    for (int i = 0; i < COLS; i++) xn[i] = frand(-1.0f, 1.0f);
    for (int i = ROWS; i < ROWS + 4; i++) logits[i] = GUARD;

    // referentie in double op de originele floats
    for (int i = 0; i < ROWS; i++) {
        ref[i] = 0.0;
        for (int j = 0; j < COLS; j++) ref[i] += (double)w[i * COLS + j] * xn[j];
    }

    tq_matrix embed = make_matrix(w, ROWS, COLS, fmt);
    int fails = 0;
    if (lm_head(&embed, xn, xq, logits) != 0) fails++;

    double refmax = 0.0, worst = 0.0;
    for (int i = 0; i < ROWS; i++) if (fabs(ref[i]) > refmax) refmax = fabs(ref[i]);
    for (int i = 0; i < ROWS; i++) {
        double err = fabs((double)logits[i] - ref[i]);
        if (err > worst) worst = err;
    }
    if (worst / refmax > tol_rel) fails++;
    for (int i = ROWS; i < ROWS + 4; i++) if (logits[i] != GUARD) fails++;

    printf("lm_head %s: %s (fout %.2f%% van de grootste logit)\n",
           name, fails ? "FAIL" : "ok", 100.0 * worst / refmax);

    free_matrix(&embed);
    return fails;
}

static int test_sample_greedy(void) {
    int fails = 0;

    float mid[]   = {0.1f, 0.3f, 0.9f, 0.2f, 0.4f};
    float first[] = {0.9f, 0.1f, 0.2f};
    float last[]  = {0.1f, 0.2f, 0.9f};
    float neg[]   = {-3.0f, -0.5f, -2.0f, -0.9f};
    float frac[]  = {0.3f, 0.9f, 0.5f};          // vangt het afkappen van decimalen
    float tie[]   = {0.5f, 0.9f, 0.9f, 0.1f};
    float one[]   = {-1.0f};

    if (sample_greedy(mid, 5)   != 2) fails++;   // maximum in het midden
    if (sample_greedy(first, 3) != 0) fails++;   // maximum op index 0
    if (sample_greedy(last, 3)  != 2) fails++;   // maximum op de laatste plek
    if (sample_greedy(neg, 4)   != 1) fails++;   // alle waarden negatief
    if (sample_greedy(frac, 3)  != 1) fails++;   // 0.9 tegenover 0.5
    if (sample_greedy(tie, 4)   != 1) fails++;   // gelijk: eerste index wint
    if (sample_greedy(one, 1)   != 0) fails++;   // n = 1
    if (sample_greedy(one, 0)   != -1) fails++;  // n = 0 geeft een fout
    if (sample_greedy(one, -1)  != -1) fails++;  // negatieve n ook

    printf("sample_greedy: %s\n", fails ? "FAIL" : "ok");
    return fails;
}

int main(void) {
    int failures = 0;

    failures += test_silu_known_values();
    failures += test_silu_mul_against_reference();
    failures += test_silu_extreme_inputs();
    failures += test_rmsnorm_known_values();
    failures += test_rmsnorm_output_has_unit_rms();
    failures += test_rmsnorm_weight_scales_output();
    failures += test_rmsnorm_in_place();
    failures += test_rmsnorm_against_reference();
    failures += test_softmax_known_values();
    failures += test_softmax_sums_to_one();
    failures += test_softmax_large_inputs();
    failures += test_softmax_constant_input();
    failures += test_softmax_against_reference();
    failures += test_rope_known_values();
    failures += test_rope_position_zero_is_identity();
    failures += test_rope_preserves_norm();
    failures += test_rope_against_reference();
    failures += test_rope_relative_position();
    failures += test_attention_single_token();
    failures += test_attention_identical_keys();
    failures += test_attention_dominant_key();
    failures += test_attention_against_reference();
    failures += test_attention_large_scores();
    failures += test_attention_q8_single_token();
    failures += test_attention_q8_matches_float_on_dequantized();
    failures += test_attention_q8_quantization_noise();
    failures += test_attention_q8_large_scores();
    failures += test_attn_validate();
    failures += test_attn_validate_cache();
    failures += test_attention_layer_pos_out_of_range();
    failures += test_attention_layer_against_reference();
    failures += test_attention_layer_q8_cache_vs_f32();
    failures += test_embed(TQ_FORMAT_Q8, 127.0f, "q8");
    failures += test_embed(TQ_FORMAT_Q4, 7.0f, "q4");
    failures += test_mlp(TQ_FORMAT_Q8, TQ_FORMAT_Q8, 0.03, "q8");
    failures += test_mlp(TQ_FORMAT_Q4, TQ_FORMAT_Q4, 0.15, "q4");
    failures += test_mlp(TQ_FORMAT_Q4, TQ_FORMAT_Q8, 0.15, "gate/up q4, down q8");
    failures += test_mlp_wrong_shape();
    failures += test_lm_head(TQ_FORMAT_Q8, 0.03, "q8");
    failures += test_lm_head(TQ_FORMAT_Q4, 0.15, "q4");
    failures += test_sample_greedy();
    if (failures == 0) {
        printf("\nAll tests passed.\n");
        return 0;
    }

    printf("\n%d check(s) failed.\n", failures);
    return 1;
}