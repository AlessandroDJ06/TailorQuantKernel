#include <math.h>
#include <stdio.h>
#include "model/model_ops.h"
#include "kernel/kernel.h"

#define MAX_N 1024

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

    /* met één token is het softmax-gewicht 1, dus out = gedequantiseerde V */
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

    if (failures == 0) {
        printf("\nAll tests passed.\n");
        return 0;
    }

    printf("\n%d check(s) failed.\n", failures);
    return 1;
}