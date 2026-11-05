#include <math.h>
#include <stdio.h>
#include "model/model_ops.h"

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

    if (failures == 0) {
        printf("\nAll tests passed.\n");
        return 0;
    }

    printf("\n%d check(s) failed.\n", failures);
    return 1;
}