#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "packer/packer.h"
#include "kernel/kernel.h"
#include "kernel/kernel_internal.h"   /* only for the scalar reference functions */

/* Tolerances */
#define TOL_X_ROUNDTRIP   0.006   /* quantize_row_q8 roundtrip, ~0.4% expected      */
#define TOL_Q8_VS_FLOAT   0.015   /* ~0.5% expected                                  */
#define TOL_Q4_VS_FLOAT   0.100   /* ~6.5% expected                                  */
#define TOL_FAST_VS_SCAL  1e-5    /* only float summation order may differ           */

static int g_failures = 0;
static int g_skipped  = 0;   /* checks that could not run because no SIMD kernel was dispatched */

static float frand(void) { return ((float)rand() / RAND_MAX - 0.5f) * 2.0f; }

static double rel_err(const float* a, const float* b, int n) {
    double num = 0.0, den = 0.0;
    for (int i = 0; i < n; i++) {
        double d = (double)a[i] - (double)b[i];
        num += d * d;
        den += (double)b[i] * (double)b[i];
    }
    return den > 0.0 ? sqrt(num / den) : sqrt(num);
}

static void check(const char* name, double value, double tol) {
    int ok = value <= tol;
    if (!ok) g_failures++;
    printf("  [%s] %-46s %.6f%%  (limit %.6f%%)\n", ok ? "PASS" : "FAIL", name, 100.0 * value, 100.0 * tol);
}

static const char* fmt_name(tq_format f) { return f == TQ_FORMAT_Q8 ? "q8" : "q4"; }
static char        fmt_id(tq_format f)   { return f == TQ_FORMAT_Q8 ? '8' : '4'; }

typedef enum { X_RANDOM, X_CONSTANT, X_ZERO } XKind;

static const char* x_name(XKind k) {
    return k == X_RANDOM ? "random x" : k == X_CONSTANT ? "constant x" : "zero x";
}

static void fill_x(float* x, int cols, XKind kind) {
    for (int i = 0; i < cols; i++)
        x[i] = kind == X_RANDOM ? frand() : kind == X_CONSTANT ? 1.0f : 0.0f;
}

/* One matmul case: dispatched kernel vs scalar kernel vs float reference, plus thread-invariance. */
static void run_case(int rows, int cols, tq_format fmt, XKind xkind) {
    const int nb = cols / QK;
    printf("%s %dx%d, %s\n", fmt_name(fmt), rows, cols, x_name(xkind));

    float*    W        = malloc((size_t)rows * cols * sizeof(float));
    float*    x        = malloc((size_t)cols * sizeof(float));
    float*    y_ref    = malloc((size_t)rows * sizeof(float));
    float*    y_fast   = malloc((size_t)rows * sizeof(float));
    float*    y_fast4  = malloc((size_t)rows * sizeof(float));
    float*    y_scalar = malloc((size_t)rows * sizeof(float));
    block_q8* xq       = malloc((size_t)nb * sizeof(block_q8));
    void*     packed   = malloc(packed_matrix_bytes(rows, cols, fmt_id(fmt)));

    for (size_t i = 0; i < (size_t)rows * cols; i++) W[i] = frand();
    fill_x(x, cols, xkind);
    quantize_row_q8(x, xq, cols);
    pack_matrix(W, rows, cols, packed, fmt_id(fmt));

    for (int r = 0; r < rows; r++) {
        double s = 0.0;
        for (int c = 0; c < cols; c++) s += (double)W[(size_t)r * cols + c] * x[c];
        y_ref[r] = (float)s;
    }

    MatmulFn fast   = get_matmul(fmt);
    /* no ?: on function names: some compilers give that a function type instead of a pointer */
    MatmulFn scalar = &matmul_q4_scalar;
    if (fmt == TQ_FORMAT_Q8) scalar = &matmul_q8_scalar;
    const int simd = fast != scalar;   /* declared before the goto below */
    if (!fast) {
        printf("  [FAIL] get_matmul returned NULL\n");
        g_failures++;
        goto done;
    }

    omp_set_num_threads(1);
    fast(packed, xq, y_fast, rows, cols);
    scalar(packed, xq, y_scalar, rows, cols);

    /* get_matmul() falls back to the scalar kernel when the CPU has no AVX2+FMA. In that case
       "fast vs scalar" would compare the scalar kernel with itself and prove nothing, so it is
       reported as SKIPPED instead of PASS. */

    if (xkind != X_ZERO) {
        /* slack: tiny matrices give noisy statistics; the fast-vs-scalar check is the strict one */
        const double slack = (size_t)rows * cols < 8192 ? 2.5 : 1.0;
        const double tol_f = slack * (fmt == TQ_FORMAT_Q8 ? TOL_Q8_VS_FLOAT : TOL_Q4_VS_FLOAT);

        check("scalar kernel vs float reference", rel_err(y_scalar, y_ref, rows), tol_f);
        if (simd) {
            check("SIMD kernel vs float reference", rel_err(y_fast, y_ref, rows), tol_f);
            check("SIMD kernel vs scalar kernel", rel_err(y_fast, y_scalar, rows), TOL_FAST_VS_SCAL);
        } else {
            g_skipped++;
            printf("  [SKIP] SIMD kernel vs scalar: dispatcher returned the scalar kernel (no AVX2+FMA)\n");
        }
    } else {
        /* all-zero input: output must be exactly zero (catches a broken offset correction) */
        double max_abs = 0.0, max_abs_s = 0.0;
        for (int r = 0; r < rows; r++) {
            if (fabs(y_fast[r])   > max_abs)   max_abs   = fabs(y_fast[r]);
            if (fabs(y_scalar[r]) > max_abs_s) max_abs_s = fabs(y_scalar[r]);
        }
        check("scalar: zero input gives zero output (max |y|)", max_abs_s, 0.0);
        if (simd) check("SIMD: zero input gives zero output (max |y|)", max_abs, 0.0);
    }

    omp_set_num_threads(4);
    fast(packed, xq, y_fast4, rows, cols);
    omp_set_num_threads(1);
    check("dispatched kernel, 1 vs 4 threads (identical)", rel_err(y_fast4, y_fast, rows), 0.0);

done:
    free(W); free(x); free(y_ref); free(y_fast); free(y_fast4); free(y_scalar); free(xq); free(packed);
    printf("\n");
}

/* quantize_row_q8 roundtrip and pack/unpack roundtrip */
static void run_roundtrips(void) {
    const int cols = 512, nb = cols / QK;
    float*    x    = malloc(cols * sizeof(float));
    float*    rt   = malloc(cols * sizeof(float));
    block_q8* xq   = malloc(nb * sizeof(block_q8));
    for (int i = 0; i < cols; i++) x[i] = frand();

    printf("roundtrips\n");
    quantize_row_q8(x, xq, cols);
    for (int b = 0; b < nb; b++)
        for (int j = 0; j < QK; j++) rt[b * QK + j] = xq[b].q[j] * xq[b].d;
    check("quantize_row_q8 roundtrip", rel_err(rt, x, cols), TOL_X_ROUNDTRIP);

    const int rows = 16;
    float* W   = malloc((size_t)rows * cols * sizeof(float));
    float* out = malloc((size_t)rows * cols * sizeof(float));
    for (size_t i = 0; i < (size_t)rows * cols; i++) W[i] = frand();

    const char  ids[]  = {'8', '4'};
    const char* nm[]   = {"pack/unpack q8 roundtrip", "pack/unpack q4 roundtrip"};
    const double tol[] = {TOL_Q8_VS_FLOAT, TOL_Q4_VS_FLOAT};
    for (int t = 0; t < 2; t++) {
        void* p = malloc(packed_matrix_bytes(rows, cols, ids[t]));
        pack_matrix(W, rows, cols, p, ids[t]);
        unpack_matrix(p, rows, cols, out, ids[t]);
        check(nm[t], rel_err(out, W, rows * cols), tol[t]);
        free(p);
    }
    printf("\n");
    free(x); free(rt); free(xq); free(W); free(out);
}

int main(int argc, char** argv) {
    /* --require-simd: treat "no SIMD kernel dispatched" as a failure (use on CI machines with AVX2) */
    int require_simd = 0;
    for (int i = 1; i < argc; i++)
        if (strcmp(argv[i], "--require-simd") == 0) require_simd = 1;

    srand(1234);
    printf("kernel: %s\n\n", kernel_isa_name());

    run_roundtrips();

    /* shapes: small, rows not divisible by 4, odd block counts, and real model shapes */
    static const int shapes[][2] = {
        {256, 512}, {7, 64}, {10, 32}, {1027, 96}, {1024, 1024}, {3072, 1024}, {1024, 3072}
    };
    const tq_format fmts[] = {TQ_FORMAT_Q8, TQ_FORMAT_Q4};

    for (int f = 0; f < 2; f++)
        for (size_t s = 0; s < sizeof shapes / sizeof shapes[0]; s++)
            run_case(shapes[s][0], shapes[s][1], fmts[f], X_RANDOM);

    /* edge inputs on one shape */
    for (int f = 0; f < 2; f++) {
        run_case(256, 512, fmts[f], X_CONSTANT);
        run_case(256, 512, fmts[f], X_ZERO);
    }

    if (g_skipped > 0) {
        printf("WARNING: %d SIMD check(s) skipped: the dispatcher used the scalar kernel.\n"
               "         Run on a CPU with AVX2+FMA to test the SIMD kernels.\n", g_skipped);
        if (require_simd) {
            printf("--require-simd was given, so this counts as a failure.\n");
            g_failures++;
        }
    }

    if (g_failures == 0) printf("ALL TESTS PASSED%s\n", g_skipped ? " (SIMD kernels NOT tested)" : "");
    else                 printf("%d CHECK(S) FAILED\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}