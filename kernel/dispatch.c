#include <stddef.h>
#include "kernel_internal.h"

static int cpu_has_avx2(void) {
#ifdef TQ_HAVE_AVX2
    static int cached = -1;
    if (cached < 0) {
        __builtin_cpu_init();
        cached = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
    }
    return cached;
#else
    return 0;
#endif
}

MatmulFn get_matmul(tq_format format) {
    int avx2 = cpu_has_avx2();
    (void)avx2;

    switch (format) {
        case TQ_FORMAT_Q8:
#ifdef TQ_HAVE_AVX2
            if (avx2) return matmul_q8_avx2;
#endif
            return matmul_q8_scalar;

        case TQ_FORMAT_Q4:
#ifdef TQ_HAVE_AVX2
            if (avx2) return matmul_q4_avx2;
#endif
            return matmul_q4_scalar;

        default:
            return NULL;
    }
}

const char* kernel_isa_name(void) {
    return cpu_has_avx2() ? "AVX2+FMA" : "scalar";
}