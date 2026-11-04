#include <stddef.h>
#include "kernel_internal.h"

static int cpu_has_avx2(void) {
#ifdef TQ_HAVE_AVX2
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2");
#else
    return 0;
#endif
}

MatmulFn get_matmul(char bit_type) {
    int avx2 = cpu_has_avx2();
    (void)avx2;   /* unused on builds without the AVX2 files */

    switch (bit_type) {
        case '8':
#ifdef TQ_HAVE_AVX2
            if (avx2) return matmul_q8_avx2;
#endif
            return matmul_q8_scalar;

        case '4':
#ifdef TQ_HAVE_AVX2
            if (avx2) return matmul_q4_avx2;
#endif
            return matmul_q4_scalar;

        default:
            return NULL;
    }
}

const char* kernel_isa_name(void) {
    return cpu_has_avx2() ? "AVX2" : "scalar";
}