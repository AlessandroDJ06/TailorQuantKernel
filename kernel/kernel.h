#ifndef KERNEL_H
#define KERNEL_H

#include "../packer/packer.h"

typedef enum {
    TQ_FORMAT_Q8,
    TQ_FORMAT_Q4
} tq_format;

void quantize_row_q8(const float* x, block_q8* out, int n);

typedef void (*MatmulFn)(const void* W, const block_q8* xq, float* y, int rows, int cols);

MatmulFn get_matmul(tq_format format);

const char* kernel_isa_name(void);

#endif