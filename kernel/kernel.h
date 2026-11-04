#ifndef KERNEL_H
#define KERNEL_H

#include "../packer/packer.h"

void quantize_row_q8(const float* x, block_q8* out, int n);

typedef void (*MatmulFn)(const void* W, const block_q8* xq, float* y, int rows, int cols);

MatmulFn get_matmul(char bit_type);

const char* kernel_isa_name(void);

#endif