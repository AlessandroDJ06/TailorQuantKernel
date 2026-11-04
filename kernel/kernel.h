#ifndef KERNEL_H
#define KERNEL_H

#include "../packer/packer.h"

void quantize_row_q8(const float* x, block_q8* out, int n);

float dot_q8_q8(const block_q8* w, const block_q8* x, int nb);

float dot_q4_q8(const block_q4* w, const block_q8* x, int nb);


typedef void (*MatmulFn)(const void* W, const block_q8* xq, float* y, int rows, int cols);
void matmul_q8(const void* W, const block_q8* xq, float* y, int rows, int cols);
void matmul_q4(const void* W, const block_q8* xq, float* y, int rows, int cols);

#endif