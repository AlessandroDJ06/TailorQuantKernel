#ifndef KERNEL_INTERNAL_H
#define KERNEL_INTERNAL_H

#include "kernel.h"

void matmul_q8_scalar(const void* W, const block_q8* xq, float* y, int rows, int cols);
void matmul_q4_scalar(const void* W, const block_q8* xq, float* y, int rows, int cols);

void matmul_q8_avx2(const void* W, const block_q8* xq, float* y, int rows, int cols);
void matmul_q4_avx2(const void* W, const block_q8* xq, float* y, int rows, int cols);

#endif