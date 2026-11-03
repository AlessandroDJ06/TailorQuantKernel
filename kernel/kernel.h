//
// Created by aless on 03/11/2026.
//

#ifndef TAILORQUANTKERNEL_KERNEL_H
#define TAILORQUANTKERNEL_KERNEL_H
#include <stdint.h>

void matmul_kernel(const uint8_t* weights, const float* input, float* output_array, int out_features);
#endif //TAILORQUANTKERNEL_KERNEL_H
