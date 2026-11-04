#ifndef TAILORQUANT_MODEL_OPS_H
#define TAILORQUANT_MODEL_OPS_H

#ifdef __cplusplus
extern "C" {
#endif

void silu_mul(const float *gate, const float *up, float *out, int n);

void rmsnorm(const float *x, const float *weight, float *out, int n, float eps);

void softmax(float *x, int n);

void rope(float *x, int head_dim, int pos, float theta);

#ifdef __cplusplus
}
#endif

#endif