#ifndef TAILORQUANT_MODEL_OPS_H
#define TAILORQUANT_MODEL_OPS_H

#ifdef __cplusplus
extern "C" {
#endif

#include "../packer/packer.h"

void silu_mul(const float *gate, const float *up, float *out, int n);

void rmsnorm(const float *x, const float *weight, float *out, int n, float eps);

void softmax(float *x, int n);

void rope(float *x, int head_dim, int pos, float theta);

/*
 * Attention voor één Q-kop over een float-cache.
 *
 *   q        : head_dim floats (na QK-norm en RoPE)
 *   k_cache  : wijst naar token 0 van de juiste KV-kop (aanroeper telt
 *              de offset (h / groepsgrootte) * head_dim er al bij op)
 *   v_cache  : idem voor V
 *   n        : aantal tokens in de cache (pos + 1); causale mask zit in deze lengte
 *   head_dim : lengte van q, een K-rij en een V-rij; bepaalt ook de schaal 1/sqrt(head_dim)
 *   stride   : afstand in floats tussen token t en t + 1 in de cache
 *              (num_kv_heads * head_dim bij layout [token][kv_kop][head_dim])
 *   scores   : werkbuffer van minstens n floats, geleverd door de aanroeper
 *   out      : head_dim floats; mag niet overlappen met scores of de caches
 */
void attention_head(const float *q,
                    const float *k_cache, const float *v_cache,
                    int n, int head_dim, int stride,
                    float *scores, float *out);

void attention_head_q8(const float *q, block_q8 *q_q8,
                       const block_q8 *k_cache, const block_q8 *v_cache,
                       int n, int head_dim,
                       float *scores, float *out);

#ifdef __cplusplus
}
#endif

#endif