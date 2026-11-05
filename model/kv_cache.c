// kv_cache.c
#include "model/model_ops.h"
#include <stdlib.h>

void tq_kv_cache_free(tq_kv_cache *cache) {
    free(cache->k_f32);
    free(cache->v_f32);
    free(cache->k_q8);
    free(cache->v_q8);
    cache->k_f32 = NULL;
    cache->v_f32 = NULL;
    cache->k_q8  = NULL;
    cache->v_q8  = NULL;
    cache->max_tokens = 0;
}

int tq_kv_cache_init(tq_kv_cache *cache, tq_kv_type type,
                     const tq_attn_config *cfg, int max_tokens) {
    cache->type       = type;
    cache->max_tokens = max_tokens;
    cache->k_f32 = NULL;
    cache->v_f32 = NULL;
    cache->k_q8  = NULL;
    cache->v_q8  = NULL;

    if (max_tokens <= 0) return -1;

    if (type == TQ_KV_F32) {
        size_t n = (size_t)cfg->n_kv_heads * max_tokens * cfg->head_dim;
        cache->k_f32 = malloc(n * sizeof(float));
        cache->v_f32 = malloc(n * sizeof(float));
        if (!cache->k_f32 || !cache->v_f32) {
            tq_kv_cache_free(cache);
            return -1;
        }
    } else if (type == TQ_KV_Q8) {
        size_t n = (size_t)cfg->n_kv_heads * max_tokens * (cfg->head_dim / QK);
        cache->k_q8 = malloc(n * sizeof(block_q8));
        cache->v_q8 = malloc(n * sizeof(block_q8));
        if (!cache->k_q8 || !cache->v_q8) {
            tq_kv_cache_free(cache);
            return -1;
        }
    } else {
        return -1;
    }
    return 0;
}