#include <stdlib.h>
#include <string.h>

#include "model/model_ops.h"

int tq_ctx_init(tq_ctx *ctx, const tq_model *m, tq_kv_type kv, int max_tokens) {
    memset(ctx, 0, sizeof *ctx);
    if (m == NULL || max_tokens <= 0) return -1;
    if (m->cfg.attn.hidden_size % QK != 0 || m->cfg.intermediate_size % QK != 0) return -1;
    if (m->cfg.n_layers <= 0 || m->cfg.vocab_size <= 0) return -1;

    for (int l = 0; l < m->cfg.n_layers; l++) {
        if (tq_attn_validate(&m->w.layers[l].attn, &m->cfg.attn) != 0) return -1;
    }

    ctx->model = m;
    ctx->max_tokens = max_tokens;
    ctx->pos = 0;

    int hidden = m->cfg.attn.hidden_size;
    int inter = m->cfg.intermediate_size;
    int vocab = m->cfg.vocab_size;
    int n_heads = m->cfg.attn.n_q_heads;
    int n_layers = m->cfg.n_layers;

    ctx->x        = calloc((size_t)hidden, sizeof(float));
    ctx->xn       = calloc((size_t)hidden, sizeof(float));
    ctx->attn_out = calloc((size_t)hidden, sizeof(float));
    ctx->mlp_out  = calloc((size_t)hidden, sizeof(float));
    ctx->gate     = calloc((size_t)inter,  sizeof(float));
    ctx->up       = calloc((size_t)inter,  sizeof(float));
    ctx->xq       = calloc((size_t)hidden / QK, sizeof(block_q8));
    ctx->hq       = calloc((size_t)inter  / QK, sizeof(block_q8));
    ctx->scores   = calloc((size_t)n_heads * max_tokens, sizeof(float));
    ctx->logits   = calloc((size_t)vocab,  sizeof(float));
    ctx->caches   = calloc((size_t)n_layers, sizeof(tq_kv_cache));

    if (!ctx->x || !ctx->xn || !ctx->attn_out || !ctx->mlp_out ||
        !ctx->gate || !ctx->up || !ctx->xq || !ctx->hq ||
        !ctx->scores || !ctx->logits || !ctx->caches) {
        tq_ctx_free(ctx);
        return -1;
        }

    for (int l = 0; l < n_layers; l++) {
        if (tq_kv_cache_init(&ctx->caches[l], kv, &m->cfg.attn, max_tokens) != 0 ||
            tq_attn_validate_cache(&ctx->caches[l]) != 0) {
            tq_ctx_free(ctx);
            return -1;
            }
    }

    return 0;
}

void tq_ctx_free(tq_ctx *ctx) {
    if (ctx == NULL) return;

    if (ctx->caches != NULL && ctx->model != NULL) {
        for (int l = 0; l < ctx->model->cfg.n_layers; l++) {
            tq_kv_cache_free(&ctx->caches[l]);
        }
    }
    free(ctx->caches);

    free(ctx->x);
    free(ctx->xn);
    free(ctx->attn_out);
    free(ctx->mlp_out);
    free(ctx->gate);
    free(ctx->up);
    free(ctx->xq);
    free(ctx->hq);
    free(ctx->scores);
    free(ctx->logits);

    memset(ctx, 0, sizeof *ctx);
}

static int attn_block(tq_ctx *ctx, const tq_layer_weights *layer, int l) {
    const tq_attn_config *cfg = &ctx->model->cfg.attn;

    rmsnorm(ctx->x, layer->input_norm, ctx->xn, cfg->hidden_size, cfg->rms_eps);

    if (attention_layer(&layer->attn, cfg, &ctx->caches[l], ctx->pos,
                        ctx->xn, ctx->scores, ctx->attn_out) != 0) return -1;

    vec_add(ctx->x, ctx->attn_out, cfg->hidden_size);
    return 0;
}

static int mlp_block(tq_ctx *ctx, const tq_layer_weights *layer) {
    const tq_attn_config *cfg = &ctx->model->cfg.attn;

    rmsnorm(ctx->x, layer->post_norm, ctx->xn, cfg->hidden_size, cfg->rms_eps);

    if (mlp_layer(&layer->mlp, ctx->xn, ctx->xq, ctx->gate, ctx->up,
                  ctx->hq, ctx->mlp_out) != 0) return -1;

    vec_add(ctx->x, ctx->mlp_out, cfg->hidden_size);
    return 0;
}

int tq_forward_token(tq_ctx *ctx, int token) {
    if (ctx == NULL) return -1;
    if (ctx->pos >= ctx->max_tokens) return -1;

    const tq_model *m = ctx->model;
    int hidden_size = m->cfg.attn.hidden_size;
    float rms_eps = m->cfg.attn.rms_eps;

    if (token < 0 || token >= m->cfg.vocab_size) return -1;
    if (embed_lookup(&m->w.embed, token, ctx->x) != 0) return -1;

    for (int l = 0; l < m->cfg.n_layers; l++) {
        const tq_layer_weights *layer = &m->w.layers[l];
        if (attn_block(ctx, layer, l) != 0) return -1;
        if (mlp_block(ctx, layer) != 0) return -1;
    }

    rmsnorm(ctx->x, m->w.final_norm, ctx->xn, hidden_size, rms_eps);
    if (lm_head(&m->w.embed, ctx->xn, ctx->xq, ctx->logits) != 0) return -1;

    ctx->pos++;
    return 0;
}
