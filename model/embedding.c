#include <stdint.h>
#include "model_ops.h"

int embed_lookup(const tq_matrix *embed, int token, float *out) {
    if (token < 0 || token >= embed->rows) return -1;
    if (embed->cols % QK != 0) return -1;

    int nb = embed->cols / QK;

    switch (embed->format) {
        case TQ_FORMAT_Q8: {
            const block_q8 *row = (const block_q8 *)embed->data + (size_t)token * nb;
            for (int b = 0; b < nb; b++) {
                for (int j = 0; j < QK; j++) {
                    out[b * QK + j] = row[b].d * (float)row[b].q[j];
                }
            }
            return 0;
        }
        case TQ_FORMAT_Q4: {
            const block_q4 *row = (const block_q4 *)embed->data + (size_t)token * nb;
            for (int b = 0; b < nb; b++) {
                for (int j = 0; j < QK / 2; j++) {
                    out[b * QK + j]          = row[b].d * (float)((row[b].q[j] & 0x0F) - 8);
                    out[b * QK + j + QK / 2] = row[b].d * (float)((row[b].q[j] >> 4) - 8);
                }
            }
            return 0;
        }
        default:
            return -1;
    }
}

int lm_head(const tq_matrix *embed, const float *xn, block_q8 *xq, float *logits) {
    if (embed->cols % QK != 0) return -1;
    quantize_row_q8(xn,xq,embed->cols);
    get_matmul(embed->format)(embed->data,xq,logits,embed->rows,embed->cols);
    return 0;
}