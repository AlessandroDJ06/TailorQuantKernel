#include "model/model_ops.h"

int mlp_layer(const tq_mlp_weights *w, const float *xn, block_q8 *xq,
              float *gate, float *up, block_q8 *hq, float *out) {
    int hidden = w->wgate.cols;
    int inter  = w->wgate.rows;
    if (w->wup.rows != inter || w->wup.cols != hidden ||
        w->wdown.rows != hidden || w->wdown.cols != inter) return -1;
    quantize_row_q8(xn, xq, hidden);
    get_matmul(w->wgate.format)(w->wgate.data, xq, gate, inter, hidden);
    get_matmul(w->wup.format)(w->wup.data, xq, up, inter, hidden);
    silu_mul(gate, up, gate, inter);
    quantize_row_q8(gate, hq, inter);
    get_matmul(w->wdown.format)(w->wdown.data, hq, out, hidden, inter);

    return 0;
}