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

int mlp_layer_batch(const tq_mlp_weights *w, int n, const float *xn,
                    float *out, tq_batch_ws *ws) {
    int hidden = w->wgate.cols;
    int inter  = w->wgate.rows;
    if (n <= 0 || n > ws->cap) return -1;
    if (w->wup.rows != inter || w->wup.cols != hidden ||
        w->wdown.rows != hidden || w->wdown.cols != inter) return -1;

    const int nbh = hidden / QK;
    const int nbi = inter / QK;

    for (int i = 0; i < n; i++)
        quantize_row_q8(xn + (size_t)i * hidden, ws->xq + (size_t)i * nbh, hidden);

    get_matmul_batch(w->wgate.format)(w->wgate.data, ws->xq, ws->gate, inter, hidden, n);
    get_matmul_batch(w->wup.format)(w->wup.data, ws->xq, ws->up, inter, hidden, n);

    for (int i = 0; i < n; i++) {
        float *g = ws->gate + (size_t)i * inter;
        silu_mul(g, ws->up + (size_t)i * inter, g, inter);
        quantize_row_q8(g, ws->hq + (size_t)i * nbi, inter);
    }

    get_matmul_batch(w->wdown.format)(w->wdown.data, ws->hq, out, hidden, inter, n);
    return 0;
}