#include "packer.h"
#include <math.h>
#include <string.h>

typedef void (*PackerBlockFunc)(const float* w, int n, float inv_scale, uint8_t* out, int* out_index);

static inline int quantize(float x, float inv_scale, int qmax) {
    int q = (int)lroundf(x * inv_scale);
    if (q >  qmax) q =  qmax;
    if (q < -qmax) q = -qmax;
    return q;
}

static void pack_block_8bit(const float* w, int n, float inv_scale, uint8_t* out, int* oi) {
    for (int j = 0; j < n; j++) {
        out[(*oi)++] = (uint8_t)(int8_t)quantize(w[j], inv_scale, 127);
    }
}

static void pack_block_4bit(const float* w, int n, float inv_scale, uint8_t* out, int* oi) {
    int half = (n + 1) / 2;
    for (int j = 0; j < half; j++) {
        int lo = quantize(w[j], inv_scale, 7) + 8;
        int hi = 8; // zero-point als padding
        if (j + half < n) hi = quantize(w[j + half], inv_scale, 7) + 8;
        out[(*oi)++] = (uint8_t)((hi << 4) | lo);
    }
}

static void pack_bits_generic(const float* w, int num_weights, uint8_t* out,
                              float qmax, PackerBlockFunc strategy) {
    int oi = 0;
    for (int i = 0; i < num_weights; i += BLOCK_SIZE) {
        int n = (i + BLOCK_SIZE > num_weights) ? num_weights - i : BLOCK_SIZE;

        float max_val = 0.0f;
        for (int j = 0; j < n; j++) {
            float v = fabsf(w[i + j]);
            if (v > max_val) max_val = v;
        }

        float scale = max_val / qmax;
        if (scale == 0.0f) scale = 1.0f;

        memcpy(&out[oi], &scale, sizeof(float));
        oi += sizeof(float);

        strategy(w + i, n, 1.0f / scale, out, &oi);
    }
}

void pack_layer(const float* w, int num_weights, uint8_t* out, char bit_type) {
    switch (bit_type) {
        case '8': pack_bits_generic(w, num_weights, out, 127.0f, pack_block_8bit); break;
        case '4': pack_bits_generic(w, num_weights, out,   7.0f, pack_block_4bit); break;
        case 'T': /* later */ break;
        default: break;
    }
}