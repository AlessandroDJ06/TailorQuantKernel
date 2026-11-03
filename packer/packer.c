#include "packer.h"
#include <math.h>

static inline int quantize(float x, float inv_scale, int qmax) {
    int q = (int)lroundf(x * inv_scale);
    if (q >  qmax) q =  qmax;
    if (q < -qmax) q = -qmax;
    return q;
}

static float block_scale(const float* w, float qmax) {
    float amax = 0.0f;
    for (int j = 0; j < QK; j++) {
        float v = fabsf(w[j]);
        if (v > amax) amax = v;
    }
    float d = amax / qmax;
    return d == 0.0f ? 1.0f : d;
}


static void pack_row_q8(const float* row, int cols, block_q8* out) {
    for (int b = 0; b < cols / QK; b++) {
        const float* w = row + b * QK;
        float d = block_scale(w, 127.0f);
        float id = 1.0f / d;
        out[b].d = d;
        for (int j = 0; j < QK; j++)
            out[b].q[j] = (int8_t)quantize(w[j], id, 127);
    }
}

static void unpack_row_q8(const block_q8* in, int cols, float* row) {
    for (int b = 0; b < cols / QK; b++)
        for (int j = 0; j < QK; j++)
            row[b * QK + j] = in[b].q[j] * in[b].d;
}


static void pack_row_q4(const float* row, int cols, block_q4* out) {
    for (int b = 0; b < cols / QK; b++) {
        const float* w = row + b * QK;
        float d = block_scale(w, 7.0f);
        float id = 1.0f / d;
        out[b].d = d;
        for (int j = 0; j < QK / 2; j++) {
            int lo = quantize(w[j],          id, 7) + 8;
            int hi = quantize(w[j + QK / 2], id, 7) + 8;
            out[b].q[j] = (uint8_t)((hi << 4) | lo);
        }
    }
}

static void unpack_row_q4(const block_q4* in, int cols, float* row) {
    for (int b = 0; b < cols / QK; b++) {
        for (int j = 0; j < QK / 2; j++) {
            row[b * QK + j]          = ((in[b].q[j] & 0x0F) - 8) * in[b].d;
            row[b * QK + j + QK / 2] = ((in[b].q[j] >> 4)   - 8) * in[b].d;
        }
    }
}

size_t packed_row_bytes(int cols, char bit_type) {
    if (cols <= 0 || cols % QK != 0) return 0;
    size_t nb = (size_t)(cols / QK);
    switch (bit_type) {
        case '8': return nb * sizeof(block_q8);
        case '4': return nb * sizeof(block_q4);
        default:  return 0;
    }
}

size_t packed_matrix_bytes(int rows, int cols, char bit_type) {
    return (size_t)rows * packed_row_bytes(cols, bit_type);
}

int pack_matrix(const float* W, int rows, int cols, void* out, char bit_type) {
    size_t rb = packed_row_bytes(cols, bit_type);
    if (rb == 0 || rows <= 0) return -1;

    uint8_t* dst = (uint8_t*)out;
    for (int r = 0; r < rows; r++) {
        const float* row = W + (size_t)r * cols;
        void* drow = dst + (size_t)r * rb;
        if (bit_type == '8') pack_row_q8(row, cols, (block_q8*)drow);
        else                 pack_row_q4(row, cols, (block_q4*)drow);
    }
    return 0;
}

int unpack_matrix(const void* packed, int rows, int cols, float* out, char bit_type) {
    size_t rb = packed_row_bytes(cols, bit_type);
    if (rb == 0 || rows <= 0) return -1;

    const uint8_t* src = (const uint8_t*)packed;
    for (int r = 0; r < rows; r++) {
        const void* srow = src + (size_t)r * rb;
        float* row = out + (size_t)r * cols;
        if (bit_type == '8') unpack_row_q8((const block_q8*)srow, cols, row);
        else                 unpack_row_q4((const block_q4*)srow, cols, row);
    }
    return 0;
}