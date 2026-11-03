#ifndef PACKER_H
#define PACKER_H

#include <stddef.h>
#include <stdint.h>

#define QK 32
typedef struct { float d; int8_t  q[QK];     } block_q8;
typedef struct { float d; uint8_t q[QK / 2]; } block_q4;

_Static_assert(sizeof(block_q8) == 36, "block_q8 has padding");
_Static_assert(sizeof(block_q4) == 20, "block_q4 has padding");
size_t packed_row_bytes(int cols, char bit_type);
size_t packed_matrix_bytes(int rows, int cols, char bit_type);
int pack_matrix(const float* W, int rows, int cols, void* out, char bit_type);
int unpack_matrix(const void* packed, int rows, int cols, float* out, char bit_type);

#endif