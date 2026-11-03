#ifndef PACKER_H
#define PACKER_H

#include <stdint.h>

#define BLOCK_SIZE 32

// Publieke routerfunctie voor de buitenwereld
void pack_layer(const float* raw_weights, int num_weights, uint8_t* packed_output, char bit_type);

#endif // PACKER_H