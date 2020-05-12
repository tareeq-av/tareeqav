#pragma once

#include <map>
#include <vector>

#include "dbc.h"

namespace tareeq {
  namespace can {

    
#define MAX_BAD_COUNTER 5

// Helper functions
unsigned int toyota_checksum(unsigned int address, uint64_t d, int l);
void init_crc_lookup_tables();
unsigned int pedal_checksum(uint64_t d, int l);
uint64_t read_u64_be(const uint8_t* v);
uint64_t read_u64_le(const uint8_t* v);
int clip(int x, int lo, int hi);
double clip(double x, double lo, double hi);

  } // namespace can
} // namespace tareeq
