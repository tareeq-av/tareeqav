#include <algorithm>

#include "common.h"

namespace tareeq {
  namespace can {

unsigned int toyota_checksum(unsigned int address, uint64_t d, int l) {
  d >>= ((8-l)*8); // remove padding
  d >>= 8; // remove checksum

  unsigned int s = l;
  while (address) { s += address & 0xFF; address >>= 8; }
  while (d) { s += d & 0xFF; d >>= 8; }

  return s & 0xFF;
}

// Static lookup table for fast computation of CRC8 poly 0x2F, aka 8H2F/AUTOSAR
uint8_t crc8_lut_8h2f[256];

void gen_crc_lookup_table(uint8_t poly, uint8_t crc_lut[]) {
  uint8_t crc;
  int i, j;

   for (i = 0; i < 256; i++) {
    crc = i;
    for (j = 0; j < 8; j++) {
      if ((crc & 0x80) != 0)
        crc = (uint8_t)((crc << 1) ^ poly);
      else
        crc <<= 1;
    }
    crc_lut[i] = crc;
  }
}

void init_crc_lookup_tables() {
  // At init time, set up static lookup tables for fast CRC computation.
  gen_crc_lookup_table(0x2F, crc8_lut_8h2f);    // CRC-8 8H2F/AUTOSAR for Volkswagen
}


unsigned int pedal_checksum(uint64_t d, int l) {
  uint8_t crc = 0xFF;
  uint8_t poly = 0xD5; // standard crc8

  d >>= ((8-l)*8); // remove padding
  d >>= 8; // remove checksum

  uint8_t *dat = (uint8_t *)&d;

  int i, j;
  for (i = 0; i < l - 1; i++) {
    crc ^= dat[i];
    for (j = 0; j < 8; j++) {
      if ((crc & 0x80) != 0) {
        crc = (uint8_t)((crc << 1) ^ poly);
      }
      else {
        crc <<= 1;
      }
    }
  }
  return crc;
}


uint64_t read_u64_be(const uint8_t* v) {
  return (((uint64_t)v[0] << 56)
          | ((uint64_t)v[1] << 48)
          | ((uint64_t)v[2] << 40)
          | ((uint64_t)v[3] << 32)
          | ((uint64_t)v[4] << 24)
          | ((uint64_t)v[5] << 16)
          | ((uint64_t)v[6] << 8)
          | (uint64_t)v[7]);
}

uint64_t read_u64_le(const uint8_t* v) {
  return ((uint64_t)v[0]
          | ((uint64_t)v[1] << 8)
          | ((uint64_t)v[2] << 16)
          | ((uint64_t)v[3] << 24)
          | ((uint64_t)v[4] << 32)
          | ((uint64_t)v[5] << 40)
          | ((uint64_t)v[6] << 48)
          | ((uint64_t)v[7] << 56));
}

int clip(int x, int lo, int hi)
{
  return std::max(lo, std::min(hi, x));
}

double clip(double x, double lo, double hi)
{
  return std::max(lo, std::min(hi, x));
}



  } // namespace can
} // namespace tareeq
