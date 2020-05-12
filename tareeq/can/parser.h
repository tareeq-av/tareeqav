 
#pragma once

#include <map>
#include <memory>
#include <vector>

#include "dbc.h"
#include "can_message.h"

/**
 *
class MessageState {
public:
  uint32_t address;
  unsigned int size;

  std::vector<Signal> parse_sigs;
  std::vector<double> vals;

  uint16_t ts;
  uint64_t seen;
  uint64_t check_threshold;

  uint8_t counter;
  uint8_t counter_fail;

  bool parse(uint64_t sec, uint16_t ts_, uint8_t * dat);
  bool update_counter_generic(int64_t v, int cnt_size);
};
*/

namespace tareeq {
  namespace can {
  
class CANParser
{
  public:
    virtual ~CANParser() = default;
    virtual std::map<std::string, double> parse(can_message msg) = 0;

};

  std::unique_ptr<CANParser> GetParser(const std::string& dbc_name);

  } // namespace can
} // namespace tareeq
