#pragma once

#include <map>
#include <memory>
#include <vector>

#include "dbc.h"
#include "can_message.h"

namespace tareeq {
    namespace can {

class CANPacker
{
  public:
    virtual ~CANPacker() = default;
    /* */
    virtual can_message make_can_msg(
      std::string &name,
      std::map<std::string, double> &values,
      int counter
    ) = 0;

private:
  /* */
  virtual uint64_t set_value(uint64_t ret, Signal sig, int64_t ival) = 0;
  /* */
  virtual uint64_t pack(
    uint32_t address,
    const std::vector<SignalPackValue> &signals,
    int counter
  ) = 0;
};

  std::unique_ptr<CANPacker> GetPacker(const std::string& dbc_name);

    } // namespace can
} // namespace tareeq
