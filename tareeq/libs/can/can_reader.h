 
#pragma once

#include <map>
#include <memory>
#include <vector>

#include "can_message.h"

namespace tareeq {
  namespace can {
  
class CANReader
{
  public:
    virtual ~CANReader() = default;
    virtual can_message receive() = 0;

};

  std::unique_ptr<CANReader> GetReader(const std::string& ifname);

  } // namespace can
} // namespace tareeq
