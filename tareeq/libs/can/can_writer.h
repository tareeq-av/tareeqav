 
#pragma once

#include <map>
#include <memory>
#include <vector>

#include "can_message.h"

namespace tareeq {
  namespace can {
  
class CANWriter
{
  public:
    virtual ~CANWriter() = default;
    virtual bool send(const can_message& msg) = 0;

};

  std::unique_ptr<CANWriter> GetWriter(const std::string& ifname);

  } // namespace can
} // namespace tareeq
