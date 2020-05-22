#pragma once

#include "cansocket_base.h"
#include "can_writer.h"

namespace tareeq {
  namespace can {

class CANWriterImpl : public CANWriter, public CANSocketBase
{
  public:

    CANWriterImpl(const std::string& ifname) : CANSocketBase(ifname){};

    virtual bool send(const can_message& msg)
    {
      frame.can_id = msg.address;
      frame.can_dlc = msg.size;
      std::memcpy(frame.data, msg.data, msg.size);

      if (write(sock, &frame, sizeof(struct can_frame)) != sizeof(struct can_frame)) {
        // perror("Write");
        return false;
      }

      return true;
    }

//   private:

};

  } // namespace can
} // namespace tareeq
