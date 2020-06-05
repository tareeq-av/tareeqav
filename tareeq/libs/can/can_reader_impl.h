#pragma once

#include "cansocket_base.h"
#include "can_reader.h"

namespace tareeq {
  namespace can {

class CANReaderImpl : public CANReader, public CANSocketBase
{
  public:

    CANReaderImpl(const std::string& ifname) : CANSocketBase(ifname){};
    
    virtual can_message receive()
    {

      can_message msg;

      // std::cout << "about to read socket " << sock << std::endl;
      nbytes = read(sock, &frame, sizeof(struct can_frame));
      // std::cout << "received nbytes" << nbytes << std::endl;
      if (nbytes < 0) {
        perror("Read Error");
        return msg;
      }

      msg.address = frame.can_id;
      msg.size = frame.can_dlc;
      std::memcpy(msg.data, frame.data, msg.size);
      // printf("0x%03X [%d] ",frame.can_id, frame.can_dlc);
      // for (i = 0; i < frame.can_dlc; i++)
      //   printf("%02X ",frame.data[i]);

      // printf("\r\n");
      return std::move(msg);
    }

  private:
    size_t nbytes;

};

  } // namespace can
} // namespace tareeq
