#include "can_writer_impl.h"

namespace tareeq {
  namespace can {

    std::unique_ptr<CANWriter> GetWriter(const std::string& ifname)
    {
      return std::make_unique<CANWriterImpl>(ifname);
    }

  } // namespace can
} // namespace tareeq
