#include "can_reader_impl.h"

namespace tareeq {
  namespace can {

    std::unique_ptr<CANReader> GetReader(const std::string& ifname)
    {
      return std::make_unique<CANReaderImpl>(ifname);
    }

  } // namespace can
} // namespace tareeq
