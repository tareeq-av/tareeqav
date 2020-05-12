#include "packer_impl.h"

#define WARN printf

namespace tareeq {
  namespace can {

    std::unique_ptr<CANPacker> GetPacker(const std::string& dbc_name)
    {
      return std::make_unique<CANPackerImpl>(dbc_name);
    }

  } // namespace can
} // namespace tareeq

