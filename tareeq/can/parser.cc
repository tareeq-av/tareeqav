#include "parser_impl.h"

namespace tareeq {
  namespace can {

    std::unique_ptr<CANParser> GetParser(const std::string& dbc_name)
    {
      return std::make_unique<CANParserImpl>(dbc_name);
    }

  } // namespace can
} // namespace tareeq
