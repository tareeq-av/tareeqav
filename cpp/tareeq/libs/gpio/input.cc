#include "input_impl.h"

namespace tareeq {
  namespace gpio {
    std::unique_ptr<Input> MakeInputPin(const int line_number)
    {
      return std::make_unique<InputImpl<gpiod::chip, gpiod::line, gpiod::line_request>>(line_number);
    }
  } // namespace gpio  
} // namespace tareeq
