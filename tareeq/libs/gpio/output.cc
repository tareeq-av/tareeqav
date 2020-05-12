#include "output_impl.h"

namespace tareeq {
  namespace gpio {
    std::unique_ptr<Output> MakeOutputPin(const int line_number)
    {
      return std::make_unique<OutputImpl<gpiod::chip, gpiod::line, gpiod::line_request>>(line_number);
    }
  } // namespace gpio  
} // namespace tareeq
