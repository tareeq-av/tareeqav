#ifndef GPIO_INPUT_INTERFACE_
#define GPIO_INPUT_INTERFACE_

#include <memory>

namespace tareeq {
  namespace gpio {
    class Input
    {
    public:
      virtual ~Input() = default;
      
      //
      virtual void WaitForEdge() = 0;
      virtual const long& GetTotalCount() = 0;
      
    };

    // simple factor method
    std::unique_ptr<Input> MakeInputPin();
    
  } // namespace gpio
} // namespace tareeq

#endif // INPUT_INTERFACE_
