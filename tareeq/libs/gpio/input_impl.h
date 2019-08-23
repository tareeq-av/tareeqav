#ifndef INPUT_IMPL_
#define INPUT_IMPL_

#include "input.h"
#include "gpio.h"

namespace tareeq {
  namespace gpio {
    template <typename TChip, typename TLine, typename TLineRequest>
      class InputImpl : public Input, public Gpio<TChip,TLine,TLineRequest>
    {
      long total_count_ = 0;
      gpiod::line_event event_;
      const std::string direction_ = std::string("input");

    public:
    InputImpl(const int& line_number): Gpio<TChip,TLine,TLineRequest>(line_number, direction_){};

      virtual const std::string &GetDirection()
      {
	return direction_;
      };

      virtual void WaitForEdge()
      {
	while(this->IsRunning())
	  {
	    while (this->line_.event_wait(std::chrono::milliseconds(1)))
	      {
		this->event_ = this->line_.event_read();
		if (this->event_.event_type == gpiod::line_event::FALLING_EDGE)
		  {
		    total_count_++;
		  }
	      }
	  }
      };
      
      virtual const long& GetTotalCount()
      {
	return total_count_;
      };
      
    };
  } // namespace gpio
} // namespace tareeq

#endif // INPUT_IMPL_
