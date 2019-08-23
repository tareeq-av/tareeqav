#ifndef GPIO_BASE_
#define GPIO_BASE_

#include <gpiod.hpp>

namespace tareeq {
  namespace gpio {
    
    template <typename TChip, typename TLine, typename TLineRequest>
      class Gpio
    {
    protected:
      TChip chip_;
      TLine line_;
      TLineRequest request_;
      
      /** gpio pin number, not board pin number */
      uint32_t line_number_;
      
      /* set by child classes */
      bool is_running_;
      
      int  request_type_;
      std::string consumer_;

      const std::string chip_number_ = std::string("0");
      
      virtual const std::string &GetDirection() = 0;

      void Stop()
      {
	this->is_running_ = false;
      };
      
      /* 
	 boolean used to activate
	 and deactivate the robot
      */
      inline const bool &IsRunning()
      {
	return is_running_;
      };

      inline void GpioOutput(int value)
      {
	this->line_.set_value(value);
      };
      
      Gpio() = default;
      ~Gpio() {this->Stop();};
      
    Gpio(const uint32_t &line_number,
	 const std::string &direction):
      line_number_(line_number)
	{
	  
	  chip_ = TChip(chip_number_, TChip::OPEN_BY_NUMBER);
	  
	  if (direction == "output")
	    {
	      request_type_ = TLineRequest::DIRECTION_OUTPUT;
	    }
	  else
	    {
	      request_type_ = TLineRequest::EVENT_BOTH_EDGES;
	    }
	  
	  consumer_ = std::string("pin_number_") + std::to_string(line_number);
	  
	  request_ = {
	  consumer: consumer_,
	  request_type: request_type_ 
	  };
	  
	  line_ = chip_.get_line(line_number_);
	  line_.request(request_);
	  
	};
    };

  } // namespace gpio
} // namespace tareeq

#endif // GPIO_BASE_
