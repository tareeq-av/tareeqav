#ifndef PWM_IMPL_
#define PWM_IMPL_

#include <chrono>
#include <thread>

#include "pwm.h"
#include "gpio.h"

namespace tareeq {
  namespace gpio {
    
    template <typename TChip, typename TLine, typename TLineRequest>
      class PwmImpl : public Pwm, public Gpio<TChip,TLine,TLineRequest>
    {
      // private members
      double frequency_;
      double base_time_;
      double slice_time_;
      double duty_cycle_;
      
      std::chrono::duration<double> on_duration_;
      std::chrono::duration<double> off_duration_;

      const double max_cycle_ = 100.;
      const std::string direction_ = std::string("output");
      
    public:
    PwmImpl(const int& pin_number) : Gpio<TChip,TLine,TLineRequest>(pin_number, direction_)
	{
	  // some basic default values which will be
	  // overriden by the PID controller
	  this->frequency_  = 100.0;
	  this->duty_cycle_ = 10.;
	  this->base_time_ = 1.0 / this->frequency_;
	  this->slice_time_ = this->base_time_ / this->max_cycle_;
	  this->CalculateDuration();
	  
	  // activate this pin
	  this->is_running_ = true;
	};

      virtual void Start()
      {
	std::thread t(&PwmImpl::Pulse, this);
	t.detach();
      };

      virtual const std::string &GetDirection()
      {
	return direction_;
      };

      virtual void SetFrequency(double freq)
      {
	this->base_time_ = 1.0 / freq;
	this->slice_time_ = this->base_time_ / this->max_cycle_;
	this->CalculateDuration();
      };
      
      virtual inline void CalculateDuration()
      {
	this->on_duration_ = std::chrono::duration<double>(this->duty_cycle_ * this->slice_time_);
	this->off_duration_ = std::chrono::duration<double>((this->max_cycle_ - this->duty_cycle_) * this->slice_time_);
      };
      
      virtual const double &GetDutyCycle()
      {
	return this->duty_cycle_;
      };

      virtual void Stop()
      {
	this->GpioOutput(0);
      };

      virtual void SetSpeed(double speed)
      {
	
	this->duty_cycle_ = speed;
	
	// ensure duty cycle is a percentage
	if (this->duty_cycle_ < 0.0)
	  this->duty_cycle_ = 0.0;
	
	if (this->duty_cycle_ > 100.0)
	  this->duty_cycle_ = 100.0;
	
	this->CalculateDuration();
      };

      
      virtual void Pulse()
      {
	while (this->IsRunning())
	  {
	    
	    if (this->duty_cycle_ >= 0.0)
	      {
		this->GpioOutput(1);
		std::this_thread::sleep_for(on_duration_);
	      }
	    
	    if (this->duty_cycle_ <= this->max_cycle_)
	      {
		this->GpioOutput(0);
		std::this_thread::sleep_for(off_duration_);
	      }
	  }
	
	//end of thread
	this->GpioOutput(0); // turn off
      };
      
    };
  } // namespace gpio
} // namespace tareeq

#endif // PWM_IMLP_
