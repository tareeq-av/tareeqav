#include "tareeq/gpio/pwm_impl.h"

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "tareeq/mocks/gpio.h"



namespace tareeq {
  namespace gpio {

    
    
    class PwmTest: public testing::Test {
    protected:
      void SetUp() override {
      
      }

      PwmImpl<tareeq::mocks::MockChip, tareeq::mocks::MockLine, tareeq::mocks::MockLineRequest> pwm_{12};
    };


    TEST_F(PwmTest, SetDutyCycle) {

      pwm_.SetSpeed(50);
      EXPECT_EQ(50, pwm_.GetDutyCycle());

      pwm_.SetSpeed(-10);
      EXPECT_EQ(0, pwm_.GetDutyCycle());

      pwm_.SetSpeed(150);
      EXPECT_EQ(100, pwm_.GetDutyCycle());
  }

 } // namespace gpio
} // namespace tareeq

