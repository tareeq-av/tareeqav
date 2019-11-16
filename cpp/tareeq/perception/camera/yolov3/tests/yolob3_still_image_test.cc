#include <gtest/gtest.h>
#include <gmock/gmock.h>

namespace tareeq {
  namespace perception {
      namespace camera {
          namespace yolov3 {

    class YOLOv3StillImageTest: public testing::Test {
    protected:
      void SetUp() override {
      
      }

    };


    TEST_F(YOLOv3Tests, StillImageTest) {
      EXPECT_EQ(1, 1);
  }

 } // namespace yolov3
} // namespace camera
} // namespace perception
} // namespace tareeq

