cc_library(
    name = "gpio",
    srcs = glob([
        "*.cc",
	"gpio.h",
	"pwm_impl.h",
	"output_impl.h",
	"input_impl.h"
	]),	
    hdrs = glob([
        "pwm.h",
	"output.h",
	"input.h"
    ]),
    linkopts = [
      "-pthread",
    ],
    deps = [
      "@linux_libs//:gpiod",
    ],
    visibility = ["//visibility:public"],
)
