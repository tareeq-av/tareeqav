

cc_library(
    name = "can",
    srcs = glob(["*.cc"]),	
    hdrs = glob(["*.h*"]),
    deps = [
      "@linux_libs//:uavcan",
    ],
    visibility = ["//visibility:public"],
)

cc_binary(
    name = "can_logger",
    srcs = ["can_logger.cc"],
    deps = [   
      ":can",
      "@linux_libs//:uavcan"
    ],
)

# cc_binary(
#     name = "can_test",
#     srcs = [
#       "can_test.cc"
#     ],
#     deps = [   
#       ":can",
#       "@linux_libs//:uavcan"
#     ],
# )
