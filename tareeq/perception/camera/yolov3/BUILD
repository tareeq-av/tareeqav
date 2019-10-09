# cc_test(
#     name = "tests",
#     srcs = glob(["tests/*.cc"]),
#     deps = [
#         "@opencv",
#         "@gtest//:gtest_main",
#     ],
# )

cc_binary(
    name = "yolov3_test",
    srcs = ["main.cc", "darknet.cc", "darknet.h"],
    deps = [
        "@opencv",
        "@libtorch"
    ]
)