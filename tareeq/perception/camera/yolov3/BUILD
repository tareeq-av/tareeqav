# cc_test(
#     name = "tests",
#     srcs = glob(["tests/*.cc"]),
#     deps = [
#         "@opencv",
#         "@gtest//:gtest_main",
#     ],
# )

cc_binary(
    name = "yolov3_single_image_test",
    srcs = ["main_single_image.cc", "darknet.cc", "darknet.h"],
    deps = [
        "@opencv",
        "@libtorch"
    ]
)

cc_binary(
    name = "yolov3_stream_test",
    srcs = ["main_stream.cc", "darknet.cc", "darknet.h"],
    deps = [
        "@opencv",
        "@libtorch"
    ]
)