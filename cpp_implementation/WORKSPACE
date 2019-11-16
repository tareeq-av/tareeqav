
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name   = "gtest",
    urls = ["https://github.com/google/googletest/archive/v1.10.x.tar.gz"],
    build_file = "BUILD.gtest"
)

# http_archive(
#     name = "opencv",
#     urls = ["https://github.com/opencv/opencv/archive/4.1.1.tar.gz"],
#     build_file = "opencv/BUILD.opencv"
# )

new_local_repository(
	name = "linux_libs",
	path = "/usr/local/lib",
	build_file_content = """
cc_library(
	name = "gpiod",
	srcs = ["libgpiodcxx.so"],
	visibility = ["//visibility:public"],
)
""",
)

new_local_repository(
    name = "opencv",
    path = "/usr/local",
    build_file = "external/opencv/prebuilt.BUILD",
)

# python3.6
# new_local_repository(
#     name = "python36",
#     build_file = "external/python36.BUILD",
#     path = "/usr",
# )

# pytorch c++ bindings
new_local_repository(
    name = "libtorch",
    build_file = "external/libtorch_gpu.BUILD",
    path = "/usr/local/lib/libtorch",
)