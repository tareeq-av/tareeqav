package(default_visibility = ["//visibility:public"])

licenses(["notice"])

cc_library(
    name = "python36",
    srcs = glob([
        "lib/python3.6/config-x86_64-linux-gnu/libpython3.6.so",
    ]),
    hdrs = glob([
        "include/python3.6/*.h",
    ]),
    includes = ["include/python3.6"],
)