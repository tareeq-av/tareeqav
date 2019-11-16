cc_library(
    name = "opencv",
    srcs = glob(["lib/*.so*"]),
    hdrs = glob(["include/**/*.hpp"]),
    includes = [
        "include/opencv4",
        "include/opencv4/opencv",
    ],
    visibility = ["//visibility:public"], 
    linkstatic = False,
)
