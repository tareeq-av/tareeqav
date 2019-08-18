load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

git_repository(
    name = "gtest",
    remote = "https://github.com/google/googletest",
    commit = "3306848f697568aacf4bcca330f6bdd5ce671899",
)

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