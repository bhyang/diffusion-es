load("@rules_python//python:defs.bzl", "py_test")
load("@pip_nuplan_devkit_deps//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

py_test(
    name = "test_tutorial_utils",
    size = "small",
    srcs = ["test_tutorial_utils.py"],
    tags = ["integration"],
    deps = [
        "//nuplan/database/tests:nuplan_db_test_utils",
        "//tutorials/utils:tutorial_utils",
        requirement("IPython"),
    ],
)
