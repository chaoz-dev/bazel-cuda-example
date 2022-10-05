cc_library(
    name = "cuda",
    srcs = glob(["lib64/**/lib*.so"]),
    hdrs = glob([
        "include/**/*.h",
        "include/**/*.hpp",
        "include/**/*.inl",
    ]),
    includes = ["include/"],
    visibility = ["//visibility:public"],
)
