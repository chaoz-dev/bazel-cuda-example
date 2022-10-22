cc_binary(
    name = "main",
    srcs = ["main.cc"],
    linkopts = [
        "-ldl",
        "-lpthread",
    ],
    deps = [
        "@cuda",  # This is equivalent to "@cuda//:cuda".
    ],
)
