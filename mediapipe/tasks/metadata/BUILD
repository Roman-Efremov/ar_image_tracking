load("@flatbuffers//:build_defs.bzl", "flatbuffer_cc_library", "flatbuffer_py_library")

package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],  # Apache 2.0
)

exports_files(["metadata_schema.fbs"])

# Generic schema for model metadata.
flatbuffer_cc_library(
    name = "metadata_schema_cc",
    srcs = ["metadata_schema.fbs"],
)

flatbuffer_py_library(
    name = "schema_py",
    srcs = ["@org_tensorflow//tensorflow/lite/schema:schema.fbs"],
)

flatbuffer_py_library(
    name = "metadata_schema_py",
    srcs = ["metadata_schema.fbs"],
)
