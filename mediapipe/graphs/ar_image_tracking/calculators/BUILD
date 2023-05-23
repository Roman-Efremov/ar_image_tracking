load("//mediapipe/framework/port:build_config.bzl", "mediapipe_proto_library")

licenses(["notice"])

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "boxes_manager_calculator",
    srcs = ["boxes_manager_calculator.cc"],
    hdrs = ["//mediapipe/graphs/instant_motion_tracking/calculators:transformations.h"],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "//mediapipe/util/tracking:box_tracker_cc_proto",
    ],
    alwayslink = 1,
)

cc_library(
    name = "prepare_input_frames_calculator",
    srcs = ["prepare_input_frames_calculator.cc"],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:matrix",
        "//mediapipe/framework/formats:image_frame_opencv",
        # "//mediapipe/framework/port:logging",
        # "//mediapipe/framework/port:opencv_core",
        # "//mediapipe/framework/port:opencv_features2d",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:source_location",
        # "//mediapipe/framework/port:status",
        "@org_tensorflow//tensorflow/lite:framework",
    ],
    alwayslink = 1,
)

cc_library(
    name = "tflite_tensor_to_image_frame_calculator",
    srcs = ["tflite_tensor_to_image_frame_calculator.cc"],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/port:opencv_core",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "@org_tensorflow//tensorflow/lite:framework",
        "//mediapipe/framework/formats:image_format_cc_proto",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:image_frame_opencv",
    ],
    alwayslink = 1,
)

cc_library(
    name = "gl_animation_overlay_calculator",
    srcs = ["gl_animation_overlay_calculator.cc"],
    hdrs = ["//mediapipe/graphs/instant_motion_tracking/calculators:transformations.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/graphs/object_detection_3d/calculators:gl_animation_overlay_calculator_cc_proto",
        "//mediapipe/graphs/object_detection_3d/calculators:model_matrix_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "//mediapipe/gpu:gl_calculator_helper",
        "//mediapipe/gpu:shader_util",
        "//mediapipe/modules/objectron/calculators:camera_parameters_cc_proto",
        "//mediapipe/util/android:asset_manager_util",
        "//mediapipe/framework/formats:image_frame_opencv",
    ],
    alwayslink = 1,
)

cc_library(
    name = "packet_splitter_calculator",
    srcs = ["packet_splitter_calculator.cc"],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "//mediapipe/gpu:gpu_buffer"
    ],
    alwayslink = 1,
)

cc_library(
    name = "matrices_manager_calculator",
    srcs = ["matrices_manager_calculator.cc"],
    hdrs = ["//mediapipe/graphs/instant_motion_tracking/calculators:transformations.h"],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:timestamp",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "//mediapipe/graphs/object_detection_3d/calculators:model_matrix_cc_proto",
        "//mediapipe/modules/objectron/calculators:box",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@eigen_archive//:eigen3",
    ],
    alwayslink = 1,
)

cc_library(
    name = "anchors_to_render_data_calculator",
    srcs = ["anchors_to_render_data_calculator.cc"],
    hdrs = ["//mediapipe/graphs/instant_motion_tracking/calculators:transformations.h"],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "//mediapipe/util/tracking:box_tracker_cc_proto",
        "//mediapipe/util:render_data_cc_proto",
    ],
    alwayslink = 1,
)