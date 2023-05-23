load(
    "//mediapipe/framework/tool:mediapipe_graph.bzl",
    "mediapipe_binary_graph",
)

licenses(["notice"])

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "mobile_calculators",
    deps = [  
        "//mediapipe/graphs/ar_image_tracking/calculators:packet_splitter_calculator",
        "//mediapipe/graphs/ar_image_tracking/subgraphs:template_matching",
        "//mediapipe/graphs/ar_image_tracking/subgraphs:object_tracking_gpu",
        "//mediapipe/graphs/ar_image_tracking/subgraphs:depth_detection",
        # "//mediapipe/graphs/ar_image_tracking/subgraphs:renderer_gpu",
        "//mediapipe/graphs/ar_image_tracking/calculators:matrices_manager_calculator",
        "//mediapipe/graphs/ar_image_tracking/calculators:gl_animation_overlay_calculator",
        "//mediapipe/calculators/core:packet_cloner_calculator",
        "//mediapipe/graphs/ar_image_tracking/calculators:anchors_to_render_data_calculator",
        "//mediapipe/calculators/util:annotation_overlay_calculator",
    ],
)

mediapipe_binary_graph(
    name = "ar_image_tracking_gpu_binary_graph",
    graph = "ar_image_tracking.pbtxt",
    output_name = "mobile_gpu.binarypb",
    deps = [":mobile_calculators"],
)