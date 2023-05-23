load(
    "//mediapipe/framework/tool:mediapipe_graph.bzl",
    "mediapipe_simple_subgraph",
)

licenses(["notice"])

package(default_visibility = ["//visibility:public"])

mediapipe_simple_subgraph(
    name = "box_tracking_gpu",
    graph = "box_tracking_gpu.pbtxt",
    register_as = "BoxTrackingSubgraphGpu",
    deps = [
        "//mediapipe/calculators/image:image_transformation_calculator",
        "//mediapipe/calculators/video:box_tracker_calculator",
        "//mediapipe/calculators/video:flow_packager_calculator",
        "//mediapipe/calculators/video:motion_analysis_calculator",
        "//mediapipe/framework/stream_handler:immediate_input_stream_handler",
        "//mediapipe/framework/stream_handler:sync_set_input_stream_handler",
        "//mediapipe/gpu:gpu_buffer_to_image_frame_calculator",
    ],
)

mediapipe_simple_subgraph(
    name = "object_tracking_gpu",
    graph = "object_tracking_gpu.pbtxt",
    register_as = "ObjectTrackingSubgraphGpu",
    deps = [
        # "//mediapipe/framework/stream_handler:sync_set_input_stream_handler",
        "//mediapipe/graphs/ar_image_tracking/subgraphs:box_tracking_gpu",
        "//mediapipe/graphs/ar_image_tracking/calculators:boxes_manager_calculator",
    ],
)

mediapipe_simple_subgraph(
    name = "renderer_gpu",
    graph = "renderer_gpu.pbtxt",
    register_as = "RendererSubgraphGpu",
    deps = [
        "//mediapipe/graphs/ar_image_tracking/calculators:matrices_manager_calculator",
        "//mediapipe/graphs/ar_image_tracking/calculators:gl_animation_overlay_calculator",
        "//mediapipe/calculators/core:packet_cloner_calculator",
        "//mediapipe/graphs/ar_image_tracking/calculators:anchors_to_render_data_calculator",
        "//mediapipe/calculators/util:annotation_overlay_calculator",
    ],
)

mediapipe_simple_subgraph(
    name = "template_matching",
    graph = "template_matching.pbtxt",
    register_as = "TemplateMatchingSubgraph",
    deps = [        
        "//mediapipe/calculators/image:feature_detector_calculator",
        "//mediapipe/calculators/image:image_properties_calculator",
        "//mediapipe/calculators/image:image_transformation_calculator",
        "//mediapipe/calculators/tflite:tflite_converter_calculator", #delete?
        "//mediapipe/calculators/tflite:tflite_inference_calculator",
        "//mediapipe/calculators/tflite:tflite_tensors_to_floats_calculator",
        "//mediapipe/calculators/video:box_detector_calculator",
        "//mediapipe/calculators/core:flow_limiter_calculator",
        "//mediapipe/gpu:gpu_buffer_to_image_frame_calculator",
        "//mediapipe/calculators/core:packet_resampler_calculator",
    ],
)

mediapipe_simple_subgraph(
    name = "depth_detection",
    graph = "depth_detection.pbtxt",
    register_as = "DepthDetectionSubgraph",
    deps = [        
        "//mediapipe/calculators/image:image_transformation_calculator",
        "//mediapipe/calculators/tflite:tflite_inference_calculator",
        "//mediapipe/calculators/core:flow_limiter_calculator",
        "//mediapipe/gpu:gpu_buffer_to_image_frame_calculator",
        "//mediapipe/calculators/tflite:tflite_converter_calculator",        
        # "//mediapipe/graphs/ar_image_tracking/calculators:gl_renderer_calculator",
        "//mediapipe/graphs/ar_image_tracking/calculators:tflite_tensor_to_image_frame_calculator",
        "//mediapipe/graphs/ar_image_tracking/calculators:gl_animation_overlay_calculator",
    ],
)