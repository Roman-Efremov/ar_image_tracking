#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/formats/matrix.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/source_location.h"
// #include "mediapipe/framework/port/status_builder.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "tensorflow/lite/interpreter.h"

namespace mediapipe {

namespace {

constexpr char kFirstFrameTag[] = "FIRST_FRAME";
constexpr char kSecondFrameTag[] = "SECOND_FRAME";
constexpr char kTensorsTag[] = "TENSORS";

} // namespace

class PrepareInputFramesCalculator : public CalculatorBase {
public:
  static absl::Status GetContract(CalculatorContract *cc) {
    if (!cc->Inputs().HasTag(kFirstFrameTag) ||
        !cc->Inputs().HasTag(kSecondFrameTag)) {
      return absl::InvalidArgumentError(
          "Missing required input streams. Both FIRST_FRAME and SECOND_FRAME "
          "must be specified.");
    }
    cc->Inputs().Tag(kFirstFrameTag).Set<ImageFrame>();
    cc->Inputs().Tag(kSecondFrameTag).Set<ImageFrame>();
    cc->Outputs().Tag(kTensorsTag).Set<std::vector<TfLiteTensor>>();

    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext *cc) override {
    cc->SetOffset(TimestampDiff(0));
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext *cc) override;
};
REGISTER_CALCULATOR(PrepareInputFramesCalculator);

absl::Status PrepareInputFramesCalculator::Process(CalculatorContext *cc) {
  const ImageFrame &first_frame =
      cc->Inputs().Tag(kFirstFrameTag).Value().Get<ImageFrame>();
  const ImageFrame &second_frame =
      cc->Inputs().Tag(kSecondFrameTag).Value().Get<ImageFrame>();
  // Convert ImageFrames
  cv::Mat first_mat = mediapipe::formats::MatView(&first_frame);
  cv::Mat second_mat = mediapipe::formats::MatView(&second_frame);
  cv::Mat grey_first_mat, grey_second_mat;
  cv::cvtColor(first_mat, grey_first_mat, cv::COLOR_RGBA2GRAY);
  cv::cvtColor(second_mat, grey_second_mat, cv::COLOR_RGBA2GRAY);
  cv::Mat normalized_first_mat, normalized_second_mat;
  grey_first_mat.convertTo(normalized_first_mat, CV_32F, 1.0 / 255, 0);
  grey_second_mat.convertTo(normalized_second_mat, CV_32F, 1.0 / 255, 0);

  // Make tensor
  // batch = 1, width = 640, height = 480, dims = 2
  // OR INPUT_DIMS    { 1, 2, 480, 640 }
  const int batch = 1;
  const int height = 480;
  const int width = 640;
  const int dims = 2;
  auto tensors = absl::make_unique<std::vector<TfLiteTensor>>();
  TfLiteTensor tensor;
  tensor.type = kTfLiteFloat32;
  tensor.dims = TfLiteIntArrayCreate(4);
  tensor.dims->data[0] = batch;
  tensor.dims->data[1] = height;
  tensor.dims->data[2] = width;
  tensor.dims->data[3] = dims;
  int num_bytes = batch * height * width * dims * sizeof(float);
  tensor.data.data = malloc(num_bytes);
  tensor.bytes = num_bytes;
  tensor.allocation_type = kTfLiteArenaRw;
  float *tensor_buffer = tensor.data.f;
  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      // *tensor_buffer++ = grey_first_mat.at<float>(i, j) / 255;
      // *tensor_buffer++ = grey_second_mat.at<float>(i, j) / 255;
      *tensor_buffer++ = normalized_first_mat.at<float>(i, j);
      *tensor_buffer++ = normalized_second_mat.at<float>(i, j);
    }
  }
  tensors->emplace_back(tensor);
  cc->Outputs().Tag(kTensorsTag).Add(tensors.release(), cc->InputTimestamp());

  return absl::OkStatus();
}

} // namespace mediapipe
