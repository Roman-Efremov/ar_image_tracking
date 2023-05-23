#include <utility> // for declval

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_format.pb.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/opencv_core_inc.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "tensorflow/lite/interpreter.h"

namespace mediapipe {

namespace {

constexpr char kTensorsTag[] = "TENSORS";
constexpr char kImageTag[] = "IMAGE";

} // namespace

class TfLiteTensorToImageFrameCalculator : public CalculatorBase {
public:
  static absl::Status GetContract(CalculatorContract *cc);
  absl::Status Open(CalculatorContext *cc) override;
  absl::Status Process(CalculatorContext *cc) override;

private:
};
REGISTER_CALCULATOR(TfLiteTensorToImageFrameCalculator);

absl::Status
TfLiteTensorToImageFrameCalculator::GetContract(CalculatorContract *cc) {
  cc->Inputs().Tag(kTensorsTag).Set<std::vector<TfLiteTensor>>();
  cc->Outputs().Tag(kImageTag).Set<ImageFrame>();

  return absl::OkStatus();
}

absl::Status TfLiteTensorToImageFrameCalculator::Open(CalculatorContext *cc) {
  cc->SetOffset(TimestampDiff(0));
  return absl::OkStatus();
}

absl::Status
TfLiteTensorToImageFrameCalculator::Process(CalculatorContext *cc) {
  const std::vector<TfLiteTensor> &tensors =
      cc->Inputs().Tag(kTensorsTag).Get<std::vector<TfLiteTensor>>();

  const int width = 256;
  const int height = 256;
  const TfLiteTensor *raw_tensor = &tensors[0];
  const float *raw_floats = raw_tensor->data.f;
  auto depth_floats = absl::make_unique<std::vector<float>>(
      raw_floats, raw_floats + width * height);
  cv::Mat depth_mat = cv::Mat(height, width, CV_32F);
  memcpy(depth_mat.data, depth_floats->data(),
         depth_floats->size() * sizeof(float));

  // double min, max;
  // cv::minMaxLoc(depth_mat, &min, &max);

  // phone: width: 720 height: 1280
  auto depth_image =
      absl::make_unique<ImageFrame>(ImageFormat::VEC32F1, height, width);
  cv::Mat depth_image_mat = formats::MatView(depth_image.get());
  // depth_mat.copyTo(depth_image_mat);
  cv::normalize(depth_mat, depth_image_mat, 0.3, 1.0, cv::NORM_MINMAX);
  // depth_mat.convertTo(depth_image_mat, CV_16U);
  // cv::resize(depth_mat.data, dst, dsize, 0, 0, cv::INTER_LINEAR);

  LOG(INFO) << "Depth matrix after copy";
  std::string my_str = "";
  for (int i = 0; i < height; i += 16) {
    for (int j = 0; j < width; j += 16) {
      my_str += std::to_string(depth_image_mat.at<float>(i, j)) + " ";
    }
    LOG(INFO) << my_str;
    my_str = "";
  }

  // LOG(INFO) << "Depth center"
  // <<std::to_string(depth_image_mat.at<float>(height / 2, width / 2));

  cc->Outputs().Tag(kImageTag).Add(depth_image.release(), cc->InputTimestamp());

  return absl::OkStatus();
}

} // namespace mediapipe