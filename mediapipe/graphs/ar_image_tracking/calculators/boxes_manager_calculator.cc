#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/graphs/instant_motion_tracking/calculators/transformations.h"
#include "mediapipe/util/tracking/box_tracker.pb.h"

namespace mediapipe {

// constexpr char kBoxesInputTag[] = "BOXES";
// See if box tracking returns duplicate boxes locations for new inputs from
// KNIFT model. If so use object_tracking_v2.
constexpr char kTrackingBoxesTag[] = "TRACKING_BOXES";
constexpr char kAnchorsTag[] = "ANCHORS";
constexpr char kRotationsTag[] = "USER_ROTATIONS";
constexpr char kScalingsTag[] = "USER_SCALINGS";
constexpr char kCancelTag[] = "CANCEL_ID";
constexpr char kRenderDataTag[] = "RENDER_DATA";

constexpr float kBoxEdgeSize =
    0.5f; // Used to establish tracking box dimensions
constexpr float kUsToMs =
    1000.0f; // Used to convert from microseconds to millis

// This calculator converts Boxes to Anchors with z axis scaling.
// Calculates user scalings and user rotations.
// ? Removes duplicate detections.
// Keeps tracking over lost boxes for specified amount of time.
// Render data: 0 - gif, 1 - 3d model.

// Example config:
// node {
//   calculator: "BoxesManagerCalculator"
//   input_stream: "BOXES:start_pos"
//   input_stream: "TRACKING_BOXES:boxes"
//   output_stream: "CANCEL_OBJECT_ID:cancel_object_id"
//   output_stream: "ANCHORS:tracked_anchor_data"
//   output_stream: "USER_ROTATIONS:user_rotation_data"
//   output_stream: "USER_SCALINGS:user_scaling_data"
//   output_stream: "RENDER_DATA:render_data"
// }

class BoxesManagerCalculator : public CalculatorBase {
private:
  bool isEmpty;

public:
  static absl::Status GetContract(CalculatorContract *cc) {
    if (cc->Inputs().HasTag(kTrackingBoxesTag)) {
      cc->Inputs().Tag(kTrackingBoxesTag).Set<mediapipe::TimedBoxProtoList>();
    }

    if (cc->Outputs().HasTag(kAnchorsTag)) {
      cc->Outputs().Tag(kAnchorsTag).Set<std::vector<Anchor>>();
    }
    if (cc->Outputs().HasTag(kRotationsTag)) {
      cc->Outputs().Tag(kRotationsTag).Set<std::vector<UserRotation>>();
    }
    if (cc->Outputs().HasTag(kScalingsTag)) {
      cc->Outputs().Tag(kScalingsTag).Set<std::vector<UserScaling>>();
    }
    if (cc->Outputs().HasTag(kRenderDataTag)) {
      cc->Outputs().Tag(kRenderDataTag).Set<std::vector<int>>();
    }
    if (cc->Outputs().HasTag(kCancelTag)) {
      cc->Outputs().Tag(kCancelTag).Set<int>();
    }

    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext *cc) override {
    isEmpty = true;
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext *cc) override;
};
REGISTER_CALCULATOR(BoxesManagerCalculator);

absl::Status BoxesManagerCalculator::Process(CalculatorContext *cc) {
  std::vector<Anchor> current_anchor_data;
  std::vector<UserRotation> current_user_rotation;
  std::vector<UserScaling> current_user_scaling;
  std::vector<int> render_data;

  mediapipe::TimedBoxProtoList tracked_boxes =
      cc->Inputs().Tag(kTrackingBoxesTag).Get<mediapipe::TimedBoxProtoList>();

  for (const mediapipe::TimedBoxProto &box : tracked_boxes.box()) {
    Anchor anchor;
    UserRotation rotation;
    UserScaling scaling;

    anchor.sticker_id = box.id();
    anchor.x = (box.left() + box.right()) * 0.5f;
    anchor.y = (box.top() + box.bottom()) * 0.5f;
    anchor.z = 30.0f * kBoxEdgeSize / (box.right() - box.left());
    LOG(INFO) << "x= " << std::to_string(anchor.x)
              << " y= " << std::to_string(anchor.y)
              << " z= " << std::to_string(anchor.z);

    rotation.sticker_id = box.id();
    rotation.rotation_radians = box.rotation();
    LOG(INFO) << "rot= " << std::to_string(rotation.rotation_radians);

    scaling.sticker_id = box.id();
    scaling.scale_factor = 1.0f;

    current_anchor_data.emplace_back(anchor);
    current_user_rotation.emplace_back(rotation);
    current_user_scaling.emplace_back(scaling);
    render_data.emplace_back(1);

    isEmpty = false;
    LOG(INFO) << "box with id: " << box.id() << "was converted";
  }
  LOG(INFO) << "-- end of loop --";

  if (isEmpty) {
    Anchor anchor;
    UserRotation rotation;
    UserScaling scaling;

    anchor.sticker_id = -1;
    anchor.x = 3.0f;
    anchor.y = 3.0f;
    anchor.z = -30.0f;

    rotation.sticker_id = -1;
    rotation.rotation_radians = 0.0f;

    scaling.sticker_id = -1;
    scaling.scale_factor = 0.1f;

    current_anchor_data.emplace_back(anchor);
    current_user_rotation.emplace_back(rotation);
    current_user_scaling.emplace_back(scaling);
    render_data.emplace_back(1);

    LOG(WARNING) << "No detection boxes, return default";
  }

  if (false && cc->Outputs().HasTag(kCancelTag)) {
    auto timestamp = cc->InputTimestamp();
    cc->Outputs()
        .Tag(kCancelTag)
        .AddPacket(mediapipe::MakePacket<int>(4).At(timestamp++));
  }

  // if (!isEmpty) {
  cc->Outputs()
      .Tag(kAnchorsTag)
      .AddPacket(MakePacket<std::vector<Anchor>>(current_anchor_data)
                     .At(cc->InputTimestamp()));
  cc->Outputs()
      .Tag(kRotationsTag)
      .AddPacket(MakePacket<std::vector<UserRotation>>(current_user_rotation)
                     .At(cc->InputTimestamp()));
  cc->Outputs()
      .Tag(kScalingsTag)
      .AddPacket(MakePacket<std::vector<UserScaling>>(current_user_scaling)
                     .At(cc->InputTimestamp()));
  cc->Outputs()
      .Tag(kRenderDataTag)
      .AddPacket(
          MakePacket<std::vector<int>>(render_data).At(cc->InputTimestamp()));

  isEmpty = true;
  // }

  return absl::OkStatus();
}

} // namespace mediapipe