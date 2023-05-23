#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/graphs/instant_motion_tracking/calculators/transformations.h"
#include "mediapipe/util/render_data.pb.h"
#include "mediapipe/util/tracking/box_tracker.pb.h"

namespace mediapipe {

constexpr char kRenderDataTag[] = "RENDER_DATA";
constexpr char kAnchorsTag[] = "ANCHORS";

class AnchorsToRenderDataCalculator : public CalculatorBase {
private:
public:
  static absl::Status GetContract(CalculatorContract *cc) {
    if (cc->Inputs().HasTag(kAnchorsTag)) {
      cc->Inputs().Tag(kAnchorsTag).Set<std::vector<Anchor>>();
    }
    if (cc->Outputs().HasTag(kRenderDataTag)) {
      cc->Outputs().Tag(kRenderDataTag).Set<std::vector<RenderData>>();
    }

    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext *cc) override { return absl::OkStatus(); }

  absl::Status Process(CalculatorContext *cc) override;
};
REGISTER_CALCULATOR(AnchorsToRenderDataCalculator);

absl::Status AnchorsToRenderDataCalculator::Process(CalculatorContext *cc) {
  std::vector<Anchor> anchors =
      cc->Inputs().Tag(kAnchorsTag).Get<std::vector<Anchor>>();

  std::vector<RenderData> render_data_out;

  for (const Anchor anchor : anchors) {
    RenderData render_data;

    auto *annotation = render_data.add_render_annotations();
    annotation->mutable_color()->set_r(0);
    annotation->mutable_color()->set_g(200);
    annotation->mutable_color()->set_b(100);
    annotation->set_thickness(2);
    RenderAnnotation::Text *text = annotation->mutable_text();
    std::string str = "id=" + std::to_string(anchor.sticker_id) +
                      " distance=" + std::to_string(anchor.z);
    text->set_display_text(str);
    text->set_normalized(true);
    text->set_left(anchor.x - 0.3);
    text->set_baseline(anchor.y / 4);
    text->set_font_height(0.014);

    render_data_out.emplace_back(render_data);
  }

  cc->Outputs()
      .Tag(kRenderDataTag)
      .AddPacket(MakePacket<std::vector<RenderData>>(render_data_out)
                     .At(cc->InputTimestamp()));

  return absl::OkStatus();
}

} // namespace mediapipe