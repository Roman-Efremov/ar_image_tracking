#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/gpu/gpu_buffer.h"

namespace mediapipe {

constexpr char kVideoTag[] = "VIDEO";
constexpr char kOutput0VideoTag[] = "VIDEO_0";
constexpr char kOutput1VideoTag[] = "VIDEO_1";

class PacketSplitterCalculator : public CalculatorBase {
private:
  int counter;

public:
  static absl::Status GetContract(CalculatorContract *cc) {
    cc->Inputs().Tag(kVideoTag).Set<GpuBuffer>();
    cc->Outputs().Tag(kOutput0VideoTag).Set<GpuBuffer>();
    cc->Outputs().Tag(kOutput1VideoTag).Set<GpuBuffer>();

    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext *cc) override {
    cc->SetOffset(TimestampDiff(0));
    counter = 2;
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext *cc) override;
};
REGISTER_CALCULATOR(PacketSplitterCalculator);

absl::Status PacketSplitterCalculator::Process(CalculatorContext *cc) {
  if (counter % 2) {
    cc->Outputs()
        .Tag(kOutput1VideoTag)
        .AddPacket(cc->Inputs().Tag(kVideoTag).Value());
  } else {
    cc->Outputs()
        .Tag(kOutput0VideoTag)
        .AddPacket(cc->Inputs().Tag(kVideoTag).Value());
    counter = 0;
  }
  counter++;

  return absl::OkStatus();
}

} // namespace mediapipe