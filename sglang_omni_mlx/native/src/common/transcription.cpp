// SPDX-License-Identifier: Apache-2.0
#include "common/transcription.h"

#include <utility>

namespace omni_common {

nlohmann::ordered_json MossPreviewTimingJson(const MossPreviewTiming &timing) {
  return {{"event", "moss_preview_timing"},
          {"window_id", timing.window_id},
          {"kind", timing.kind},
          {"audio_sample_count", timing.audio_sample_count},
          {"append_ms", timing.append_ms},
          {"decision_ms", timing.decision_ms},
          {"submit_ms", timing.submit_ms},
          {"transcribe_ms", timing.transcribe_ms},
          {"publish_ms", timing.publish_ms}};
}

nlohmann::ordered_json
SpeakerSegmentsJson(const std::vector<SpeakerSegment> &segments) {
  nlohmann::ordered_json array = nlohmann::ordered_json::array();
  for (const SpeakerSegment &segment : segments) {
    nlohmann::ordered_json json = {{"start", segment.start_seconds},
                                   {"end", segment.end_seconds},
                                   {"speaker", segment.speaker},
                                   {"text", segment.text}};
    for (const auto &[name, value] : {std::pair{"language", &segment.language},
                                      std::pair{"emotion", &segment.emotion},
                                      std::pair{"event", &segment.event}}) {
      if (value->has_value()) {
        json[name] = **value;
      } else {
      }
    }
    array.push_back(std::move(json));
  }
  return array;
}

} // namespace omni_common
