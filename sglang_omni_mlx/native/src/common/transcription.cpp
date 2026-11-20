// SPDX-License-Identifier: Apache-2.0
#include "common/transcription.h"

#include <utility>

namespace omni_common {

nlohmann::ordered_json MossPreviewTimingJson(const MossPreviewTiming &timing) {
  return {{"event", "moss_preview_timing"},
          {"window_id", timing.window_id},
          {"kind", timing.kind},
          {"audio_sample_count", timing.audio_sample_count},
          {"append_at_ms", timing.append_at_ms},
          {"decision_at_ms", timing.decision_at_ms},
          {"submit_at_ms", timing.submit_at_ms},
          {"transcribe_started_at_ms", timing.transcribe_started_at_ms},
          {"transcribe_at_ms", timing.transcribe_at_ms},
          {"publish_at_ms", timing.publish_at_ms},
          {"min_preview_samples", timing.min_preview_samples}};
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
