// SPDX-License-Identifier: Apache-2.0
// Model-independent transcription results and cancellation.
#pragma once

#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"

namespace omni_common {

enum class FinishReason { kStop, kLength };

const char *FinishReasonName(FinishReason reason);

struct SpeakerSegment {
  double start_seconds = 0.0;
  double end_seconds = 0.0;
  // Empty when the output carried no speaker segments.
  std::string speaker;
  std::string text;
  // Note (Dayuxiaoshui): SenseVoice's language, emotion and audio event of
  // the segment's audio; none for other models.
  std::optional<std::string> language;
  std::optional<std::string> emotion;
  std::optional<std::string> event;
};

struct TranscriptionResult {
  std::string text;
  std::optional<std::string> language;
  int generated_token_count = 0;
  FinishReason finish_reason = FinishReason::kLength;
  // Timestamped speaker segments, for models that produce them.
  std::vector<SpeakerSegment> segments;
  // Rounds of draft verification; 0 when decoding without a draft.
  int verify_round_count = 0;
};

class TranscriptionCancelled : public std::runtime_error {
public:
  TranscriptionCancelled() : std::runtime_error("transcription cancelled") {}
};

// Segments as the HTTP and realtime APIs send them: start, end, speaker, text,
// and language, emotion and event when set.
nlohmann::ordered_json
SpeakerSegmentsJson(const std::vector<SpeakerSegment> &segments);

} // namespace omni_common
