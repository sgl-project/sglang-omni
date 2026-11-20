// SPDX-License-Identifier: Apache-2.0
// Model-independent transcription results and their JSON form.
#pragma once

#include <chrono>
#include <optional>
#include <string>
#include <vector>

#include "common/generation.h"
#include "nlohmann/json.hpp"

namespace omni_common {

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
  // Rounds of draft verification, and the draft proposals the target checked
  // and accepted; all 0 when decoding without a draft.
  int verify_round_count = 0;
  int proposed_draft_token_count = 0;
  int accepted_draft_token_count = 0;
  // Ordered text of each window in a multi-window Whisper request.
  std::vector<std::string> window_texts;
};

// Realtime preview timing without prompt, transcript or speaker content.
// The *_at_ms fields are absolute steady_clock milliseconds, not stage
// durations: a stage's cost is the difference of two consecutive fields
// (e.g. publish_at_ms - transcribe_at_ms is the publish cost).
struct MossPreviewTiming {
  int window_id = 0;
  std::string kind = "preview";
  int audio_sample_count = 0;
  int64_t append_at_ms = 0;
  int64_t decision_at_ms = 0;
  int64_t submit_at_ms = 0;
  int64_t transcribe_at_ms = 0;
  int64_t publish_at_ms = 0;
};

nlohmann::ordered_json MossPreviewTimingJson(const MossPreviewTiming &timing);

// Segments as the HTTP and realtime APIs send them: start, end, speaker, text,
// and language, emotion and event when set.
nlohmann::ordered_json
SpeakerSegmentsJson(const std::vector<SpeakerSegment> &segments);

} // namespace omni_common
