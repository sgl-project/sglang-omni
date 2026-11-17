// SPDX-License-Identifier: Apache-2.0
// Model-independent transcription results and their JSON form.
#pragma once

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
  // Existing synchronization boundaries only; no extra MLX evaluation is added.
  double native_split_ms = 0.0;
  double frontend_build_ms = 0.0;
  double frontend_eval_ms = 0.0;
  double decoder_prefill_ms = 0.0;
  double autoregressive_decode_ms = 0.0;
  double result_rebuild_ms = 0.0;
  double cache_clear_ms = 0.0;
  double native_total_ms = 0.0;
};

// Segments as the HTTP and realtime APIs send them: start, end, speaker, text,
// and language, emotion and event when set.
nlohmann::ordered_json
SpeakerSegmentsJson(const std::vector<SpeakerSegment> &segments);

} // namespace omni_common
