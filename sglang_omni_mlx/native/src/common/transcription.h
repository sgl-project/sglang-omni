// SPDX-License-Identifier: Apache-2.0
// Model-independent transcription results, their JSON form, and long audio
// decoded one 30 s window at a time.
#pragma once

#include <chrono>
#include <cstddef>
#include <functional>
#include <optional>
#include <string>
#include <utility>
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
  double prompt_embeddings_ms = 0.0;
  double prefill_ms = 0.0;
  double decode_rounds_ms = 0.0;
  double chunk_result_ms = 0.0;
  // Ordered text of the windows of a request cut into several.
  std::vector<std::string> window_texts;
  // Audio tokens a live session had already encoded.
  int reused_audio_token_count = 0;
  // Prompt tokens a live session had already computed.
  int reused_prompt_token_count = 0;
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
  int64_t transcribe_started_at_ms = 0;
  int64_t transcribe_at_ms = 0;
  int64_t publish_at_ms = 0;
  int min_preview_samples = 0;
  int verify_round_count = 0;
  int proposed_draft_token_count = 0;
  int accepted_draft_token_count = 0;
  double prompt_embeddings_ms = 0.0;
  double prefill_ms = 0.0;
  double decode_rounds_ms = 0.0;
  double chunk_result_ms = 0.0;
};

// Offline Final timing without prompt, transcript or speaker content. Each
// window is one chunk of the full request, not a realtime preview window.
struct MossOfflineFinalTiming {
  int window_id = 0;
  int audio_sample_count = 0;
  int64_t transcribe_started_at_ms = 0;
  int64_t transcribe_at_ms = 0;
  int verify_round_count = 0;
  int proposed_draft_token_count = 0;
  int accepted_draft_token_count = 0;
  double prompt_embeddings_ms = 0.0;
  double prefill_ms = 0.0;
  double decode_rounds_ms = 0.0;
  double chunk_result_ms = 0.0;
};

nlohmann::ordered_json MossPreviewTimingJson(const MossPreviewTiming &timing);
nlohmann::ordered_json
MossOfflineFinalTimingJson(const MossOfflineFinalTiming &timing);

// Segments as the HTTP and realtime APIs send them: start, end, speaker, text,
// and language, emotion and event when set.
nlohmann::ordered_json
SpeakerSegmentsJson(const std::vector<SpeakerSegment> &segments);

// Sample ranges of at most 30 s that cover samples, each cut at the quietest
// 0.1 s in the 5 s before its limit, the last at least 1 s long; one range
// for shorter audio.
std::vector<std::pair<size_t, size_t>>
ThirtySecondWindows(const std::vector<float> &samples);

// samples transcribed one window at a time and the texts joined with
// JoinTranscriptParts, token and draft counts summed. max_new_tokens gives a
// window's budget from the tokens the windows before it generated; below 1,
// the request stops there. Truncation by either ends it with kLength, and
// window_texts lists every window's text when there are several.
TranscriptionResult TranscribeEachWindow(
    const std::vector<float> &samples,
    const std::vector<std::pair<size_t, size_t>> &windows,
    const std::function<int(const std::pair<size_t, size_t> &window,
                            int generated_token_count)> &max_new_tokens,
    const std::function<TranscriptionResult(
        const std::vector<float> &window_samples, int max_new_tokens)>
        &transcribe_window);

} // namespace omni_common
