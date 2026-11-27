// SPDX-License-Identifier: Apache-2.0
#include "common/transcription.h"

#include <utility>

#include "common/audio.h"
#include "common/audio_chunks.h"
#include "common/text.h"
#include "mlx/mlx.h"

namespace omni_common {

namespace {

constexpr size_t kWindowSampleCount = 30 * kSampleRate;
constexpr size_t kCutSearchSampleCount = 5 * kSampleRate;
constexpr size_t kCutEnergySampleCount = kSampleRate / 10;
// Note (XXXXRT666): the search stops a second before the audio ends, so the
// last window never becomes a sliver of padding the model transcribes as
// rubbish.
constexpr size_t kMinTailSampleCount = kSampleRate;

} // namespace

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

std::vector<std::pair<size_t, size_t>>
ThirtySecondWindows(const std::vector<float> &samples) {
  return BoundedEnergyCutChunks(samples, kWindowSampleCount,
                                kCutSearchSampleCount, kCutEnergySampleCount,
                                kMinTailSampleCount);
}

TranscriptionResult TranscribeEachWindow(
    const std::vector<float> &samples,
    const std::vector<std::pair<size_t, size_t>> &windows,
    const std::function<int(const std::pair<size_t, size_t> &window,
                            int generated_token_count)> &max_new_tokens,
    const std::function<TranscriptionResult(
        const std::vector<float> &window_samples, int max_new_tokens)>
        &transcribe_window) {
  TranscriptionResult combined;
  combined.finish_reason = FinishReason::kStop;
  std::vector<std::string> texts;
  for (const std::pair<size_t, size_t> &window : windows) {
    const int window_max_new_tokens =
        max_new_tokens(window, combined.generated_token_count);
    if (window_max_new_tokens < 1) {
      combined.finish_reason = FinishReason::kLength;
      break;
    } else {
    }
    const TranscriptionResult result = transcribe_window(
        {samples.begin() + window.first, samples.begin() + window.second},
        window_max_new_tokens);
    texts.push_back(result.text);
    combined.generated_token_count += result.generated_token_count;
    combined.verify_round_count += result.verify_round_count;
    combined.proposed_draft_token_count += result.proposed_draft_token_count;
    combined.accepted_draft_token_count += result.accepted_draft_token_count;
    if (result.finish_reason == FinishReason::kLength) {
      combined.finish_reason = FinishReason::kLength;
    } else {
    }
    mlx::core::clear_cache();
  }
  combined.text = JoinTranscriptParts(texts);
  if (windows.size() > 1) {
    combined.window_texts = std::move(texts);
  } else {
  }
  return combined;
}

} // namespace omni_common
