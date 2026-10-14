// SPDX-License-Identifier: Apache-2.0
// MOSS-Transcribe-Diarize transcription: prompt with time markers, greedy
// decoding, and timestamped speaker segments, as Voxt's Swift port runs it.
#pragma once

#include <atomic>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "moss_model.h"
#include "tokenizer.h"
#include "transcriber.h"

namespace moss {

struct MossOptions {
  // The task instruction; blank is the model's default diarization prompt.
  std::optional<std::string> prompt;
  // Note (Dayuxiaoshui): unset or zero is Voxt's final-pass budget for the
  // audio's duration, applied per chunk.
  std::optional<int> max_new_tokens;
  // Note (Dayuxiaoshui): decoding always stops at <|im_end|>; these add the
  // checks Voxt's Swift port always applied.
  bool stop_at_end_of_text = false;
  bool stop_on_token_loop = false;
  // Realtime: decode the samples as one window that starts this far into the
  // session, without chunking or padding.
  std::optional<double> window_offset_seconds;
};

class MossTranscriber {
public:
  explicit MossTranscriber(const std::filesystem::path &model_directory);

  qwen3_asr::TranscriptionResult
  Transcribe(const std::vector<float> &samples, const MossOptions &options,
             const std::atomic<bool> &cancel) const;
  std::vector<int> PromptIds(int audio_token_count,
                             const std::optional<std::string> &prompt) const;

private:
  qwen3_asr::TranscriptionResult
  TranscribeChunk(const std::vector<float> &samples, const MossOptions &options,
                  int max_new_tokens, double offset_seconds,
                  const std::atomic<bool> &cancel) const;
  std::vector<int> AudioSpanIds(int audio_token_count) const;

  MossTranscribeDiarize model_;
  qwen3_asr::Tokenizer tokenizer_;
  std::vector<int> digit_token_ids_;
  int end_of_text_id_ = 0;
  int im_end_id_ = 0;
  float audio_tokens_per_second_ = 12.5f;
  int time_marker_every_seconds_ = 5;
  bool enable_time_marker_ = true;
};

// Throws std::invalid_argument for a prompt with more than one audio
// placeholder, so the server can reject it before decoding.
void ValidatePrompt(const std::optional<std::string> &prompt);

// Timestamp tags [12.34] shifted by offset_seconds, printed with 2 decimals.
std::string OffsetTimestampTags(const std::string &text, double offset_seconds);
// [start][Sxx]text[end] segments; one unlabelled segment spanning the audio
// when none parse.
std::vector<qwen3_asr::SpeakerSegment> ParseSegments(const std::string &text,
                                                     double duration_seconds,
                                                     double offset_seconds);

} // namespace moss
