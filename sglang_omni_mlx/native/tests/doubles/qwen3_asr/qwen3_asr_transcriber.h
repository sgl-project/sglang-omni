// SPDX-License-Identifier: Apache-2.0
// Checkpoint-free Qwen3ASRTranscriber double for realtime session tests.
#pragma once

#include <atomic>
#include <optional>
#include <string>
#include <vector>

#include "common/transcription.h"

namespace qwen3_asr {

struct TranscriptionOptions {
  std::optional<std::string> language;
  bool stop_on_token_loop = false;
  std::vector<int> prefix_token_ids;
  std::string prefix_text;
  std::vector<int> guessed_token_ids;
  std::optional<int> max_new_tokens;
};

struct RefreshReuse {};

struct RetainedText {
  std::vector<int> prefix_ids;
  std::string prefix_text;
  std::vector<int> rest_ids;
};

inline std::optional<std::string>
NormalizeLanguage(const std::string &language) {
  if (language.empty()) {
    return std::nullopt;
  }
  return language;
}

class Qwen3ASRTranscriber {
public:
  RetainedText RetainedPrefix(const std::string &, int) const { return {}; }

  omni_common::TranscriptionResult Transcribe(const std::vector<float> &,
                                              const TranscriptionOptions &,
                                              const std::atomic<bool> &,
                                              RefreshReuse * = nullptr) const {
    omni_common::TranscriptionResult result;
    result.text = "ok";
    result.finish_reason = omni_common::FinishReason::kStop;
    return result;
  }
};

} // namespace qwen3_asr
