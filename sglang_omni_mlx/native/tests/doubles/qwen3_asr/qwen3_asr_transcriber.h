// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <atomic>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "common/audio.h"
#include "common/transcription.h"

namespace qwen3_asr {

struct TranscriptionOptions {
  std::optional<std::string> language;
  std::optional<std::string> context;
  std::optional<int> max_new_tokens;
  bool stop_at_end_of_text = false;
  bool stop_on_token_loop = false;
  std::vector<int> prefix_token_ids;
  std::string prefix_text;
  std::vector<int> guessed_token_ids;
};

struct RetainedText {
  std::vector<int> prefix_ids;
  std::string prefix_text;
  std::vector<int> rest_ids;
};

struct RefreshReuse {
  std::vector<int> head_ids;
};

inline constexpr size_t kMaxChunkSamples = 120 * omni_common::kSampleRate;
inline constexpr float kSilentPeak = 1e-3f;

inline std::optional<std::string>
NormalizeLanguage(const std::string &language) {
  return language.empty() ? std::nullopt : std::optional<std::string>(language);
}

class Qwen3ASRTranscriber {
public:
  using Decoder = std::function<omni_common::TranscriptionResult(
      const std::vector<float> &, const std::atomic<bool> &, RefreshReuse *)>;

  Decoder decoder;

  omni_common::TranscriptionResult
  Transcribe(const std::vector<float> &samples, const TranscriptionOptions &,
             const std::atomic<bool> &cancel,
             RefreshReuse *reuse = nullptr) const {
    return decoder ? decoder(samples, cancel, reuse)
                   : omni_common::TranscriptionResult{};
  }

  RetainedText RetainedPrefix(const std::string &, int) const { return {}; }

  std::vector<int> AutoLanguageGuesses(const std::string &,
                                       const std::string &) const {
    return {};
  }

  bool ExtendFinalPrefill(const RefreshReuse &, const std::vector<float> &,
                          long, const TranscriptionOptions &,
                          const std::atomic<bool> &, RefreshReuse &) const {
    return false;
  }
};

} // namespace qwen3_asr
