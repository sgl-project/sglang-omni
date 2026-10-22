// SPDX-License-Identifier: Apache-2.0
// ARK-ASR transcription: the checkpoint processor's prompt over one window of
// at most 30 s, greedy decoding to <|im_end|>, and longer audio cut at its
// quietest point near each 30 s boundary.
#pragma once

#include <atomic>
#include <filesystem>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include "arkasr/arkasr_model.h"
#include "common/tokenizer.h"
#include "qwen3_asr/qwen3_asr_transcriber.h"

namespace arkasr {

struct ArkOptions {
  // Reported with the result; the model detects the language itself.
  std::optional<std::string> language;
  // Per window; unset is the model's default.
  std::optional<int> max_new_tokens;
};

class ArkTranscriber {
public:
  explicit ArkTranscriber(const std::filesystem::path &model_directory);

  qwen3_asr::TranscriptionResult
  Transcribe(const std::vector<float> &samples, const ArkOptions &options,
             const std::atomic<bool> &cancel) const;
  std::vector<int> PromptIds(int audio_token_count) const;

private:
  qwen3_asr::TranscriptionResult
  TranscribeWindow(const std::vector<float> &samples, int max_new_tokens,
                   const std::atomic<bool> &cancel) const;

  ArkASR model_;
  qwen3_asr::Tokenizer tokenizer_;
  std::vector<int> prompt_prefix_ids_;
  std::vector<int> prompt_suffix_ids_;
  int end_of_turn_id_ = 0;
  // Reserved marker ids the checkpoint never bans in generation, dropped from
  // the text as the reference server drops them.
  std::set<int> marker_token_ids_;
};

} // namespace arkasr
