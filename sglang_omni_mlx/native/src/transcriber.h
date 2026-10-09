// SPDX-License-Identifier: Apache-2.0
// Qwen3-ASR transcription: prompt, greedy decoding and stop rules.
#pragma once

#include <atomic>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "audio.h"
#include "model.h"
#include "tokenizer.h"

namespace qwen3_asr {

enum class FinishReason { kStop, kLength };

const char *FinishReasonName(FinishReason reason);

struct TranscriptionOptions {
  std::optional<std::string> language;
  std::optional<std::string> context;
  std::optional<int> max_new_tokens;
  bool stop_at_end_of_text = false;
  bool stop_on_token_loop = false;
  AudioLayout layout = AudioLayout::kReference;
  // Realtime refreshes continue from text already shown, as ids and text.
  std::vector<int> prefix_token_ids;
  std::string prefix_text;
};

struct TranscriptionResult {
  std::string text;
  std::optional<std::string> language;
  int generated_token_count = 0;
  FinishReason finish_reason = FinishReason::kLength;
};

class TranscriptionCancelled : public std::runtime_error {
public:
  TranscriptionCancelled() : std::runtime_error("transcription cancelled") {}
};

// The canonical prompt name for a Qwen3-ASR language code or name. Like
// Voxt's Swift port, a name outside the table is used as given; blank is none.
std::optional<std::string> NormalizeLanguage(const std::string &language);

class Qwen3ASRTranscriber {
public:
  explicit Qwen3ASRTranscriber(const std::filesystem::path &model_directory);

  std::vector<int> PromptIds(int audio_token_count,
                             const TranscriptionOptions &options) const;
  // Text already shown minus its last tokens, cut back to whole characters.
  std::pair<std::vector<int>, std::string>
  RetainedPrefix(const std::string &text, int rollback_token_count) const;
  TranscriptionResult Transcribe(const std::vector<float> &samples,
                                 const TranscriptionOptions &options,
                                 const std::atomic<bool> &cancel) const;

private:
  std::pair<std::string, std::optional<std::string>>
  SplitOutput(const std::vector<int> &output_ids,
              const TranscriptionOptions &options) const;

  Qwen3ASR model_;
  Tokenizer tokenizer_;
  int audio_pad_id_;
  int end_of_text_id_;
  int im_end_id_;
  std::vector<int> asr_text_ids_;
};

} // namespace qwen3_asr
