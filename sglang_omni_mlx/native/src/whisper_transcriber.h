// SPDX-License-Identifier: Apache-2.0
// Whisper transcription as Voxt's Swift port runs it: fixed 30 s windows,
// greedy decoding without timestamps, and the checkpoint's suppressed tokens.
#pragma once

#include <atomic>
#include <filesystem>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include "tokenizer.h"
#include "transcriber.h"
#include "whisper_model.h"

namespace whisper {

struct WhisperOptions {
  // An ISO code or an English language name; blank or unknown leaves the
  // language token out.
  std::string language;
  // Zero is the Swift port's default budget; each window keeps within the
  // text context.
  int max_new_tokens = 0;
  // Zero decodes greedily.
  float temperature = 0.0f;
};

class WhisperTranscriber {
public:
  // Reads the checkpoint and the tokenizer files Voxt places beside it:
  // vocab.json, merges.txt, tokenizer_config.json and generation_config.json.
  explicit WhisperTranscriber(const std::filesystem::path &model_directory);

  qwen3_asr::TranscriptionResult
  Transcribe(const std::vector<float> &samples, const WhisperOptions &options,
             const std::atomic<bool> &cancel) const;

private:
  struct WindowResult {
    std::string text;
    int generated_token_count = 0;
    bool reached_token_limit = false;
  };

  // The code of the language token a language names, when there is one.
  std::optional<std::string> LanguageCode(const std::string &language) const;
  // Text of generated ids: special and timestamp ids dropped, byte-level
  // decoded, tokenization spaces cleaned up.
  std::string DecodeText(const std::vector<int> &token_ids) const;
  WindowResult TranscribeWindow(const std::vector<float> &window_samples,
                                const std::vector<int> &prompt_ids,
                                int max_new_tokens, float temperature,
                                const std::atomic<bool> &cancel) const;

  WhisperModel model_;
  qwen3_asr::Tokenizer tokenizer_;
  bool clean_up_tokenization_spaces_ = true;
  bool is_multilingual_ = false;
  int start_of_transcript_id_ = 0;
  int end_of_text_id_ = 0;
  int no_timestamps_id_ = 0;
  int timestamp_begin_id_ = 0;
  std::optional<int> transcribe_id_;
  std::map<std::string, int> language_ids_;
  std::set<int> special_token_ids_;
  // Additive logit masks, each absent when it suppresses nothing: the first
  // step's, every step's, and the timestamps'.
  std::optional<mlx::core::array> begin_suppression_;
  std::optional<mlx::core::array> step_suppression_;
  std::optional<mlx::core::array> timestamp_suppression_;
};

} // namespace whisper
