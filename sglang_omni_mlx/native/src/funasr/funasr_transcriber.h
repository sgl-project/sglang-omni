// SPDX-License-Identifier: Apache-2.0
// Fun-ASR-Nano transcription: the reference's chat prompt (language and
// hotwords) over one window of at most 30 s, greedy decoding to <|im_end|>,
// and longer audio cut at its quietest point near each 30 s boundary.
#pragma once

#include <atomic>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "common/tokenizer.h"
#include "common/transcription.h"
#include "funasr/funasr_model.h"

namespace funasr {

struct FunAsrOptions {
  // Reported with the result and named in the prompt: English and other
  // languages are, Chinese and auto leave the model to choose.
  std::optional<std::string> language;
  // Hotwords separated by commas, which the prompt lists as context.
  std::optional<std::string> prompt;
  // One budget for the whole request; unset or 0 is the reference's default
  // for each window's duration.
  std::optional<int> max_new_tokens;
  // Divides the logits of tokens already generated (multiplies them when
  // negative); 1 turns it off.
  float repetition_penalty = 1.0f;
};

class FunAsrTranscriber {
public:
  explicit FunAsrTranscriber(const std::filesystem::path &model_directory);

  omni_common::TranscriptionResult
  Transcribe(const std::vector<float> &samples, const FunAsrOptions &options,
             const std::atomic<bool> &cancel) const;

private:
  // One window decoded after prompt_embeddings [1, tokens, hidden], the
  // prompt before the audio.
  omni_common::TranscriptionResult
  TranscribeWindow(const std::vector<float> &samples,
                   const mlx::core::array &prompt_embeddings,
                   int max_new_tokens, float repetition_penalty,
                   const std::atomic<bool> &cancel) const;
  // text's token ids as embeddings [1, tokens, hidden].
  mlx::core::array PromptEmbeddings(const std::string &text) const;

  FunAsrNano model_;
  omni_common::Tokenizer tokenizer_;
  // The prompt after the audio, embedded once.
  mlx::core::array suffix_embeddings_;
  int im_end_id_ = 0;
};

} // namespace funasr
