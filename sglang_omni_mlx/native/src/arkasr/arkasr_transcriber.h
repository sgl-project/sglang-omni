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
#include "common/transcription.h"

namespace arkasr {

struct ArkOptions {
  // Reported with the result; the model detects the language itself.
  std::optional<std::string> language;
  // One budget for the whole request; unset or 0 is the default of 256 tokens
  // per 30 s window, scaled to the request's window count.
  std::optional<int> max_new_tokens;
};

class ArkTranscriber {
public:
  explicit ArkTranscriber(const std::filesystem::path &model_directory);

  omni_common::TranscriptionResult
  Transcribe(const std::vector<float> &samples, const ArkOptions &options,
             const std::atomic<bool> &cancel) const;

private:
  omni_common::TranscriptionResult
  TranscribeWindow(const std::vector<float> &samples, int max_new_tokens,
                   const std::atomic<bool> &cancel) const;
  // text's token ids as embeddings [1, tokens, hidden].
  mlx::core::array PromptEmbeddings(const char *text) const;

  ArkASR model_;
  omni_common::Tokenizer tokenizer_;
  // The prompt's text parts embedded once, [1, tokens, hidden]; each window's
  // prompt is them around its audio features.
  mlx::core::array prefix_embeddings_;
  mlx::core::array suffix_embeddings_;
  int end_of_turn_id_ = 0;
  // Reserved marker ids the checkpoint never bans in generation, dropped from
  // the text as the reference server drops them.
  std::set<int> marker_token_ids_;
};

} // namespace arkasr
