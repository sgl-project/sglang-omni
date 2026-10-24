// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <filesystem>
#include <vector>

#include "common/layers.h"
#include "qwen3_asr/qwen3_decoder.h"

namespace fun_cosyvoice3 {

inline constexpr int kSpeechTokenSize = 6561;
inline constexpr int kTotalSpeechVocabularySize = kSpeechTokenSize + 200;
inline constexpr int kSosTokenId = kSpeechTokenSize;
inline constexpr int kTaskTokenId = kSpeechTokenSize + 2;

class CosyVoice3Model {
public:
  explicit CosyVoice3Model(const std::filesystem::path &model_directory);
  CosyVoice3Model(const CosyVoice3Model &) = delete;
  CosyVoice3Model &operator=(const CosyVoice3Model &) = delete;
  CosyVoice3Model(CosyVoice3Model &&) = delete;
  CosyVoice3Model &operator=(CosyVoice3Model &&) = delete;

  // Text ids include any reference transcript before the target text.
  mlx::core::array
  BuildPromptEmbeddings(const std::vector<int> &text_token_ids,
                        const std::vector<int> &prompt_speech_token_ids) const;
  // Includes the extended control-token vocabulary for teacher forcing.
  mlx::core::array
  SpeechEmbeddings(const std::vector<int> &speech_token_ids) const;
  // Batch-one embeddings; returns [1, 1, 6761] logits and appends to caches.
  mlx::core::array
  ForwardEmbeddings(const mlx::core::array &embeddings,
                    std::vector<qwen3_asr::KVCache> &caches) const;
  std::vector<qwen3_asr::KVCache> NewCaches() const;

private:
  qwen3_asr::Checkpoint checkpoint_;
  qwen3_asr::Qwen3Decoder decoder_;
};

} // namespace fun_cosyvoice3
