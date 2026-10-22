// SPDX-License-Identifier: Apache-2.0
// Voxtral text decoding and acoustic code generation.
#pragma once

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "common/layers.h"
#include "mlx/mlx.h"
#include "qwen3_asr/qwen3_decoder.h"
#include "voxtral_tts/audio_tokenizer.h"
#include "voxtral_tts/config.h"

namespace voxtral_tts {

struct GeneratedCodes {
  std::vector<std::vector<int>> frames;
  std::string finish_reason;
};

class VoxtralModel {
public:
  explicit VoxtralModel(const VoxtralConfig &config);
  ~VoxtralModel();

  const VoxtralConfig &config() const;
  mlx::core::array LoadVoiceEmbedding(const std::string &voice) const;
  GeneratedCodes Generate(const std::vector<int> &prompt_token_ids,
                          const mlx::core::array &voice_embedding,
                          int max_new_tokens, uint64_t seed,
                          const std::atomic<bool> &cancelled) const;
  std::vector<float> DecodeAudio(const std::vector<std::vector<int>> &frames,
                                 const std::atomic<bool> &cancelled) const;

private:
  struct TextDecoder;
  struct AcousticHead;

  VoxtralConfig config_;
  std::unique_ptr<qwen3_asr::Checkpoint> text_checkpoint_;
  std::unique_ptr<qwen3_asr::Checkpoint> acoustic_checkpoint_;
  std::unique_ptr<AudioTokenizerDecoder> audio_tokenizer_;
  mlx::core::array audio_code_embeddings_;
  std::unique_ptr<TextDecoder> text_decoder_;
  std::unique_ptr<AcousticHead> acoustic_head_;
};

} // namespace voxtral_tts
