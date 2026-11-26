// SPDX-License-Identifier: Apache-2.0
// ARK-ASR on MLX: Whisper encoder with rotary self-attention, frame-merging
// MLP adapter and Qwen2 text decoder, loaded from the checkpoint as published.
#pragma once

#include <filesystem>
#include <string>

#include "common/layers.h"
#include "common/qwen3_decoder.h"
#include "mlx/mlx.h"

namespace arkasr {

struct WhisperEncoderConfig {
  int encoder_layers = 0;
  int encoder_attention_heads = 0;
};

class ArkASR {
public:
  explicit ArkASR(const std::filesystem::path &model_directory);

  // Log-mel [mel_bins, frames] of one clip to audio token embeddings
  // [tokens, hidden].
  mlx::core::array EncodeAudio(const mlx::core::array &mel) const;
  const omni_common::Qwen3Decoder &decoder() const { return decoder_; }

  int audio_token_id() const { return audio_token_id_; }

private:
  ArkASR(const nlohmann::json &config,
         const std::filesystem::path &model_directory);

  WhisperEncoderConfig whisper_;
  int audio_token_id_ = 0;
  int merge_factor_ = 0;
  omni_common::Checkpoint checkpoint_;
  omni_common::Qwen3Decoder decoder_;
};

} // namespace arkasr
