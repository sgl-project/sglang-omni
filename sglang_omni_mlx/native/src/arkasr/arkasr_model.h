// SPDX-License-Identifier: Apache-2.0
// ARK-ASR on MLX: Whisper encoder with rotary self-attention, frame-merging
// MLP adapter and Qwen2 text decoder, loaded from the checkpoint as published.
#pragma once

#include <filesystem>
#include <string>

#include "common/layers.h"
#include "mlx/mlx.h"
#include "qwen3_asr/qwen3_decoder.h"

namespace arkasr {

struct WhisperEncoderConfig {
  int encoder_layers = 0;
  int encoder_attention_heads = 0;
};

class ArkASR {
public:
  explicit ArkASR(const std::filesystem::path &model_directory);

  // Log-mel [mel_bins, frames] of one clip to audio token embeddings
  // [AudioTokenCount(frames), hidden].
  mlx::core::array EncodeAudio(const mlx::core::array &mel) const;
  // The checkpoint processor's count: conv2 halves the frames, the adapter
  // merges merge_factor of them, and at least one token remains.
  int AudioTokenCount(int mel_frame_count) const;
  const qwen3_asr::Qwen3Decoder &decoder() const { return decoder_; }

  int audio_token_id() const { return audio_token_id_; }

private:
  ArkASR(const nlohmann::json &config,
         const std::filesystem::path &model_directory);
  mlx::core::array Conv1d(const mlx::core::array &x, const std::string &prefix,
                          int stride) const;
  mlx::core::array WhisperEncoderLayer(const mlx::core::array &x,
                                       int layer) const;

  WhisperEncoderConfig whisper_;
  int audio_token_id_ = 0;
  int merge_factor_ = 0;
  qwen3_asr::Checkpoint checkpoint_;
  qwen3_asr::Qwen3Decoder decoder_;
};

} // namespace arkasr
