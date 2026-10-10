// SPDX-License-Identifier: Apache-2.0
// Qwen3-ASR on MLX: audio encoder, Qwen3 text decoder and checkpoint loading.
#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "audio.h"
#include "layers.h"
#include "mlx/mlx.h"
#include "qwen3_decoder.h"

namespace qwen3_asr {

struct AudioEncoderConfig {
  int num_mel_bins = 0;
  int encoder_layers = 0;
  int encoder_attention_heads = 0;
  int d_model = 0;
  int n_window = 0;
  int n_window_infer = 0;
};

class Qwen3ASR {
public:
  // Builds the model from an MLX checkpoint directory; layers stored with
  // .scales are quantized as the checkpoint was.
  explicit Qwen3ASR(const std::filesystem::path &model_directory);

  // [mel_bins, frames] to [audio_tokens, output_dim].
  mlx::core::array EncodeAudio(const mlx::core::array &mel,
                               AudioLayout layout) const;
  // Token ids [1, length] to embeddings [1, length, hidden].
  mlx::core::array EmbedTokens(const mlx::core::array &ids) const;
  // Logits for the last position only, [vocab].
  mlx::core::array Decode(const mlx::core::array &embeddings,
                          std::vector<KVCache> &caches) const;
  std::vector<KVCache> NewCaches() const;

private:
  Qwen3ASR(const nlohmann::json &config,
           const std::filesystem::path &model_directory);
  mlx::core::array Conv2d(const mlx::core::array &x,
                          const std::string &prefix) const;
  mlx::core::array AudioEncoderLayer(const mlx::core::array &x,
                                     int layer) const;

  AudioEncoderConfig audio_;
  Checkpoint checkpoint_;
  Qwen3Decoder decoder_;
};

} // namespace qwen3_asr
