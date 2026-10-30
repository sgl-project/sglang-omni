// SPDX-License-Identifier: Apache-2.0
// Qwen3-ASR on MLX: audio encoder, Qwen3 text decoder and checkpoint loading.
#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "common/layers.h"
#include "mlx/mlx.h"
#include "qwen3_asr/qwen3_asr_decoder.h"
#include "qwen3_asr/qwen3_audio.h"

namespace qwen3_asr {

// Consecutive attention windows of one length: the first one's first token,
// the window count and tokens per window.
struct WindowRun {
  int start = 0;
  int windows = 0;
  int length = 0;
};

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
  // Logits for the last rows positions, [rows, vocab].
  mlx::core::array Decode(const mlx::core::array &embeddings,
                          std::vector<Qwen3ASRKVCache> &caches, int rows) const;
  std::vector<Qwen3ASRKVCache> NewCaches() const;
  const omni_common::WeightMap &Weights() const {
    return checkpoint_.Weights();
  }

private:
  Qwen3ASR(const nlohmann::json &config,
           const std::filesystem::path &model_directory);
  // One layer over tokens [tokens, d_model], attention within the windows of
  // runs.
  mlx::core::array AudioEncoderLayer(const mlx::core::array &x, int layer,
                                     const std::vector<WindowRun> &runs) const;

  AudioEncoderConfig audio_;
  omni_common::Checkpoint checkpoint_;
  Qwen3ASRDecoder decoder_;
};

} // namespace qwen3_asr
