// SPDX-License-Identifier: Apache-2.0
// Cohere Transcribe on MLX: checkpoint loading, the log-mel front end, the
// Conformer encoder and the Transformer decoder, op for op as Voxt's Swift
// port runs them.
#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "mlx/mlx.h"

namespace cohere_transcribe {

struct CohereConfig {
  int encoder_width = 0;
  int encoder_head_count = 0;
  int encoder_layer_count = 0;
  int convolution_kernel_size = 0;
  int relative_position_count = 0;
  int mel_bin_count = 0;
  int decoder_width = 0;
  int decoder_head_count = 0;
  int decoder_layer_count = 0;
  int max_sequence_length = 0;
};

// Self-attention keys and values grow by one token per step; cross-attention
// keys and values are projected once from the encoder output.
struct DecoderLayerCache {
  std::optional<mlx::core::array> self_keys;
  std::optional<mlx::core::array> self_values;
  std::optional<mlx::core::array> cross_keys;
  std::optional<mlx::core::array> cross_values;
};

class CohereModel {
public:
  // Reads config.json and model.safetensors of an MLX Cohere Transcribe
  // checkpoint.
  explicit CohereModel(const std::filesystem::path &model_directory);

  const CohereConfig &config() const { return config_; }
  // 16 kHz samples to encoder states projected to the decoder width [1,
  // frames, width].
  mlx::core::array Encode(const std::vector<float> &samples) const;
  // Token ids [1, length] placed from start_position on, to the logits of the
  // last position [vocabulary].
  mlx::core::array Decode(const mlx::core::array &token_ids, int start_position,
                          const mlx::core::array &encoder_states,
                          std::vector<DecoderLayerCache> &caches) const;

private:
  const mlx::core::array &Weight(const std::string &name) const;
  mlx::core::array Linear(const mlx::core::array &x,
                          const std::string &prefix) const;
  mlx::core::array LayerNorm(const mlx::core::array &x,
                             const std::string &prefix) const;
  mlx::core::array Conv1d(const mlx::core::array &x, const std::string &prefix,
                          int padding, int groups) const;
  mlx::core::array Conv2d(const mlx::core::array &x, const std::string &prefix,
                          int stride, int padding, int groups) const;
  mlx::core::array RelativePositions(int length, mlx::core::Dtype dtype) const;
  mlx::core::array EncoderSelfAttention(const mlx::core::array &x,
                                        const mlx::core::array &positions,
                                        const std::string &prefix) const;
  mlx::core::array EncoderConvolution(const mlx::core::array &x,
                                      const std::string &prefix) const;
  mlx::core::array EncoderLayer(const mlx::core::array &x,
                                const mlx::core::array &positions,
                                int layer) const;
  mlx::core::array
  DecoderAttention(const mlx::core::array &queries_input,
                   const mlx::core::array &keys, const mlx::core::array &values,
                   const std::string &prefix,
                   const std::optional<mlx::core::array> &mask) const;

  CohereConfig config_;
  // The log-mel front end's centered Hann window [512] and mel filters
  // [257, mel_bins].
  mlx::core::array hann_window_;
  mlx::core::array mel_filters_;
  // The encoder's relative positions [1, 2 * relative_position_count - 1,
  // width], from relative_position_count - 1 down.
  mlx::core::array relative_position_table_;
  // The decoder's fixed sinusoidal positions [max_sequence_length, width].
  mlx::core::array decoder_positions_;
  std::unordered_map<std::string, mlx::core::array> weights_;
};

} // namespace cohere_transcribe
