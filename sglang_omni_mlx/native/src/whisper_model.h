// SPDX-License-Identifier: Apache-2.0
// Whisper on MLX: checkpoint loading, the log-mel front end, encoder and
// decoder, op for op as Voxt's Swift port runs them.
#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "mlx/mlx.h"

namespace whisper {

// Whisper encodes 30 s windows of 16 kHz audio into 3000 mel frames.
inline constexpr int kWindowSampleCount = 30 * 16000;

struct WhisperConfig {
  int mel_bin_count = 0;
  int audio_context_length = 0;
  int audio_width = 0;
  int audio_head_count = 0;
  int audio_layer_count = 0;
  int vocabulary_size = 0;
  int text_context_length = 0;
  int text_head_count = 0;
  int text_layer_count = 0;
};

// Self-attention keys and values grow by one token per step; cross-attention
// keys and values are projected once per window.
struct DecoderLayerCache {
  std::optional<mlx::core::array> self_keys;
  std::optional<mlx::core::array> self_values;
  std::optional<mlx::core::array> cross_keys;
  std::optional<mlx::core::array> cross_values;
};

class WhisperModel {
public:
  // Reads config.json and the weights (*.safetensors, else weights.npz) of an
  // mlx-community Whisper checkpoint.
  explicit WhisperModel(const std::filesystem::path &model_directory);

  const WhisperConfig &config() const { return config_; }
  // One window of kWindowSampleCount samples to encoder states [1, 1500,
  // width].
  mlx::core::array Encode(const std::vector<float> &window_samples) const;
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
                          int stride) const;
  mlx::core::array Attention(const mlx::core::array &queries_input,
                             const mlx::core::array &keys,
                             const mlx::core::array &values,
                             const std::string &prefix, int head_count,
                             const std::optional<mlx::core::array> &mask) const;
  mlx::core::array EncoderLayer(const mlx::core::array &x, int layer) const;
  mlx::core::array DecoderLayer(const mlx::core::array &x, int layer,
                                const mlx::core::array &encoder_states,
                                const std::optional<mlx::core::array> &mask,
                                DecoderLayerCache &cache) const;

  WhisperConfig config_;
  // The log-mel front end's periodic Hann window [400] and mel filters
  // [201, mel_bins].
  mlx::core::array hann_window_;
  mlx::core::array mel_filters_;
  std::unordered_map<std::string, mlx::core::array> weights_;
};

} // namespace whisper
