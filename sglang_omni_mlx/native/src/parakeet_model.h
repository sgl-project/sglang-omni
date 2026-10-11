// SPDX-License-Identifier: Apache-2.0
// Parakeet TDT on MLX: the NeMo log-mel front end with per-feature
// normalization, the FastConformer encoder, and one step of the TDT
// prediction and joint networks, as Voxt's Swift port computes them in
// bfloat16.
#pragma once

#include <filesystem>
#include <functional>
#include <string>
#include <vector>

#include "layers.h"
#include "mlx/mlx.h"
#include "sortformer_features.h"

namespace parakeet {

struct ParakeetConfig {
  int feature_count = 0;
  int model_width = 0;
  int head_count = 0;
  int layer_count = 0;
  int subsampling_factor = 0;
  int prediction_layer_count = 0;
  // Ids of the output vocabulary; the blank id is its size.
  std::vector<std::string> vocabulary;
  // Frames a TDT duration choice advances, by duration class.
  std::vector<int> durations;
  // Symbols emitted on one frame before moving on.
  int max_symbols_per_frame = 0;
};

// One TDT decoder step: the predicted token and duration class, and the
// prediction network state after feeding the step's input token.
struct DecoderStep {
  int token = 0;
  int duration_index = 0;
  mlx::core::array hidden;
  mlx::core::array cell;
};

class ParakeetModel {
public:
  // Reads config.json and model.safetensors, cast to bfloat16 as the Swift
  // port loads them.
  explicit ParakeetModel(const std::filesystem::path &model_directory);
  // Note (Jiaxin Deng): the compiled decoder step captures this.
  ParakeetModel(const ParakeetModel &) = delete;
  ParakeetModel &operator=(const ParakeetModel &) = delete;

  // Normalized log-mel frames [1, frames, features] in float32.
  mlx::core::array Features(const std::vector<float> &samples) const;
  // Encoder frames [1, frames, width] of features, in bfloat16.
  mlx::core::array Encode(const mlx::core::array &features) const;
  // The prediction network fed token (the blank feeds zeros) from state
  // [layers, 1, hidden], joined with one encoder frame [1, 1, width].
  DecoderStep Decode(const mlx::core::array &frame, int token,
                     const mlx::core::array &hidden,
                     const mlx::core::array &cell) const;
  // Zero prediction state [layers, 1, hidden] in bfloat16.
  std::pair<mlx::core::array, mlx::core::array> InitialState() const;

  const ParakeetConfig &config() const { return config_; }
  int blank_id() const { return static_cast<int>(config_.vocabulary.size()); }
  // Seconds of audio in frames encoder frames, computed as the Swift port
  // does.
  double FrameSeconds(int frames) const;

private:
  ParakeetModel(const nlohmann::json &config,
                const std::filesystem::path &model_directory);
  mlx::core::array Conv2d(const mlx::core::array &x, const std::string &prefix,
                          int groups) const;
  mlx::core::array FeedForward(const mlx::core::array &x,
                               const std::string &prefix) const;
  mlx::core::array Attention(const mlx::core::array &x,
                             const mlx::core::array &positions,
                             const std::string &prefix) const;
  mlx::core::array Convolution(const mlx::core::array &x,
                               const std::string &prefix) const;
  // [frame, token, hidden, cell] to [token and duration class, hidden, cell].
  std::vector<mlx::core::array>
  DecodeGraph(const std::vector<mlx::core::array> &inputs) const;

  ParakeetConfig config_;
  sortformer::ProcessorConfig front_end_;
  sortformer::FeatureExtractor features_;
  qwen3_asr::Checkpoint checkpoint_;
  int prediction_width_ = 0;
  // Note (Dayuxiaoshui): the decoder step compiled, as the Swift port runs it.
  std::function<std::vector<mlx::core::array>(
      const std::vector<mlx::core::array> &)>
      compiled_decode_;
};

} // namespace parakeet
