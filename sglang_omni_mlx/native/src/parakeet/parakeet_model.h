// SPDX-License-Identifier: Apache-2.0
// Parakeet TDT on MLX: the NeMo log-mel front end with per-feature
// normalization, the FastConformer encoder, and one step of the TDT
// prediction and joint networks, as Voxt's Swift port computes them in
// bfloat16.
#pragma once

#include <atomic>
#include <filesystem>
#include <functional>
#include <string>
#include <vector>

#include "common/layers.h"
#include "common/nemo_features.h"
#include "mlx/mlx.h"

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

// The prediction network's joint.pred output [rows, 1, joint] for each
// row's state, and the states after feeding its token, kept once a token is
// emitted.
struct Prediction {
  mlx::core::array projection;
  mlx::core::array hidden;
  mlx::core::array cell;
};

class ParakeetModel {
public:
  // Reads config.json and model.safetensors, cast to bfloat16 as the Swift
  // port loads them.
  explicit ParakeetModel(const std::filesystem::path &model_directory);
  // Note (Jiaxin Deng): the compiled prediction network captures this.
  ParakeetModel(const ParakeetModel &) = delete;
  ParakeetModel &operator=(const ParakeetModel &) = delete;

  // Normalized log-mel frames [1, frames, features] in float32.
  mlx::core::array Features(const std::vector<float> &samples) const;
  // Encoder frames [1, frames, width] of features, in bfloat16.
  mlx::core::array Encode(const mlx::core::array &features,
                          const std::atomic<bool> *cancel = nullptr) const;
  // joint.enc of encoder frames [1, frames, width]: [1, frames, joint].
  mlx::core::array ProjectFrames(const mlx::core::array &encoded) const;
  // The prediction network fed each row's token, int32 [rows, 1] (the blank
  // feeds zeros), from states [layers, rows, hidden].
  Prediction Predict(const mlx::core::array &tokens,
                     const mlx::core::array &hidden,
                     const mlx::core::array &cell) const;
  // Token and duration classes, int32 [2, rows, frames], of projected frames
  // [rows, frames, joint] joined with each row's prediction.
  mlx::core::array Decisions(const mlx::core::array &projected_frames,
                             const Prediction &prediction) const;
  // Zero prediction states [layers, rows, hidden] in bfloat16.
  std::pair<mlx::core::array, mlx::core::array> InitialState(int rows) const;

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
  // [token, hidden, cell] to [joint.pred projection, hidden, cell].
  std::vector<mlx::core::array>
  PredictGraph(const std::vector<mlx::core::array> &inputs) const;

  ParakeetConfig config_;
  omni_common::NemoProcessorConfig front_end_;
  omni_common::NemoFeatureExtractor features_;
  omni_common::Checkpoint checkpoint_;
  int prediction_width_ = 0;
  // Note (Dayuxiaoshui): the prediction network compiled, as the Swift port
  // runs its decoder step.
  std::function<std::vector<mlx::core::array>(
      const std::vector<mlx::core::array> &)>
      compiled_predict_;
};

} // namespace parakeet
