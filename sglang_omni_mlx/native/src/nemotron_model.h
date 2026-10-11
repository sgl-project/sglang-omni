// SPDX-License-Identifier: Apache-2.0
// Nemotron 3.5 ASR streaming on MLX: causal FastConformer encoder run in
// cache-aware chunks, language prompt fusion, and the RNN-T prediction and
// joint networks, as Voxt's Swift port computes them.
#pragma once

#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "layers.h"
#include "mlx/mlx.h"

namespace nemotron {

struct NemotronConfig {
  int feature_count = 0;
  int model_width = 0;
  int head_count = 0;
  int layer_count = 0;
  int subsampling_factor = 0;
  int conv_kernel_size = 0;
  // Attention frames kept from earlier chunks, and the frames a chunk adds
  // past its first: [56, 13] is 56 cached frames and 14-frame chunks.
  int left_context_frames = 0;
  int right_context_frames = 0;
  int prompt_count = 0;
  std::map<std::string, int> prompt_indices;
  std::string default_language;
  int prediction_layer_count = 0;
  // Ids of the output vocabulary; the blank id is its size.
  std::vector<std::string> vocabulary;
  // Symbols emitted on one encoder frame before moving on; none is no limit.
  std::optional<int> max_symbols_per_frame;
};

// One conformer layer's state between chunks: its last attention inputs
// (after the attention norm) and its last GLU outputs, which the causal
// depthwise convolution reads.
struct LayerCache {
  std::optional<mlx::core::array> attention_inputs;
  std::optional<mlx::core::array> convolution_inputs;
};

// The prediction network's per-layer LSTM state, each [1, hidden].
struct LstmState {
  std::vector<mlx::core::array> hidden;
  std::vector<mlx::core::array> cell;
};

class NemotronModel {
public:
  explicit NemotronModel(const std::filesystem::path &model_directory);

  // Log-mel frames [1, frames, features] to subsampled frames [1, out, width]
  // in the checkpoint's dtype; each of the three stages maps n frames to
  // n / 2 + 1, so a window yields one frame beyond its eighth.
  mlx::core::array Subsample(const mlx::core::array &mel) const;
  // Relative positions length - 1 down to -(length - 1), [1, 2 * length - 1,
  // width] in dtype; every layer of a chunk shares them.
  mlx::core::array RelativePositions(int length, mlx::core::Dtype dtype) const;
  // One conformer layer over a chunk [1, frames, width], reading and replacing
  // the layer's cache.
  mlx::core::array StreamLayer(const mlx::core::array &x, int layer,
                               const mlx::core::array &relative_positions,
                               LayerCache &cache) const;
  // Encoder frames conditioned on the language prompt.
  mlx::core::array ApplyPrompt(const mlx::core::array &encoded,
                               int prompt_index) const;
  // The prompt index for a language; unknown or none is the default
  // language's, else the first prompt.
  int PromptIndex(const std::optional<std::string> &language) const;
  // Prediction network output [1, 1, hidden] for the last emitted token, none
  // at the start of a stream, and its new state.
  std::pair<mlx::core::array, LstmState>
  Predict(std::optional<int> token,
          const std::optional<LstmState> &state) const;
  // Joint logits [1, 1, 1, classes] for one encoder frame [1, 1, width].
  mlx::core::array JointLogits(const mlx::core::array &frame,
                               const mlx::core::array &prediction) const;

  const NemotronConfig &config() const { return config_; }
  int blank_id() const { return static_cast<int>(config_.vocabulary.size()); }

private:
  NemotronModel(const nlohmann::json &config,
                const std::filesystem::path &model_directory);
  mlx::core::array Conv2d(const mlx::core::array &x, const std::string &prefix,
                          int stride, int groups) const;
  mlx::core::array FeedForward(const mlx::core::array &x,
                               const std::string &prefix) const;
  mlx::core::array RelativeAttention(const mlx::core::array &queries,
                                     const mlx::core::array &keys_values,
                                     const mlx::core::array &relative_positions,
                                     const std::string &prefix) const;

  NemotronConfig config_;
  qwen3_asr::Checkpoint checkpoint_;
};

} // namespace nemotron
