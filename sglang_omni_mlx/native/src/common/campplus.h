// SPDX-License-Identifier: Apache-2.0
// The CAM++ speaker encoder CosyVoice3 and Ming-Omni-TTS share: Kaldi log-mel
// filterbank features of 16 kHz audio to a 192-dim speaker embedding.
#pragma once

#include <array>
#include <string>
#include <vector>

#include "common/layers.h"
#include "mlx/mlx.h"

namespace omni_common {

// Layers in each of CAM++'s three densely connected blocks.
inline constexpr std::array<int, 3> kCamPlusBlockLayerCounts = {12, 24, 16};
// Their weights' prefix: blocks.<block>.layers.<layer>.
std::string CamPlusLayerPrefix(int block, int layer);

class CamPlusEncoder {
public:
  // weights are float32 in MLX layout under the reference module's names
  // (head.conv1.weight, blocks.0.layers.0.linear1.weight, ...). A
  // convolution the reference follows with batch norm may instead carry that
  // norm folded into its .bias, as an ONNX export stores it. Evaluates them.
  explicit CamPlusEncoder(WeightMap weights);

  // The speaker embedding of 16 kHz samples, float32 [1, 192].
  mlx::core::array Embed(const std::vector<float> &samples_16k) const;

private:
  const mlx::core::array &Weight(const std::string &name) const;
  bool Has(const std::string &name) const;
  mlx::core::array BatchNorm(const mlx::core::array &x,
                             const std::string &prefix) const;
  // x, the output of convolution, through the batch norm that follows it or
  // the bias it was folded into.
  mlx::core::array Normalized(const mlx::core::array &x,
                              const std::string &convolution,
                              const std::string &norm) const;
  mlx::core::array ResidualBlock(const mlx::core::array &x,
                                 const std::string &prefix, int stride) const;
  mlx::core::array DenseLayer(const mlx::core::array &x,
                              const std::string &prefix, int dilation) const;

  WeightMap weights_;
};

} // namespace omni_common
