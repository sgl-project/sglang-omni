// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "qwen3_omni/model.h"
namespace qwen3_omni {
struct VisualGrid {
  int frames;
  int height;
  int width;
  int token_count() const { return frames * height * width / 4; }
};
class VisionEncoder {
public:
  VisionEncoder(Weights &weights, Json configuration);
  std::pair<mx::array, std::vector<mx::array>>
  forward(const mx::array &patches, const VisualGrid &grid,
          const Cancellation &cancelled) const;

private:
  mx::array merge(const mx::array &hidden_states, int index) const;
  Weights &weights;
  Json configuration;
};
class AudioEncoder {
public:
  AudioEncoder(Weights &weights, Json configuration);
  static int output_length(int frames) {
    return frames / 100 * 13 + (frames % 100 + 7) / 8;
  }
  mx::array forward(const mx::array &mel, const Cancellation &cancelled) const;

private:
  Weights &weights;
  Json configuration;
};
} // namespace qwen3_omni
