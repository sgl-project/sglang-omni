// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Prince Canuma and contributors.
#pragma once

#include <functional>
#include <map>
#include <string>
#include <vector>

#include "common/layers.h"

namespace qwen3_tts {

class AudioCodec {
public:
  struct State {
    std::map<std::string, mlx::core::array> tails;
    std::map<std::string, mlx::core::array> keys;
    std::map<std::string, mlx::core::array> values;
    int offset = 0;
  };

  explicit AudioCodec(const std::filesystem::path &directory, bool encoder);
  mlx::core::array Decode(const mlx::core::array &codes, State &state,
                          const std::function<void()> &check_cancelled) const;
  mlx::core::array Encode(const std::vector<float> &samples,
                          const std::function<void()> &check_cancelled) const;

private:
  static qwen3_asr::WeightMap
  LoadWeights(const std::filesystem::path &directory,
              const nlohmann::json &config, bool encoder);
  mlx::core::array Convolve(const mlx::core::array &hidden,
                            const std::string &prefix, State *state,
                            int stride = 1, int dilation = 1, int groups = 1,
                            bool transpose = false,
                            const std::string &pad_mode = "constant") const;
  mlx::core::array
  Transformer(mlx::core::array hidden, const std::string &prefix,
              const nlohmann::json &config, State *state,
              const std::function<void()> &check_cancelled) const;
  mlx::core::array Snake(const mlx::core::array &hidden,
                         const std::string &prefix) const;

  nlohmann::json config_;
  qwen3_asr::Checkpoint checkpoint_;
};

} // namespace qwen3_tts
