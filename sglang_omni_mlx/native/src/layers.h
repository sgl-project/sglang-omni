// SPDX-License-Identifier: Apache-2.0
// Checkpoint weights and the layers built on them: linear (quantized when the
// checkpoint is), norms, embeddings and activations.
#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <unordered_map>

#include "mlx/mlx.h"
#include "nlohmann/json.hpp"

namespace qwen3_asr {

struct QuantizationConfig {
  int group_size = 0;
  int bits = 0;
  std::string mode = "affine";
};

using WeightMap = std::unordered_map<std::string, mlx::core::array>;

nlohmann::json ReadJson(const std::filesystem::path &path);
// Every .safetensors file in the directory, names as stored.
WeightMap LoadSafetensors(const std::filesystem::path &model_directory);

class Checkpoint {
public:
  // Evaluates the weights; layers stored with .scales are quantized with the
  // config's top-level quantization, as the checkpoint was.
  Checkpoint(WeightMap weights, const nlohmann::json &config);

  const mlx::core::array &Weight(const std::string &name) const;
  bool Has(const std::string &name) const;
  mlx::core::array Linear(const mlx::core::array &x,
                          const std::string &prefix) const;
  mlx::core::array LayerNorm(const mlx::core::array &x,
                             const std::string &prefix, float eps) const;
  mlx::core::array RmsNorm(const mlx::core::array &x, const std::string &prefix,
                           float eps) const;
  // Rows of an embedding table for ids.
  mlx::core::array Embed(const mlx::core::array &ids,
                         const std::string &prefix) const;
  // An embedding table used as the output projection.
  mlx::core::array TiedProjection(const mlx::core::array &x,
                                  const std::string &prefix) const;

private:
  WeightMap weights_;
  QuantizationConfig quantization_;
};

// Activations in x's dtype.
mlx::core::array Gelu(const mlx::core::array &x);
mlx::core::array Silu(const mlx::core::array &x);

} // namespace qwen3_asr
