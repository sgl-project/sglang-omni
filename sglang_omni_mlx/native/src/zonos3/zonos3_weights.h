// SPDX-License-Identifier: Apache-2.0
// One converted ZONOS3-Mini component (backbone, codec or speaker encoder):
// its config.json metadata and the tensors its "tensors" table lists, each a
// table or a matrix, dense or affine-quantized in groups of 32.
#pragma once

#include <filesystem>
#include <string>
#include <unordered_map>

#include "mlx/mlx.h"
#include "nlohmann/json.hpp"

namespace zonos3 {

class Weights {
public:
  // Reads directory/config.json and directory/model.safetensors, whose keys
  // start with key_prefix. Throws std::runtime_error when a listed tensor is
  // missing, has another shape or dtype, or the file holds one not listed.
  Weights(const std::filesystem::path &directory,
          const std::string &key_prefix);

  const nlohmann::json &metadata() const { return metadata_; }
  // A metadata value under the component's architecture, "<arch>.<key>".
  const nlohmann::json &Meta(const std::string &key) const;
  const std::string &architecture() const { return architecture_; }

  bool HasTable(const std::string &name) const;
  const mlx::core::array &Table(const std::string &name) const;
  bool HasMatrix(const std::string &name) const;
  // x @ matrix^T in float32: a dense matrix cast to float32, or a quantized
  // one through quantized_matmul.
  mlx::core::array Linear(const mlx::core::array &x,
                          const std::string &name) const;
  // Linear(x, name) plus the table bias_name.
  mlx::core::array Linear(const mlx::core::array &x, const std::string &name,
                          const std::string &bias_name) const;
  // Stacked expert matrices [experts, out, in] applied per row of
  // expert_indices, through gather_qmm.
  mlx::core::array GatherLinear(const mlx::core::array &x,
                                const std::string &name,
                                const mlx::core::array &expert_indices) const;

private:
  struct Matrix {
    mlx::core::array weight;
    std::optional<mlx::core::array> scales;
    std::optional<mlx::core::array> biases;
    int bits = 0;
  };
  const Matrix &FindMatrix(const std::string &name) const;

  nlohmann::json metadata_;
  std::string architecture_;
  std::unordered_map<std::string, mlx::core::array> tables_;
  std::unordered_map<std::string, Matrix> matrices_;
};

} // namespace zonos3
