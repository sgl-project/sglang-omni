// SPDX-License-Identifier: Apache-2.0
// Sortformer config.json and safetensors loading, as Swift
// SortformerModel.fromModelDirectory.
#pragma once

#include <filesystem>
#include <string>
#include <unordered_map>

#include "mlx/mlx.h"
#include "sortformer.h"

namespace sortformer {

// config.json with the Swift defaults for absent keys; only use_aosc (v2.1)
// checkpoints are accepted.
Config LoadConfig(const std::filesystem::path &model_directory);

// Every *.safetensors in the directory, sanitized as Swift does (PyTorch
// layouts converted, num_batches_tracked dropped) and checked strictly: each
// key must be a model parameter and each parameter present. Evaluated.
std::unordered_map<std::string, mlx::core::array>
LoadCheckpoint(const std::filesystem::path &model_directory,
               const Config &config);

} // namespace sortformer
