// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <atomic>
#include <functional>
#include <set>

#include "common/layers.h"

namespace qwen3_omni {
namespace mx = mlx::core;
using Json = nlohmann::json;
Json validate_configuration(Json configuration);
void validate_text_configuration(const Json &configuration);

struct Cancelled : std::runtime_error {
  Cancelled() : std::runtime_error("generation cancelled") {}
};
using Cancellation = std::function<bool()>;
void check_cancelled(const Cancellation &cancelled);
mx::array rows(const mx::array &tensor, int start, int stop, int axis = 0);

class Weights {
public:
  Weights(const std::filesystem::path &directory, const Json &configuration);
  Weights(omni_common::WeightMap tensors, int group_size, int bits);
  const mx::array &operator[](const std::string &name) const;
  bool contains(const std::string &name) const;
  void require(const std::string &name, const mx::Shape &shape);
  void require_linear(const std::string &name, int input_dimensions,
                      int output_dimensions, bool bias = false,
                      int experts = 0);
  void require_norm(const std::string &name, int dimensions, bool bias = false);
  void finish_validation();
  mx::array linear(const std::string &name, const mx::array &input,
                   const std::optional<mx::array> &experts = {}) const;
  mx::array embedding(const std::string &name, const mx::array &tokens) const;
  mx::array rms(const std::string &name, const mx::array &input,
                float epsilon) const;
  mx::array norm(const std::string &name, const mx::array &input,
                 float epsilon = 1e-5) const;

private:
  omni_common::WeightMap tensors;
  std::set<std::string> consumed;
  int group_size;
  int bits;
};

std::pair<mx::array, mx::array> rotary_factors(const mx::array &positions,
                                               int dimensions, float theta,
                                               bool interleaved);
mx::array rotary(const mx::array &input, const mx::array &positions,
                 float theta, bool interleaved);
mx::array rotary(const mx::array &input,
                 const std::pair<mx::array, mx::array> &factors);

struct KVCache {
  std::optional<mx::array> keys;
  std::optional<mx::array> values;
  int offset = 0;
};

class TextModel {
public:
  TextModel(Weights &weights, Json configuration, std::string prefix,
            bool interleaved = true);
  std::vector<KVCache> make_cache() const;
  std::pair<mx::array, std::optional<mx::array>>
  forward(const mx::array &embeddings, const mx::array &positions,
          std::vector<KVCache> &cache, const Cancellation &cancelled,
          const std::vector<mx::array> &deepstack = {},
          int capture_layer = -1) const;

private:
  mx::array feed_forward(const std::string &name, const mx::array &input,
                         const std::optional<mx::array> &experts = {}) const;
  Weights &weights;
  Json configuration;
  std::string prefix;
  bool interleaved;
};
} // namespace qwen3_omni
