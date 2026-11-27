// SPDX-License-Identifier: Apache-2.0
#include <stdexcept>
#include <string>

#include "moss/moss_model.h"

namespace {

namespace mx = mlx::core;

void Require(bool condition, const std::string &message) {
  if (!condition) {
    throw std::runtime_error(message);
  } else {
  }
}

void RequireEqual(const mx::array &actual, const mx::array &expected,
                  const std::string &message) {
  Require(mx::all(mx::array_equal(actual, expected)).item<bool>(), message);
}

} // namespace

int main() {
  namespace mx = mlx::core;
  const std::string decoder = "model.language_model";
  const std::string embedding = decoder + ".embed_tokens.weight";
  const std::string projection = decoder + ".output_projection";
  const std::string projection_weight = projection + ".weight";
  const auto embedding_values = mx::ones({2, 64}, mx::float32);
  const auto projection_values = mx::add(embedding_values, embedding_values);
  const auto quantized = mx::quantize(projection_values, 64, 8);
  mx::async_eval(quantized);

  omni_common::WeightMap original;
  original.emplace(embedding, embedding_values);
  auto loaded = moss::LoadedWeights(original);
  Require(loaded.at(projection_weight).dtype() == mx::float32,
          "the tied projection keeps the embedding dtype without a preexisting "
          "projection");
  RequireEqual(loaded.at(projection_weight), embedding_values,
               "the tied projection keeps the embedding values");

  omni_common::WeightMap prequantized;
  prequantized.emplace(embedding, embedding_values);
  prequantized.emplace(projection_weight, quantized[0]);
  prequantized.emplace(projection + ".scales", quantized[1]);
  prequantized.emplace(projection + ".biases", quantized[2]);
  loaded = moss::LoadedWeights(prequantized);
  const nlohmann::json config = {
      {"quantization", {{"group_size", 64}, {"bits", 8}}}};
  auto checkpoint = omni_common::QuantizedCheckpoint(
      loaded, config, 64, [](const std::string &) { return 0; });
  const auto input = mx::ones({1, 64}, mx::float32);
  RequireEqual(checkpoint.Linear(input, projection),
               mx::matmul(input, mx::transpose(projection_values)),
               "the prequantized projection produces the source linear output");
  Require(loaded.at(projection_weight).dtype() == mx::uint32,
          "a prequantized output projection must not be replaced by the tied "
          "embedding");
  RequireEqual(loaded.at(projection_weight), quantized[0],
               "the prequantized projection's values must be preserved");
  RequireEqual(loaded.at(projection + ".scales"), quantized[1],
               "the prequantized projection's scales must be preserved");
  RequireEqual(loaded.at(projection + ".biases"), quantized[2],
               "the prequantized projection's biases must be preserved");
  return 0;
}
