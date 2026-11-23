// SPDX-License-Identifier: Apache-2.0
#include <stdexcept>
#include <string>

#include "../src/moss/moss_model.cpp"

namespace {

void Require(bool condition, const std::string &message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

} // namespace

int main() {
  namespace mx = mlx::core;
  const std::string decoder = "model.language_model";
  const std::string embedding = decoder + ".embed_tokens.weight";
  const std::string projection = decoder + ".output_projection.weight";

  omni_common::WeightMap original;
  original.emplace(embedding, mx::ones({2, 2}, mx::float32));
  auto loaded = moss::LoadedWeights(original);
  Require(loaded.at(projection).dtype() == mx::float32,
          "the tied projection keeps the embedding dtype without a preexisting projection");

  omni_common::WeightMap prequantized;
  prequantized.emplace(embedding, mx::ones({2, 2}, mx::float32));
  prequantized.emplace(projection, mx::zeros({2, 2}, mx::uint32));
  loaded = moss::LoadedWeights(prequantized);
  Require(loaded.at(projection).dtype() == mx::uint32,
          "a prequantized output projection must not be replaced by the tied embedding");
  Require(mx::all(mx::array_equal(loaded.at(projection), mx::zeros({2, 2}, mx::uint32)))
              .item<bool>(),
          "the prequantized projection's values must be preserved");
  return 0;
}
