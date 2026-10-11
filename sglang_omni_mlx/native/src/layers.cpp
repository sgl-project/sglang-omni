// SPDX-License-Identifier: Apache-2.0
#include "layers.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace qwen3_asr {

namespace mx = mlx::core;

namespace {

std::vector<mx::array> GeluGraph(const std::vector<mx::array> &inputs) {
  // Note (Jiaxin Deng): mlx.nn.gelu's op order, kept for bit identity.
  // Note (Dayuxiaoshui): constants take x's dtype as Python scalars do, so a
  // half-precision encoder is not promoted to float32.
  const mx::array &x = inputs[0];
  const mx::Dtype dtype = x.dtype();
  return {mx::divide(
      mx::multiply(
          x, mx::add(mx::array(1.0f, dtype),
                     mx::erf(mx::divide(
                         x, mx::array(static_cast<float>(M_SQRT2), dtype))))),
      mx::array(2.0f, dtype))};
}

std::vector<mx::array> SiluGraph(const std::vector<mx::array> &inputs) {
  return {mx::multiply(inputs[0], mx::sigmoid(inputs[0]))};
}

} // namespace

// Note (Jiaxin Deng): compiled shapeless as mlx.nn does, so each activation
// is one fused kernel instead of one per elementwise op.
mx::array Gelu(const mx::array &x) {
  static const auto compiled = mx::compile(GeluGraph, true);
  return compiled({x})[0];
}

mx::array Silu(const mx::array &x) {
  static const auto compiled = mx::compile(SiluGraph, true);
  return compiled({x})[0];
}

nlohmann::json ReadJson(const std::filesystem::path &path) {
  std::ifstream stream(path);
  if (!stream) {
    throw std::runtime_error("cannot read " + path.string());
  } else {
  }
  return nlohmann::json::parse(stream);
}

WeightMap LoadSafetensors(const std::filesystem::path &model_directory) {
  std::vector<std::filesystem::path> weight_files;
  for (const auto &entry :
       std::filesystem::directory_iterator(model_directory)) {
    if (entry.path().extension() == ".safetensors") {
      weight_files.push_back(entry.path());
    } else {
    }
  }
  if (weight_files.empty()) {
    throw std::runtime_error("no .safetensors files in " +
                             model_directory.string());
  } else {
  }
  std::sort(weight_files.begin(), weight_files.end());
  WeightMap weights;
  for (const auto &path : weight_files) {
    auto [loaded, metadata] = mx::load_safetensors(path.string());
    for (auto &[name, array] : loaded) {
      weights.insert_or_assign(name, array);
    }
  }
  return weights;
}

Checkpoint::Checkpoint(WeightMap weights, const nlohmann::json &config)
    : weights_(std::move(weights)) {
  if (config.contains("quantization")) {
    const nlohmann::json &quantization = config.at("quantization");
    quantization_ = {quantization.at("group_size").get<int>(),
                     quantization.at("bits").get<int>(),
                     quantization.value("mode", std::string("affine"))};
  } else {
  }
  std::vector<mx::array> parameters;
  parameters.reserve(weights_.size());
  for (const auto &[name, array] : weights_)
    parameters.push_back(array);
  mx::eval(parameters);
}

const mx::array &Checkpoint::Weight(const std::string &name) const {
  const auto found = weights_.find(name);
  if (found == weights_.end()) {
    throw std::runtime_error("checkpoint is missing " + name);
  } else {
  }
  return found->second;
}

bool Checkpoint::Has(const std::string &name) const {
  return weights_.count(name) > 0;
}

mx::array Checkpoint::Linear(const mx::array &x,
                             const std::string &prefix) const {
  if (Has(prefix + ".scales")) {
    const std::optional<mx::array> biases =
        Has(prefix + ".biases")
            ? std::optional<mx::array>(Weight(prefix + ".biases"))
            : std::nullopt;
    mx::array y = mx::quantized_matmul(
        x, Weight(prefix + ".weight"), Weight(prefix + ".scales"), biases, true,
        quantization_.group_size, quantization_.bits, quantization_.mode);
    if (Has(prefix + ".bias")) {
      return mx::add(y, Weight(prefix + ".bias"));
    } else {
      return y;
    }
  } else if (Has(prefix + ".bias")) {
    return mx::addmm(Weight(prefix + ".bias"), x,
                     mx::transpose(Weight(prefix + ".weight")));
  } else {
    return mx::matmul(x, mx::transpose(Weight(prefix + ".weight")));
  }
}

mx::array Checkpoint::LayerNorm(const mx::array &x, const std::string &prefix,
                                float eps) const {
  return mx::fast::layer_norm(x, Weight(prefix + ".weight"),
                              Weight(prefix + ".bias"), eps);
}

mx::array Checkpoint::RmsNorm(const mx::array &x, const std::string &prefix,
                              float eps) const {
  return mx::fast::rms_norm(x, Weight(prefix + ".weight"), eps);
}

mx::array Checkpoint::Embed(const mx::array &ids,
                            const std::string &prefix) const {
  if (Has(prefix + ".scales")) {
    const std::optional<mx::array> biases =
        Has(prefix + ".biases") ? std::optional<mx::array>(mx::take(
                                      Weight(prefix + ".biases"), ids, 0))
                                : std::nullopt;
    return mx::dequantize(mx::take(Weight(prefix + ".weight"), ids, 0),
                          mx::take(Weight(prefix + ".scales"), ids, 0), biases,
                          quantization_.group_size, quantization_.bits,
                          quantization_.mode);
  } else {
    return mx::take(Weight(prefix + ".weight"), ids, 0);
  }
}

mx::array Checkpoint::TiedProjection(const mx::array &x,
                                     const std::string &prefix) const {
  if (Has(prefix + ".scales")) {
    const std::optional<mx::array> biases =
        Has(prefix + ".biases")
            ? std::optional<mx::array>(Weight(prefix + ".biases"))
            : std::nullopt;
    return mx::quantized_matmul(
        x, Weight(prefix + ".weight"), Weight(prefix + ".scales"), biases, true,
        quantization_.group_size, quantization_.bits, quantization_.mode);
  } else {
    return mx::matmul(x, mx::transpose(Weight(prefix + ".weight")));
  }
}

} // namespace qwen3_asr
