// SPDX-License-Identifier: Apache-2.0
#include "zonos3/zonos3_weights.h"

#include <set>
#include <stdexcept>

#include "common/layers.h"

namespace zonos3 {

namespace mx = mlx::core;

namespace {

// The quantized matrices' group size, which the converter fixes.
constexpr int kQuantizationGroupSize = 32;

// A tensor name as the converter stores it: "." is an attribute separator in
// the stored module tree, so "blk.0.attn_q.weight" is "blk__0__attn_q__weight".
std::string StoredName(const std::string &name) {
  if (name.find("__") != std::string::npos) {
    throw std::runtime_error("unsupported tensor name " + name);
  } else {
  }
  std::string stored;
  for (const char character : name) {
    if (character == '.') {
      stored += "__";
    } else {
      stored += character;
    }
  }
  return stored;
}

mx::Dtype DtypeNamed(const std::string &name) {
  if (name == "float32") {
    return mx::float32;
  } else if (name == "float16") {
    return mx::float16;
  } else if (name == "bfloat16") {
    return mx::bfloat16;
  } else if (name == "uint32") {
    return mx::uint32;
  } else {
    throw std::runtime_error("unsupported tensor dtype " + name);
  }
}

mx::Shape ShapeOf(const nlohmann::json &dimensions) {
  mx::Shape shape;
  for (const auto &dimension : dimensions) {
    shape.push_back(dimension.get<int>());
  }
  return shape;
}

std::string ShapeText(const mx::Shape &shape) {
  std::string text = "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    text += (i == 0 ? "" : ", ") + std::to_string(shape[i]);
  }
  return text + "]";
}

} // namespace

Weights::Weights(const std::filesystem::path &directory,
                 const std::string &key_prefix) {
  const nlohmann::json config = omni_common::ReadJson(directory / "config.json");
  metadata_ = config.value("metadata", nlohmann::json::object());
  if (metadata_.contains("general.architecture")) {
    architecture_ = metadata_.at("general.architecture").get<std::string>();
  } else {
  }
  auto [stored, unused_metadata] =
      mx::load_safetensors((directory / "model.safetensors").string());
  std::set<std::string> unread;
  for (const auto &[key, array] : stored) {
    unread.insert(key);
  }
  const auto take = [&](const std::string &key, const mx::Shape &shape,
                        mx::Dtype dtype) {
    const auto found = stored.find(key);
    if (found == stored.end()) {
      throw std::runtime_error((directory / "model.safetensors").string() +
                               " is missing " + key);
    } else if (found->second.shape() != shape ||
               found->second.dtype() != dtype) {
      throw std::runtime_error(key + " has shape " +
                               ShapeText(found->second.shape()) +
                               "; the config lists " + ShapeText(shape));
    } else {
    }
    unread.erase(key);
    return found->second;
  };
  std::vector<mx::array> loaded;
  for (const auto &[name, spec] : config.at("tensors").items()) {
    const mx::Shape shape = ShapeOf(spec.at("shape"));
    const mx::Dtype dtype = DtypeNamed(spec.at("dtype").get<std::string>());
    const std::string stored_name = StoredName(name);
    if (!spec.at("matrix").get<bool>()) {
      tables_.emplace(name,
                      take(key_prefix + "tables." + stored_name, shape, dtype));
      loaded.push_back(tables_.at(name));
      continue;
    } else {
    }
    const std::string prefix = key_prefix + "matrices." + stored_name;
    const nlohmann::json bits = spec.value("bits", nlohmann::json());
    if (bits.is_null()) {
      matrices_.emplace(name, Matrix{take(prefix + ".weight", shape, dtype),
                                     std::nullopt, std::nullopt, 0});
    } else {
      const int bit_count = bits.get<int>();
      if (shape.size() < 2 || shape.back() % kQuantizationGroupSize != 0 ||
          shape.back() * bit_count % 32 != 0) {
        throw std::runtime_error(name + " cannot be quantized in groups of " +
                                 std::to_string(kQuantizationGroupSize));
      } else {
      }
      mx::Shape packed = shape;
      packed.back() = shape.back() * bit_count / 32;
      mx::Shape groups = shape;
      groups.back() = shape.back() / kQuantizationGroupSize;
      matrices_.emplace(name,
                        Matrix{take(prefix + ".weight", packed, mx::uint32),
                               take(prefix + ".scales", groups, mx::float32),
                               take(prefix + ".biases", groups, mx::float32),
                               bit_count});
    }
    const Matrix &matrix = matrices_.at(name);
    loaded.push_back(matrix.weight);
    if (matrix.scales.has_value()) {
      loaded.push_back(*matrix.scales);
      loaded.push_back(*matrix.biases);
    } else {
    }
  }
  if (!unread.empty()) {
    throw std::runtime_error((directory / "model.safetensors").string() +
                             " holds " + *unread.begin() +
                             ", which config.json does not list");
  } else {
  }
  mx::eval(loaded);
}

const nlohmann::json &Weights::Meta(const std::string &key) const {
  const std::string full_key = architecture_ + "." + key;
  if (!metadata_.contains(full_key)) {
    throw std::runtime_error("config.json metadata is missing " + full_key);
  } else {
  }
  return metadata_.at(full_key);
}

bool Weights::HasTable(const std::string &name) const {
  return tables_.count(name) > 0;
}

const mx::array &Weights::Table(const std::string &name) const {
  const auto found = tables_.find(name);
  if (found == tables_.end()) {
    throw std::runtime_error("checkpoint is missing the table " + name);
  } else {
  }
  return found->second;
}

bool Weights::HasMatrix(const std::string &name) const {
  return matrices_.count(name) > 0;
}

const Weights::Matrix &Weights::FindMatrix(const std::string &name) const {
  const auto found = matrices_.find(name);
  if (found == matrices_.end()) {
    throw std::runtime_error("checkpoint is missing the matrix " + name);
  } else {
  }
  return found->second;
}

mx::array Weights::Linear(const mx::array &x, const std::string &name) const {
  const Matrix &matrix = FindMatrix(name);
  if (matrix.scales.has_value()) {
    return mx::quantized_matmul(x, matrix.weight, *matrix.scales,
                                matrix.biases, true, kQuantizationGroupSize,
                                matrix.bits, "affine");
  } else {
    return mx::matmul(x,
                      mx::swapaxes(mx::astype(matrix.weight, mx::float32), -1,
                                   -2));
  }
}

mx::array Weights::Linear(const mx::array &x, const std::string &name,
                          const std::string &bias_name) const {
  return mx::add(Linear(x, name), Table(bias_name));
}

mx::array Weights::GatherLinear(const mx::array &x, const std::string &name,
                                const mx::array &expert_indices) const {
  const Matrix &matrix = FindMatrix(name);
  if (!matrix.scales.has_value()) {
    throw std::runtime_error(name + " must be quantized");
  } else {
  }
  return mx::gather_qmm(x, matrix.weight, *matrix.scales, matrix.biases,
                        std::nullopt, expert_indices, true,
                        kQuantizationGroupSize, matrix.bits, "affine");
}

} // namespace zonos3
