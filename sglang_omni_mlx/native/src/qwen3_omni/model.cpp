// SPDX-License-Identifier: Apache-2.0
#include "qwen3_omni/model.h"
#include "qwen3_omni/generation.h"

#include <algorithm>
#include <iostream>

namespace qwen3_omni {
OmniModel::OmniModel(const std::filesystem::path &directory)
    : configuration(validate_configuration(
          omni_common::ReadJson(directory / "config.json"))),
      tokenizer(directory),
      weights(directory, configuration.at("quantization")),
      thinker(weights, configuration.at("thinker_config").at("text_config"),
              "thinker.language_model.model"),
      talker(weights, configuration.at("talker_config").at("text_config"),
             "talker.model"),
      predictor(weights,
                configuration.at("talker_config").at("code_predictor_config"),
                "talker.code_predictor.model", false),
      vision(weights, configuration.at("thinker_config").at("vision_config")),
      audio(weights, configuration.at("thinker_config").at("audio_config")),
      vocoder(weights, configuration.at("code2wav_config")) {
  const auto &thinker_config = configuration.at("thinker_config");
  const auto &talker_config = configuration.at("talker_config");
  const auto &text_config = thinker_config.at("text_config");
  const auto &speech_config = talker_config.at("text_config");
  const auto &predictor_config = talker_config.at("code_predictor_config");
  for (const auto &name : {"thinker.language_model.model.embed_tokens",
                           "thinker.language_model.lm_head"}) {
    weights.require_linear(name, text_config.at("hidden_size"),
                           text_config.at("vocab_size"));
  }
  for (const auto &name :
       {"talker.model.codec_embedding", "talker.codec_head"}) {
    weights.require_linear(name, speech_config.at("hidden_size"),
                           speech_config.at("vocab_size"));
  }
  for (const auto &name : {"text_projection", "hidden_projection"}) {
    weights.require_linear(std::string("talker.") + name + ".linear_fc1",
                           talker_config.at("thinker_hidden_size"),
                           speech_config.at("intermediate_size"), true);
    weights.require_linear(std::string("talker.") + name + ".linear_fc2",
                           speech_config.at("intermediate_size"),
                           speech_config.at("hidden_size"), true);
  }
  for (int i = 0; i < talker_config.at("num_code_groups").get<int>() - 1; ++i) {
    for (const auto &name : {"model.codec_embedding", "lm_head"}) {
      weights.require_linear(std::string("talker.code_predictor.") + name +
                                 "." + std::to_string(i),
                             predictor_config.at("hidden_size"),
                             predictor_config.at("vocab_size"));
    }
  }
  weights.finish_validation();
  std::cerr << "native Omni checkpoint loaded\n";
}
void check_cancelled(const Cancellation &cancelled) {
  if (cancelled && cancelled()) {
    throw Cancelled();
  } else {
  }
}

mx::array rows(const mx::array &tensor, int start, int stop, int axis) {
  axis = axis < 0 ? axis + tensor.ndim() : axis;
  auto begin = mx::Shape(tensor.ndim(), 0);
  auto end = tensor.shape();
  begin.at(axis) = start < 0 ? tensor.shape(axis) + start : start;
  end.at(axis) = stop < 0 ? tensor.shape(axis) + stop : stop;
  return mx::slice(tensor, begin, end);
}

Weights::Weights(omni_common::WeightMap tensors, int group_size, int bits)
    : tensors(std::move(tensors)), group_size(group_size), bits(bits) {}

Weights::Weights(const std::filesystem::path &directory,
                 const Json &configuration)
    : group_size(configuration.at("group_size")),
      bits(configuration.at("bits")) {
  if (group_size != 64 || bits != 4 ||
      configuration.value("mode", "") != "affine") {
    throw std::invalid_argument("unsupported Omni checkpoint quantization");
  } else {
  }
  const auto index =
      omni_common::ReadJson(directory / "model.safetensors.index.json")
          .at("weight_map");
  std::set<std::string> shards;
  for (const auto &[name, shard] : index.items()) {
    const auto filename = shard.get<std::string>();
    if (filename.find('/') != std::string::npos ||
        std::filesystem::path(filename).extension() != ".safetensors") {
      throw std::invalid_argument("invalid checkpoint shard path");
    } else {
      shards.insert(filename);
    }
  }
  for (const auto &shard : shards) {
    auto loaded = mx::load_safetensors((directory / shard).string()).first;
    for (auto &[name, tensor] : loaded) {
      if (!index.contains(name) || index.at(name) != shard ||
          tensors.contains(name)) {
        throw std::invalid_argument("unexpected or duplicate weight: " + name);
      } else {
        tensors.emplace(name, std::move(tensor));
      }
    }
  }
  if (index.empty() || tensors.size() != index.size()) {
    throw std::invalid_argument("checkpoint index/weights mismatch");
  } else {
  }
  for (const auto &[name, tensor] : tensors) {
    if (tensor.dtype() == mx::uint32) {
      if (!name.ends_with(".weight") ||
          !contains(name.substr(0, name.size() - 7) + ".scales")) {
        throw std::invalid_argument("packed weight missing scales: " + name);
      } else {
      }
    } else if (tensor.dtype() != mx::float32 && tensor.dtype() != mx::float16 &&
               tensor.dtype() != mx::bfloat16) {
      throw std::invalid_argument("unsupported weight dtype: " + name);
    } else {
    }
  }
}

const mx::array &Weights::operator[](const std::string &name) const {
  return tensors.at(name);
}
bool Weights::contains(const std::string &name) const {
  return tensors.contains(name);
}
void Weights::require(const std::string &name, const mx::Shape &shape) {
  if (!contains(name) || tensors.at(name).shape() != shape) {
    throw std::invalid_argument("missing or incorrectly shaped weight: " +
                                name);
  } else {
    consumed.insert(name);
  }
}
void Weights::require_linear(const std::string &name, int input_dimensions,
                             int output_dimensions, bool bias, int experts) {
  mx::Shape leading;
  if (experts > 0) {
    leading.push_back(experts);
  } else {
  }
  leading.push_back(output_dimensions);
  if (contains(name + ".scales")) {
    if ((*this)[name + ".weight"].dtype() != mx::uint32 ||
        input_dimensions % group_size != 0) {
      throw std::invalid_argument("invalid packed linear: " + name);
    } else {
    }
    auto packed_shape = leading;
    packed_shape.push_back(input_dimensions * bits / 32);
    require(name + ".weight", packed_shape);
    leading.push_back(input_dimensions / group_size);
    require(name + ".scales", leading);
    require(name + ".biases", leading);
  } else {
    leading.push_back(input_dimensions);
    require(name + ".weight", leading);
  }
  if (bias) {
    require(name + ".bias", {output_dimensions});
  } else {
  }
}
void Weights::require_norm(const std::string &name, int dimensions, bool bias) {
  require(name + ".weight", {dimensions});
  if (bias) {
    require(name + ".bias", {dimensions});
  } else {
  }
}
void Weights::finish_validation() {
  for (const auto &[name, tensor] : tensors) {
    if (!consumed.contains(name)) {
      throw std::invalid_argument("unconsumed checkpoint weight: " + name);
    } else {
      mx::eval(tensor);
    }
  }
}
mx::array Weights::linear(const std::string &name, const mx::array &input,
                          const std::optional<mx::array> &experts) const {
  const auto &weight = (*this)[name + ".weight"];
  auto output = [&]() {
    if (contains(name + ".scales")) {
      const auto &scales = (*this)[name + ".scales"];
      const auto &biases = (*this)[name + ".biases"];
      if (experts) {
        return mx::gather_qmm(input, weight, scales, biases, {}, experts, true,
                              group_size, bits);
      } else {
        return mx::quantized_matmul(input, weight, scales, biases, true,
                                    group_size, bits);
      }
    } else if (experts) {
      return mx::gather_mm(input, mx::swapaxes(weight, -1, -2), {}, experts);
    } else {
      return mx::matmul(input, mx::swapaxes(weight, -1, -2));
    }
  }();
  return contains(name + ".bias") ? output + (*this)[name + ".bias"] : output;
}
mx::array Weights::embedding(const std::string &name,
                             const mx::array &tokens) const {
  auto selected = mx::take((*this)[name + ".weight"], tokens, 0);
  if (contains(name + ".scales")) {
    return mx::dequantize(
        selected, mx::take((*this)[name + ".scales"], tokens, 0),
        mx::take((*this)[name + ".biases"], tokens, 0), group_size, bits);
  } else {
    return selected;
  }
}
mx::array Weights::rms(const std::string &name, const mx::array &input,
                       float epsilon) const {
  return mx::fast::rms_norm(input, (*this)[name + ".weight"], epsilon);
}
mx::array Weights::norm(const std::string &name, const mx::array &input,
                        float epsilon) const {
  return mx::fast::layer_norm(input, (*this)[name + ".weight"],
                              (*this)[name + ".bias"], epsilon);
}
} // namespace qwen3_omni
