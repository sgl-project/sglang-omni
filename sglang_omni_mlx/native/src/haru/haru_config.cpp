// SPDX-License-Identifier: Apache-2.0
#include "haru/haru_config.h"

#include <fstream>

namespace haru {

namespace {

int RequireInt(const nlohmann::json &obj, const char *key) {
  if (!obj.contains(key) || !obj[key].is_number_integer()) {
    throw std::invalid_argument(std::string("config field ") + key + " must be an integer");
  }
  return obj[key].get<int>();
}

std::vector<int> IntVector(const nlohmann::json &obj, const char *key) {
  if (!obj.contains(key) || !obj[key].is_array()) {
    throw std::invalid_argument(std::string("config field ") + key + " must be an array");
  }
  std::vector<int> out;
  for (const auto &value : obj[key]) out.push_back(value.get<int>());
  return out;
}

TextConfig ParseText(const nlohmann::json &lm) {
  TextConfig text;
  text.vocab_size = RequireInt(lm, "vocab_size");
  text.hidden_size = RequireInt(lm, "hidden_size");
  text.intermediate_size = RequireInt(lm, "intermediate_size");
  text.num_attention_heads = RequireInt(lm, "num_attention_heads");
  text.num_key_value_heads = RequireInt(lm, "num_key_value_heads");
  text.head_dim = RequireInt(lm, "head_dim");
  text.linear_num_key_heads = RequireInt(lm, "linear_num_key_heads");
  text.linear_num_value_heads = RequireInt(lm, "linear_num_value_heads");
  text.linear_key_head_dim = RequireInt(lm, "linear_key_head_dim");
  text.linear_value_head_dim = RequireInt(lm, "linear_value_head_dim");
  text.linear_conv_kernel_dim = RequireInt(lm, "linear_conv_kernel_dim");
  text.rms_norm_eps = lm.value("rms_norm_eps", 1e-6f);

  // layer_types: either explicit, or derived from full_attention_interval.
  const int layers = RequireInt(lm, "num_hidden_layers");
  if (lm.contains("layer_types") && lm["layer_types"].is_array()) {
    for (const auto &kind : lm["layer_types"]) {
      text.layer_types.push_back(kind.get<std::string>());
    }
  } else {
    const int interval = lm.value("full_attention_interval", 4);
    for (int i = 0; i < layers; ++i) {
      text.layer_types.push_back((i + 1) % interval == 0 ? kFull : kLinear);
    }
  }
  if (static_cast<int>(text.layer_types.size()) != layers) {
    throw std::invalid_argument("layer_types must name one kind per hidden layer");
  }

  // Rotary: prefer rope_parameters, fall back to the flat fields.
  const auto *rope = lm.contains("rope_parameters") ? &lm["rope_parameters"] : nullptr;
  text.rope_theta = rope != nullptr && rope->contains("rope_theta")
                        ? (*rope)["rope_theta"].get<float>()
                        : lm.value("rope_theta", 10'000'000.0f);
  text.partial_rotary_factor =
      rope != nullptr && rope->contains("partial_rotary_factor")
          ? (*rope)["partial_rotary_factor"].get<float>()
          : lm.value("partial_rotary_factor", 0.25f);
  return text;
}

CodecConfig ParseCodec(const nlohmann::json &codec) {
  CodecConfig out;
  if (codec.contains("latent_dim")) out.latent_dim = codec["latent_dim"].get<int>();
  if (codec.contains("base_channels")) out.base_channels = codec["base_channels"].get<int>();
  if (codec.contains("ratios")) out.ratios = IntVector(codec, "ratios");
  if (codec.contains("encoder_depths")) out.encoder_depths = IntVector(codec, "encoder_depths");
  if (codec.contains("decoder_depths")) out.decoder_depths = IntVector(codec, "decoder_depths");
  if (codec.contains("norm_eps")) out.norm_eps = codec["norm_eps"].get<float>();
  if (codec.contains("latent_std")) out.latent_std = codec["latent_std"].get<float>();
  return out;
}

HeadConfig ParseHead(const nlohmann::json &head) {
  HeadConfig out;
  if (head.contains("hidden_size")) out.hidden_size = head["hidden_size"].get<int>();
  if (head.contains("latent_dim")) out.latent_dim = head["latent_dim"].get<int>();
  if (head.contains("num_layers")) out.num_layers = head["num_layers"].get<int>();
  if (head.contains("ffn_ratio")) out.ffn_ratio = head["ffn_ratio"].get<float>();
  if (head.contains("norm_eps")) out.norm_eps = head["norm_eps"].get<float>();
  if (head.contains("interval_embed")) out.interval_embed = head["interval_embed"].get<float>();
  if (head.contains("num_steps")) out.num_steps = head["num_steps"].get<int>();
  if (head.contains("guidance_scale")) out.guidance_scale = head["guidance_scale"].get<float>();
  if (head.contains("guidance_range") && head["guidance_range"].is_array()) {
    out.guidance_range = {head["guidance_range"][0].get<float>(),
                          head["guidance_range"][1].get<float>()};
  }
  return out;
}

} // namespace

HaruConfig HaruConfig::FromDict(const nlohmann::json &params) {
  const std::string model_type = params.value("model_type", "");
  if (model_type != "haru") {
    throw std::invalid_argument("not a Haru checkpoint (model_type=" + model_type + ")");
  }
  HaruConfig config;
  config.sample_rate = RequireInt(params, "sample_rate");
  config.frame_samples = RequireInt(params, "frame_samples");
  config.latent_dim = RequireInt(params, "latent_dim");
  config.pad_token_id = RequireInt(params, "pad_token_id");
  config.eop_token_id = RequireInt(params, "eop_token_id");

  if (!params.contains("lm") || !params["lm"].is_object()) {
    throw std::invalid_argument("config field lm must be an object");
  }
  config.lm = ParseText(params["lm"]);
  // The end-of-speech id lives under lm, like HaruConfig.eos_token_id.
  if (params["lm"].contains("eos_token_id")) {
    config.eos_token_id = params["lm"]["eos_token_id"].get<int>();
  }
  if (params.contains("codec") && params["codec"].is_object()) {
    config.codec = ParseCodec(params["codec"]);
  }
  if (params.contains("head") && params["head"].is_object()) {
    config.head = ParseHead(params["head"]);
  }
  if (config.codec.frame_samples() != config.frame_samples) {
    throw std::invalid_argument("codec ratios do not match frame_samples");
  }
  return config;
}

HaruConfig HaruConfig::FromFile(const std::filesystem::path &path) {
  std::ifstream file(path);
  if (!file.is_open()) {
    throw std::invalid_argument("cannot open config: " + path.string());
  }
  nlohmann::json params;
  file >> params;
  return FromDict(params);
}

} // namespace haru
