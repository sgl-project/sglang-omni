// SPDX-License-Identifier: Apache-2.0
#include "fishaudio_drama3/fishaudio_drama3_model.h"

#include <stdexcept>

namespace fishaudio_drama3 {

namespace mx = mlx::core;

namespace {

// Tensors of the backbone, the semantic head and the tower; the checkpoint's
// codec.safetensors loads into AudioCodec.
constexpr const char *kModelPrefixes[] = {"backbone.", "semantic_head.",
                                          "tower."};
// The defaults the reference takes for settings a config leaves out.
constexpr double kDefaultPartialRotaryFactor = 0.25;
constexpr float kDefaultRopeTheta = 100000.0f;
constexpr double kDefaultTowerRotaryPercent = 0.25;
constexpr float kDefaultTowerRopeBase = 1e7f;
constexpr float kDefaultTowerNormEps = 1e-6f;

void CheckSetting(const nlohmann::json &text, const char *name,
                  const nlohmann::json &expected) {
  if (!text.contains(name) || text.at(name) != expected) {
    throw std::runtime_error(std::string("unsupported text_config ") + name);
  } else {
  }
}

// The config of a checkpoint this model runs, checked at load: three gated
// delta layers per attention layer, mixture-of-experts MLPs, and the settings
// this implementation hardcodes.
nlohmann::json CheckedConfig(const std::filesystem::path &model_directory) {
  nlohmann::json config =
      omni_common::ReadJson(model_directory / "config.json");
  if (config.value("model_type", std::string()) != "fish_drama3") {
    throw std::runtime_error("config.json is not a fish_drama3 checkpoint");
  } else if (config.value("speech_template_version", std::string()) !=
             "reference-bracket-v1") {
    throw std::runtime_error("unsupported speech_template_version");
  } else {
  }
  const nlohmann::json &text = config.at("text_config");
  const size_t layer_count = text.at("num_hidden_layers").get<size_t>();
  const nlohmann::json &layer_types = text.at("layer_types");
  const nlohmann::json mlp_layer_types =
      text.value("mlp_layer_types", nlohmann::json(std::vector<std::string>(
                                        layer_count, std::string("moe"))));
  if (layer_types.size() != layer_count) {
    throw std::runtime_error("expected one layer type per layer");
  } else if (mlp_layer_types !=
             nlohmann::json(std::vector<std::string>(layer_count, "moe"))) {
    throw std::runtime_error(
        "expected a mixture-of-experts MLP in every layer");
  } else {
  }
  for (size_t layer = 0; layer < layer_count; ++layer) {
    const std::string expected =
        (layer + 1) % 4 == 0 ? "full_attention" : "linear_attention";
    if (layer_types.at(layer) != expected) {
      throw std::runtime_error(
          "expected three linear-attention layers per full one");
    } else {
    }
  }
  CheckSetting(text, "attn_output_gate", true);
  CheckSetting(text, "hidden_act", "silu");
  CheckSetting(text, "layernorm_zero_centered_gamma", true);
  CheckSetting(text, "rotary_interleaved", false);
  CheckSetting(text, "shared_expert_gate", true);
  CheckSetting(text, "use_qk_norm", true);
  const nlohmann::json &rope = text.at("rope_parameters");
  if (rope.value("type", rope.value("rope_type", std::string("default"))) !=
      "default") {
    throw std::runtime_error("unsupported rope_parameters");
  } else if (!text.value("norm_topk_prob", true)) {
    throw std::runtime_error("unsupported norm_topk_prob");
  } else {
  }
  const nlohmann::json &tower = config.at("audio_decoder_config");
  if (tower.at("vq_backbone_codebooks").get<int>() >=
          tower.at("num_codebooks").get<int>() ||
      !tower.value("attention_output_gate", false) ||
      !tower.value("rotary_interleaved", false)) {
    throw std::runtime_error("unsupported audio_decoder_config");
  } else {
  }
  return config;
}

omni_common::WeightMap
LoadModelWeights(const std::filesystem::path &model_directory) {
  omni_common::WeightMap weights =
      omni_common::LoadSafetensors(model_directory);
  std::erase_if(weights, [](const auto &named) {
    for (const char *prefix : kModelPrefixes) {
      if (named.first.starts_with(prefix)) {
        return false;
      } else {
      }
    }
    return true;
  });
  return weights;
}

} // namespace

Drama3Model::Drama3Model(const std::filesystem::path &model_directory)
    : config_(CheckedConfig(model_directory)),
      checkpoint_(LoadModelWeights(model_directory), config_),
      tokenizer_(model_directory), codec_(model_directory),
      candidate_ids_(mx::array(0)) {
  const nlohmann::json &text = config_.at("text_config");
  const nlohmann::json &tower = config_.at("audio_decoder_config");
  layer_count_ = text.at("num_hidden_layers").get<int>();
  rms_norm_eps_ = text.at("rms_norm_eps").get<float>();
  attention_head_count_ = text.at("num_attention_heads").get<int>();
  key_value_head_count_ = text.at("num_key_value_heads").get<int>();
  head_dim_ = text.at("head_dim").get<int>();
  const nlohmann::json &rope = text.at("rope_parameters");
  rotary_dims_ =
      static_cast<int>(head_dim_ * rope.value("partial_rotary_factor",
                                              kDefaultPartialRotaryFactor));
  rope_theta_ = rope.value("rope_theta", kDefaultRopeTheta);
  linear_key_head_count_ = text.at("linear_num_key_heads").get<int>();
  linear_value_head_count_ = text.at("linear_num_value_heads").get<int>();
  linear_key_head_dim_ = text.at("linear_key_head_dim").get<int>();
  linear_value_head_dim_ = text.at("linear_value_head_dim").get<int>();
  convolution_kernel_size_ = text.at("linear_conv_kernel_dim").get<int>();
  expert_count_ = text.at("num_experts").get<int>();
  experts_per_token_ = text.at("num_experts_per_tok").get<int>();
  semantic_begin_ = config_.at("semantic_start_token_id").get<int>();
  semantic_end_ = config_.at("semantic_end_token_id").get<int>();
  end_of_speech_id_ = config_.at("eos_token_id").get<int>();
  codebook_count_ = tower.at("num_codebooks").get<int>();
  codebook_size_ = tower.at("vocab_size").get<int>();
  backbone_codebook_count_ = tower.at("vq_backbone_codebooks").get<int>();
  shared_tower_layer_count_ = tower.at("shared_enrichment_layers").get<int>();
  tower_head_count_ = tower.at("n_head").get<int>();
  tower_key_value_head_count_ = tower.at("n_local_heads").get<int>();
  tower_head_dim_ = tower.at("head_dim").get<int>();
  tower_rotary_dims_ = static_cast<int>(
      tower_head_dim_ *
      tower.value("rotary_percent", kDefaultTowerRotaryPercent));
  tower_rope_base_ = tower.value("rope_base", kDefaultTowerRopeBase);
  tower_norm_eps_ = tower.value("norm_eps", kDefaultTowerNormEps);
  if (semantic_end_ - semantic_begin_ + 1 != codebook_size_) {
    throw std::runtime_error(
        "the semantic band and the codebook size disagree");
  } else {
  }
  for (const auto &[token, expected] : std::vector<std::pair<std::string, int>>{
           {"<|semantic:0|>", semantic_begin_},
           {"<|semantic:" + std::to_string(codebook_size_ - 1) + "|>",
            semantic_end_},
           {"<|im_end|>", end_of_speech_id_}}) {
    if (tokenizer_.AddedTokenId(token) != expected) {
      throw std::runtime_error("tokenizer and config disagree on " + token);
    } else {
    }
  }
  if (checkpoint_.Weight("semantic_head.weight").shape(0) !=
      codebook_size_ + 1) {
    throw std::runtime_error("semantic_head must hold the band and the end");
  } else {
  }
  candidate_ids_ = mx::concatenate(
      {mx::arange(semantic_begin_, semantic_end_ + 1, mx::int32),
       mx::array({end_of_speech_id_}, mx::int32)});
  mx::eval(candidate_ids_);
}

} // namespace fishaudio_drama3
