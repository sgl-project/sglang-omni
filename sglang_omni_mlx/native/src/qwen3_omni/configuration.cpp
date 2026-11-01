// SPDX-License-Identifier: Apache-2.0
#include "qwen3_omni/model.h"
#include <cmath>

namespace qwen3_omni {
namespace {
void require(bool supported, const std::string &description) {
  if (!supported) {
    throw std::invalid_argument("unsupported Omni configuration: " +
                                description);
  } else {
  }
}
void positive(const Json &config, std::initializer_list<const char *> names) {
  for (const auto *name : names) {
    require(config.at(name).is_number_integer() &&
                config.at(name).get<int>() > 0,
            std::string(name) + " must be positive");
  }
}
void rotary_parameters(const Json &config) {
  for (const auto *name : {"rms_norm_eps", "rope_theta"}) {
    const double value = config.at(name);
    require(std::isfinite(value) && value > 0, name);
  }
}
} // namespace

void validate_text_configuration(const Json &config) {
  positive(config,
           {"hidden_size", "head_dim", "num_attention_heads",
            "num_key_value_heads", "num_hidden_layers", "intermediate_size"});
  rotary_parameters(config);
  require(config.at("hidden_act") == "silu" &&
              !config.value("attention_bias", false) &&
              config.at("num_attention_heads").get<int>() %
                      config.at("num_key_value_heads").get<int>() ==
                  0 &&
              config.at("head_dim").get<int>() % 2 == 0 &&
              config.value("decoder_sparse_step", 1) == 1 &&
              config.value("mlp_only_layers", Json::array()).empty() &&
              config.value("norm_topk_prob", true),
          "text attention or expert routing");
  const int experts = config.value("num_experts", 0);
  require(experts >= 0 &&
              config.value("shared_expert_intermediate_size", 0) >= 0,
          "expert counts");
  if (experts > 0) {
    positive(config, {"num_experts_per_tok", "moe_intermediate_size"});
    require(config.at("num_experts_per_tok").get<int>() <= experts,
            "num_experts_per_tok exceeds num_experts");
  } else {
  }
}

Json validate_configuration(Json configuration) {
  const auto &thinker = configuration.at("thinker_config");
  const auto &talker = configuration.at("talker_config");
  const auto &text = thinker.at("text_config");
  const auto &speech = talker.at("text_config");
  const auto &predictor = talker.at("code_predictor_config");
  const auto &vision = thinker.at("vision_config");
  const auto &audio = thinker.at("audio_config");
  const auto &codec = configuration.at("code2wav_config");
  const auto &quantization = configuration.at("quantization");
  require(configuration.at("model_type") == "qwen3_omni_moe" &&
              configuration.at("enable_audio_output").get<bool>() &&
              quantization.at("bits") == 4 &&
              quantization.at("group_size") == 64 &&
              quantization.at("mode") == "affine",
          "model type, audio output or quantization");
  for (const auto *config : {&text, &speech, &predictor}) {
    validate_text_configuration(*config);
    positive(*config, {"vocab_size"});
  }
  require(text.at("rope_scaling").at("mrope_section") ==
                  Json::array({24, 20, 20}) &&
              speech.at("rope_scaling").at("mrope_section") ==
                  Json::array({24, 20, 20}) &&
              text.at("head_dim") == 128 && speech.at("head_dim") == 128 &&
              talker.at("num_code_groups") == 16 &&
              talker.at("speaker_id").at("chelsie") == 2301 &&
              thinker.at("image_token_id") == 151655 &&
              thinker.at("video_token_id") == 151656 &&
              thinker.at("audio_token_id") == 151675,
          "M-RoPE, codec groups, speaker or media tokens");
  positive(vision, {"hidden_size", "intermediate_size", "depth", "num_heads",
                    "out_hidden_size"});
  require(vision.at("hidden_size").get<int>() %
                      vision.at("num_heads").get<int>() ==
                  0 &&
              (vision.at("hidden_size").get<int>() /
               vision.at("num_heads").get<int>()) %
                      4 ==
                  0 &&
              vision.at("patch_size") == 16 &&
              vision.at("temporal_patch_size") == 2 &&
              vision.at("spatial_merge_size") == 2,
          "vision attention or patch geometry");
  for (const auto &layer : vision.at("deepstack_visual_indexes")) {
    require(layer.is_number_integer() && layer.get<int>() >= 0 &&
                layer.get<int>() < vision.at("depth").get<int>(),
            "deepstack layer index");
  }
  positive(audio, {"d_model", "encoder_attention_heads", "encoder_ffn_dim",
                   "encoder_layers", "downsample_hidden_size", "output_dim",
                   "n_window_infer"});
  require(audio.at("d_model").get<int>() > 2 &&
              audio.at("d_model").get<int>() % 2 == 0 &&
              audio.at("d_model").get<int>() %
                      audio.at("encoder_attention_heads").get<int>() ==
                  0 &&
              audio.at("num_mel_bins") == 128 && audio.at("n_window") == 50 &&
              audio.at("n_window_infer").get<int>() >= 100,
          "audio attention, mel bins or n_window (must be 50)");
  positive(codec,
           {"hidden_size", "intermediate_size", "num_hidden_layers",
            "num_attention_heads", "num_key_value_heads", "sliding_window",
            "decoder_dim", "codebook_size", "num_quantizers"});
  rotary_parameters(codec);
  require(codec.at("num_attention_heads") == codec.at("num_key_value_heads") &&
              codec.at("hidden_size").get<int>() %
                      codec.at("num_attention_heads").get<int>() ==
                  0 &&
              (codec.at("hidden_size").get<int>() /
               codec.at("num_attention_heads").get<int>()) %
                      2 ==
                  0 &&
              codec.at("num_quantizers") == talker.at("num_code_groups"),
          "vocoder attention or quantizers");
  for (const auto *name : {"upsample_rates", "upsampling_ratios"}) {
    const auto &ratios = codec.at(name);
    require(ratios.is_array() && !ratios.empty() && ratios.size() < 16, name);
    for (const auto &ratio : ratios) {
      require(ratio.is_number_integer() && ratio.get<int>() > 0 &&
                  ratio.get<int>() <= 16,
              name);
    }
  }
  require(codec.at("decoder_dim").get<int>() %
                  (1 << codec.at("upsample_rates").size()) ==
              0,
          "vocoder channel halving");
  require(vision.at("out_hidden_size") == text.at("hidden_size") &&
              audio.at("output_dim") == text.at("hidden_size") &&
              talker.at("thinker_hidden_size") == text.at("hidden_size") &&
              predictor.at("hidden_size") == speech.at("hidden_size"),
          "cross-component hidden dimensions");
  return configuration;
}
} // namespace qwen3_omni
