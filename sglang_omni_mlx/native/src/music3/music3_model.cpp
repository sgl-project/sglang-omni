// SPDX-License-Identifier: Apache-2.0
#include "music3/music3_model.h"

#include <algorithm>
#include <stdexcept>

namespace music3 {
namespace mx = mlx::core;

namespace {

nlohmann::json ReadConfig(const std::filesystem::path &directory) {
  const auto config = qwen3_asr::ReadJson(directory / "config.json");
  const nlohmann::json expected = {{"model_type", "minimax_music3"},
                                   {"num_codebooks", 8},
                                   {"num_condition_layers", 8},
                                   {"audio_code_offset", 151675},
                                   {"audio_cfg_token_id", 151654},
                                   {"audio_end_token_id", 151670},
                                   {"input_sampling_rate", 24000},
                                   {"input_hop_length", 960},
                                   {"output_sampling_rate", 44100},
                                   {"output_hop_length", 512},
                                   {"vocoder_upsampling_ratios", {8, 8, 4, 2}}};
  for (const auto &[name, value] : expected.items()) {
    if (!config.contains(name) || config.at(name) != value) {
      throw std::invalid_argument("unsupported Music3 config: " + name);
    } else {
    }
  }
  const std::vector<std::string> dimensions = {"hidden_size",
                                               "num_hidden_layers",
                                               "intermediate_size",
                                               "num_attention_heads",
                                               "num_key_value_heads",
                                               "head_dim",
                                               "vocab_size",
                                               "semantic_vocab_size",
                                               "audio_vocab_size",
                                               "depth_num_layers",
                                               "depth_num_heads",
                                               "depth_intermediate_size",
                                               "depth_max_position_embeddings",
                                               "condition_out_dim",
                                               "dit_in_channels",
                                               "dit_num_layers",
                                               "dit_num_heads",
                                               "dit_head_dim",
                                               "dit_ff_inner_dim",
                                               "dit_fourier_dim",
                                               "dit_rotary_dim",
                                               "vocoder_input_dim",
                                               "vocoder_hidden_dim",
                                               "max_position_embeddings"};
  for (const auto &name : dimensions) {
    if (!config.at(name).is_number_integer() ||
        config.at(name).get<int>() <= 0) {
      throw std::invalid_argument("Music3 requires a positive integer " + name);
    } else {
    }
  }
  if (config.at("hidden_size").get<int>() %
              config.at("depth_num_heads").get<int>() !=
          0 ||
      config.at("num_attention_heads").get<int>() %
              config.at("num_key_value_heads").get<int>() !=
          0 ||
      config.at("dit_in_channels").get<int>() % 2 != 0 ||
      config.at("dit_fourier_dim").get<int>() % 2 != 0 ||
      config.at("dit_rotary_dim").get<int>() % 2 != 0 ||
      config.at("dit_rotary_dim").get<int>() >
          config.at("dit_head_dim").get<int>() ||
      config.at("depth_max_position_embeddings").get<int>() < 8 ||
      config.at("vocoder_hidden_dim").get<int>() % 16 != 0 ||
      config.at("vocab_size").get<int>() <
          151675 + config.at("semantic_vocab_size").get<int>() ||
      !(config.at("rms_norm_eps").get<float>() > 0) ||
      !(config.at("rope_theta").get<float>() > 0)) {
    throw std::invalid_argument("incompatible Music3 architecture dimensions");
  } else {
  }
  return config;
}

qwen3_asr::WeightMap LoadWeights(const std::filesystem::path &directory,
                                 const nlohmann::json &config) {
  auto weights = qwen3_asr::LoadSafetensors(directory);
  const auto shape = [&](const std::string &name, const mx::Shape &expected) {
    const auto found = weights.find(name);
    if (found == weights.end() || found->second.shape() != expected) {
      throw std::invalid_argument("missing or incompatible Music3 tensor: " +
                                  name);
    } else {
    }
  };
  const auto linear = [&](const std::string &prefix, int input, int output,
                          bool bias = false) {
    if (weights.count(prefix + ".scales") != 0) {
      const auto &quantization = config.at("quantization");
      const int bits = quantization.at("bits");
      const int group_size = quantization.at("group_size");
      const std::string mode =
          quantization.value("mode", std::string("affine"));
      if ((mode != "affine" && mode != "mxfp8" && mode != "mxfp4") ||
          (bits != 4 && bits != 8) || group_size <= 0 ||
          input % group_size != 0) {
        throw std::invalid_argument("unsupported Music3 quantization");
      } else {
      }
      shape(prefix + ".weight", {output, input * bits / 32});
      shape(prefix + ".scales", {output, input / group_size});
      if (mode == "affine") {
        shape(prefix + ".biases", {output, input / group_size});
      } else {
      }
    } else {
      shape(prefix + ".weight", {output, input});
    }
    if (bias) {
      shape(prefix + ".bias", {output});
    } else {
    }
  };
  const auto convolution = [&](const std::string &prefix, int input, int output,
                               int kernel, bool bias = true) {
    shape(prefix + ".weight", {output, kernel, input});
    if (bias) {
      shape(prefix + ".bias", {output});
    } else {
    }
  };
  const int hidden = config.at("hidden_size");
  const int vocabulary = config.at("vocab_size");
  shape("language_model.model.embed_tokens.weight", {vocabulary, hidden});
  shape("language_model.model.norm.weight", {hidden});
  if (!config.at("tie_word_embeddings").get<bool>()) {
    linear("language_model.lm_head", hidden, vocabulary);
  } else {
  }
  for (int layer = 0; layer < config.at("num_hidden_layers").get<int>();
       ++layer) {
    const std::string prefix =
        "language_model.model.layers." + std::to_string(layer);
    const int heads = config.at("num_attention_heads");
    const int kv_heads = config.at("num_key_value_heads");
    const int head_dim = config.at("head_dim");
    shape(prefix + ".input_layernorm.weight", {hidden});
    shape(prefix + ".post_attention_layernorm.weight", {hidden});
    shape(prefix + ".self_attn.q_norm.weight", {head_dim});
    shape(prefix + ".self_attn.k_norm.weight", {head_dim});
    linear(prefix + ".self_attn.q_proj", hidden, heads * head_dim);
    linear(prefix + ".self_attn.k_proj", hidden, kv_heads * head_dim);
    linear(prefix + ".self_attn.v_proj", hidden, kv_heads * head_dim);
    linear(prefix + ".self_attn.o_proj", heads * head_dim, hidden);
    const int intermediate = config.at("intermediate_size");
    linear(prefix + ".mlp.gate_proj", hidden, intermediate);
    linear(prefix + ".mlp.up_proj", hidden, intermediate);
    linear(prefix + ".mlp.down_proj", intermediate, hidden);
  }
  shape("rvq_depth_decoder.audio_embeddings.weight",
        {7 * config.at("audio_vocab_size").get<int>(), hidden});
  shape("rvq_depth_decoder.pos_embedding.weight",
        {config.at("depth_max_position_embeddings").get<int>(), hidden});
  shape("rvq_depth_decoder.norm.weight", {hidden});
  linear("rvq_depth_decoder.projection", hidden, hidden);
  for (int head = 0; head < 7; ++head)
    linear("rvq_depth_decoder.audio_heads." + std::to_string(head), hidden,
           config.at("audio_vocab_size"));
  for (int layer = 0; layer < config.at("depth_num_layers").get<int>();
       ++layer) {
    const std::string prefix =
        "rvq_depth_decoder.layers." + std::to_string(layer);
    shape(prefix + ".input_layernorm.weight", {hidden});
    shape(prefix + ".post_attention_layernorm.weight", {hidden});
    for (const auto *projection : {"to_q", "to_k", "to_v", "to_out"})
      linear(prefix + ".attn." + projection, hidden, hidden);
    const int intermediate = config.at("depth_intermediate_size");
    linear(prefix + ".gate_proj", hidden, intermediate);
    linear(prefix + ".up_proj", hidden, intermediate);
    linear(prefix + ".down_proj", intermediate, hidden);
  }
  const int condition = config.at("condition_out_dim");
  shape("condition_encoder.layer_weight_logits", {8});
  shape("condition_encoder.layer_scale", {1});
  convolution("condition_encoder.proj", hidden, condition, 3);
  const int inner = config.at("dit_num_heads").get<int>() *
                    config.at("dit_head_dim").get<int>();
  const int channels = config.at("dit_in_channels");
  const int combined = 2 * channels + condition;
  const int fourier = config.at("dit_fourier_dim");
  shape("transformer.time_proj.weight", {fourier / 2, 1});
  linear("transformer.time_embed.linear_1", fourier, inner, true);
  linear("transformer.time_embed.linear_2", inner, inner, true);
  convolution("transformer.preprocess_conv", combined, combined, 1, false);
  linear("transformer.proj_in", combined, inner);
  linear("transformer.proj_out", inner, channels);
  convolution("transformer.postprocess_conv", channels, channels, 1, false);
  for (int layer = 0; layer < config.at("dit_num_layers").get<int>(); ++layer) {
    const std::string prefix =
        "transformer.transformer_blocks." + std::to_string(layer);
    for (const auto *norm : {"norm1", "norm2"}) {
      shape(prefix + "." + norm + ".weight", {inner});
      shape(prefix + "." + norm + ".bias", {inner});
    }
    for (const auto *projection : {"to_q", "to_k", "to_v", "to_out.0"})
      linear(prefix + ".attn." + projection, inner, inner);
    const int intermediate = config.at("dit_ff_inner_dim");
    linear(prefix + ".ff_in", inner, intermediate * 2, true);
    linear(prefix + ".ff_out", intermediate, inner, true);
  }
  convolution("vocoder.dec_in_proj", channels / 2,
              config.at("vocoder_input_dim"), 1);
  int vocoder_hidden = config.at("vocoder_hidden_dim");
  convolution("vocoder.conv_in", config.at("vocoder_input_dim"), vocoder_hidden,
              7);
  const auto ratios =
      config.at("vocoder_upsampling_ratios").get<std::vector<int>>();
  for (size_t index = 0; index < ratios.size(); ++index) {
    const std::string prefix = "vocoder.blocks." + std::to_string(index);
    shape(prefix + ".snake1.alpha", {1, vocoder_hidden, 1});
    convolution(prefix + ".conv_t1", vocoder_hidden, vocoder_hidden / 2,
                2 * ratios[index]);
    vocoder_hidden /= 2;
    for (int unit = 1; unit <= 3; ++unit) {
      const std::string residual = prefix + ".res_unit" + std::to_string(unit);
      shape(residual + ".snake1.alpha", {1, vocoder_hidden, 1});
      shape(residual + ".snake2.alpha", {1, vocoder_hidden, 1});
      convolution(residual + ".conv1", vocoder_hidden, vocoder_hidden, 7);
      convolution(residual + ".conv2", vocoder_hidden, vocoder_hidden, 1);
    }
  }
  shape("vocoder.snake_out.alpha", {1, vocoder_hidden, 1});
  convolution("vocoder.conv_out", vocoder_hidden, 1, 7);
  return weights;
}

} // namespace

Model::Model(const std::filesystem::path &directory)
    : config_(ReadConfig(directory)),
      checkpoint_(LoadWeights(directory, config_), config_),
      decoder_(checkpoint_, "language_model.model",
               qwen3_asr::Qwen3DecoderConfig::FromJson(config_)),
      tokenizer_(directory / "tokenizer") {
  const std::vector<std::pair<std::string, int>> special_tokens = {
      {"<|im_start|>", 151644},    {"<|im_end|>", 151645},
      {"<|audio_cfg|>", 151654},   {"<|audio_start|>", 151669},
      {"<|audio_end|>", 151670},   {"<|caption_start|>", 151671},
      {"<|caption_end|>", 151672}, {"<|lyrics_start|>", 151673},
      {"<|lyrics_end|>", 151674}};
  for (const auto &[token, expected] : special_tokens) {
    if (tokenizer_.AddedTokenId(token) != expected) {
      throw std::invalid_argument("Music3 tokenizer mismatch for " + token);
    } else {
    }
  }
}

} // namespace music3
