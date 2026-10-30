#include "ming3_asr/ming3_asr_model.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "qwen3_asr/qwen3_asr_transcriber.h"

namespace ming3_asr {

namespace mx = mlx::core;

MingConfig::MingConfig(const nlohmann::json &config) {
  const auto require = [](bool condition, const char *message) {
    if (!condition) {
      throw std::invalid_argument(message);
    } else {
    }
  };
  const auto integer_value = [](const nlohmann::json &value,
                                const char *field_name, int minimum,
                                int maximum) {
    if (!value.is_number_integer() || value < minimum || value > maximum) {
      throw std::invalid_argument(std::string("Ming ASR ") + field_name +
                                  " must be an integer within [" +
                                  std::to_string(minimum) + ", " +
                                  std::to_string(maximum) + "]");
    } else {
      return value.get<int>();
    }
  };
  const auto positive_dimension = [&](const nlohmann::json &section,
                                      const char *field_name) {
    return integer_value(section.at(field_name), field_name, 1,
                         std::numeric_limits<int>::max());
  };
  const auto token_ids = [&](const nlohmann::json &value,
                             const char *field_name, int vocabulary_size) {
    std::vector<int> ids;
    if (value.is_array()) {
      for (const auto &entry : value) {
        ids.push_back(integer_value(entry, field_name, 0, vocabulary_size - 1));
      }
    } else {
      ids.push_back(integer_value(value, field_name, 0, vocabulary_size - 1));
    }
    if (ids.empty()) {
      throw std::invalid_argument(std::string("Ming ASR ") + field_name +
                                  " must not be empty");
    } else {
    }
    return ids;
  };
  const auto &text_config = config.at("text_config");
  const auto &audio_config = config.at("audio_config");
  hidden_size = positive_dimension(text_config, "hidden_size");
  vocabulary_size = positive_dimension(text_config, "vocab_size");
  intermediate_size = positive_dimension(text_config, "intermediate_size");
  expert_intermediate_size =
      positive_dimension(text_config, "moe_intermediate_size");
  shared_intermediate_size =
      positive_dimension(text_config, "moe_shared_expert_intermediate_size");
  layer_count = positive_dimension(text_config, "num_hidden_layers");
  attention_head_count = positive_dimension(text_config, "num_attention_heads");
  attention_head_dimension = positive_dimension(text_config, "head_dim");
  layers_per_attention_group =
      positive_dimension(text_config, "layer_group_size");
  key_value_lora_rank = positive_dimension(text_config, "kv_lora_rank");
  query_lora_rank = positive_dimension(text_config, "q_lora_rank");
  rotary_head_dimension = positive_dimension(text_config, "qk_rope_head_dim");
  non_rotary_head_dimension =
      positive_dimension(text_config, "qk_nope_head_dim");
  value_head_dimension = positive_dimension(text_config, "v_head_dim");
  convolution_kernel_token_count =
      positive_dimension(text_config, "short_conv_kernel_size");
  expert_count = positive_dimension(text_config, "num_experts");
  selected_expert_count =
      positive_dimension(text_config, "num_experts_per_tok");
  expert_group_count = positive_dimension(text_config, "n_group");
  selected_expert_group_count = positive_dimension(text_config, "topk_group");
  dense_layer_count = integer_value(text_config.at("first_k_dense_replace"),
                                    "first_k_dense_replace", 0, layer_count);
  max_context_token_count =
      positive_dimension(text_config, "max_position_embeddings");
  eos_token_ids = token_ids(text_config.at("eos_token_id"), "eos_token_id",
                            vocabulary_size);
  audio_token_id = integer_value(config.at("audio_token_id"), "audio_token_id",
                                 0, vocabulary_size - 1);
  audio_hidden_dimension = positive_dimension(audio_config, "n_audio_state");
  audio_attention_head_count = positive_dimension(audio_config, "n_audio_head");
  audio_layer_count = positive_dimension(audio_config, "n_audio_layer");
  audio_context_frame_count = positive_dimension(audio_config, "n_audio_ctx");
  constexpr int kDefaultAudioChunkFrameCount = 3000;
  audio_chunk_frame_count =
      audio_config.contains("whisper_chunk_size")
          ? positive_dimension(audio_config, "whisper_chunk_size")
          : kDefaultAudioChunkFrameCount;
  const auto finite_number = [](const nlohmann::json &section,
                                const char *field_name) {
    const auto &value = section.at(field_name);
    if (!value.is_number() || !std::isfinite(value.get<double>()) ||
        std::abs(value.get<double>()) > std::numeric_limits<float>::max()) {
      throw std::invalid_argument(std::string("Ming ASR ") + field_name +
                                  " must be a finite float");
    } else {
      return static_cast<float>(value.get<double>());
    }
  };
  rms_norm_epsilon = finite_number(text_config, "rms_norm_eps");
  rope_theta = finite_number(text_config, "rope_theta");
  routed_scaling_factor = finite_number(text_config, "routed_scaling_factor");
  kda_lower_bound = finite_number(text_config, "kda_lower_bound");
  should_normalize_audio_embeddings = config.value("norm_query_embeds", true);
  if (config.contains("quantization")) {
    const auto &quantization_config = config.at("quantization");
    quantization = {positive_dimension(quantization_config, "group_size"),
                    positive_dimension(quantization_config, "bits"),
                    quantization_config.value("mode", std::string("affine"))};
    require(quantization.group_size == 64 && quantization.bits == 4 &&
                quantization.mode == "affine",
            "Ming ASR requires affine 4-bit weights with group_size=64");
  } else {
    throw std::invalid_argument("Ming ASR checkpoint is missing quantization");
  }
  require(config.at("model_type") == "ming_v3_asr",
          "Ming ASR model_type must be ming_v3_asr");
  require(text_config.at("no_kda_lora").get<bool>(),
          "Ming ASR requires no_kda_lora=true");
  require(text_config.at("kda_safe_gate").get<bool>(),
          "Ming ASR requires kda_safe_gate=true");
  require(text_config.at("rope_interleave").get<bool>(),
          "Ming ASR requires rope_interleave=true");
  require(text_config.at("hidden_act") == "silu",
          "Ming ASR requires hidden_act=silu");
  require(text_config.at("use_qkv_bias") == false,
          "Ming ASR requires use_qkv_bias=false");
  require(text_config.at("use_bias") == false,
          "Ming ASR requires use_bias=false");
  require(text_config.at("tie_word_embeddings") == false,
          "Ming ASR requires tie_word_embeddings=false");
  require(text_config.at("gated_attention_proj_granularity_type") ==
              "head_wise",
          "Ming ASR requires head-wise attention projections");
  require(positive_dimension(text_config, "num_shared_experts") == 1,
          "Ming ASR requires one shared expert");
  require(text_config.at("rope_scaling").is_null(),
          "Ming ASR requires rope_scaling=null");
  require(text_config.at("scoring_func") == "sigmoid",
          "Ming ASR requires scoring_func=sigmoid");
  require(!text_config.contains("expert_swiglu_limit_list"),
          "Ming ASR does not support expert_swiglu_limit_list");
  require(!text_config.contains("share_expert_swiglu_limit_list"),
          "Ming ASR does not support share_expert_swiglu_limit_list");
  require(positive_dimension(audio_config, "n_mels") == 128,
          "Ming ASR requires n_mels=128");
  require(audio_config.at("downsampler_kernel_size_list") ==
              nlohmann::json::array({3}),
          "Ming ASR requires one downsampler kernel of size 3");
  require(audio_config.at("downsampler_stride_list") ==
              nlohmann::json::array({2}),
          "Ming ASR requires one downsampler stride of 2");
  require(audio_config.value("downsampler_padding_list",
                             nlohmann::json::array({1})) ==
              nlohmann::json::array({1}),
          "Ming ASR requires one downsampler padding of 1");
  require(attention_head_dimension % 32 == 0,
          "Ming ASR head_dim must be divisible by 32");
  require(rotary_head_dimension % 2 == 0,
          "Ming ASR qk_rope_head_dim must be even");
  require(expert_count % expert_group_count == 0,
          "Ming ASR num_experts must be divisible by n_group");
  require(selected_expert_group_count <= expert_group_count,
          "Ming ASR topk_group must not exceed n_group");
  require(selected_expert_count <=
              selected_expert_group_count * (expert_count / expert_group_count),
          "Ming ASR num_experts_per_tok exceeds the selected expert groups");
  require(audio_hidden_dimension % audio_attention_head_count == 0,
          "Ming ASR n_audio_state must be divisible by n_audio_head");
  require((audio_chunk_frame_count + 1LL) / 2 <= audio_context_frame_count,
          "Ming ASR whisper_chunk_size exceeds the encoder positional context");
  require(rms_norm_epsilon > 0 && std::isfinite(rms_norm_epsilon),
          "Ming ASR rms_norm_eps must be finite and positive");
  require(rope_theta > 0 && std::isfinite(rope_theta),
          "Ming ASR rope_theta must be finite and positive");
  require(kda_lower_bound < 0 && std::isfinite(kda_lower_bound),
          "Ming ASR kda_lower_bound must be finite and negative");
  require(routed_scaling_factor > 0 && std::isfinite(routed_scaling_factor),
          "Ming ASR routed_scaling_factor must be finite and positive");
}

bool MingConfig::UsesMla(int layer_index) const {
  return (layer_index + 1) % layers_per_attention_group == 0 ||
         layer_index >= layer_count / layers_per_attention_group *
                            layers_per_attention_group;
}

int MingConfig::AudioTokenCount(int frame_count) const {
  int audio_token_count = 0;
  for (int start = 0; start < frame_count; start += audio_chunk_frame_count) {
    const int chunk_frames =
        std::min(audio_chunk_frame_count, frame_count - start);
    audio_token_count += ((chunk_frames + 1) / 2 + 1) / 2;
  }
  return audio_token_count;
}

MingModel::MingModel(const std::filesystem::path &model_directory)
    : MingModel(omni_common::ReadJson(model_directory / "config.json"),
                model_directory) {}

MingModel::MingModel(const nlohmann::json &configuration,
                     const std::filesystem::path &model_directory)
    : config_(configuration),
      checkpoint_(omni_common::LoadSafetensors(model_directory),
                  configuration) {
  ValidateWeights();
}

void MingModel::ValidateWeights() const {
  const auto expect = [&](const std::string &weight_name,
                          const mx::Shape &shape) {
    if (!checkpoint_.Has(weight_name)) {
      throw std::invalid_argument("Ming checkpoint missing weight: " +
                                  weight_name);
    } else if (checkpoint_.Weight(weight_name).shape() != shape) {
      throw std::invalid_argument(
          "Ming checkpoint " + weight_name + " shape: expected " +
          nlohmann::json(shape).dump() + ", got " +
          nlohmann::json(checkpoint_.Weight(weight_name).shape()).dump());
    } else {
    }
  };
  const auto linear = [&](const std::string &weight_prefix, int input_dimension,
                          int output_dimension, int expert_count = 0,
                          bool has_bias = false) {
    mx::Shape shape =
        expert_count > 0
            ? mx::Shape{expert_count, output_dimension, input_dimension}
            : mx::Shape{output_dimension, input_dimension};
    if (checkpoint_.Has(weight_prefix + ".scales")) {
      if (config_.quantization.bits == 0 ||
          input_dimension % config_.quantization.group_size != 0 ||
          checkpoint_.Weight(weight_prefix + ".weight").dtype() != mx::uint32) {
        throw std::invalid_argument("invalid Ming quantized projection: " +
                                    weight_prefix);
      } else {
      }
      shape.back() = input_dimension / config_.quantization.group_size;
      expect(weight_prefix + ".scales", shape);
      expect(weight_prefix + ".biases", shape);
      shape.back() = input_dimension / (32 / config_.quantization.bits);
    } else {
    }
    expect(weight_prefix + ".weight", shape);
    if (has_bias) {
      expect(weight_prefix + ".bias", {output_dimension});
    } else {
    }
  };
  const int hidden_dimension = config_.hidden_size;
  const int audio_hidden_dimension = config_.audio_hidden_dimension;
  const int projection_dimension =
      config_.attention_head_count * config_.attention_head_dimension;
  linear("model.word_embeddings", hidden_dimension, config_.vocabulary_size);
  linear("lm_head", hidden_dimension, config_.vocabulary_size);
  expect("model.norm.weight", {hidden_dimension});
  expect("model.audio.conv1.weight", {audio_hidden_dimension, 3, 128});
  expect("model.audio.conv1.bias", {audio_hidden_dimension});
  expect("model.audio.conv2.weight",
         {audio_hidden_dimension, 3, audio_hidden_dimension});
  expect("model.audio.conv2.bias", {audio_hidden_dimension});
  expect("model.audio.positional_embedding",
         {config_.audio_context_frame_count, audio_hidden_dimension});
  for (int layer_index = 0; layer_index < config_.audio_layer_count;
       ++layer_index) {
    const std::string weight_prefix =
        "model.audio.blocks." + std::to_string(layer_index);
    for (const std::string norm : {"attn_ln", "mlp_ln"}) {
      expect(weight_prefix + "." + norm + ".weight", {audio_hidden_dimension});
      expect(weight_prefix + "." + norm + ".bias", {audio_hidden_dimension});
    }
    for (const std::string weight_name : {"query", "key", "value", "out"}) {
      linear(weight_prefix + ".attn." + weight_name, audio_hidden_dimension,
             audio_hidden_dimension, 0, weight_name != "key");
    }
    linear(weight_prefix + ".mlp.0", audio_hidden_dimension,
           4 * audio_hidden_dimension, 0, true);
    linear(weight_prefix + ".mlp.2", 4 * audio_hidden_dimension,
           audio_hidden_dimension, 0, true);
  }
  expect("model.audio.ln_post.weight", {audio_hidden_dimension});
  expect("model.audio.ln_post.bias", {audio_hidden_dimension});
  expect("linear_proj_audio.0.weight",
         {hidden_dimension, 3, audio_hidden_dimension});
  expect("linear_proj_audio.0.bias", {hidden_dimension});
  linear("linear_proj_audio.3", hidden_dimension, hidden_dimension, 0, true);
  for (int layer_index = 0; layer_index < config_.layer_count; ++layer_index) {
    const std::string weight_prefix =
        "model.layers." + std::to_string(layer_index);
    expect(weight_prefix + ".input_layernorm.weight", {hidden_dimension});
    expect(weight_prefix + ".post_attention_layernorm.weight",
           {hidden_dimension});
    const std::string attention = weight_prefix + ".attention";
    if (config_.UsesMla(layer_index)) {
      linear(attention + ".q_a_proj", hidden_dimension,
             config_.query_lora_rank);
      expect(attention + ".q_a_layernorm.weight", {config_.query_lora_rank});
      linear(attention + ".q_b_proj", config_.query_lora_rank,
             config_.attention_head_count * (config_.non_rotary_head_dimension +
                                             config_.rotary_head_dimension));
      linear(attention + ".kv_a_proj_with_mqa", hidden_dimension,
             config_.key_value_lora_rank + config_.rotary_head_dimension);
      expect(attention + ".kv_a_layernorm.weight",
             {config_.key_value_lora_rank});
      linear(attention + ".kv_b_proj", config_.key_value_lora_rank,
             config_.attention_head_count * (config_.non_rotary_head_dimension +
                                             config_.value_head_dimension));
      linear(attention + ".g_proj", hidden_dimension,
             config_.attention_head_count);
      linear(attention + ".dense",
             config_.attention_head_count * config_.value_head_dimension,
             hidden_dimension);
    } else {
      for (const std::string weight_name : {"q", "k", "v"}) {
        linear(attention + "." + weight_name + "_proj", hidden_dimension,
               projection_dimension);
        expect(
            attention + "." + weight_name + "_conv1d.weight",
            {projection_dimension, config_.convolution_kernel_token_count, 1});
      }
      linear(attention + ".b_proj", hidden_dimension,
             config_.attention_head_count);
      linear(attention + ".f_proj", hidden_dimension, projection_dimension);
      linear(attention + ".g_proj", hidden_dimension, projection_dimension);
      linear(attention + ".o_proj", projection_dimension, hidden_dimension);
      expect(attention + ".A_log", {config_.attention_head_count});
      expect(attention + ".dt_bias", {projection_dimension});
      expect(attention + ".o_norm.weight", {config_.attention_head_dimension});
    }
    const std::string feed_forward = weight_prefix + ".mlp";
    if (layer_index < config_.dense_layer_count) {
      linear(feed_forward + ".gate_proj", hidden_dimension,
             config_.intermediate_size);
      linear(feed_forward + ".up_proj", hidden_dimension,
             config_.intermediate_size);
      linear(feed_forward + ".down_proj", config_.intermediate_size,
             hidden_dimension);
    } else {
      expect(feed_forward + ".gate.weight",
             {config_.expert_count, hidden_dimension});
      expect(feed_forward + ".gate.expert_bias", {config_.expert_count});
      for (const std::string weight_name : {"gate_proj", "up_proj"}) {
        linear(feed_forward + ".switch_mlp." + weight_name, hidden_dimension,
               config_.expert_intermediate_size, config_.expert_count);
        linear(feed_forward + ".shared_experts." + weight_name,
               hidden_dimension, config_.shared_intermediate_size);
      }
      linear(feed_forward + ".switch_mlp.down_proj",
             config_.expert_intermediate_size, hidden_dimension,
             config_.expert_count);
      linear(feed_forward + ".shared_experts.down_proj",
             config_.shared_intermediate_size, hidden_dimension);
    }
  }
}

mx::array MingModel::EmbedTokens(const std::vector<int> &token_ids) const {
  return checkpoint_.Embed(
      mx::array(token_ids.data(), {1, static_cast<int>(token_ids.size())}),
      "model.word_embeddings");
}

MingCache::MingCache(const MingModel *owner) : owner_(owner) {
  layers_.reserve(owner->config().layer_count);
  for (int layer_index = 0; layer_index < owner->config().layer_count;
       ++layer_index) {
    if (owner->config().UsesMla(layer_index)) {
      layers_.emplace_back(LatentAttentionCache{});
    } else {
      layers_.emplace_back(DeltaCache{});
    }
  }
}

MingCache MingModel::NewCache() const { return MingCache(this); }

mx::array MingModel::Forward(const mx::array &embeddings, MingCache &cache,
                             const std::atomic<bool> &cancel) const {
  if (cache.owner_ != this ||
      cache.layers_.size() != static_cast<size_t>(config_.layer_count)) {
    throw std::invalid_argument("Ming cache does not belong to this model");
  } else if (!cache.is_valid_) {
    throw std::invalid_argument(
        "Ming cache is invalid after a failed or cancelled Forward; create a "
        "new cache");
  } else if (embeddings.ndim() != 3 || embeddings.shape(0) != 1 ||
             embeddings.shape(1) < 1 ||
             embeddings.shape(2) != config_.hidden_size) {
    throw std::invalid_argument(
        "Ming Forward embeddings must have shape [1, positive token_count, " +
        std::to_string(config_.hidden_size) + "]");
  } else if (embeddings.shape(1) >
             config_.max_context_token_count - cache.token_count_) {
    throw std::invalid_argument("Ming Forward exceeds max_position_embeddings");
  } else {
  }
  cache.is_valid_ = false;
  mx::array hidden_states = embeddings;
  std::vector<mx::array> evaluation_tensors;
  for (int layer_index = 0; layer_index < config_.layer_count; ++layer_index) {
    if (cancel.load()) {
      throw qwen3_asr::TranscriptionCancelled();
    } else {
    }
    const std::string weight_prefix =
        "model.layers." + std::to_string(layer_index);
    auto normalized_hidden_states =
        checkpoint_.RmsNorm(hidden_states, weight_prefix + ".input_layernorm",
                            config_.rms_norm_epsilon);
    hidden_states = mx::add(
        hidden_states,
        config_.UsesMla(layer_index)
            ? LatentAttention(
                  normalized_hidden_states, weight_prefix + ".attention",
                  std::get<LatentAttentionCache>(cache.layers_.at(layer_index)))
            : DeltaAttention(
                  normalized_hidden_states, weight_prefix + ".attention",
                  std::get<DeltaCache>(cache.layers_.at(layer_index))));
    normalized_hidden_states = checkpoint_.RmsNorm(
        hidden_states, weight_prefix + ".post_attention_layernorm",
        config_.rms_norm_epsilon);
    hidden_states = mx::add(hidden_states,
                            FeedForward(normalized_hidden_states,
                                        weight_prefix + ".mlp", layer_index));
    const auto &layer_cache = cache.layers_.at(layer_index);
    if (const auto *delta = std::get_if<DeltaCache>(&layer_cache)) {
      const auto &state = delta->state.value();
      evaluation_tensors.push_back(state.recurrent);
      for (const auto &convolution : state.convolution) {
        evaluation_tensors.push_back(convolution);
      }
    } else {
      const auto &state =
          std::get<LatentAttentionCache>(layer_cache).state.value();
      evaluation_tensors.push_back(state.keys);
      evaluation_tensors.push_back(state.values);
    }
  }
  hidden_states = checkpoint_.RmsNorm(hidden_states, "model.norm",
                                      config_.rms_norm_epsilon);
  evaluation_tensors.push_back(hidden_states);
  mx::eval(evaluation_tensors);
  cache.token_count_ += embeddings.shape(1);
  cache.is_valid_ = true;
  return hidden_states;
}

mx::array MingModel::LastTokenLogits(const mx::array &hidden_states) const {
  return mx::astype(
      checkpoint_.Linear(mx::slice(hidden_states,
                                   {0, hidden_states.shape(1) - 1, 0},
                                   hidden_states.shape()),
                         "lm_head"),
      mx::float32);
}

} // namespace ming3_asr
