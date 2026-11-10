// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team.
#include "auk/auk_conditioner.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "common/audio.h"

namespace auk {

namespace mx = mlx::core;

namespace {

constexpr const char *kNoPromptAudio = "|<no_prompt_audio>|";

omni_common::WeightMap
LoadConditionerWeights(const std::filesystem::path &directory,
                       mx::Dtype dtype) {
  auto loaded = omni_common::LoadSafetensors(directory);
  omni_common::WeightMap weights;
  for (auto &[name, weight] : loaded) {
    const std::string normalized =
        name.starts_with("thinker.") ? name.substr(8) : name;
    if ((!normalized.starts_with("model.") &&
         !normalized.starts_with("audio_tower.")) ||
        normalized.ends_with("audio_bos_eos_token.weight") ||
        normalized.ends_with("rotary_emb.inv_freq")) {
      continue;
    } else {
    }
    if (weight.dtype() != mx::float32 && weight.dtype() != mx::float16 &&
        weight.dtype() != mx::bfloat16) {
      throw std::invalid_argument(
          "AuK conditioner weights must be floating point");
    } else {
    }
    mx::array value = mx::astype(weight, dtype);
    if (normalized == "audio_tower.conv1.weight" ||
        normalized == "audio_tower.conv2.weight") {
      value = mx::transpose(value, {0, 2, 1});
    } else {
    }
    if (!weights.emplace(normalized, std::move(value)).second) {
      throw std::invalid_argument("duplicate AuK conditioner weight: " +
                                  normalized);
    } else {
    }
  }
  return weights;
}

} // namespace

Conditioner::Config
Conditioner::Config::Read(const std::filesystem::path &directory) {
  const auto root = omni_common::ReadJson(directory / "config.json");
  const auto &thinker = root.at("thinker_config");
  const auto &text = thinker.at("text_config");
  const auto &audio = thinker.at("audio_config");
  const auto processor =
      omni_common::ReadJson(directory / "preprocessor_config.json");
  Config config;
  config.hidden_size = text.at("hidden_size");
  config.vocabulary_size = text.at("vocab_size");
  config.text_layer_count = text.at("num_hidden_layers");
  config.text_head_count = text.at("num_attention_heads");
  config.text_kv_head_count = text.at("num_key_value_heads");
  config.text_intermediate_size = text.at("intermediate_size");
  config.max_tokens = text.at("max_position_embeddings");
  config.rms_norm_epsilon = text.at("rms_norm_eps");
  config.rope_theta = text.at("rope_theta");
  config.audio_hidden_size = audio.at("d_model");
  config.audio_layer_count = audio.at("encoder_layers");
  config.audio_head_count = audio.at("encoder_attention_heads");
  config.audio_intermediate_size = audio.at("encoder_ffn_dim");
  config.audio_window_frames = 2 * audio.at("n_window").get<int>();
  config.max_audio_samples = processor.at("n_samples");
  config.audio_token_id = thinker.at("audio_token_index");
  config.image_token_id = thinker.at("image_token_index");
  config.video_token_id = thinker.at("video_token_index");
  if (text.at("hidden_act") != "silu" ||
      text.at("rope_scaling").at("rope_type") != "default" ||
      text.at("use_sliding_window") != false ||
      audio.at("activation_function") != "gelu" ||
      audio.at("num_mel_bins") != omni_common::kMelBinCount ||
      audio.at("output_dim") != config.hidden_size ||
      processor.at("sampling_rate") != omni_common::kSampleRate ||
      processor.at("feature_size") != omni_common::kMelBinCount ||
      processor.at("n_fft") != omni_common::kFftSize ||
      processor.at("hop_length") != omni_common::kHopLength ||
      processor.at("padding_side") != "right" ||
      processor.value("dither", 0.0f) != 0.0f ||
      processor.value("padding_value", 0.0f) != 0.0f) {
    throw std::invalid_argument("unsupported AuK conditioner configuration");
  } else if (config.hidden_size <= 0 || config.vocabulary_size <= 0 ||
             config.text_layer_count <= 0 || config.text_head_count <= 0 ||
             config.text_kv_head_count <= 0 ||
             config.text_intermediate_size <= 0 || config.max_tokens <= 0 ||
             config.audio_hidden_size <= 2 || config.audio_layer_count <= 0 ||
             config.audio_head_count <= 0 ||
             config.audio_intermediate_size <= 0 ||
             config.audio_window_frames <= 0 ||
             config.max_audio_samples < omni_common::kFftSize ||
             !(config.rms_norm_epsilon > 0) || !(config.rope_theta > 0) ||
             !std::isfinite(config.rms_norm_epsilon) ||
             !std::isfinite(config.rope_theta)) {
    throw std::invalid_argument("invalid AuK conditioner dimensions");
  } else if (config.hidden_size % config.text_head_count != 0 ||
             (config.hidden_size / config.text_head_count) % 2 != 0 ||
             config.text_head_count % config.text_kv_head_count != 0 ||
             config.audio_hidden_size % config.audio_head_count != 0 ||
             config.audio_hidden_size % 2 != 0 ||
             config.audio_window_frames % 2 != 0 ||
             config.max_audio_samples % omni_common::kHopLength != 0) {
    throw std::invalid_argument(
        "incompatible AuK conditioner attention dimensions");
  } else {
    return config;
  }
}

Conditioner::Conditioner(const std::filesystem::path &directory,
                         mx::Dtype dtype)
    : config_(Config::Read(directory)),
      checkpoint_(LoadConditionerWeights(directory, dtype),
                  nlohmann::json::object()),
      tokenizer_(directory) {
  const int hidden = config_.hidden_size;
  const int head_size = hidden / config_.text_head_count;
  const int key_width = config_.text_kv_head_count * head_size;
  checkpoint_.RequireShape("model.embed_tokens.weight",
                           {config_.vocabulary_size, hidden});
  checkpoint_.RequireShape("model.norm.weight", {hidden});
  for (int layer = 0; layer < config_.text_layer_count; ++layer) {
    const std::string prefix = "model.layers." + std::to_string(layer);
    for (const auto &[name, width] :
         std::vector<std::pair<std::string, int>>{{"q_proj", hidden},
                                                  {"k_proj", key_width},
                                                  {"v_proj", key_width}}) {
      checkpoint_.RequireShape(prefix + ".self_attn." + name + ".weight",
                               {width, hidden});
      checkpoint_.RequireShape(prefix + ".self_attn." + name + ".bias",
                               {width});
    }
    checkpoint_.RequireShape(prefix + ".self_attn.o_proj.weight",
                             {hidden, hidden});
    checkpoint_.RequireShape(prefix + ".input_layernorm.weight", {hidden});
    checkpoint_.RequireShape(prefix + ".post_attention_layernorm.weight",
                             {hidden});
    for (const std::string name : {"gate_proj", "up_proj"}) {
      checkpoint_.RequireShape(prefix + ".mlp." + name + ".weight",
                               {config_.text_intermediate_size, hidden});
    }
    checkpoint_.RequireShape(prefix + ".mlp.down_proj.weight",
                             {hidden, config_.text_intermediate_size});
  }
  const int audio_hidden = config_.audio_hidden_size;
  checkpoint_.RequireShape("audio_tower.conv1.weight",
                           {audio_hidden, 3, omni_common::kMelBinCount});
  checkpoint_.RequireShape("audio_tower.conv2.weight",
                           {audio_hidden, 3, audio_hidden});
  for (const std::string name : {"conv1", "conv2", "ln_post"}) {
    checkpoint_.RequireShape("audio_tower." + name + ".bias", {audio_hidden});
  }
  checkpoint_.RequireShape("audio_tower.ln_post.weight", {audio_hidden});
  checkpoint_.RequireShape("audio_tower.proj.weight", {hidden, audio_hidden});
  checkpoint_.RequireShape("audio_tower.proj.bias", {hidden});
  for (int layer = 0; layer < config_.audio_layer_count; ++layer) {
    const std::string prefix = "audio_tower.layers." + std::to_string(layer);
    for (const std::string name : {"q_proj", "k_proj", "v_proj", "out_proj"}) {
      checkpoint_.RequireShape(prefix + ".self_attn." + name + ".weight",
                               {audio_hidden, audio_hidden});
      if (name != "k_proj") {
        checkpoint_.RequireShape(prefix + ".self_attn." + name + ".bias",
                                 {audio_hidden});
      } else {
      }
    }
    for (const std::string name :
         {"self_attn_layer_norm", "final_layer_norm"}) {
      checkpoint_.RequireShape(prefix + "." + name + ".weight", {audio_hidden});
      checkpoint_.RequireShape(prefix + "." + name + ".bias", {audio_hidden});
    }
    checkpoint_.RequireShape(prefix + ".fc1.weight",
                             {config_.audio_intermediate_size, audio_hidden});
    checkpoint_.RequireShape(prefix + ".fc1.bias",
                             {config_.audio_intermediate_size});
    checkpoint_.RequireShape(prefix + ".fc2.weight",
                             {audio_hidden, config_.audio_intermediate_size});
    checkpoint_.RequireShape(prefix + ".fc2.bias", {audio_hidden});
  }
  if (tokenizer_.AddedTokenId("<|AUDIO|>") != config_.audio_token_id ||
      tokenizer_.AddedTokenId("<|im_start|>") < 0 ||
      tokenizer_.AddedTokenId("<|im_end|>") < 0 ||
      tokenizer_.AddedTokenId("<|audio_bos|>") < 0 ||
      tokenizer_.AddedTokenId("<|audio_eos|>") < 0) {
    throw std::invalid_argument(
        "AuK conditioner tokenizer has incompatible special tokens");
  } else {
  }
}

std::vector<int> Conditioner::Tokenize(const std::string &instruction,
                                       int audio_frame_count) const {
  std::string text = "<|im_start|>system\nYou are a helpful "
                     "assistant.<|im_end|>\n<|im_start|>user\n" +
                     instruction;
  if (audio_frame_count > 0) {
    const int audio_tokens = ((audio_frame_count + 1) / 2) / 2;
    text += "<|audio_bos|>";
    for (int token = 0; token < audio_tokens; ++token) {
      text += "<|AUDIO|>";
    }
    text += "<|audio_eos|>";
  } else if (!instruction.ends_with(kNoPromptAudio)) {
    text += kNoPromptAudio;
  } else {
  }
  text += "<|im_end|>\n<|im_start|>assistant\n";
  return tokenizer_.Encode(text);
}

mx::array Conditioner::Encode(
    const std::string &instruction,
    const std::optional<std::vector<float>> &reference_audio) const {
  std::optional<mx::array> features;
  if (reference_audio.has_value()) {
    if (reference_audio->size() < omni_common::kFftSize ||
        reference_audio->size() >
            static_cast<size_t>(config_.max_audio_samples)) {
      throw std::invalid_argument(
          "AuK reference audio must fit the checkpoint's audio window");
    } else {
    }
    // Note (XXXXRT666): the processor zero-pads to n_samples, 300 s for
    // Qwen2.5-Omni-3B, before its centered STFT, so the final frames see
    // zeros; padding past the last window is enough for the same frames.
    std::vector<float> padded = *reference_audio;
    const int frames = static_cast<int>(
        (reference_audio->size() + omni_common::kHopLength - 1) /
        omni_common::kHopLength);
    padded.resize(
        std::min(config_.max_audio_samples,
                 frames * omni_common::kHopLength + omni_common::kFftSize),
        0.0f);
    const mx::array full =
        omni_common::LogMel(padded, omni_common::FinalStftFrame::kDropped);
    features = mx::slice(full, {0, 0}, {omni_common::kMelBinCount, frames});
    mx::eval(*features);
  } else {
  }
  const auto token_ids =
      Tokenize(instruction, features.has_value() ? features->shape(1) : 0);
  return EncodeInputs(token_ids, features);
}

mx::array Conditioner::TextLayer(const mx::array &hidden, int layer,
                                 const mx::array &mask) const {
  const std::string prefix = "model.layers." + std::to_string(layer);
  const int length = hidden.shape(1);
  const int head_size = config_.hidden_size / config_.text_head_count;
  const auto dtype =
      checkpoint_.Weight(prefix + ".self_attn.q_proj.weight").dtype();
  const mx::array normed =
      mx::astype(checkpoint_.RmsNorm(hidden, prefix + ".input_layernorm",
                                     config_.rms_norm_epsilon),
                 dtype);
  const auto project = [&](const std::string &name, int head_count) {
    return mx::transpose(
        mx::reshape(checkpoint_.Linear(normed, prefix + ".self_attn." + name),
                    {1, length, head_count, head_size}),
        {0, 2, 1, 3});
  };
  const mx::array queries =
      mx::fast::rope(project("q_proj", config_.text_head_count), head_size,
                     false, config_.rope_theta, 1.0f, 0);
  const mx::array keys =
      mx::fast::rope(project("k_proj", config_.text_kv_head_count), head_size,
                     false, config_.rope_theta, 1.0f, 0);
  const mx::array values = project("v_proj", config_.text_kv_head_count);
  const mx::array attended = mx::fast::scaled_dot_product_attention(
      queries, keys, values,
      static_cast<float>(std::pow(static_cast<double>(head_size), -0.5)), "",
      mask);
  const mx::array residual = mx::add(
      hidden,
      checkpoint_.Linear(mx::reshape(mx::transpose(attended, {0, 2, 1, 3}),
                                     {1, length, config_.hidden_size}),
                         prefix + ".self_attn.o_proj"));
  const mx::array mlp_input = mx::astype(
      checkpoint_.RmsNorm(residual, prefix + ".post_attention_layernorm",
                          config_.rms_norm_epsilon),
      dtype);
  return mx::add(
      residual,
      checkpoint_.Linear(
          mx::multiply(omni_common::Silu(checkpoint_.Linear(
                           mlp_input, prefix + ".mlp.gate_proj")),
                       checkpoint_.Linear(mlp_input, prefix + ".mlp.up_proj")),
          prefix + ".mlp.down_proj"));
}

mx::array
Conditioner::EncodeInputs(const std::vector<int> &token_ids,
                          const std::optional<mx::array> &features) const {
  if (token_ids.size() > static_cast<size_t>(config_.max_tokens)) {
    throw std::invalid_argument(
        "AuK conditioning exceeds the text model's positions");
  } else {
  }
  const int length = static_cast<int>(token_ids.size());
  std::vector<int> audio_positions;
  for (int token = 0; token < length; ++token) {
    const int id = token_ids[token];
    if (id < 0 || id >= config_.vocabulary_size ||
        id == config_.image_token_id || id == config_.video_token_id) {
      throw std::invalid_argument("invalid AuK text/audio token IDs");
    } else {
    }
    if (id == config_.audio_token_id) {
      audio_positions.push_back(token);
    } else {
    }
  }
  mx::array embeddings = checkpoint_.Embed(
      mx::array(token_ids.data(), {length}, mx::int32), "model.embed_tokens");
  if (features.has_value()) {
    const mx::array audio =
        mx::astype(EncodeAudio(*features), embeddings.dtype());
    if (audio_positions.size() != static_cast<size_t>(audio.shape(0))) {
      throw std::invalid_argument(
          "AuK audio features and placeholder token counts differ");
    } else {
      embeddings = mx::scatter(
          embeddings,
          mx::array(audio_positions.data(), {audio.shape(0)}, mx::int32),
          mx::expand_dims(audio, 1), 0);
    }
  } else if (!audio_positions.empty()) {
    throw std::invalid_argument(
        "AuK audio placeholder tokens require reference features");
  } else {
  }
  const mx::array indices = mx::arange(length);
  const mx::array distances =
      mx::subtract(mx::reshape(indices, {length, 1}), indices);
  const mx::array mask = mx::reshape(mx::greater_equal(distances, mx::array(0)),
                                     {1, 1, length, length});
  mx::array hidden = mx::reshape(mx::astype(embeddings, mx::float32),
                                 {1, length, config_.hidden_size});
  std::vector<mx::array> states;
  for (int layer = 0; layer < config_.text_layer_count; ++layer) {
    states.push_back(hidden);
    hidden = TextLayer(hidden, layer, mask);
    mx::eval(hidden);
  }
  states.push_back(
      checkpoint_.RmsNorm(hidden, "model.norm", config_.rms_norm_epsilon));
  const mx::array result = mx::squeeze(mx::stack(states, 1), 0);
  mx::eval(result);
  return result;
}

} // namespace auk
