// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team.
#include "auk/auk_conditioner.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "common/audio.h"
#include "qwen3_asr/qwen3_audio.h"

namespace auk {

namespace mx = mlx::core;

namespace {

constexpr const char *kNoPromptAudio = "|<no_prompt_audio>|";

omni_common::WeightMap
LoadConditionerWeights(const std::filesystem::path &directory,
                       mx::Dtype dtype) {
  if (dtype != mx::bfloat16 && dtype != mx::float32) {
    throw std::invalid_argument(
        "AuK conditioning requires bfloat16 or float32");
  } else {
  }
  auto loaded = omni_common::LoadSafetensors(directory);
  omni_common::WeightMap weights;
  for (auto &[name, weight] : loaded) {
    const std::string normalized =
        name.starts_with("thinker.") ? name.substr(8) : name;
    if ((!normalized.starts_with("model.") &&
         !normalized.starts_with("audio_tower.") &&
         !normalized.starts_with("frontend.")) ||
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
    mx::array value = mx::astype(
        weight, normalized.starts_with("frontend.") ? mx::float32 : dtype);
    if (normalized == "audio_tower.conv1.weight" ||
        normalized == "audio_tower.conv2.weight") {
      if (value.ndim() != 3 || value.shape(2) != 3) {
        throw std::invalid_argument(
            "AuK audio convolutions must use Torch layout");
      } else {
        value = mx::transpose(value, {0, 2, 1});
      }
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

void RequireWeight(const omni_common::Checkpoint &checkpoint,
                   const std::string &name, const mx::Shape &shape) {
  if (checkpoint.Weight(name).shape() != shape) {
    throw std::invalid_argument("invalid AuK conditioner weight shape: " +
                                name);
  } else {
  }
}

mx::array HalfRoPE(const mx::array &value, const mx::array &cosine,
                   const mx::array &sine) {
  const auto halves = mx::split(value, 2, -1);
  const mx::array rotated =
      mx::concatenate({mx::negative(halves[1]), halves[0]}, -1);
  return mx::astype(
      mx::add(mx::multiply(value, cosine), mx::multiply(rotated, sine)),
      value.dtype());
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
  const auto rope = text.value(
      "rope_parameters", text.value("rope_scaling", nlohmann::json::object()));
  Config config;
  config.hidden_size = text.at("hidden_size");
  config.vocabulary_size = text.at("vocab_size");
  config.text_layer_count = text.at("num_hidden_layers");
  config.text_head_count = text.at("num_attention_heads");
  config.text_kv_head_count = text.at("num_key_value_heads");
  config.text_intermediate_size = text.at("intermediate_size");
  config.max_tokens = text.at("max_position_embeddings");
  config.rms_norm_epsilon = text.at("rms_norm_eps");
  config.rope_theta =
      rope.value("rope_theta", text.value("rope_theta", 1000000.0f));
  config.sliding_window = text.value("use_sliding_window", false)
                              ? text.at("sliding_window").get<int>()
                              : 0;
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
      rope.value("rope_type", rope.value("type", std::string("default"))) !=
          "default" ||
      audio.at("activation_function") != "gelu" ||
      audio.at("num_mel_bins") != omni_common::kMelBinCount ||
      audio.at("output_dim") != config.hidden_size ||
      processor.at("sampling_rate") != omni_common::kSampleRate ||
      processor.at("feature_size") != omni_common::kMelBinCount ||
      processor.at("n_fft") != omni_common::kFftSize ||
      processor.at("hop_length") != omni_common::kHopLength ||
      processor.at("padding_side") != "right" ||
      audio.value("num_hidden_layers", config.audio_layer_count) !=
          config.audio_layer_count ||
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
  }
  const bool explicit_layers =
      text.contains("layer_types") && !text.at("layer_types").is_null();
  if (explicit_layers && text.at("layer_types").size() !=
                             static_cast<size_t>(config.text_layer_count)) {
    throw std::invalid_argument("AuK layer_types must name every text layer");
  } else {
  }
  for (int layer = 0; layer < config.text_layer_count; ++layer) {
    const std::string type =
        explicit_layers ? text.at("layer_types").at(layer).get<std::string>()
        : config.sliding_window > 0 &&
                layer >=
                    text.value("max_window_layers", config.text_layer_count)
            ? "sliding_attention"
            : "full_attention";
    if (type != "full_attention" &&
        (type != "sliding_attention" || config.sliding_window <= 0)) {
      throw std::invalid_argument("unsupported AuK text attention type");
    } else {
      config.sliding_layers.push_back(type == "sliding_attention");
    }
  }
  return config;
}

Conditioner::Conditioner(const std::filesystem::path &directory,
                         mx::Dtype dtype)
    : config_(Config::Read(directory)),
      checkpoint_(LoadConditionerWeights(directory, dtype),
                  nlohmann::json::object()),
      tokenizer_(directory) {
  RequireWeight(checkpoint_, "frontend.hann_window", {omni_common::kFftSize});
  RequireWeight(checkpoint_, "frontend.mel_filters",
                {omni_common::kFftSize / 2 + 1, omni_common::kMelBinCount});
  const int hidden = config_.hidden_size;
  const int head_size = hidden / config_.text_head_count;
  const int key_width = config_.text_kv_head_count * head_size;
  RequireWeight(checkpoint_, "model.embed_tokens.weight",
                {config_.vocabulary_size, hidden});
  RequireWeight(checkpoint_, "model.norm.weight", {hidden});
  for (int layer = 0; layer < config_.text_layer_count; ++layer) {
    const std::string prefix = "model.layers." + std::to_string(layer);
    for (const auto &[name, width] :
         std::vector<std::pair<std::string, int>>{{"q_proj", hidden},
                                                  {"k_proj", key_width},
                                                  {"v_proj", key_width}}) {
      RequireWeight(checkpoint_, prefix + ".self_attn." + name + ".weight",
                    {width, hidden});
      RequireWeight(checkpoint_, prefix + ".self_attn." + name + ".bias",
                    {width});
    }
    RequireWeight(checkpoint_, prefix + ".self_attn.o_proj.weight",
                  {hidden, hidden});
    RequireWeight(checkpoint_, prefix + ".input_layernorm.weight", {hidden});
    RequireWeight(checkpoint_, prefix + ".post_attention_layernorm.weight",
                  {hidden});
    for (const std::string name : {"gate_proj", "up_proj"}) {
      RequireWeight(checkpoint_, prefix + ".mlp." + name + ".weight",
                    {config_.text_intermediate_size, hidden});
    }
    RequireWeight(checkpoint_, prefix + ".mlp.down_proj.weight",
                  {hidden, config_.text_intermediate_size});
  }
  const int audio_hidden = config_.audio_hidden_size;
  RequireWeight(checkpoint_, "audio_tower.conv1.weight",
                {audio_hidden, 3, omni_common::kMelBinCount});
  RequireWeight(checkpoint_, "audio_tower.conv2.weight",
                {audio_hidden, 3, audio_hidden});
  for (const std::string name : {"conv1", "conv2", "ln_post"}) {
    RequireWeight(checkpoint_, "audio_tower." + name + ".bias", {audio_hidden});
  }
  RequireWeight(checkpoint_, "audio_tower.ln_post.weight", {audio_hidden});
  RequireWeight(checkpoint_, "audio_tower.proj.weight", {hidden, audio_hidden});
  RequireWeight(checkpoint_, "audio_tower.proj.bias", {hidden});
  for (int layer = 0; layer < config_.audio_layer_count; ++layer) {
    const std::string prefix = "audio_tower.layers." + std::to_string(layer);
    for (const std::string name : {"q_proj", "k_proj", "v_proj", "out_proj"}) {
      RequireWeight(checkpoint_, prefix + ".self_attn." + name + ".weight",
                    {audio_hidden, audio_hidden});
      if (name != "k_proj") {
        RequireWeight(checkpoint_, prefix + ".self_attn." + name + ".bias",
                      {audio_hidden});
      } else {
      }
    }
    for (const std::string name :
         {"self_attn_layer_norm", "final_layer_norm"}) {
      RequireWeight(checkpoint_, prefix + "." + name + ".weight",
                    {audio_hidden});
      RequireWeight(checkpoint_, prefix + "." + name + ".bias", {audio_hidden});
    }
    RequireWeight(checkpoint_, prefix + ".fc1.weight",
                  {config_.audio_intermediate_size, audio_hidden});
    RequireWeight(checkpoint_, prefix + ".fc1.bias",
                  {config_.audio_intermediate_size});
    RequireWeight(checkpoint_, prefix + ".fc2.weight",
                  {audio_hidden, config_.audio_intermediate_size});
    RequireWeight(checkpoint_, prefix + ".fc2.bias", {audio_hidden});
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
  if (instruction.empty() || audio_frame_count < 0 ||
      audio_frame_count > config_.max_audio_samples / omni_common::kHopLength) {
    throw std::invalid_argument(
        "invalid AuK conditioning instruction or audio length");
  } else {
  }
  std::string text = "<|im_start|>system\nYou are a helpful "
                     "assistant.<|im_end|>\n<|im_start|>user\n" +
                     instruction;
  if (audio_frame_count > 0) {
    const int audio_tokens = ((audio_frame_count + 1) / 2) / 2;
    if (audio_tokens == 0) {
      throw std::invalid_argument(
          "AuK reference audio must contain at least three mel frames");
    } else {
    }
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
            static_cast<size_t>(config_.max_audio_samples) ||
        !std::all_of(reference_audio->begin(), reference_audio->end(),
                     [](float sample) { return std::isfinite(sample); })) {
      throw std::invalid_argument("AuK reference audio must be finite and "
                                  "within the checkpoint's duration limit");
    } else {
    }
    // Note (Adrian Wei): CPU FFT and pinned frontend constants preserve the
    // processor's BF16 inputs; padding only needs to cover the final window.
    mx::StreamContext cpu(mx::Device::cpu);
    std::vector<float> padded = *reference_audio;
    const int frames = static_cast<int>(
        (reference_audio->size() + omni_common::kHopLength - 1) /
        omni_common::kHopLength);
    padded.resize(
        std::min(config_.max_audio_samples,
                 frames * omni_common::kHopLength + omni_common::kFftSize),
        0.0f);
    const mx::array full =
        qwen3_asr::LogMel(padded, qwen3_asr::AudioLayout::kReference,
                          checkpoint_.Weight("frontend.hann_window"),
                          checkpoint_.Weight("frontend.mel_filters"));
    features = mx::slice(full, {0, 0}, {omni_common::kMelBinCount, frames});
    mx::eval(*features);
  } else {
  }
  const auto token_ids =
      Tokenize(instruction, features.has_value() ? features->shape(1) : 0);
  return EncodeInputs(token_ids, std::vector<int>(token_ids.size(), 1),
                      features);
}

mx::array Conditioner::TextLayer(const mx::array &hidden, int layer,
                                 const mx::array &mask, const mx::array &cosine,
                                 const mx::array &sine) const {
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
      HalfRoPE(project("q_proj", config_.text_head_count), cosine, sine);
  const mx::array keys =
      HalfRoPE(project("k_proj", config_.text_kv_head_count), cosine, sine);
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
                          const std::vector<int> &attention_mask,
                          const std::optional<mx::array> &features) const {
  if (token_ids.empty() || token_ids.size() != attention_mask.size() ||
      token_ids.size() > static_cast<size_t>(config_.max_tokens) ||
      std::none_of(attention_mask.begin(), attention_mask.end(),
                   [](int mask) { return mask == 1; })) {
    throw std::invalid_argument(
        "AuK token IDs and attention mask must be matching nonempty sequences");
  } else {
  }
  const int length = static_cast<int>(token_ids.size());
  std::vector<int> audio_positions;
  std::vector<float> positions;
  int valid_tokens = 0;
  for (int token = 0; token < length; ++token) {
    const int id = token_ids[token];
    const int mask = attention_mask[token];
    if (id < 0 || id >= config_.vocabulary_size ||
        id == config_.image_token_id || id == config_.video_token_id ||
        (mask != 0 && mask != 1)) {
      throw std::invalid_argument(
          "invalid AuK text/audio token IDs or attention mask");
    } else {
    }
    if (id == config_.audio_token_id) {
      audio_positions.push_back(token);
    } else {
    }
    valid_tokens += mask;
    positions.push_back(mask ? static_cast<float>(valid_tokens - 1) : 1.0f);
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
  const int head_size = config_.hidden_size / config_.text_head_count;
  const mx::array inv_freq = mx::divide(
      mx::array(1.0f),
      mx::power(mx::array(config_.rope_theta),
                mx::divide(mx::arange(0, head_size, 2, mx::float32),
                           mx::array(static_cast<float>(head_size)))));
  const mx::array angles = mx::multiply(
      mx::array(positions.data(), {length, 1}, mx::float32), inv_freq);
  const mx::array doubled = mx::reshape(mx::concatenate({angles, angles}, -1),
                                        {1, 1, length, head_size});
  const mx::array cosine = mx::cos(doubled);
  const mx::array sine = mx::sin(doubled);
  const mx::array indices = mx::arange(length);
  const mx::array distances =
      mx::subtract(mx::reshape(indices, {length, 1}), indices);
  const mx::array mask =
      mx::reshape(mx::logical_and(mx::greater_equal(distances, mx::array(0)),
                                  mx::astype(mx::array(attention_mask.data(),
                                                       {length}, mx::int32),
                                             mx::bool_)),
                  {1, 1, length, length});
  mx::array hidden = mx::reshape(mx::astype(embeddings, mx::float32),
                                 {1, length, config_.hidden_size});
  std::vector<mx::array> states;
  for (int layer = 0; layer < config_.text_layer_count; ++layer) {
    states.push_back(hidden);
    const mx::array layer_mask =
        config_.sliding_layers[layer]
            ? mx::logical_and(
                  mask, mx::less(distances, mx::array(config_.sliding_window)))
            : mask;
    hidden = TextLayer(hidden, layer, layer_mask, cosine, sine);
    mx::eval(hidden);
  }
  states.push_back(
      checkpoint_.RmsNorm(hidden, "model.norm", config_.rms_norm_epsilon));
  const mx::array result = mx::squeeze(mx::stack(states, 1), 0);
  mx::eval(result);
  return result;
}

} // namespace auk
