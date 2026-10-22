// SPDX-License-Identifier: Apache-2.0
#include "arkasr/arkasr_model.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "common/audio.h"

namespace arkasr {

namespace mx = mlx::core;
using qwen3_asr::Gelu;
using qwen3_asr::LoadSafetensors;
using qwen3_asr::Qwen3DecoderConfig;
using qwen3_asr::ReadJson;
using qwen3_asr::WeightMap;

namespace {

constexpr float kLayerNormEps = 1e-5f;
constexpr float kRotaryBase = 10000.0f;
constexpr const char *kWhisperPrefix = "audio_encoder.whisper";
constexpr const char *kAdapterPrefix = "audio_encoder";
constexpr const char *kDecoderPrefix = "model";

// Note (AzizbekMuminoff): the checkpoint keeps PyTorch's Conv1d layout, and its
// lm_head is a copy of the tied embedding table.
WeightMap MlxLayoutWeights(WeightMap weights) {
  weights.erase("lm_head.weight");
  const std::string conv_prefix = std::string(kWhisperPrefix) + ".conv";
  for (auto &[name, array] : weights) {
    if (name.rfind(conv_prefix, 0) == 0 && array.ndim() == 3) {
      array = mx::transpose(array, {0, 2, 1});
    } else {
    }
  }
  return weights;
}

Qwen3DecoderConfig Qwen2DecoderConfig(const nlohmann::json &config) {
  Qwen3DecoderConfig decoder = Qwen3DecoderConfig::FromJson(config);
  decoder.has_query_key_norm = false;
  return decoder;
}

} // namespace

ArkASR::ArkASR(const std::filesystem::path &model_directory)
    : ArkASR(ReadJson(model_directory / "config.json"), model_directory) {}

ArkASR::ArkASR(const nlohmann::json &config,
               const std::filesystem::path &model_directory)
    : checkpoint_(MlxLayoutWeights(LoadSafetensors(model_directory)), config),
      decoder_(checkpoint_, kDecoderPrefix, Qwen2DecoderConfig(config)) {
  const nlohmann::json &whisper = config.at("whisper_config");
  whisper_ = {whisper.at("encoder_layers").get<int>(),
              whisper.at("encoder_attention_heads").get<int>()};
  audio_token_id_ = config.at("audio_token_id").get<int>();
  merge_factor_ = config.at("merge_factor").get<int>();
  if (whisper.at("num_mel_bins").get<int>() != qwen3_asr::kMelBinCount) {
    throw std::runtime_error("the audio encoder must take 128 mel bins");
  } else if (!config.value("use_rope", false)) {
    throw std::runtime_error("only the rotary audio encoder is supported");
  } else if (config.value("adapter_type", std::string()) != "mlp" ||
             config.value("mlp_adapter_act", std::string()) != "gelu") {
    throw std::runtime_error("only the GELU MLP adapter is supported");
  } else if (!config.value("tie_word_embeddings", true)) {
    throw std::runtime_error("untied output embeddings are not supported");
  } else {
  }
}

int ArkASR::AudioTokenCount(int mel_frame_count) const {
  return std::max((mel_frame_count + 1) / 2 / merge_factor_, 1);
}

mx::array ArkASR::Conv1d(const mx::array &x, const std::string &prefix,
                         int stride) const {
  return mx::add(
      mx::conv1d(x, checkpoint_.Weight(prefix + ".weight"), stride, 1),
      checkpoint_.Weight(prefix + ".bias"));
}

mx::array ArkASR::WhisperEncoderLayer(const mx::array &x, int layer) const {
  const std::string prefix =
      std::string(kWhisperPrefix) + ".layers." + std::to_string(layer);
  const int batch = x.shape(0);
  const int length = x.shape(1);
  const int width = x.shape(2);
  const int head_count = whisper_.encoder_attention_heads;
  const int head_dim = width / head_count;
  const mx::array normed =
      checkpoint_.LayerNorm(x, prefix + ".self_attn_layer_norm", kLayerNormEps);
  const auto project = [&](const std::string &name) {
    return mx::transpose(
        mx::reshape(checkpoint_.Linear(normed, prefix + ".self_attn." + name),
                    {batch, length, head_count, head_dim}),
        {0, 2, 1, 3});
  };
  // Note (AzizbekMuminoff): rotary positions turn the first half of each head
  // in adjacent pairs; the second half passes through.
  const auto rotate = [&](const mx::array &states) {
    return mx::fast::rope(states, head_dim / 2, true, kRotaryBase, 1.0f, 0);
  };
  const mx::array attended = mx::fast::scaled_dot_product_attention(
      rotate(project("q_proj")), rotate(project("k_proj")), project("v_proj"),
      static_cast<float>(std::pow(static_cast<double>(head_dim), -0.5)));
  const mx::array hidden = mx::add(
      x, checkpoint_.Linear(mx::reshape(mx::transpose(attended, {0, 2, 1, 3}),
                                        {batch, length, width}),
                            prefix + ".self_attn.out_proj"));
  return mx::add(
      hidden, checkpoint_.Linear(
                  Gelu(checkpoint_.Linear(
                      checkpoint_.LayerNorm(
                          hidden, prefix + ".final_layer_norm", kLayerNormEps),
                      prefix + ".fc1")),
                  prefix + ".fc2"));
}

mx::array ArkASR::EncodeAudio(const mx::array &mel) const {
  const std::string whisper = kWhisperPrefix;
  const std::string adapter = kAdapterPrefix;
  mx::array x =
      mx::astype(mx::expand_dims(mx::transpose(mel), 0),
                 checkpoint_.Weight(whisper + ".conv1.weight").dtype());
  x = Gelu(Conv1d(x, whisper + ".conv1", 1));
  x = Gelu(Conv1d(x, whisper + ".conv2", 2));
  for (int layer = 0; layer < whisper_.encoder_layers; ++layer) {
    x = WhisperEncoderLayer(x, layer);
  }
  // Note (AzizbekMuminoff): the checkpoint replaces the Whisper encoder's own
  // final norm with the adapter's.
  x = checkpoint_.LayerNorm(x, adapter + ".layer_norm", kLayerNormEps);

  const int frame_count = x.shape(1);
  const int width = x.shape(2);
  const int token_count = std::max(frame_count / merge_factor_, 1);
  const int kept_frame_count = token_count * merge_factor_;
  if (kept_frame_count > frame_count) {
    x = mx::pad(x, {{0, 0}, {0, kept_frame_count - frame_count}, {0, 0}});
  } else {
    x = mx::slice(x, {0, 0, 0}, {1, kept_frame_count, width});
  }
  mx::array merged = mx::reshape(x, {1, token_count, width * merge_factor_});
  merged = checkpoint_.Linear(
      Gelu(checkpoint_.Linear(merged, adapter + ".adapting.0")),
      adapter + ".adapting.2");
  return mx::astype(
      mx::reshape(merged, {token_count, merged.shape(2)}),
      checkpoint_.Weight(std::string(kDecoderPrefix) + ".embed_tokens.weight")
          .dtype());
}

} // namespace arkasr
