// SPDX-License-Identifier: Apache-2.0
#include "arkasr/arkasr_model.h"

#include <algorithm>
#include <stdexcept>

#include "common/audio.h"
#include "common/whisper_encoder.h"

namespace arkasr {

namespace mx = mlx::core;
using omni_common::Gelu;
using omni_common::LoadSafetensors;
using omni_common::Qwen3DecoderConfig;
using omni_common::ReadJson;
using omni_common::WeightMap;

namespace {

constexpr float kLayerNormEps = 1e-5f;
constexpr const char *kWhisperPrefix = "audio_encoder.whisper";
constexpr const char *kAdapterPrefix = "audio_encoder";
constexpr const char *kDecoderPrefix = "model";

// Note (AzizbekMuminoff): the checkpoint's lm_head is a copy of the tied
// embedding table.
WeightMap MlxLayoutWeights(WeightMap weights) {
  weights.erase("lm_head.weight");
  return omni_common::WhisperConvMlxLayout(std::move(weights), kWhisperPrefix);
}

// Qwen2's config leaves head_dim out, as each head is hidden_size /
// num_attention_heads wide, and its attention has no query and key norms.
Qwen3DecoderConfig Qwen2DecoderConfig(nlohmann::json config) {
  config["head_dim"] = config.at("hidden_size").get<int>() /
                       config.at("num_attention_heads").get<int>();
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
  if (whisper.at("num_mel_bins").get<int>() != omni_common::kMelBinCount) {
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

mx::array ArkASR::EncodeAudio(const mx::array &mel) const {
  const std::string whisper = kWhisperPrefix;
  const std::string adapter = kAdapterPrefix;
  mx::array x =
      mx::astype(mx::expand_dims(mx::transpose(mel), 0),
                 checkpoint_.Weight(whisper + ".conv1.weight").dtype());
  x = Gelu(checkpoint_.Conv1d(x, whisper + ".conv1", 1, 1));
  x = Gelu(checkpoint_.Conv1d(x, whisper + ".conv2", 2, 1));
  for (int layer = 0; layer < whisper_.encoder_layers; ++layer) {
    x = omni_common::WhisperEncoderLayer(
        checkpoint_, x, whisper + ".layers." + std::to_string(layer),
        whisper_.encoder_attention_heads, true);
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
