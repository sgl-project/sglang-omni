// SPDX-License-Identifier: Apache-2.0
#include "moss_model.h"

#include <cmath>
#include <stdexcept>

namespace moss {

namespace mx = mlx::core;
using qwen3_asr::Gelu;
using qwen3_asr::LoadSafetensors;
using qwen3_asr::Qwen3DecoderConfig;
using qwen3_asr::ReadJson;
using qwen3_asr::Silu;
using qwen3_asr::WeightMap;

namespace {

constexpr float kWhisperLayerNormEps = 1e-5f;
constexpr const char *kWhisperPrefix = "model.whisper_encoder";
constexpr const char *kAdaptorPrefix = "model.vq_adaptor.layers";
constexpr const char *kDecoderPrefix = "model.language_model";

// Note (Dayuxiaoshui): the checkpoint keeps PyTorch's Conv1d layout, and its
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

} // namespace

MossTranscribeDiarize::MossTranscribeDiarize(
    const std::filesystem::path &model_directory)
    : MossTranscribeDiarize(ReadJson(model_directory / "config.json"),
                            model_directory) {}

MossTranscribeDiarize::MossTranscribeDiarize(
    const nlohmann::json &config, const std::filesystem::path &model_directory)
    : checkpoint_(MlxLayoutWeights(LoadSafetensors(model_directory)), config),
      decoder_(checkpoint_, kDecoderPrefix,
               Qwen3DecoderConfig::FromJson(config.at("text_config"))) {
  const nlohmann::json &audio = config.at("audio_config");
  whisper_ = {audio.at("num_mel_bins").get<int>(),
              audio.at("encoder_layers").get<int>(),
              audio.at("encoder_attention_heads").get<int>()};
  adaptor_norm_eps_ = config.at("text_config").at("rms_norm_eps").get<float>();
  audio_token_id_ = config.at("audio_token_id").get<int>();
  audio_merge_size_ = config.at("audio_merge_size").get<int>();
  if (!config.value("tie_word_embeddings", true)) {
    throw std::runtime_error("untied output embeddings are not supported");
  } else {
  }
}

mx::array MossTranscribeDiarize::Conv1d(const mx::array &x,
                                        const std::string &prefix,
                                        int stride) const {
  return mx::add(
      mx::conv1d(x, checkpoint_.Weight(prefix + ".weight"), stride, 1),
      checkpoint_.Weight(prefix + ".bias"));
}

mx::array MossTranscribeDiarize::WhisperEncoderLayer(const mx::array &x,
                                                     int layer) const {
  const std::string prefix =
      std::string(kWhisperPrefix) + ".layers." + std::to_string(layer);
  const int batch = x.shape(0);
  const int length = x.shape(1);
  const int width = x.shape(2);
  const int head_count = whisper_.encoder_attention_heads;
  const int head_dim = width / head_count;
  const mx::array normed = checkpoint_.LayerNorm(
      x, prefix + ".self_attn_layer_norm", kWhisperLayerNormEps);
  const auto project = [&](const std::string &name) {
    return mx::transpose(
        mx::reshape(checkpoint_.Linear(normed, prefix + ".self_attn." + name),
                    {batch, length, head_count, head_dim}),
        {0, 2, 1, 3});
  };
  const mx::array attended = mx::fast::scaled_dot_product_attention(
      project("q_proj"), project("k_proj"), project("v_proj"),
      static_cast<float>(std::pow(static_cast<double>(head_dim), -0.5)));
  const mx::array hidden = mx::add(
      x, checkpoint_.Linear(mx::reshape(mx::transpose(attended, {0, 2, 1, 3}),
                                        {batch, length, width}),
                            prefix + ".self_attn.out_proj"));
  return mx::add(
      hidden,
      checkpoint_.Linear(
          Gelu(checkpoint_.Linear(
              checkpoint_.LayerNorm(hidden, prefix + ".final_layer_norm",
                                    kWhisperLayerNormEps),
              prefix + ".fc1")),
          prefix + ".fc2"));
}

mx::array MossTranscribeDiarize::EncodeAudio(
    const mx::array &window_features,
    const std::vector<int> &window_token_counts) const {
  const std::string whisper = kWhisperPrefix;
  mx::array x = mx::astype(
      window_features, checkpoint_.Weight(whisper + ".conv1.weight").dtype());
  x = Gelu(Conv1d(x, whisper + ".conv1", 1));
  x = Gelu(Conv1d(x, whisper + ".conv2", 2));
  const int frame_count = x.shape(1);
  const int width = x.shape(2);
  x = mx::add(x,
              mx::slice(checkpoint_.Weight(whisper + ".embed_positions.weight"),
                        {0, 0}, {frame_count, width}));
  for (int layer = 0; layer < whisper_.encoder_layers; ++layer) {
    x = WhisperEncoderLayer(x, layer);
  }
  x = checkpoint_.LayerNorm(x, whisper + ".layer_norm", kWhisperLayerNormEps);

  // Note (Dayuxiaoshui): a window keeps a merge-size multiple of the frames its
  // samples cover, so no merged token mixes two windows.
  std::vector<mx::array> kept_frames;
  for (size_t window = 0; window < window_token_counts.size(); ++window) {
    const int kept = window_token_counts[window] * audio_merge_size_;
    const int index = static_cast<int>(window);
    kept_frames.push_back(
        mx::slice(x, {index, 0, 0}, {index + 1, kept, width}));
  }
  const mx::array frames = mx::concatenate(kept_frames, 1);
  const int merged_count = frames.shape(1) / audio_merge_size_;
  const std::string adaptor = kAdaptorPrefix;
  mx::array merged =
      mx::reshape(frames, {1, merged_count, width * audio_merge_size_});
  merged = checkpoint_.Linear(Silu(checkpoint_.Linear(merged, adaptor + ".0")),
                              adaptor + ".2");
  merged = checkpoint_.LayerNorm(merged, adaptor + ".3", adaptor_norm_eps_);
  return mx::astype(
      mx::reshape(merged, {merged_count, merged.shape(2)}),
      checkpoint_.Weight(std::string(kDecoderPrefix) + ".embed_tokens.weight")
          .dtype());
}

} // namespace moss
