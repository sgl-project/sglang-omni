// SPDX-License-Identifier: Apache-2.0
#include "moss/moss_model.h"

#include <stdexcept>

#include "common/whisper_encoder.h"

namespace moss {

namespace mx = mlx::core;
using omni_common::Gelu;
using omni_common::LoadSafetensors;
using omni_common::Qwen3DecoderConfig;
using omni_common::ReadJson;
using omni_common::Silu;
using omni_common::WeightMap;

namespace {

constexpr float kWhisperLayerNormEps = 1e-5f;
constexpr const char *kWhisperPrefix = "model.whisper_encoder";
constexpr const char *kAdaptorPrefix = "model.vq_adaptor.layers";
constexpr const char *kDecoderPrefix = "model.language_model";
// Note (XXXXRT666): the decoder's layers are quantized at load, attention to
// 8 bits and the MLPs to 4, which keeps the transcripts and speakers, and so
// is its output projection, to 8 bits; its tied embedding table stays in the
// checkpoint dtype, since quantizing the lookup changed them, as did 4-bit
// attention.
constexpr int kAttentionQuantizationBits = 8;
constexpr int kOutputQuantizationBits = 8;
constexpr int kMlpQuantizationBits = 4;
constexpr int kDecoderQuantizationGroupSize = 64;

// Note (Dayuxiaoshui): the checkpoint keeps PyTorch's Conv1d layout, and its
// lm_head is a copy of the tied embedding table, kept as the decoder's output
// projection.
WeightMap LoadedWeights(WeightMap weights) {
  weights.insert_or_assign(
      std::string(kDecoderPrefix) + ".output_projection.weight",
      weights.at(std::string(kDecoderPrefix) + ".embed_tokens.weight"));
  weights.erase("lm_head.weight");
  return omni_common::WhisperConvMlxLayout(std::move(weights), kWhisperPrefix);
}

} // namespace

MossTranscribeDiarize::MossTranscribeDiarize(
    const std::filesystem::path &model_directory)
    : MossTranscribeDiarize(ReadJson(model_directory / "config.json"),
                            model_directory) {}

MossTranscribeDiarize::MossTranscribeDiarize(
    const nlohmann::json &config, const std::filesystem::path &model_directory)
    : checkpoint_(omni_common::QuantizedCheckpoint(
          LoadedWeights(LoadSafetensors(model_directory)), config,
          kDecoderQuantizationGroupSize,
          [](const std::string &name) {
            if (name ==
                std::string(kDecoderPrefix) + ".output_projection.weight") {
              return kOutputQuantizationBits;
            } else if (!name.starts_with(std::string(kDecoderPrefix) +
                                         ".layers.")) {
              return 0;
            } else if (name.find(".self_attn.") != std::string::npos) {
              return kAttentionQuantizationBits;
            } else {
              return kMlpQuantizationBits;
            }
          })),
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
    x = omni_common::WhisperEncoderLayer(
        checkpoint_, x, whisper + ".layers." + std::to_string(layer),
        whisper_.encoder_attention_heads, false);
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
