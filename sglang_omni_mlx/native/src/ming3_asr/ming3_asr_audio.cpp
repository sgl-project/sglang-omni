#include "ming3_asr/ming3_asr_model.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <stdexcept>

#include "common/audio.h"
#include "qwen3_asr/qwen3_asr_transcriber.h"

namespace ming3_asr {

namespace mx = mlx::core;

namespace {

constexpr float kAudioLayerNormEpsilon = 1e-5f;
constexpr float kAudioEmbeddingNormFloor = 1e-12f;

} // namespace

void ValidateAudio(const std::vector<float> &samples) {
  if (samples.size() <= omni_common::kFftSize / 2 ||
      !std::all_of(samples.begin(), samples.end(),
                   [](float sample) { return std::isfinite(sample); })) {
    throw std::invalid_argument("Ming ASR requires finite mono audio with more "
                                "than 200 samples at 16 kHz");
  } else {
  }
}

mx::array ComputeAudioFeatures(const std::vector<float> &samples) {
  ValidateAudio(samples);
  constexpr int fft_size = omni_common::kFftSize;
  constexpr int mel_bin_count = omni_common::kMelBinCount;
  static const auto window = [] {
    std::vector<float> values(fft_size);
    for (int sample = 0; sample < fft_size; ++sample) {
      values[sample] = static_cast<float>(
          0.5 * (1 - std::cos(2 * M_PI * sample / fft_size)));
    }
    return mx::array(values.data(), {fft_size});
  }();
  static const auto filters = [] {
    const double log_step = std::log(6.4) / 27.0;
    const double max_mel = 15.0 + std::log(8.0) / log_step;
    const auto mel_points = mx::linspace(0.0, max_mel, mel_bin_count + 2);
    const auto frequency_points = mx::where(
        mx::greater_equal(mel_points, mx::array(15.0f)),
        mx::multiply(
            mx::array(1000.0f),
            mx::exp(mx::multiply(mx::array(static_cast<float>(log_step)),
                                 mx::subtract(mel_points, mx::array(15.0f))))),
        mx::multiply(mx::array(static_cast<float>(200.0 / 3.0)), mel_points));
    const auto differences =
        mx::subtract(mx::slice(frequency_points, {1}, {mel_bin_count + 2}),
                     mx::slice(frequency_points, {0}, {mel_bin_count + 1}));
    const auto slopes = mx::subtract(
        mx::expand_dims(frequency_points, 0),
        mx::expand_dims(mx::linspace(0.0, 8000.0, fft_size / 2 + 1), 1));
    const auto down =
        mx::divide(mx::negative(mx::slice(slopes, {0, 0},
                                          {fft_size / 2 + 1, mel_bin_count})),
                   mx::slice(differences, {0}, {mel_bin_count}));
    const auto up = mx::divide(
        mx::slice(slopes, {0, 2}, {fft_size / 2 + 1, mel_bin_count + 2}),
        mx::slice(differences, {1}, {mel_bin_count + 1}));
    const auto normalization = mx::divide(
        mx::array(2.0f),
        mx::subtract(mx::slice(frequency_points, {2}, {mel_bin_count + 2}),
                     mx::slice(frequency_points, {0}, {mel_bin_count})));
    auto result =
        mx::multiply(mx::maximum(mx::zeros_like(down), mx::minimum(down, up)),
                     normalization);
    mx::eval(result);
    return result;
  }();
  return mx::astype(mx::transpose(omni_common::LogMel(
                        samples, window, filters,
                        {.drop_final_frame = true, .pad_short_audio = false})),
                    mx::bfloat16);
}

mx::array MingModel::EncodeAudioChunk(const mx::array &audio_features,
                                      const std::atomic<bool> &cancel) const {
  auto hidden_states = omni_common::Gelu(
      mx::add(mx::conv1d(audio_features,
                         checkpoint_.Weight("model.audio.conv1.weight"), 1, 1),
              checkpoint_.Weight("model.audio.conv1.bias")));
  hidden_states = omni_common::Gelu(
      mx::add(mx::conv1d(hidden_states,
                         checkpoint_.Weight("model.audio.conv2.weight"), 2, 1),
              checkpoint_.Weight("model.audio.conv2.bias")));
  const int encoder_frame_count = hidden_states.shape(1);
  const int audio_hidden_dimension = config_.audio_hidden_dimension;
  hidden_states = mx::astype(
      mx::add(hidden_states,
              mx::slice(checkpoint_.Weight("model.audio.positional_embedding"),
                        {0, 0}, {encoder_frame_count, audio_hidden_dimension})),
      hidden_states.dtype());
  for (int layer_index = 0; layer_index < config_.audio_layer_count;
       ++layer_index) {
    if (cancel.load()) {
      throw qwen3_asr::TranscriptionCancelled();
    } else {
    }
    const std::string weight_prefix =
        "model.audio.blocks." + std::to_string(layer_index);
    const auto normalized_hidden_states = checkpoint_.LayerNorm(
        hidden_states, weight_prefix + ".attn_ln", kAudioLayerNormEpsilon);
    std::vector<mx::array> projected_states;
    for (const std::string projection_name : {"query", "key", "value"}) {
      projected_states.push_back(mx::transpose(
          mx::reshape(
              checkpoint_.Linear(normalized_hidden_states,
                                 weight_prefix + ".attn." + projection_name),
              {1, encoder_frame_count, config_.audio_attention_head_count,
               audio_hidden_dimension / config_.audio_attention_head_count}),
          {0, 2, 1, 3}));
    }
    auto attention_output = mx::fast::scaled_dot_product_attention(
        projected_states[0], projected_states[1], projected_states[2],
        1.0f /
            std::sqrt(static_cast<float>(audio_hidden_dimension /
                                         config_.audio_attention_head_count)));
    attention_output =
        mx::reshape(mx::transpose(attention_output, {0, 2, 1, 3}),
                    {1, encoder_frame_count, audio_hidden_dimension});
    hidden_states =
        mx::add(hidden_states, checkpoint_.Linear(attention_output,
                                                  weight_prefix + ".attn.out"));
    const auto feed_forward = omni_common::Gelu(checkpoint_.Linear(
        checkpoint_.LayerNorm(hidden_states, weight_prefix + ".mlp_ln",
                              kAudioLayerNormEpsilon),
        weight_prefix + ".mlp.0"));
    hidden_states =
        mx::add(hidden_states,
                checkpoint_.Linear(feed_forward, weight_prefix + ".mlp.2"));
  }
  hidden_states = checkpoint_.LayerNorm(hidden_states, "model.audio.ln_post",
                                        kAudioLayerNormEpsilon);
  hidden_states = omni_common::Gelu(mx::add(
      mx::conv1d(hidden_states,
                 checkpoint_.Weight("linear_proj_audio.0.weight"), 2, 1),
      checkpoint_.Weight("linear_proj_audio.0.bias")));
  hidden_states = checkpoint_.Linear(hidden_states, "linear_proj_audio.3");
  if (config_.should_normalize_audio_embeddings) {
    const auto embedding_norm = mx::sqrt(mx::sum(
        mx::power(mx::astype(hidden_states, mx::float32), mx::array(2.0f)), -1,
        true));
    hidden_states =
        mx::astype(mx::divide(hidden_states,
                              mx::maximum(embedding_norm,
                                          mx::array(kAudioEmbeddingNormFloor))),
                   audio_features.dtype());
  } else {
  }
  mx::eval(hidden_states);
  return hidden_states;
}

mx::array MingModel::EncodeAudio(const std::vector<float> &samples,
                                 const std::atomic<bool> &cancel) const {
  const auto model_dtype =
      checkpoint_.Weight("model.audio.conv1.weight").dtype();
  const auto audio_features = mx::expand_dims(
      mx::astype(ComputeAudioFeatures(samples), model_dtype), 0);
  std::vector<mx::array> encoded_chunks;
  for (int start = 0; start < audio_features.shape(1);
       start += config_.audio_chunk_frame_count) {
    encoded_chunks.push_back(EncodeAudioChunk(
        mx::slice(audio_features, {0, start, 0},
                  {1,
                   std::min(start + config_.audio_chunk_frame_count,
                            audio_features.shape(1)),
                   128}),
        cancel));
  }
  auto audio_embeddings = mx::concatenate(encoded_chunks, 1);
  assert(audio_embeddings.shape(1) ==
         config_.AudioTokenCount(audio_features.shape(1)));
  return audio_embeddings;
}

} // namespace ming3_asr
