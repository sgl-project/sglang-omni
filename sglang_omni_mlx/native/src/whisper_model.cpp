// SPDX-License-Identifier: Apache-2.0
#include "whisper_model.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>

#include "nlohmann/json.hpp"
#include "npz.h"
#include "swift_port.h"

namespace whisper {

namespace mx = mlx::core;

namespace {

constexpr int kFftSize = 400;
constexpr int kHopLength = 160;
constexpr int kReflectPadding = kFftSize / 2;
constexpr float kLogMelFloor = 1e-10f;
constexpr float kLogMelDynamicRange = 8.0f;
constexpr float kLayerNormEpsilon = 1e-5f;
// Note (khazic): the decoder's additive causal mask value; it becomes -inf in
// float16.
constexpr float kMaskedScore = -1e9f;

std::vector<mx::array> GeluGraph(const std::vector<mx::array> &inputs) {
  // Note (khazic): x * (1 + erf(x / sqrt(2))) / 2, its scalars in x's dtype as
  // in Swift.
  const mx::array &x = inputs[0];
  const mx::Dtype dtype = x.dtype();
  return {mx::divide(
      mx::multiply(
          x, mx::add(mx::array(1.0f, dtype),
                     mx::erf(mx::divide(
                         x, mx::array(static_cast<float>(M_SQRT2), dtype))))),
      mx::array(2.0f, dtype))};
}

// Note (khazic): compiled shapeless, as the Swift port's gelu is: one fused
// kernel.
mx::array Gelu(const mx::array &x) {
  static const auto compiled = mx::compile(GeluGraph, true);
  return compiled({x})[0];
}

// Note (khazic): fixed sinusoids of the encoder positions, computed in double
// precision.
mx::array Sinusoids(int length, int channels) {
  const int half = channels / 2;
  const double log_timescale_increment =
      std::log(10000.0) / static_cast<double>(std::max(half - 1, 1));
  std::vector<double> inverse_timescales(half);
  for (int i = 0; i < half; ++i)
    inverse_timescales[i] =
        std::exp(-log_timescale_increment * static_cast<double>(i));
  std::vector<float> values(static_cast<size_t>(length) * channels);
  for (int position = 0; position < length; ++position) {
    for (int i = 0; i < half; ++i) {
      const double scaled_time =
          static_cast<double>(position) * inverse_timescales[i];
      values[static_cast<size_t>(position) * channels + i] =
          static_cast<float>(std::sin(scaled_time));
      values[static_cast<size_t>(position) * channels + half + i] =
          static_cast<float>(std::cos(scaled_time));
    }
  }
  return mx::array(values.data(), {length, channels}, mx::float32);
}

// Note (khazic): the periodic Hann window, built with MLX ops from Swift's
// Float.pi.
mx::array HannWindow() {
  const float swift_pi = std::nextafter(static_cast<float>(M_PI), 0.0f);
  const mx::array sample_indices =
      mx::astype(mx::arange(kFftSize, mx::int32), mx::float32);
  return mx::multiply(
      mx::array(0.5f),
      mx::subtract(mx::array(1.0f),
                   mx::cos(mx::divide(
                       mx::multiply(mx::array(2.0f * swift_pi), sample_indices),
                       mx::array(static_cast<float>(kFftSize))))));
}

mx::array MelFilters(int mel_bin_count) {
  const std::vector<float> values =
      swift_port::SlaneyMelFilterBank(kFftSize, mel_bin_count);
  return mx::array(values.data(), {kFftSize / 2 + 1, mel_bin_count},
                   mx::float32);
}

WhisperConfig ReadConfig(const std::filesystem::path &model_directory) {
  std::ifstream config_stream(model_directory / "config.json");
  if (!config_stream) {
    throw std::runtime_error("cannot read config.json in " +
                             model_directory.string());
  } else {
  }
  const nlohmann::json config = nlohmann::json::parse(config_stream);
  if (config.contains("quantization")) {
    throw std::runtime_error("quantized Whisper checkpoints are not served");
  } else {
  }
  return {config.at("n_mels").get<int>(),
          config.at("n_audio_ctx").get<int>(),
          config.at("n_audio_state").get<int>(),
          config.at("n_audio_head").get<int>(),
          config.at("n_audio_layer").get<int>(),
          config.at("n_vocab").get<int>(),
          config.at("n_text_ctx").get<int>(),
          config.at("n_text_head").get<int>(),
          config.at("n_text_layer").get<int>()};
}

// [batch, length, width] to [batch, heads, length, width / heads].
mx::array SplitHeads(const mx::array &x, int head_count) {
  return mx::transpose(mx::reshape(x, {x.shape(0), x.shape(1), head_count,
                                       x.shape(2) / head_count}),
                       {0, 2, 1, 3});
}

} // namespace

WhisperModel::WhisperModel(const std::filesystem::path &model_directory)
    : config_(ReadConfig(model_directory)), hann_window_(HannWindow()),
      mel_filters_(MelFilters(config_.mel_bin_count)) {
  std::vector<std::filesystem::path> weight_files;
  for (const auto &entry :
       std::filesystem::directory_iterator(model_directory)) {
    if (entry.path().extension() == ".safetensors") {
      weight_files.push_back(entry.path());
    } else {
    }
  }
  std::sort(weight_files.begin(), weight_files.end());
  // Note (khazic): mlx-community converted the older checkpoints to weights.npz
  // only.
  if (!weight_files.empty()) {
    for (const auto &path : weight_files) {
      auto [loaded, metadata] = mx::load_safetensors(path.string());
      for (auto &[name, array] : loaded) {
        weights_.insert_or_assign(name, array);
      }
    }
  } else if (std::filesystem::exists(model_directory / "weights.npz")) {
    weights_ = npz::Load(model_directory / "weights.npz");
  } else {
    throw std::runtime_error("no Whisper weights in " +
                             model_directory.string());
  }
  weights_.erase("alignment_heads");
  // Note (khazic): the checkpoint leaves out the fixed encoder positions.
  if (weights_.count("encoder.positional_embedding") == 0) {
    weights_.insert_or_assign(
        "encoder.positional_embedding",
        Sinusoids(config_.audio_context_length, config_.audio_width));
  } else {
  }
  std::vector<mx::array> parameters = {hann_window_, mel_filters_};
  for (const auto &[name, array] : weights_)
    parameters.push_back(array);
  mx::eval(parameters);
}

const mx::array &WhisperModel::Weight(const std::string &name) const {
  const auto found = weights_.find(name);
  if (found == weights_.end()) {
    throw std::runtime_error("checkpoint is missing " + name);
  } else {
  }
  return found->second;
}

mx::array WhisperModel::Linear(const mx::array &x,
                               const std::string &prefix) const {
  const auto bias = weights_.find(prefix + ".bias");
  if (bias == weights_.end()) {
    return mx::matmul(x, mx::transpose(Weight(prefix + ".weight")));
  } else {
    return mx::addmm(bias->second, x,
                     mx::transpose(Weight(prefix + ".weight")));
  }
}

mx::array WhisperModel::LayerNorm(const mx::array &x,
                                  const std::string &prefix) const {
  return mx::fast::layer_norm(x, Weight(prefix + ".weight"),
                              Weight(prefix + ".bias"), kLayerNormEpsilon);
}

mx::array WhisperModel::Conv1d(const mx::array &x, const std::string &prefix,
                               int stride) const {
  return mx::add(mx::conv1d(x, Weight(prefix + ".weight"), stride, 1),
                 Weight(prefix + ".bias"));
}

mx::array WhisperModel::Attention(const mx::array &queries_input,
                                  const mx::array &keys,
                                  const mx::array &values,
                                  const std::string &prefix, int head_count,
                                  const std::optional<mx::array> &mask) const {
  const int batch = queries_input.shape(0);
  const int length = queries_input.shape(1);
  const int width = queries_input.shape(2);
  const mx::array queries =
      SplitHeads(Linear(queries_input, prefix + ".query"), head_count);
  const float scale = std::pow(static_cast<float>(width / head_count), -0.5f);
  const mx::array attended =
      mask.has_value() ? mx::fast::scaled_dot_product_attention(
                             queries, keys, values, scale, "", *mask)
                       : mx::fast::scaled_dot_product_attention(queries, keys,
                                                                values, scale);
  return Linear(mx::reshape(mx::transpose(attended, {0, 2, 1, 3}),
                            {batch, length, width}),
                prefix + ".out");
}

mx::array WhisperModel::EncoderLayer(const mx::array &x, int layer) const {
  const std::string prefix = "encoder.blocks." + std::to_string(layer);
  const int head_count = config_.audio_head_count;
  const mx::array normed = LayerNorm(x, prefix + ".attn_ln");
  const mx::array keys =
      SplitHeads(Linear(normed, prefix + ".attn.key"), head_count);
  const mx::array values =
      SplitHeads(Linear(normed, prefix + ".attn.value"), head_count);
  const mx::array hidden =
      mx::add(x, Attention(normed, keys, values, prefix + ".attn", head_count,
                           std::nullopt));
  return mx::add(hidden,
                 Linear(Gelu(Linear(LayerNorm(hidden, prefix + ".mlp_ln"),
                                    prefix + ".mlp1")),
                        prefix + ".mlp2"));
}

mx::array WhisperModel::Encode(const std::vector<float> &window_samples) const {
  const int sample_count = static_cast<int>(window_samples.size());
  const mx::array audio(window_samples.data(), {sample_count}, mx::float32);
  // Note (khazic): reflect padding: audio[1:201] reversed, audio,
  // audio[-201:-1] reversed.
  const mx::array head = mx::slice(audio, {kReflectPadding}, {0}, {-1});
  const mx::array tail = mx::slice(audio, {sample_count - 2},
                                   {sample_count - kReflectPadding - 2}, {-1});
  const mx::array padded = mx::concatenate({head, audio, tail});
  const int frame_count = 1 + (padded.shape(0) - kFftSize) / kHopLength;
  const mx::array frames =
      mx::as_strided(padded, {frame_count, kFftSize}, {kHopLength, 1}, 0);
  mx::array magnitudes = mx::square(mx::abs(mx::fft::rfft(
      mx::multiply(frames, mx::expand_dims(hann_window_, 0)), -1)));
  // Note (khazic): the final centered STFT frame is dropped, as Whisper's
  // reference front end drops it.
  magnitudes = mx::transpose(
      mx::slice(magnitudes, {0, 0}, {frame_count - 1, magnitudes.shape(1)}),
      {1, 0});
  const mx::array mel_power =
      mx::matmul(mx::transpose(mel_filters_, {1, 0}), magnitudes);
  mx::array log_mel =
      mx::log10(mx::maximum(mel_power, mx::array(kLogMelFloor)));
  log_mel = mx::maximum(
      log_mel, mx::subtract(mx::max(log_mel), mx::array(kLogMelDynamicRange)));
  log_mel = mx::divide(mx::add(log_mel, mx::array(4.0f)), mx::array(4.0f));

  mx::array x = mx::expand_dims(mx::transpose(log_mel, {1, 0}), 0);
  x = Gelu(Conv1d(x, "encoder.conv1", 1));
  x = Gelu(Conv1d(x, "encoder.conv2", 2));
  x = mx::add(x, mx::slice(Weight("encoder.positional_embedding"), {0, 0},
                           {x.shape(1), x.shape(2)}));
  for (int layer = 0; layer < config_.audio_layer_count; ++layer) {
    x = EncoderLayer(x, layer);
  }
  return LayerNorm(x, "encoder.ln_post");
}

mx::array WhisperModel::DecoderLayer(const mx::array &x, int layer,
                                     const mx::array &encoder_states,
                                     const std::optional<mx::array> &mask,
                                     DecoderLayerCache &cache) const {
  const std::string prefix = "decoder.blocks." + std::to_string(layer);
  const int head_count = config_.text_head_count;
  const mx::array normed = LayerNorm(x, prefix + ".attn_ln");
  mx::array keys = SplitHeads(Linear(normed, prefix + ".attn.key"), head_count);
  mx::array values =
      SplitHeads(Linear(normed, prefix + ".attn.value"), head_count);
  if (cache.self_keys.has_value()) {
    keys = mx::concatenate({*cache.self_keys, keys}, 2);
    values = mx::concatenate({*cache.self_values, values}, 2);
  } else {
  }
  cache.self_keys = keys;
  cache.self_values = values;
  mx::array hidden = mx::add(
      x, Attention(normed, keys, values, prefix + ".attn", head_count, mask));
  if (!cache.cross_keys.has_value()) {
    cache.cross_keys = SplitHeads(
        Linear(encoder_states, prefix + ".cross_attn.key"), head_count);
    cache.cross_values = SplitHeads(
        Linear(encoder_states, prefix + ".cross_attn.value"), head_count);
  } else {
  }
  hidden = mx::add(hidden,
                   Attention(LayerNorm(hidden, prefix + ".cross_attn_ln"),
                             *cache.cross_keys, *cache.cross_values,
                             prefix + ".cross_attn", head_count, std::nullopt));
  return mx::add(hidden,
                 Linear(Gelu(Linear(LayerNorm(hidden, prefix + ".mlp_ln"),
                                    prefix + ".mlp1")),
                        prefix + ".mlp2"));
}

mx::array WhisperModel::Decode(const mx::array &token_ids, int start_position,
                               const mx::array &encoder_states,
                               std::vector<DecoderLayerCache> &caches) const {
  const int length = token_ids.shape(1);
  const int total_length = start_position + length;
  std::vector<int32_t> position_values(length);
  for (int i = 0; i < length; ++i)
    position_values[i] = start_position + i;
  const mx::array position_ids(position_values.data(), {length}, mx::int32);
  mx::array hidden =
      mx::add(mx::take(Weight("decoder.token_embedding.weight"), token_ids, 0),
              mx::expand_dims(mx::take(Weight("decoder.positional_embedding"),
                                       position_ids, 0),
                              0));
  std::optional<mx::array> mask;
  if (length > 1) {
    std::vector<int32_t> column_values(total_length);
    for (int i = 0; i < total_length; ++i)
      column_values[i] = i;
    const mx::array rows = mx::expand_dims(position_ids, 1);
    const mx::array columns = mx::expand_dims(
        mx::array(column_values.data(), {total_length}, mx::int32), 0);
    mask = mx::where(mx::less_equal(columns, rows),
                     mx::zeros({length, total_length}, hidden.dtype()),
                     mx::full({length, total_length}, mx::array(kMaskedScore),
                              hidden.dtype()));
  } else {
  }
  for (int layer = 0; layer < config_.text_layer_count; ++layer) {
    hidden = DecoderLayer(hidden, layer, encoder_states, mask, caches[layer]);
  }
  hidden = LayerNorm(hidden, "decoder.ln");
  const int width = hidden.shape(2);
  const mx::array last = mx::reshape(
      mx::slice(hidden, {0, length - 1, 0}, {1, length, width}), {width});
  // Note (khazic): tied output projection: the token embedding used as a linear
  // layer.
  return mx::matmul(last,
                    mx::transpose(Weight("decoder.token_embedding.weight")));
}

} // namespace whisper
