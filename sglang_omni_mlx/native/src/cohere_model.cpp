// SPDX-License-Identifier: Apache-2.0
#include "cohere_model.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>

#include "nlohmann/json.hpp"
#include "swift_port.h"

namespace cohere_transcribe {

namespace mx = mlx::core;

namespace {

constexpr int kFftSize = 512;
constexpr int kWindowLength = 400;
constexpr int kHopLength = 160;
constexpr float kPreEmphasis = 0.97f;
constexpr float kLayerNormEpsilon = 1e-5f;
constexpr float kBatchNormEpsilon = 1e-5f;
constexpr float kNormalizationEpsilon = 1e-5f;
constexpr int kSubsamplingStride = 2;
// Note (khazic): the decoder's additive causal mask value.
constexpr float kMaskedScore = -1e9f;

// The symmetric Hann window of kWindowLength, centered in kFftSize zeros.
mx::array CenteredHannWindow() {
  // Note (khazic): Swift's Float.pi: pi rounded toward zero, one step below
  // (float)M_PI.
  const float swift_pi = std::nextafter(static_cast<float>(M_PI), 0.0f);
  const float denominator = static_cast<float>(kWindowLength - 1);
  std::vector<float> window(kFftSize, 0.0f);
  const int left_padding = (kFftSize - kWindowLength) / 2;
  for (int n = 0; n < kWindowLength; ++n) {
    window[left_padding + n] =
        0.5f * (1.0f - std::cos(2.0f * swift_pi * static_cast<float>(n) /
                                denominator));
  }
  return mx::array(window.data(), {kFftSize}, mx::float32);
}

mx::array MelFilters(int mel_bin_count) {
  const std::vector<float> values =
      swift_port::SlaneyMelFilterBank(kFftSize, mel_bin_count);
  return mx::array(values.data(), {kFftSize / 2 + 1, mel_bin_count},
                   mx::float32);
}

// Interleaved sin and cos [positions, channels] of positions [positions, 1]
// over the even channel indices.
mx::array Sinusoids(const mx::array &positions, int channels) {
  std::vector<int32_t> even_values(channels / 2);
  for (int i = 0; i < channels / 2; ++i)
    even_values[i] = 2 * i;
  const mx::array even_indices = mx::astype(
      mx::array(even_values.data(), {channels / 2}, mx::int32), mx::float32);
  const float scale = -std::log(10000.0f) / static_cast<float>(channels);
  const mx::array angles = mx::multiply(
      positions, mx::exp(mx::multiply(even_indices, mx::array(scale))));
  return mx::reshape(mx::stack({mx::sin(angles), mx::cos(angles)}, -1),
                     {positions.shape(0), channels});
}

// Sinusoids of the relative positions table_length - 1 down to
// -(table_length - 1), [1, 2 * table_length - 1, width].
mx::array RelativePositionTable(int table_length, int width) {
  const int position_count = 2 * table_length - 1;
  std::vector<float> position_values(position_count);
  for (int i = 0; i < position_count; ++i)
    position_values[i] = static_cast<float>(table_length - 1 - i);
  return mx::expand_dims(Sinusoids(mx::array(position_values.data(),
                                             {position_count, 1}, mx::float32),
                                   width),
                         0);
}

// Fixed sinusoids of the decoder positions scaled down by sqrt(width).
mx::array DecoderPositions(int length, int width) {
  return mx::divide(
      Sinusoids(
          mx::reshape(mx::astype(mx::arange(length, mx::int32), mx::float32),
                      {length, 1}),
          width),
      mx::array(static_cast<float>(std::sqrt(static_cast<double>(width)))));
}

CohereConfig ReadConfig(const std::filesystem::path &model_directory) {
  std::ifstream config_stream(model_directory / "config.json");
  if (!config_stream) {
    throw std::runtime_error("cannot read config.json in " +
                             model_directory.string());
  } else {
  }
  const nlohmann::json config = nlohmann::json::parse(config_stream);
  if (config.contains("quantization") ||
      config.contains("quantization_config")) {
    throw std::runtime_error("quantized Cohere checkpoints are not served");
  } else {
  }
  const nlohmann::json &encoder = config.at("encoder");
  const nlohmann::json &decoder = config.at("transf_decoder").at("config_dict");
  return {encoder.at("d_model").get<int>(),
          encoder.at("n_heads").get<int>(),
          encoder.at("n_layers").get<int>(),
          encoder.at("conv_kernel_size").get<int>(),
          encoder.at("pos_emb_max_len").get<int>(),
          encoder.at("feat_in").get<int>(),
          decoder.at("hidden_size").get<int>(),
          decoder.at("num_attention_heads").get<int>(),
          decoder.at("num_layers").get<int>(),
          decoder.at("max_sequence_length").get<int>()};
}

// [batch, length, width] to [batch, heads, length, width / heads].
mx::array SplitHeads(const mx::array &x, int head_count) {
  return mx::transpose(
      mx::reshape(x, {x.shape(0), -1, head_count, x.shape(2) / head_count}),
      {0, 2, 1, 3});
}

} // namespace

CohereModel::CohereModel(const std::filesystem::path &model_directory)
    : config_(ReadConfig(model_directory)), hann_window_(CenteredHannWindow()),
      mel_filters_(MelFilters(config_.mel_bin_count)),
      relative_position_table_(RelativePositionTable(
          config_.relative_position_count, config_.encoder_width)),
      decoder_positions_(DecoderPositions(config_.max_sequence_length,
                                          config_.decoder_width)) {
  std::vector<std::filesystem::path> weight_files;
  for (const auto &entry :
       std::filesystem::directory_iterator(model_directory)) {
    if (entry.path().extension() == ".safetensors") {
      weight_files.push_back(entry.path());
    } else {
    }
  }
  if (weight_files.empty()) {
    throw std::runtime_error("no Cohere weights in " +
                             model_directory.string());
  } else {
  }
  for (const auto &path : weight_files) {
    auto [loaded, metadata] = mx::load_safetensors(path.string());
    for (auto &[raw_name, array] : loaded) {
      if (raw_name.rfind("preprocessor.", 0) == 0 ||
          raw_name.rfind("decoder.embedding.position_embedding", 0) == 0 ||
          (raw_name.size() > 20 &&
           raw_name.compare(raw_name.size() - 20, 20, ".num_batches_tracked") ==
               0)) {
        continue;
      } else {
      }
      // Note (khazic): encoder.subsampling.conv.N. is the subsampling's convN.
      std::string name = raw_name;
      const std::string subsampling_prefix = "encoder.subsampling.conv.";
      if (name.rfind(subsampling_prefix, 0) == 0) {
        name =
            "encoder.subsampling.conv" + name.substr(subsampling_prefix.size());
      } else {
      }
      // Note (khazic): channels-first convolution weights move their channels
      // last: a subsampling kernel is 3x3, or 1x1 for the pointwise conv3 and
      // conv6.
      const bool is_weight =
          name.size() > 7 && name.compare(name.size() - 7, 7, ".weight") == 0;
      const int subsampling_kernel =
          name == "encoder.subsampling.conv3.weight" ||
                  name == "encoder.subsampling.conv6.weight"
              ? 1
              : 3;
      if (is_weight && array.ndim() == 4 &&
          name.rfind("encoder.subsampling.conv", 0) == 0 &&
          array.shape(1) != subsampling_kernel &&
          array.shape(2) == subsampling_kernel &&
          array.shape(3) == subsampling_kernel) {
        array = mx::transpose(array, {0, 2, 3, 1});
      } else if (is_weight && array.ndim() == 3 &&
                 name.find(".conv.") != std::string::npos) {
        const bool is_channels_first =
            name.find("depthwise_conv") != std::string::npos
                ? array.shape(1) == 1 && array.shape(2) > 1
                : array.shape(2) == 1 && array.shape(1) > 1;
        if (is_channels_first) {
          array = mx::transpose(array, {0, 2, 1});
        } else {
        }
      } else {
      }
      weights_.insert_or_assign(name, array);
    }
  }

  std::vector<mx::array> parameters = {
      hann_window_, mel_filters_, relative_position_table_, decoder_positions_};
  for (const auto &[name, array] : weights_)
    parameters.push_back(array);
  mx::eval(parameters);
}

const mx::array &CohereModel::Weight(const std::string &name) const {
  const auto found = weights_.find(name);
  if (found == weights_.end()) {
    throw std::runtime_error("checkpoint is missing " + name);
  } else {
  }
  return found->second;
}

mx::array CohereModel::Linear(const mx::array &x,
                              const std::string &prefix) const {
  const auto bias = weights_.find(prefix + ".bias");
  if (bias == weights_.end()) {
    return mx::matmul(x, mx::transpose(Weight(prefix + ".weight")));
  } else {
    return mx::addmm(bias->second, x,
                     mx::transpose(Weight(prefix + ".weight")));
  }
}

mx::array CohereModel::LayerNorm(const mx::array &x,
                                 const std::string &prefix) const {
  return mx::fast::layer_norm(x, Weight(prefix + ".weight"),
                              Weight(prefix + ".bias"), kLayerNormEpsilon);
}

mx::array CohereModel::Conv1d(const mx::array &x, const std::string &prefix,
                              int padding, int groups) const {
  return mx::add(
      mx::conv1d(x, Weight(prefix + ".weight"), 1, padding, 1, groups),
      Weight(prefix + ".bias"));
}

mx::array CohereModel::Conv2d(const mx::array &x, const std::string &prefix,
                              int stride, int padding, int groups) const {
  return mx::add(mx::conv2d(x, Weight(prefix + ".weight"), {stride, stride},
                            {padding, padding}, {1, 1}, groups),
                 Weight(prefix + ".bias"));
}

mx::array CohereModel::RelativePositions(int length, mx::Dtype dtype) const {
  // Note (khazic): relative positions from length - 1 down to -(length - 1),
  // from the table built at load, or for longer input a table of its own.
  const mx::array table =
      mx::astype(length <= config_.relative_position_count
                     ? relative_position_table_
                     : RelativePositionTable(length, config_.encoder_width),
                 dtype);
  const int center = table.shape(1) / 2 + 1;
  return mx::slice(table, {0, center - length, 0},
                   {1, center + length - 1, table.shape(2)});
}

mx::array CohereModel::EncoderSelfAttention(const mx::array &x,
                                            const mx::array &positions,
                                            const std::string &prefix) const {
  const int batch = x.shape(0);
  const int head_count = config_.encoder_head_count;
  const int head_width = config_.encoder_width / head_count;
  const std::vector<mx::array> parts =
      mx::split(Linear(x, prefix + ".qkv_proj"), 3, -1);
  const mx::array queries = SplitHeads(parts[0], head_count);
  const mx::array keys = SplitHeads(parts[1], head_count);
  const mx::array values = SplitHeads(parts[2], head_count);
  const mx::array projected_positions =
      SplitHeads(Linear(positions, prefix + ".pos_proj"), head_count);
  const mx::array content_scores = mx::matmul(
      mx::add(queries, mx::expand_dims(Weight(prefix + ".pos_bias_u"), {0, 2})),
      mx::transpose(keys, {0, 1, 3, 2}));
  mx::array position_scores = mx::matmul(
      mx::add(queries, mx::expand_dims(Weight(prefix + ".pos_bias_v"), {0, 2})),
      mx::transpose(projected_positions, {0, 1, 3, 2}));
  // Note (khazic): relative shift: row i of [length, 2 * length - 1] scores
  // moves left by length - 1 - i, so column j is relative position j - i.
  const int length = position_scores.shape(2);
  const int position_count = position_scores.shape(3);
  position_scores = mx::pad(position_scores, {{0, 0}, {0, 0}, {0, 0}, {1, 0}});
  position_scores = mx::reshape(
      position_scores, {batch, head_count, position_count + 1, length});
  position_scores = mx::slice(position_scores, {0, 0, 1, 0},
                              {batch, head_count, position_count + 1, length});
  position_scores =
      mx::reshape(position_scores, {batch, head_count, length, position_count});
  position_scores =
      mx::slice(position_scores, {0, 0, 0, 0},
                {batch, head_count, length, content_scores.shape(3)});
  const float scale = std::pow(static_cast<float>(head_width), -0.5f);
  const mx::array attention = mx::softmax(
      mx::multiply(mx::add(content_scores, position_scores), mx::array(scale)),
      -1);
  return Linear(
      mx::reshape(mx::transpose(mx::matmul(attention, values), {0, 2, 1, 3}),
                  {batch, -1, head_count * head_width}),
      prefix + ".out_proj");
}

mx::array CohereModel::EncoderConvolution(const mx::array &x,
                                          const std::string &prefix) const {
  const std::vector<mx::array> halves =
      mx::split(Conv1d(x, prefix + ".pointwise_conv1", 0, 1), 2, -1);
  mx::array hidden = mx::multiply(halves[0], mx::sigmoid(halves[1]));
  hidden =
      Conv1d(hidden, prefix + ".depthwise_conv",
             (config_.convolution_kernel_size - 1) / 2, config_.encoder_width);
  // Note (khazic): batch norm with its running statistics; eps takes the
  // statistics' dtype.
  const mx::array &variance = Weight(prefix + ".batch_norm.running_var");
  hidden = mx::multiply(
      mx::subtract(hidden, Weight(prefix + ".batch_norm.running_mean")),
      mx::rsqrt(
          mx::add(variance, mx::array(kBatchNormEpsilon, variance.dtype()))));
  hidden = mx::add(mx::multiply(Weight(prefix + ".batch_norm.weight"), hidden),
                   Weight(prefix + ".batch_norm.bias"));
  return Conv1d(swift_port::Silu(hidden), prefix + ".pointwise_conv2", 0, 1);
}

mx::array CohereModel::EncoderLayer(const mx::array &x,
                                    const mx::array &positions,
                                    int layer) const {
  const std::string prefix = "encoder.layers." + std::to_string(layer);
  const mx::array half(0.5f, x.dtype());
  const auto feed_forward = [&](const mx::array &input,
                                const std::string &name) {
    return Linear(
        swift_port::Silu(Linear(input, prefix + "." + name + ".linear1")),
        prefix + "." + name + ".linear2");
  };
  mx::array hidden = mx::add(
      x, mx::multiply(half,
                      feed_forward(LayerNorm(x, prefix + ".norm_feed_forward1"),
                                   "feed_forward1")));
  hidden = mx::add(
      hidden, EncoderSelfAttention(LayerNorm(hidden, prefix + ".norm_self_att"),
                                   positions, prefix + ".self_attn"));
  hidden = mx::add(hidden,
                   EncoderConvolution(LayerNorm(hidden, prefix + ".norm_conv"),
                                      prefix + ".conv"));
  hidden = mx::add(
      hidden,
      mx::multiply(
          half, feed_forward(LayerNorm(hidden, prefix + ".norm_feed_forward2"),
                             "feed_forward2")));
  return LayerNorm(hidden, prefix + ".norm_out");
}

mx::array CohereModel::Encode(const std::vector<float> &samples) const {
  const int sample_count = static_cast<int>(samples.size());
  const mx::array audio(samples.data(), {sample_count}, mx::float32);
  mx::array emphasized = audio;
  if (sample_count > 1) {
    emphasized = mx::concatenate(
        {mx::slice(audio, {0}, {1}),
         mx::subtract(
             mx::slice(audio, {1}, {sample_count}),
             mx::multiply(mx::array(kPreEmphasis),
                          mx::slice(audio, {0}, {sample_count - 1})))});
  } else {
  }
  // Note (khazic): centered frames over zero padding of half an FFT on each
  // side.
  const mx::array padding = mx::zeros({kFftSize / 2}, mx::float32);
  const mx::array padded = mx::concatenate({padding, emphasized, padding});
  const int frame_count = 1 + (padded.shape(0) - kFftSize) / kHopLength;
  const mx::array frames =
      mx::as_strided(padded, {frame_count, kFftSize}, {kHopLength, 1}, 0);
  const mx::array power =
      mx::square(mx::abs(mx::fft::rfft(mx::multiply(frames, hann_window_), 1)));
  mx::array mel = mx::log(mx::add(mx::matmul(power, mel_filters_),
                                  mx::array(std::ldexp(1.0f, -24))));
  mel = mx::expand_dims(mx::transpose(mel, {1, 0}), 0);
  // Note (khazic): per-feature normalization over time.
  const mx::array mean = mx::mean(mel, std::vector<int>{2}, true);
  const mx::array deviation =
      mx::add(mx::sqrt(mx::var(mel, std::vector<int>{2}, true)),
              mx::array(kNormalizationEpsilon));
  const mx::array features = mx::divide(mx::subtract(mel, mean), deviation);

  // Note (khazic): subsampling on [batch, frames, mel_bins, 1] images.
  const std::string prefix = "encoder.subsampling.";
  const int channels = Weight(prefix + "conv0.bias").shape(0);
  mx::array x = mx::expand_dims(mx::transpose(features, {0, 2, 1}), -1);
  x = swift_port::Relu(Conv2d(x, prefix + "conv0", kSubsamplingStride, 1, 1));
  x = Conv2d(x, prefix + "conv2", kSubsamplingStride, 1, channels);
  x = swift_port::Relu(Conv2d(x, prefix + "conv3", 1, 0, 1));
  x = Conv2d(x, prefix + "conv5", kSubsamplingStride, 1, channels);
  x = swift_port::Relu(Conv2d(x, prefix + "conv6", 1, 0, 1));
  x = mx::reshape(mx::transpose(x, {0, 1, 3, 2}),
                  {x.shape(0), x.shape(1), x.shape(3) * x.shape(2)});
  x = Linear(x, prefix + "out");
  const mx::array positions = RelativePositions(x.shape(1), x.dtype());
  for (int layer = 0; layer < config_.encoder_layer_count; ++layer) {
    x = EncoderLayer(x, positions, layer);
  }
  return Linear(x, "bridge_proj");
}

mx::array CohereModel::DecoderAttention(
    const mx::array &queries, const mx::array &keys, const mx::array &values,
    const std::string &prefix, const std::optional<mx::array> &mask) const {
  const int batch = queries.shape(0);
  const int length = queries.shape(2);
  const float scale = std::pow(
      static_cast<float>(config_.decoder_width / config_.decoder_head_count),
      -0.5f);
  const mx::array attended =
      mask.has_value() ? mx::fast::scaled_dot_product_attention(
                             queries, keys, values, scale, "", *mask)
                       : mx::fast::scaled_dot_product_attention(queries, keys,
                                                                values, scale);
  return Linear(mx::reshape(mx::transpose(attended, {0, 2, 1, 3}),
                            {batch, length, config_.decoder_width}),
                prefix + ".out_proj");
}

mx::array CohereModel::Decode(const mx::array &token_ids, int start_position,
                              const mx::array &encoder_states,
                              std::vector<DecoderLayerCache> &caches) const {
  const int length = token_ids.shape(1);
  const int width = config_.decoder_width;
  const int head_count = config_.decoder_head_count;
  std::vector<int32_t> position_values(length);
  for (int i = 0; i < length; ++i)
    position_values[i] = start_position + i;
  const mx::array positions = mx::reshape(
      mx::take(decoder_positions_,
               mx::array(position_values.data(), {length}, mx::int32), 0),
      {1, length, width});
  mx::array hidden = LayerNorm(
      mx::add(mx::take(Weight("decoder.embedding.token_embedding.weight"),
                       token_ids, 0),
              positions),
      "decoder.embedding.layer_norm");
  std::optional<mx::array> mask;
  if (length > 1) {
    std::vector<int32_t> index_values(length);
    for (int i = 0; i < length; ++i)
      index_values[i] = i;
    const mx::array indices(index_values.data(), {length}, mx::int32);
    mask = mx::astype(
        mx::multiply(mx::astype(mx::less(mx::expand_dims(indices, 1),
                                         mx::expand_dims(indices, 0)),
                                mx::float32),
                     mx::array(kMaskedScore)),
        encoder_states.dtype());
  } else {
  }
  for (int layer = 0; layer < config_.decoder_layer_count; ++layer) {
    const std::string prefix = "decoder.core.layers." + std::to_string(layer);
    DecoderLayerCache &cache = caches[layer];
    const std::vector<mx::array> self_parts =
        mx::split(Linear(LayerNorm(hidden, prefix + ".layer_norm_1"),
                         prefix + ".first_sub_layer.qkv_proj"),
                  3, -1);
    mx::array keys = SplitHeads(self_parts[1], head_count);
    mx::array values = SplitHeads(self_parts[2], head_count);
    if (cache.self_keys.has_value()) {
      keys = mx::concatenate({*cache.self_keys, keys}, 2);
      values = mx::concatenate({*cache.self_values, values}, 2);
    } else {
    }
    cache.self_keys = keys;
    cache.self_values = values;
    hidden = mx::add(
        hidden, DecoderAttention(SplitHeads(self_parts[0], head_count), keys,
                                 values, prefix + ".first_sub_layer", mask));
    if (!cache.cross_keys.has_value()) {
      const std::vector<mx::array> source_parts = mx::split(
          Linear(encoder_states, prefix + ".second_sub_layer.qkv_proj"), 3, -1);
      cache.cross_keys = SplitHeads(source_parts[1], head_count);
      cache.cross_values = SplitHeads(source_parts[2], head_count);
    } else {
    }
    const std::vector<mx::array> cross_parts =
        mx::split(Linear(LayerNorm(hidden, prefix + ".layer_norm_2"),
                         prefix + ".second_sub_layer.qkv_proj"),
                  3, -1);
    hidden = mx::add(
        hidden, DecoderAttention(SplitHeads(cross_parts[0], head_count),
                                 *cache.cross_keys, *cache.cross_values,
                                 prefix + ".second_sub_layer", std::nullopt));
    hidden =
        mx::add(hidden, Linear(swift_port::Relu(Linear(
                                   LayerNorm(hidden, prefix + ".layer_norm_3"),
                                   prefix + ".third_sub_layer.dense_in")),
                               prefix + ".third_sub_layer.dense_out"));
  }
  hidden = LayerNorm(hidden, "decoder.core.final_layer_norm");
  const mx::array last = mx::reshape(
      mx::slice(hidden, {0, length - 1, 0}, {1, length, width}), {width});
  return Linear(last, "lm_head");
}

} // namespace cohere_transcribe
