// SPDX-License-Identifier: Apache-2.0
// Chatterbox-Turbo's S3 speech tokenizer and the shared 16 kHz spectrum.
#include "chatterbox/chatterbox_speech_tokenizer.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace chatterbox {

namespace mx = mlx::core;

namespace {

constexpr int kMelCount = 128;
constexpr int kLayerCount = 6;
constexpr int kHiddenSize = 1280;
constexpr int kHeadCount = 20;
constexpr int kHeadDim = kHiddenSize / kHeadCount;
constexpr int kFsmnKernelSize = 31;
constexpr int kRotaryPositions = 2048;
constexpr double kRotaryTheta = 10000.0;
constexpr float kAttentionNormEps = 1e-6f;
constexpr float kMlpNormEps = 1e-5f;
// Longer input is tokenized in overlapping 30 s windows, which is not
// supported here.
constexpr int kMaxMelFrames = 3000;
constexpr float kMelFloor = 1e-10f;
constexpr float kLogMelDynamicRange = 8.0f;
constexpr int kQuantizerDimensions = 8;
constexpr float kQuantizerLevelCount = 3.0f;
// tanh output is shrunk just inside (-1, 1) before rounding to {-1, 0, 1}.
constexpr float kQuantizerShrink = 0.999f;

std::string LayerPrefix(int layer) {
  return "encoder.blocks." + std::to_string(layer);
}

// x [1, positions, heads, head dim] rotated by halves.
mx::array ApplyRotary(const mx::array &x, const mx::array &cos,
                      const mx::array &sin) {
  const int half = x.shape(-1) / 2;
  const mx::array left =
      mx::slice(x, {0, 0, 0, 0}, {x.shape(0), x.shape(1), x.shape(2), half});
  const mx::array right = mx::slice(
      x, {0, 0, 0, half}, {x.shape(0), x.shape(1), x.shape(2), x.shape(3)});
  const mx::array rotated = mx::concatenate({mx::negative(right), left}, -1);
  return mx::add(mx::multiply(x, cos), mx::multiply(rotated, sin));
}

} // namespace

mx::array Spectrum16k(const std::vector<float> &samples_16k) {
  const int sample_count = static_cast<int>(samples_16k.size());
  const int padding = kSpectrumFftSize / 2;
  if (sample_count <= padding) {
    throw std::invalid_argument("need more than " + std::to_string(padding) +
                                " samples for a spectrum");
  } else {
  }
  std::vector<float> padded(sample_count + 2 * padding);
  for (int i = 0; i < padding; ++i) {
    padded[i] = samples_16k[padding - i];
    padded[padding + sample_count + i] = samples_16k[sample_count - 2 - i];
  }
  std::copy(samples_16k.begin(), samples_16k.end(), padded.begin() + padding);
  // Note (Jiaxin Deng): the symmetric Hann window is built in double and
  // rounded once, as the float32 constant it becomes.
  std::vector<float> window(kSpectrumFftSize);
  const double denominator = kSpectrumFftSize - 1;
  for (int n = 0; n < kSpectrumFftSize; ++n) {
    window[n] =
        static_cast<float>(0.5 * (1 - std::cos(2 * M_PI * n / denominator)));
  }
  const int frame_count =
      1 + (static_cast<int>(padded.size()) - kSpectrumFftSize) /
              kSpectrumHopSamples;
  const mx::array signal(padded.begin(), {static_cast<int>(padded.size())});
  const mx::array frames = mx::as_strided(
      signal, {frame_count, kSpectrumFftSize}, {kSpectrumHopSamples, 1}, 0);
  return mx::fft::rfft(
      mx::multiply(frames, mx::array(window.begin(), {kSpectrumFftSize})));
}

mx::array SlaneyMelFilters16k(int mel_count) {
  const double maximum_hz = kSpeechTokenizerSampleRate / 2;
  const double linear_hz_per_mel = 200.0 / 3;
  const double log_region_hz = 1000.0;
  const double log_region_mel = log_region_hz / linear_hz_per_mel;
  const double log_step = std::log(6.4) / 27.0;
  const double maximum_mel =
      log_region_mel + std::log(maximum_hz / log_region_hz) / log_step;
  const int bin_count = kSpectrumFftSize / 2 + 1;
  const mx::array bin_hz =
      mx::linspace(0, kSpeechTokenizerSampleRate / 2, bin_count);
  const mx::array mel_points = mx::linspace(0.0, maximum_mel, mel_count + 2);
  const mx::array linear_hz =
      mx::add(mx::array(0.0f),
              mx::multiply(mx::array(static_cast<float>(linear_hz_per_mel)),
                           mel_points));
  const mx::array log_hz = mx::multiply(
      mx::array(static_cast<float>(log_region_hz)),
      mx::exp(mx::multiply(
          mx::array(static_cast<float>(log_step)),
          mx::subtract(mel_points,
                       mx::array(static_cast<float>(log_region_mel))))));
  const mx::array point_hz =
      mx::where(mx::greater_equal(
                    mel_points, mx::array(static_cast<float>(log_region_mel))),
                log_hz, linear_hz);
  const int point_count = mel_count + 2;
  const mx::array point_gaps =
      mx::subtract(mx::slice(point_hz, {1}, {point_count}),
                   mx::slice(point_hz, {0}, {point_count - 1}));
  const mx::array slopes =
      mx::subtract(mx::expand_dims(point_hz, 0), mx::expand_dims(bin_hz, 1));
  const mx::array down_slopes = mx::divide(
      mx::negative(mx::slice(slopes, {0, 0}, {bin_count, point_count - 2})),
      mx::slice(point_gaps, {0}, {point_count - 2}));
  const mx::array up_slopes =
      mx::divide(mx::slice(slopes, {0, 2}, {bin_count, point_count}),
                 mx::slice(point_gaps, {1}, {point_count - 1}));
  mx::array filters = mx::maximum(mx::zeros_like(down_slopes),
                                  mx::minimum(down_slopes, up_slopes));
  const mx::array area_normalization = mx::divide(
      mx::array(2.0f), mx::subtract(mx::slice(point_hz, {2}, {point_count}),
                                    mx::slice(point_hz, {0}, {mel_count})));
  filters = mx::multiply(filters, mx::expand_dims(area_normalization, 0));
  return mx::moveaxis(filters, 0, 1);
}

SpeechTokenizer::SpeechTokenizer(const omni_common::Checkpoint &checkpoint)
    : checkpoint_(checkpoint), mel_filters_(SlaneyMelFilters16k(kMelCount)),
      rotary_cos_(0.0f), rotary_sin_(0.0f) {
  const mx::array exponents =
      mx::divide(mx::astype(mx::arange(0, kHeadDim, 2), mx::float32),
                 mx::array(static_cast<float>(kHeadDim)));
  const mx::array frequencies = mx::divide(
      mx::array(1.0f),
      mx::power(mx::array(static_cast<float>(kRotaryTheta)), exponents));
  const mx::array angles = mx::astype(
      mx::outer(mx::arange(kRotaryPositions), frequencies), mx::float32);
  const mx::array cos = mx::cos(angles);
  const mx::array sin = mx::sin(angles);
  rotary_cos_ = mx::concatenate({cos, cos}, -1);
  rotary_sin_ = mx::concatenate({sin, sin}, -1);
  mx::eval({mel_filters_, rotary_cos_, rotary_sin_});
}

mx::array SpeechTokenizer::LogMel(const std::vector<float> &samples_16k) const {
  mx::array spectrum = mx::expand_dims(Spectrum16k(samples_16k), 0);
  // The last frame is dropped, so the mel has samples / hop frames.
  const int frame_count = spectrum.shape(1) - 1;
  spectrum =
      mx::slice(spectrum, {0, 0, 0}, {1, frame_count, spectrum.shape(2)});
  const mx::array power = mx::power(mx::abs(spectrum), mx::array(2.0f));
  const mx::array mel =
      mx::transpose(mx::matmul(power, mx::transpose(mel_filters_)), {0, 2, 1});
  mx::array log_mel = mx::log10(mx::maximum(mel, mx::array(kMelFloor)));
  log_mel = mx::maximum(
      log_mel, mx::subtract(mx::max(log_mel), mx::array(kLogMelDynamicRange)));
  return mx::divide(mx::add(log_mel, mx::array(4.0f)), mx::array(4.0f));
}

mx::array SpeechTokenizer::Block(const mx::array &x, int layer,
                                 const mx::array &attention_mask) const {
  const std::string prefix = LayerPrefix(layer);
  const int positions = x.shape(1);
  const mx::array normed =
      checkpoint_.LayerNorm(x, prefix + ".attn_ln", kAttentionNormEps);
  const mx::array query = checkpoint_.Linear(normed, prefix + ".attn.query");
  const mx::array key = checkpoint_.Linear(normed, prefix + ".attn.key");
  const mx::array value = checkpoint_.Linear(normed, prefix + ".attn.value");
  const mx::Shape head_shape = {1, positions, kHeadCount, kHeadDim};
  const mx::array cos = mx::expand_dims(
      mx::slice(rotary_cos_, {0, 0}, {positions, kHeadDim}), {0, 2});
  const mx::array sin = mx::expand_dims(
      mx::slice(rotary_sin_, {0, 0}, {positions, kHeadDim}), {0, 2});
  const mx::array q = ApplyRotary(mx::reshape(query, head_shape), cos, sin);
  const mx::array k = ApplyRotary(mx::reshape(key, head_shape), cos, sin);
  const mx::array v = mx::reshape(value, head_shape);

  // The FSMN memory: a centered depthwise convolution over the values.
  const mx::array flat_value = mx::reshape(v, {1, positions, kHiddenSize});
  const int left_padding = (kFsmnKernelSize - 1) / 2;
  const int right_padding = kFsmnKernelSize - 1 - left_padding;
  const mx::array padded_value = mx::concatenate(
      {mx::zeros({1, left_padding, kHiddenSize}, flat_value.dtype()),
       flat_value,
       mx::zeros({1, right_padding, kHiddenSize}, flat_value.dtype())},
      1);
  const mx::array memory =
      mx::add(mx::conv1d(padded_value,
                         checkpoint_.Weight(prefix + ".attn.fsmn_block.weight"),
                         1, 0, 1, kHiddenSize),
              flat_value);

  const mx::array scale(
      static_cast<float>(std::pow(static_cast<double>(kHeadDim), -0.25)));
  const mx::array attention = mx::fast::scaled_dot_product_attention(
      mx::multiply(mx::transpose(q, {0, 2, 1, 3}), scale),
      mx::multiply(mx::transpose(k, {0, 2, 1, 3}), scale),
      mx::transpose(v, {0, 2, 1, 3}), 1.0f, "array", attention_mask);
  const mx::array attended = mx::reshape(mx::transpose(attention, {0, 2, 1, 3}),
                                         {1, positions, kHiddenSize});
  const mx::array after_attention = mx::add(
      x, mx::add(checkpoint_.Linear(attended, prefix + ".attn.out"), memory));

  const mx::array mlp_input =
      checkpoint_.LayerNorm(after_attention, prefix + ".mlp_ln", kMlpNormEps);
  const mx::array mlp_output =
      checkpoint_.Linear(omni_common::Gelu(checkpoint_.Linear(
                             mlp_input, prefix + ".mlp.layers.0")),
                         prefix + ".mlp.layers.2");
  return mx::add(after_attention, mlp_output);
}

mx::array SpeechTokenizer::Encode(const std::vector<float> &samples_16k) const {
  if (samples_16k.size() / kSpectrumHopSamples > kMaxMelFrames) {
    throw std::invalid_argument(
        "the speech tokenizer takes at most 30 s of audio");
  } else {
  }
  // [1, mel frames, mel bands]
  mx::array x = mx::transpose(LogMel(samples_16k), {0, 2, 1});
  x = omni_common::Gelu(
      mx::add(mx::conv1d(x, checkpoint_.Weight("encoder.conv1.weight"), 2, 1),
              checkpoint_.Weight("encoder.conv1.bias")));
  x = omni_common::Gelu(
      mx::add(mx::conv1d(x, checkpoint_.Weight("encoder.conv2.weight"), 2, 1),
              checkpoint_.Weight("encoder.conv2.bias")));
  const int token_count = x.shape(1);
  // Note (Jiaxin Deng): a single clip masks nothing, but the all-pass
  // additive mask (-0.0) keeps the masked attention kernel.
  const mx::array attention_mask = mx::full({1, 1, token_count}, -0.0f);
  for (int layer = 0; layer < kLayerCount; ++layer) {
    x = Block(x, layer, attention_mask);
  }

  mx::array levels = mx::tanh(
      mx::astype(checkpoint_.Linear(mx::reshape(x, {-1, kHiddenSize}),
                                    "quantizer.fsq_codebook.project_down"),
                 mx::float32));
  levels = mx::add(mx::round(mx::multiply(levels, mx::array(kQuantizerShrink))),
                   mx::array(1.0f));
  const mx::array place_values =
      mx::power(mx::array(kQuantizerLevelCount),
                mx::arange(0, kQuantizerDimensions, mx::float32));
  const mx::array tokens =
      mx::sum(mx::multiply(levels, mx::expand_dims(place_values, 0)), -1);
  return mx::astype(mx::reshape(tokens, {1, token_count}), mx::int32);
}

} // namespace chatterbox
