// SPDX-License-Identifier: Apache-2.0
#include "sortformer_features.h"

#include <cmath>
#include <stdexcept>

// Note (Jiaxin Deng): Swift never fuses a multiply and an add; contracting here
// breaks bit identity of the filter bank and window.
#pragma STDC FP_CONTRACT OFF

namespace sortformer {

namespace mx = mlx::core;

namespace {

struct SlaneyScale {
  float min_hz = 0.0f;
  float linear_step_hz = 200.0f / 3.0f;
  float min_log_hz = 1000.0f;
  float min_log_mel = (1000.0f - 0.0f) / (200.0f / 3.0f);
  float log_step = std::log(6.4f) / 27.0f;

  float HertzToMel(float frequency_hz) const {
    if (frequency_hz < min_log_hz) {
      return (frequency_hz - min_hz) / linear_step_hz;
    } else {
      return min_log_mel + std::log(frequency_hz / min_log_hz) / log_step;
    }
  }

  float MelToHertz(float mel) const {
    if (mel < min_log_mel) {
      return min_hz + linear_step_hz * mel;
    } else {
      return min_log_hz * std::exp(log_step * (mel - min_log_mel));
    }
  }
};

// Note (Jiaxin Deng): 2^-24, the log guard NeMo adds before the log.
constexpr float kLogGuard = 5.9604644775390625e-08f;

} // namespace

std::vector<float> MelFilterBank(int sample_rate, int fft_size,
                                 int mel_bin_count) {
  const SlaneyScale scale;
  const float max_hz = static_cast<float>(sample_rate) / 2.0f;
  const int frequency_bin_count = fft_size / 2 + 1;
  std::vector<float> bin_frequencies_hz(frequency_bin_count);
  for (int i = 0; i < frequency_bin_count; ++i) {
    bin_frequencies_hz[i] = static_cast<float>(i) *
                            static_cast<float>(sample_rate) /
                            static_cast<float>(fft_size);
  }
  const float mel_min = scale.HertzToMel(scale.min_hz);
  const float mel_max = scale.HertzToMel(max_hz);
  std::vector<float> edges_hz(mel_bin_count + 2);
  for (int i = 0; i < mel_bin_count + 2; ++i) {
    const float mel = mel_min + static_cast<float>(i) * (mel_max - mel_min) /
                                    static_cast<float>(mel_bin_count + 1);
    edges_hz[i] = scale.MelToHertz(mel);
  }
  std::vector<float> filters(
      static_cast<size_t>(frequency_bin_count) * mel_bin_count, 0.0f);
  for (int frequency_bin = 0; frequency_bin < frequency_bin_count;
       ++frequency_bin) {
    const float frequency_hz = bin_frequencies_hz[frequency_bin];
    for (int mel_bin = 0; mel_bin < mel_bin_count; ++mel_bin) {
      const float low = edges_hz[mel_bin];
      const float center = edges_hz[mel_bin + 1];
      const float high = edges_hz[mel_bin + 2];
      float &weight =
          filters[static_cast<size_t>(frequency_bin) * mel_bin_count + mel_bin];
      if (frequency_hz >= low && frequency_hz < center) {
        weight = (frequency_hz - low) / (center - low);
      } else if (frequency_hz >= center && frequency_hz <= high) {
        weight = (high - frequency_hz) / (high - center);
      } else {
      }
    }
  }
  // Note (Jiaxin Deng): normalized after the triangles, in Swift's order.
  for (int mel_bin = 0; mel_bin < mel_bin_count; ++mel_bin) {
    const float normalization =
        2.0f / (edges_hz[mel_bin + 2] - edges_hz[mel_bin]);
    for (int frequency_bin = 0; frequency_bin < frequency_bin_count;
         ++frequency_bin) {
      filters[static_cast<size_t>(frequency_bin) * mel_bin_count + mel_bin] *=
          normalization;
    }
  }
  return filters;
}

std::vector<float> SymmetricHannWindow(int size) {
  const float denominator = static_cast<float>(size - 1);
  // Note (Jiaxin Deng): Swift's Float.pi rounds toward zero, one step below
  // (float)M_PI.
  const float pi = std::nextafter(static_cast<float>(M_PI), 0.0f);
  std::vector<float> window(size);
  for (int n = 0; n < size; ++n) {
    window[n] =
        0.5f *
        (1.0f - std::cos(2.0f * pi * static_cast<float>(n) / denominator));
  }
  return window;
}

std::vector<float> CenteredWindow(int window_length, int fft_size) {
  std::vector<float> window = SymmetricHannWindow(window_length);
  if (window_length < fft_size) {
    const int left = (fft_size - window_length) / 2;
    window.insert(window.begin(), left, 0.0f);
    window.resize(fft_size, 0.0f);
  } else {
  }
  return window;
}

FeatureExtractor::FeatureExtractor(const ProcessorConfig &config)
    : config_(config),
      mel_filters_(MelFilterBank(config.sampling_rate, config.fft_size,
                                 config.feature_size)),
      window_(CenteredWindow(config.window_length, config.fft_size)),
      mel_filters_array_(mel_filters_.data(),
                         {config.fft_size / 2 + 1, config.feature_size},
                         mx::float32),
      window_array_(window_.data(), {static_cast<int>(window_.size())},
                    mx::float32) {
  if (config.window_length > config.fft_size) {
    throw std::runtime_error("Sortformer win_length must not exceed n_fft");
  } else {
  }
}

mx::array
FeatureExtractor::operator()(const std::vector<float> &samples) const {
  const int sample_count = static_cast<int>(samples.size());
  if (sample_count < 2) {
    throw std::invalid_argument("Sortformer needs at least two samples");
  } else {
  }
  const mx::array waveform(samples.data(), {1, sample_count}, mx::float32);
  // Note (Jiaxin Deng): preemphasis as separate MLX ops, in Swift's op order.
  const mx::array first = mx::slice(waveform, {0, 0}, {1, 1});
  const mx::array rest = mx::subtract(
      mx::slice(waveform, {0, 1}, {1, sample_count}),
      mx::multiply(mx::array(config_.preemphasis),
                   mx::slice(waveform, {0, 0}, {1, sample_count - 1})));
  const mx::array emphasized = mx::concatenate({first, rest}, -1);

  const int padding = config_.fft_size / 2;
  const mx::array audio = mx::reshape(emphasized, {sample_count});
  const mx::array padded =
      mx::concatenate({mx::zeros({padding}, mx::float32), audio,
                       mx::zeros({padding}, mx::float32)});
  const int frame_count =
      1 + (padded.shape(0) - config_.fft_size) / config_.hop_length;
  const mx::array frames = mx::as_strided(
      padded, {frame_count, config_.fft_size}, {config_.hop_length, 1}, 0);
  const mx::array spectrum =
      mx::fft::rfft(mx::multiply(frames, window_array_), 1);
  const mx::array power = mx::square(mx::abs(spectrum));
  const mx::array mel = mx::matmul(power, mel_filters_array_);
  const mx::array log_mel = mx::log(mx::add(mel, mx::array(kLogGuard)));
  return mx::stack({mx::transpose(log_mel, {1, 0})});
}

} // namespace sortformer
