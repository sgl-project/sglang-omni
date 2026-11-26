// SPDX-License-Identifier: Apache-2.0
#include "common/kaldi_fbank.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace omni_common {

namespace mx = mlx::core;

namespace {

constexpr int kKaldiSampleRate = 16000;
constexpr double kKaldiLowHz = 20.0;
constexpr float kInt16Scale = 32768.0f;
constexpr float kPreemphasis = 0.97f;

} // namespace

std::vector<double> KaldiMelBanks(int fft_size, int mel_bin_count) {
  const int frequency_bin_count = fft_size / 2 + 1;
  const double nyquist_hz = 0.5 * kKaldiSampleRate;
  const double bin_width_hz = static_cast<double>(kKaldiSampleRate) / fft_size;
  const double low_mel = 1127.0 * std::log(1.0 + kKaldiLowHz / 700.0);
  const double high_mel = 1127.0 * std::log(1.0 + nyquist_hz / 700.0);
  const double mel_step = (high_mel - low_mel) / (mel_bin_count + 1);
  std::vector<double> banks(
      static_cast<size_t>(mel_bin_count) * frequency_bin_count, 0.0);
  for (int mel_bin = 0; mel_bin < mel_bin_count; ++mel_bin) {
    const double left_mel = low_mel + mel_bin * mel_step;
    const double center_mel = low_mel + (mel_bin + 1) * mel_step;
    const double right_mel = low_mel + (mel_bin + 2) * mel_step;
    for (int bin = 0; bin < fft_size / 2; ++bin) {
      const double mel = 1127.0 * std::log(1.0 + bin_width_hz * bin / 700.0);
      double weight = 0.0;
      if (mel > left_mel && mel <= center_mel) {
        weight = (mel - left_mel) / (center_mel - left_mel);
      } else if (mel > center_mel && mel < right_mel) {
        weight = (right_mel - mel) / (right_mel - center_mel);
      } else {
      }
      banks[mel_bin * frequency_bin_count + bin] = weight;
    }
  }
  return banks;
}

int PaddedFftSize(int window_length) {
  int power = 1;
  while (power < window_length) {
    power <<= 1;
  }
  return power;
}

std::vector<float> SymmetricHammingWindow(int size) {
  std::vector<float> window(size);
  const float denominator = static_cast<float>(size - 1);
  for (int n = 0; n < size; ++n) {
    const float phase =
        2.0f * static_cast<float>(M_PI) * static_cast<float>(n) / denominator;
    window[n] = 0.54f - 0.46f * std::cos(phase);
  }
  return window;
}

mx::array KaldiFbank(const std::vector<float> &samples,
                     const std::vector<float> &window, int hop_length,
                     const mx::array &mel_filters, float log_floor) {
  const int window_length = static_cast<int>(window.size());
  const int sample_count = static_cast<int>(samples.size());
  if (sample_count < window_length) {
    return mx::zeros({0, mel_filters.shape(1)}, mx::float32);
  } else {
  }
  const mx::array audio =
      mx::multiply(mx::array(samples.data(), {sample_count}, mx::float32),
                   mx::array(kInt16Scale));
  const int frame_count = 1 + (sample_count - window_length) / hop_length;
  mx::array frames =
      mx::as_strided(audio, {frame_count, window_length}, {hop_length, 1}, 0);
  frames = mx::subtract(frames, mx::mean(frames, 1, true));
  const mx::array preemphasis(kPreemphasis);
  const mx::array head = mx::slice(frames, {0, 0}, {frame_count, 1});
  const mx::array first = mx::subtract(head, mx::multiply(preemphasis, head));
  const mx::array rest = mx::subtract(
      mx::slice(frames, {0, 1}, {frame_count, window_length}),
      mx::multiply(preemphasis, mx::slice(frames, {0, 0},
                                          {frame_count, window_length - 1})));
  frames = mx::concatenate({first, rest}, 1);
  frames = mx::multiply(frames,
                        mx::array(window.data(), {window_length}, mx::float32));
  const int fft_size = PaddedFftSize(window_length);
  if (fft_size > window_length) {
    frames = mx::concatenate(
        {frames,
         mx::zeros({frame_count, fft_size - window_length}, mx::float32)},
        1);
  } else {
  }
  const mx::array power = mx::square(mx::abs(mx::fft::rfft(frames, 1)));
  return mx::log(
      mx::maximum(mx::matmul(power, mel_filters), mx::array(log_floor)));
}

mx::array LowFrameRateFrames(const mx::array &fbank, int lfr_m, int lfr_n) {
  const int time = fbank.shape(0);
  const int mel_count = fbank.shape(1);
  const int lfr_frames = (time + lfr_n - 1) / lfr_n;
  if (lfr_frames == 0) {
    return mx::zeros({0, lfr_m * mel_count}, fbank.dtype());
  } else {
  }
  // Note (Dayuxiaoshui): a gather of the clamped frame indices moves the same
  // values as the reference's edge padding.
  const int left_pad = (lfr_m - 1) / 2;
  std::vector<int32_t> indices;
  indices.reserve(static_cast<size_t>(lfr_frames) * lfr_m);
  for (int frame = 0; frame < lfr_frames; ++frame) {
    for (int offset = 0; offset < lfr_m; ++offset) {
      indices.push_back(
          std::clamp(frame * lfr_n + offset - left_pad, 0, time - 1));
    }
  }
  return mx::reshape(
      mx::take(fbank,
               mx::array(indices.data(), {static_cast<int>(indices.size())},
                         mx::int32),
               0),
      {lfr_frames, lfr_m * mel_count});
}

} // namespace omni_common
