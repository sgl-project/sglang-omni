// SPDX-License-Identifier: Apache-2.0
#include "chatterbox/chatterbox_vocoder.h"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <optional>

namespace chatterbox {

namespace mx = mlx::core;

namespace {

constexpr const char *kPrefix = "s3gen.mel2wav";
constexpr int kSampleRate = 24000;
constexpr int kHarmonicCount = 8;
constexpr float kSineAmplitude = 0.1f;
constexpr float kVoicedNoiseStd = 0.003f;
constexpr float kVoicedThresholdHz = 10.0f;
constexpr int kF0PredictorLayers = 5;
constexpr int kUpsampleRates[] = {8, 5, 3};
constexpr int kUpsampleKernelSizes[] = {16, 11, 7};
constexpr int kResidualKernelSizes[] = {3, 7, 11};
constexpr int kResidualDilations[] = {1, 3, 5};
constexpr int kSourceResidualKernelSizes[] = {7, 7, 11};
// Source STFT frames per stage's step: the stages still ahead's upsampling.
constexpr int kSourceDownsampleRates[] = {15, 3, 1};
constexpr int kFftSize = 16;
constexpr int kHopSamples = 4;
constexpr int kFrequencyBins = kFftSize / 2 + 1;
constexpr int kSamplesPerMelFrame = 8 * 5 * 3 * kHopSamples;
constexpr float kMagnitudeLimit = 100.0f;
constexpr float kAudioLimit = 0.99f;
constexpr float kWindowSumFloor = 1e-8f;
constexpr float kSnakeAlphaFloor = 1e-4f;
constexpr float kSnakeAlphaZero = 1e-9f;

double CosineOfSixteenth(int multiple) {
  // Note (Jiaxin Deng): exact zeros and mirrored values at multiples of
  // 2 pi / 16; the double sum then rounds to the same float32 a float64 FFT
  // does.
  static const double first_quadrant[] = {
      1.0, std::cos(M_PI / 8), std::sqrt(0.5), std::sin(M_PI / 8), 0.0};
  const int m = ((multiple % kFftSize) + kFftSize) % kFftSize;
  if (m <= 4) {
    return first_quadrant[m];
  } else if (m <= 8) {
    return -first_quadrant[8 - m];
  } else if (m <= 12) {
    return -first_quadrant[m - 8];
  } else {
    return first_quadrant[16 - m];
  }
}

} // namespace

Vocoder::Vocoder(const omni_common::Checkpoint &checkpoint)
    : checkpoint_(checkpoint), window_(0.0f) {
  for (int n = 0; n < kFftSize; ++n) {
    window_values_.push_back(
        static_cast<float>(0.5 * (1 - std::cos(2 * M_PI * n / kFftSize))));
  }
  window_ = mx::array(window_values_.begin(), {kFftSize}, mx::float32);
}

mx::array Vocoder::Convolution(const mx::array &x, const std::string &prefix,
                               int stride, int padding, int dilation) const {
  return mx::add(mx::conv1d(x, checkpoint_.Weight(prefix + ".weight"), stride,
                            padding, dilation),
                 checkpoint_.Weight(prefix + ".bias"));
}

mx::array Vocoder::Snake(const mx::array &x, const std::string &prefix) const {
  const mx::array &alpha = checkpoint_.Weight(prefix + ".alpha");
  const mx::array alpha_magnitude = mx::abs(alpha);
  const mx::array floor = mx::array(kSnakeAlphaFloor);
  const mx::array clamped_alpha = mx::where(
      mx::less(alpha_magnitude, mx::array(kSnakeAlphaZero)), floor,
      mx::multiply(mx::sign(alpha), mx::maximum(alpha_magnitude, floor)));
  return mx::add(x, mx::multiply(mx::divide(mx::array(1.0f), clamped_alpha),
                                 mx::power(mx::sin(mx::multiply(x, alpha)),
                                           mx::array(2.0f))));
}

mx::array Vocoder::ResidualBlock(const mx::array &x, const std::string &prefix,
                                 int kernel_size) const {
  mx::array y = x;
  for (int i = 0; i < 3; ++i) {
    const std::string index = std::to_string(i);
    const int dilation = kResidualDilations[i];
    mx::array branch = Snake(y, prefix + ".activations1." + index);
    branch = Convolution(branch, prefix + ".convs1." + index + ".conv", 1,
                         (kernel_size * dilation - dilation) / 2, dilation);
    branch = Snake(branch, prefix + ".activations2." + index);
    branch = Convolution(branch, prefix + ".convs2." + index + ".conv", 1,
                         (kernel_size - 1) / 2, 1);
    y = mx::add(branch, y);
  }
  return y;
}

mx::array Vocoder::Source(const mx::array &f0_hz) const {
  const int samples = f0_hz.shape(1);
  // The sine generator works on [1, harmonics, samples].
  const mx::array f0_row = mx::transpose(f0_hz, {0, 2, 1});
  const mx::array harmonics = mx::reshape(
      mx::arange(1, kHarmonicCount + 2, mx::int32), {1, kHarmonicCount + 1, 1});
  const mx::array cycles_per_sample =
      mx::divide(mx::multiply(f0_row, harmonics),
                 mx::array(static_cast<float>(kSampleRate)));
  const mx::array two_pi = mx::array(static_cast<float>(2 * M_PI));
  const mx::array phase = mx::remainder(
      mx::multiply(two_pi, mx::cumsum(cycles_per_sample, -1)), two_pi);
  const mx::array harmonic_phases = mx::random::uniform(
      mx::array(static_cast<float>(-M_PI)), mx::array(static_cast<float>(M_PI)),
      {1, kHarmonicCount, 1});
  const mx::array phase_offsets =
      mx::concatenate({mx::zeros({1, 1, 1}, mx::float32), harmonic_phases}, 1);
  mx::array sines = mx::multiply(mx::array(kSineAmplitude),
                                 mx::sin(mx::add(phase, phase_offsets)));
  const mx::array voiced = mx::astype(
      mx::greater(f0_row, mx::array(kVoicedThresholdHz)), mx::float32);
  const mx::array noise_amplitude =
      mx::add(mx::multiply(voiced, mx::array(kVoicedNoiseStd)),
              mx::divide(mx::multiply(mx::subtract(mx::array(1.0f), voiced),
                                      mx::array(kSineAmplitude)),
                         mx::array(3.0f)));
  const mx::array noise =
      mx::multiply(noise_amplitude, mx::random::normal(sines.shape()));
  sines = mx::add(mx::multiply(sines, voiced), noise);
  const mx::array merged =
      mx::tanh(checkpoint_.Linear(mx::transpose(sines, {0, 2, 1}),
                                  std::string(kPrefix) + ".m_source.l_linear"));
  // Note (Jiaxin Deng): the source's separate noise branch is unused, but its
  // draw still advances the global key.
  mx::random::normal({1, samples, 1});
  return merged;
}

mx::array Vocoder::SourceSpectrum(const mx::array &source) const {
  const mx::array samples_array =
      mx::astype(mx::reshape(source, {-1}), mx::float32);
  mx::eval(samples_array);
  const float *samples = samples_array.data<float>();
  const int sample_count = samples_array.size();
  const int frame_count = (sample_count - kFftSize) / kHopSamples + 1;
  double cosines[kFrequencyBins][kFftSize];
  double negative_sines[kFrequencyBins][kFftSize];
  for (int bin = 0; bin < kFrequencyBins; ++bin) {
    for (int n = 0; n < kFftSize; ++n) {
      cosines[bin][n] = CosineOfSixteenth(bin * n);
      negative_sines[bin][n] = -CosineOfSixteenth(bin * n - 4);
    }
  }
  std::vector<float> spectrum(static_cast<size_t>(frame_count) * 2 *
                              kFrequencyBins);
  float frame[kFftSize];
  for (int f = 0; f < frame_count; ++f) {
    for (int n = 0; n < kFftSize; ++n) {
      frame[n] = samples[f * kHopSamples + n] * window_values_[n];
    }
    float *row = spectrum.data() + static_cast<size_t>(f) * 2 * kFrequencyBins;
    for (int bin = 0; bin < kFrequencyBins; ++bin) {
      double real = 0.0;
      double imaginary = 0.0;
      for (int n = 0; n < kFftSize; ++n) {
        real += static_cast<double>(frame[n]) * cosines[bin][n];
        imaginary += static_cast<double>(frame[n]) * negative_sines[bin][n];
      }
      row[bin] = static_cast<float>(real);
      row[kFrequencyBins + bin] = static_cast<float>(imaginary);
    }
  }
  return mx::array(spectrum.begin(), {1, frame_count, 2 * kFrequencyBins},
                   mx::float32);
}

mx::array Vocoder::OverlapAdd(const mx::array &frames) const {
  const int frame_count = frames.shape(1);
  const int hops_per_frame = kFftSize / kHopSamples;
  const int block_count = frame_count + hops_per_frame - 1;
  const mx::array windowed = mx::multiply(frames, window_);
  // Each hop-sized block sums its frames' pieces from the oldest frame on, as
  // a running float32 sum over the frames does.
  std::optional<mx::array> blocks;
  for (int piece = hops_per_frame - 1; piece >= 0; --piece) {
    const mx::array part =
        mx::pad(mx::slice(windowed, {0, 0, piece * kHopSamples},
                          {1, frame_count, (piece + 1) * kHopSamples}),
                std::vector<int>{1}, mx::Shape{piece},
                mx::Shape{hops_per_frame - 1 - piece});
    if (blocks.has_value()) {
      blocks = mx::add(*blocks, part);
    } else {
      blocks = part;
    }
  }
  const int length = block_count * kHopSamples;
  std::vector<float> window_sum(length, 0.0f);
  for (int f = 0; f < frame_count; ++f) {
    for (int n = 0; n < kFftSize; ++n) {
      window_sum[f * kHopSamples + n] += window_values_[n] * window_values_[n];
    }
  }
  for (float &value : window_sum) {
    value = std::max(value, kWindowSumFloor);
  }
  // Note (Jiaxin Deng): divided on the CPU, whose float32 division rounds
  // correctly.
  const mx::array audio = mx::divide(
      mx::reshape(*blocks, {1, length}),
      mx::array(window_sum.begin(), {1, length}, mx::float32), mx::Device::cpu);
  const int start = kFftSize / 2;
  return mx::slice(audio, {0, start},
                   {1, start + (frame_count - 1) * kHopSamples});
}

mx::array Vocoder::Waveform(const mx::array &mel) const {
  const std::string prefix = kPrefix;
  const mx::array mel_frames = mx::transpose(mel, {0, 2, 1});

  mx::array hidden = mel_frames;
  for (int layer = 0; layer < kF0PredictorLayers; ++layer) {
    hidden = Convolution(hidden,
                         prefix + ".f0_predictor.condnet." +
                             std::to_string(layer) + ".conv",
                         1, 1, 1);
    hidden =
        mx::where(mx::greater(hidden, mx::array(0.0f)), hidden,
                  mx::multiply(mx::array(1.0f),
                               mx::subtract(mx::exp(hidden), mx::array(1.0f))));
  }
  const mx::array f0_hz =
      mx::abs(checkpoint_.Linear(hidden, prefix + ".f0_predictor.classifier"));
  const mx::array source = Source(mx::repeat(f0_hz, kSamplesPerMelFrame, 1));
  const mx::array source_spectrum = SourceSpectrum(source);

  mx::array x = Convolution(mel_frames, prefix + ".conv_pre.conv", 1, 3, 1);
  const int stage_count = std::size(kUpsampleRates);
  for (int stage = 0; stage < stage_count; ++stage) {
    const std::string index = std::to_string(stage);
    const int rate = kUpsampleRates[stage];
    x = mx::maximum(mx::multiply(mx::array(0.1f), x), x);
    x = mx::add(
        mx::conv_transpose1d(
            x, checkpoint_.Weight(prefix + ".ups." + index + ".conv.weight"),
            rate, (kUpsampleKernelSizes[stage] - rate) / 2),
        checkpoint_.Weight(prefix + ".ups." + index + ".conv.bias"));
    if (stage == stage_count - 1) {
      x = mx::pad(x, std::vector<int>{1}, mx::Shape{1}, mx::Shape{0});
    } else {
    }
    const int downsample = kSourceDownsampleRates[stage];
    mx::array source_branch =
        downsample == 1
            ? Convolution(source_spectrum,
                          prefix + ".source_downs." + index + ".conv", 1, 0, 1)
            : Convolution(source_spectrum,
                          prefix + ".source_downs." + index + ".conv",
                          downsample, downsample / 2, 1);
    source_branch =
        ResidualBlock(source_branch, prefix + ".source_resblocks." + index,
                      kSourceResidualKernelSizes[stage]);
    const int steps = std::min(x.shape(1), source_branch.shape(1));
    x = mx::add(mx::slice(x, {0, 0, 0}, {1, steps, x.shape(2)}),
                mx::slice(source_branch, {0, 0, 0},
                          {1, steps, source_branch.shape(2)}));
    const int kernel_count = std::size(kResidualKernelSizes);
    std::optional<mx::array> kernels_sum;
    for (int j = 0; j < kernel_count; ++j) {
      const mx::array y = ResidualBlock(
          x, prefix + ".resblocks." + std::to_string(stage * kernel_count + j),
          kResidualKernelSizes[j]);
      if (kernels_sum.has_value()) {
        kernels_sum = mx::add(*kernels_sum, y);
      } else {
        kernels_sum = y;
      }
    }
    x = mx::divide(*kernels_sum, mx::array(static_cast<float>(kernel_count)));
  }
  x = mx::maximum(mx::multiply(mx::array(0.01f), x), x);
  x = Convolution(x, prefix + ".conv_post.conv", 1, 3, 1);

  const int frame_count = x.shape(1);
  const mx::array magnitude = mx::clip(
      mx::exp(mx::slice(x, {0, 0, 0}, {1, frame_count, kFrequencyBins})),
      std::nullopt, mx::array(kMagnitudeLimit));
  const mx::array phase = mx::sin(mx::slice(
      x, {0, 0, kFrequencyBins}, {1, frame_count, 2 * kFrequencyBins}));
  const mx::array spectrum =
      mx::add(mx::multiply(magnitude, mx::cos(phase)),
              mx::multiply(mx::array(mx::complex64_t(0.0f, 1.0f)),
                           mx::multiply(magnitude, mx::sin(phase))));
  const mx::array frames = mx::fft::irfft(spectrum, kFftSize, -1);
  return mx::clip(OverlapAdd(frames), mx::array(-kAudioLimit),
                  mx::array(kAudioLimit));
}

} // namespace chatterbox
