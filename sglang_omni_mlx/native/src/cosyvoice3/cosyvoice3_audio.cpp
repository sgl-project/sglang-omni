// SPDX-License-Identifier: Apache-2.0
#include "cosyvoice3/cosyvoice3_audio.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace cosyvoice3 {

namespace mx = mlx::core;

namespace {

constexpr int kMelSampleRate = 24000;
constexpr int kMelFftSize = 1920;
constexpr int kMelHopSamples = 480;
constexpr int kMelBinCount = 80;
constexpr float kMagnitudeEpsilon = 1e-9f;
constexpr float kMelFloor = 1e-5f;

// Slaney mel scale, as librosa computes it.
double HzToMel(double hz) {
  constexpr double kLinearHzPerMel = 200.0 / 3;
  constexpr double kLogRegionHz = 1000.0;
  const double log_step = std::log(6.4) / 27.0;
  return hz >= kLogRegionHz ? kLogRegionHz / kLinearHzPerMel +
                                  std::log(hz / kLogRegionHz) / log_step
                            : hz / kLinearHzPerMel;
}

double MelToHz(double mel) {
  constexpr double kLinearHzPerMel = 200.0 / 3;
  constexpr double kLogRegionHz = 1000.0;
  const double log_region_mel = kLogRegionHz / kLinearHzPerMel;
  const double log_step = std::log(6.4) / 27.0;
  return mel >= log_region_mel
             ? kLogRegionHz * std::exp(log_step * (mel - log_region_mel))
             : kLinearHzPerMel * mel;
}

// librosa.filters.mel(sr=24000, n_fft=1920, n_mels=80), Slaney-normalized,
// [frequency_bins, mel_bins]: built in double and stored in float32 as
// librosa stores it.
mx::array PromptMelFilters() {
  constexpr int bin_count = kMelFftSize / 2 + 1;
  constexpr int edge_count = kMelBinCount + 2;
  const double max_mel = HzToMel(kMelSampleRate / 2.0);
  std::vector<double> edges_hz(edge_count);
  for (int i = 0; i < edge_count; ++i) {
    edges_hz[i] = MelToHz(max_mel * i / (edge_count - 1));
  }
  std::vector<float> filters(static_cast<size_t>(bin_count) * kMelBinCount);
  for (int mel_bin = 0; mel_bin < kMelBinCount; ++mel_bin) {
    const double normalization =
        2.0 / (edges_hz[mel_bin + 2] - edges_hz[mel_bin]);
    for (int bin = 0; bin < bin_count; ++bin) {
      const double hz = static_cast<double>(bin) * kMelSampleRate / kMelFftSize;
      const double lower = (hz - edges_hz[mel_bin]) /
                           (edges_hz[mel_bin + 1] - edges_hz[mel_bin]);
      const double upper = (edges_hz[mel_bin + 2] - hz) /
                           (edges_hz[mel_bin + 2] - edges_hz[mel_bin + 1]);
      const float weight =
          static_cast<float>(std::max(0.0, std::min(lower, upper)));
      filters[bin * kMelBinCount + mel_bin] =
          static_cast<float>(static_cast<double>(weight) * normalization);
    }
  }
  return mx::array(filters.data(), {bin_count, kMelBinCount}, mx::float32);
}

} // namespace

mx::array PromptMel(const std::vector<float> &samples_24k) {
  static const mx::array filters = [] {
    mx::array built = PromptMelFilters();
    mx::eval(built);
    return built;
  }();
  constexpr int padding = (kMelFftSize - kMelHopSamples) / 2;
  const int sample_count = static_cast<int>(samples_24k.size());
  if (sample_count <= padding) {
    throw std::invalid_argument("reference audio is too short for a mel frame");
  } else {
  }
  std::vector<float> padded(sample_count + 2 * padding);
  for (int i = 0; i < padding; ++i) {
    padded[i] = samples_24k[padding - i];
    padded[padding + sample_count + i] = samples_24k[sample_count - 2 - i];
  }
  std::copy(samples_24k.begin(), samples_24k.end(), padded.begin() + padding);
  std::vector<float> window(kMelFftSize);
  for (int i = 0; i < kMelFftSize; ++i) {
    window[i] =
        static_cast<float>(0.5 * (1 - std::cos(2 * M_PI * i / kMelFftSize)));
  }
  const int frame_count =
      1 + (static_cast<int>(padded.size()) - kMelFftSize) / kMelHopSamples;
  const mx::array signal(padded.begin(), {static_cast<int>(padded.size())});
  const mx::array frames = mx::as_strided(signal, {frame_count, kMelFftSize},
                                          {kMelHopSamples, 1}, 0);
  const mx::array spectrum = mx::fft::rfft(
      mx::multiply(frames, mx::array(window.begin(), {kMelFftSize})), -1);
  const mx::array magnitude = mx::sqrt(mx::add(
      mx::add(mx::square(mx::real(spectrum)), mx::square(mx::imag(spectrum))),
      mx::array(kMagnitudeEpsilon)));
  const mx::array mel = mx::log(
      mx::maximum(mx::matmul(magnitude, filters), mx::array(kMelFloor)));
  return mx::expand_dims(mel, 0);
}

} // namespace cosyvoice3
