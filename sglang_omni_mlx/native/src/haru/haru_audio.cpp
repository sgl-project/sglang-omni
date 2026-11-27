// SPDX-License-Identifier: Apache-2.0
#include "haru/haru_audio.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>

#include "common/reference_audio.h"

namespace haru {

namespace {

constexpr float kSilenceDbFs = -45.0f;
constexpr float kSilenceMarginSeconds = 0.2f;

// Cut leading and trailing audio quieter than -45 dBFS (20 ms RMS windows).
std::vector<float> TrimSilence(const std::vector<float> &audio,
                               int sample_rate) {
  const int window = sample_rate / 50;
  const int count = static_cast<int>(audio.size()) / window;
  if (count == 0) {
    return audio;
  } else {
  }
  int first = -1;
  int last = -1;
  for (int i = 0; i < count; ++i) {
    double sum = 0.0;
    for (int j = 0; j < window; ++j) {
      sum += audio[i * window + j] * audio[i * window + j];
    }
    const double rms = std::sqrt(sum / window + 1e-12);
    if (20.0 * std::log10(rms) > kSilenceDbFs) {
      if (first < 0) {
        first = i;
      } else {
      }
      last = i;
    } else {
    }
  }
  if (first < 0) {
    return {};
  } else {
  }
  const int margin = static_cast<int>(kSilenceMarginSeconds * sample_rate);
  const int start = std::max(0, first * window - margin);
  const int end =
      std::min(static_cast<int>(audio.size()), (last + 1) * window + margin);
  return std::vector<float>(audio.begin() + start, audio.begin() + end);
}

} // namespace

std::vector<float> PrepareReference(std::string_view wav_bytes, int sample_rate,
                                    int max_seconds) {
  const omni_common::DecodedAudio decoded =
      omni_common::DecodeAnyWav(wav_bytes);
  std::vector<float> samples = decoded.samples;
  if (decoded.sample_rate != sample_rate) {
    samples = omni_common::Resample(samples, decoded.sample_rate, sample_rate);
  } else {
  }
  samples = TrimSilence(samples, sample_rate);
  const int max_samples = max_seconds * sample_rate;
  if (static_cast<int>(samples.size()) > max_samples) {
    samples.erase(samples.begin(), samples.end() - max_samples);
  } else {
  }
  return samples;
}

} // namespace haru
