// SPDX-License-Identifier: Apache-2.0
#include "haru/haru_audio.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <stdexcept>

#include "common/audio.h"
#include "common/audio_resample.h"

namespace haru {

namespace {

constexpr int kSampleRate = 24000;
constexpr float kSilenceDbFs = -45.0f;
constexpr float kSilenceMarginSeconds = 0.2f;

uint32_t ReadU32(std::string_view bytes, size_t offset) {
  return static_cast<uint8_t>(bytes[offset]) |
         (static_cast<uint8_t>(bytes[offset + 1]) << 8) |
         (static_cast<uint8_t>(bytes[offset + 2]) << 16) |
         (static_cast<uint8_t>(bytes[offset + 3]) << 24);
}

// Cut leading and trailing audio quieter than -45 dBFS (20 ms RMS windows).
std::vector<float> TrimSilence(const std::vector<float> &audio) {
  const int window = kSampleRate / 50;
  const int count = static_cast<int>(audio.size()) / window;
  if (count == 0) {
    return audio;
  }
  int first = -1, last = -1;
  for (int i = 0; i < count; ++i) {
    double sum = 0.0;
    for (int j = 0; j < window; ++j) {
      sum += audio[i * window + j] * audio[i * window + j];
    }
    const double rms = std::sqrt(sum / window + 1e-12);
    if (20.0 * std::log10(rms) > kSilenceDbFs) {
      if (first < 0) {
        first = i;
      }
      last = i;
    }
  }
  if (first < 0) {
    return {};
  }
  const int margin = static_cast<int>(kSilenceMarginSeconds * kSampleRate);
  const int start = std::max(0, first * window - margin);
  const int end =
      std::min(static_cast<int>(audio.size()), (last + 1) * window + margin);
  return std::vector<float>(audio.begin() + start, audio.begin() + end);
}

} // namespace

std::vector<float> PrepareReferenceBytes(std::string_view wav_bytes,
                                         int max_seconds) {
  if (wav_bytes.size() < 28) {
    throw std::invalid_argument("WAV file is too short");
  }
  const uint32_t source_rate = ReadU32(wav_bytes, 24);
  std::vector<float> samples = omni_common::DecodeWav(wav_bytes, source_rate);
  if (source_rate != kSampleRate) {
    samples = omni_common::ResampleAudio(samples, source_rate, kSampleRate);
  }
  samples = TrimSilence(samples);
  const int max_samples = max_seconds * kSampleRate;
  if (static_cast<int>(samples.size()) > max_samples) {
    samples.erase(samples.begin(), samples.end() - max_samples);
  }
  return samples;
}

std::vector<float> PrepareReference(const std::string &wav_path,
                                    int max_seconds) {
  std::ifstream file(wav_path, std::ios::binary);
  if (!file) {
    throw std::invalid_argument("cannot open " + wav_path);
  }
  const std::string bytes((std::istreambuf_iterator<char>(file)),
                          std::istreambuf_iterator<char>());
  return PrepareReferenceBytes(bytes, max_seconds);
}

} // namespace haru
