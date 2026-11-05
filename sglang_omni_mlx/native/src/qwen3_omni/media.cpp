// SPDX-License-Identifier: Apache-2.0
#include "qwen3_omni/media.h"
#include "common/audio.h"

#include <cmath>

namespace qwen3_omni {
mx::array patchify(const std::vector<float> &pixels, int frames, int height,
                   int width) {
  if (frames < 2 || frames % 2 || height % 32 || width % 32 ||
      pixels.size() != size_t(frames) * height * width * 3) {
    throw std::invalid_argument("invalid image patch geometry");
  } else {
  }
  auto tensor = mx::array(
      pixels.data(), {frames / 2, 2, height / 32, 2, 16, width / 32, 2, 16, 3});
  return mx::reshape(mx::transpose(tensor, {0, 2, 5, 3, 6, 8, 1, 4, 7}),
                     {-1, 1536});
}
mx::array audio_mel(const std::vector<float> &samples) {
  if (samples.size() <= 200 || samples.size() > 120 * 16000 ||
      !std::all_of(samples.begin(), samples.end(),
                   [](float sample) { return std::isfinite(sample); })) {
    throw std::invalid_argument(
        "unsupported audio length or non-finite samples");
  } else {
  }
  int frames = int(samples.size() / 160);
  std::vector<float> centered;
  centered.reserve(samples.size() + 400);
  for (int i = 200; i > 0; --i) {
    centered.push_back(samples[i]);
  }
  centered.insert(centered.end(), samples.begin(), samples.end());
  for (int i = 0; i < 200; ++i) {
    centered.push_back(samples[samples.size() - 2 - i]);
  }
  auto indices = mx::reshape(mx::arange(frames, mx::int32), {-1, 1}) * 160 +
                 mx::reshape(mx::arange(400, mx::int32), {1, -1});
  auto window = omni_common::SwiftPeriodicHannWindow();
  auto transformed = mx::fft::rfft(
      mx::take(mx::array(centered.data(), {int(centered.size())}), indices) *
          window,
      400, -1);
  auto magnitudes = mx::abs(transformed);
  auto hertz_to_mel = [](double frequency) {
    return frequency < 1000
               ? frequency / (200.0 / 3)
               : 15 + std::log(frequency / 1000) / (std::log(6.4) / 27);
  };
  auto mel_to_hertz = [](double mel) {
    return mel < 15 ? mel * (200.0 / 3)
                    : 1000 * std::exp((mel - 15) * (std::log(6.4) / 27));
  };
  std::vector<double> edges(130);
  for (int i = 0; i < 130; ++i) {
    edges[i] = mel_to_hertz(i * hertz_to_mel(8000) / 129);
  }
  std::vector<float> filters;
  for (int bin = 0; bin < 201; ++bin) {
    for (int band = 0; band < 128; ++band) {
      double lower = (bin * 40 - edges[band]) / (edges[band + 1] - edges[band]);
      double upper =
          (edges[band + 2] - bin * 40) / (edges[band + 2] - edges[band + 1]);
      filters.push_back(float(std::max(0.0, std::min(lower, upper)) * 2 /
                              (edges[band + 2] - edges[band])));
    }
  }
  auto log_mel =
      mx::log10(mx::maximum(mx::matmul(magnitudes * magnitudes,
                                       mx::array(filters.data(), {201, 128})),
                            mx::array(1e-10f)));
  return mx::transpose((mx::maximum(log_mel, mx::max(log_mel) - 8) + 4) / 4);
}
std::vector<uint8_t> encode_wav(const std::vector<float> &samples) {
  std::vector<uint8_t> result;
  auto append_text = [&](const char *text) {
    result.insert(result.end(), text, text + 4);
  };
  auto append_integer = [&](uint32_t value) {
    for (int i = 0; i < 4; ++i) {
      result.push_back(uint8_t(value >> (8 * i)));
    }
  };
  append_text("RIFF");
  append_integer(uint32_t(36 + samples.size() * 2));
  append_text("WAVE");
  append_text("fmt ");
  append_integer(16);
  result.insert(result.end(), {1, 0, 1, 0});
  append_integer(24000);
  append_integer(48000);
  result.insert(result.end(), {2, 0, 16, 0});
  append_text("data");
  append_integer(uint32_t(samples.size() * 2));
  for (float sample : samples) {
    if (!std::isfinite(sample)) {
      throw std::invalid_argument("cannot encode non-finite audio");
    } else {
      int16_t pcm = int16_t(std::clamp(sample, -1.f, 1.f) * 32767);
      result.push_back(uint8_t(pcm));
      result.push_back(uint8_t(uint16_t(pcm) >> 8));
    }
  }
  return result;
}
} // namespace qwen3_omni
