// SPDX-License-Identifier: Apache-2.0
#include "swift_port.h"

#include <cmath>
#include <cstdint>

namespace swift_port {

namespace {

namespace mx = mlx::core;

constexpr int kSampleRate = 16000;

std::vector<mx::array> SiluGraph(const std::vector<mx::array> &inputs) {
  return {mx::multiply(inputs[0], mx::sigmoid(inputs[0]))};
}

std::vector<mx::array> ReluGraph(const std::vector<mx::array> &inputs) {
  return {mx::maximum(inputs[0], mx::array(0.0f, inputs[0].dtype()))};
}

float HertzToMel(float frequency_hz, float linear_step_hz, float min_log_hz,
                 float min_log_mel, float log_step) {
  if (frequency_hz < min_log_hz) {
    return frequency_hz / linear_step_hz;
  } else {
    return min_log_mel + std::log(frequency_hz / min_log_hz) / log_step;
  }
}

float MelToHertz(float mel, float linear_step_hz, float min_log_hz,
                 float min_log_mel, float log_step) {
  if (mel < min_log_mel) {
    return linear_step_hz * mel;
  } else {
    return min_log_hz * std::exp(log_step * (mel - min_log_mel));
  }
}

bool IsWhitespace(uint32_t code_point, bool newlines) {
  const bool is_newline = (code_point >= 0x0A && code_point <= 0x0D) ||
                          code_point == 0x85 || code_point == 0x2028 ||
                          code_point == 0x2029;
  return code_point == 0x09 || code_point == 0x20 || code_point == 0xA0 ||
         code_point == 0x1680 ||
         (code_point >= 0x2000 && code_point <= 0x200A) ||
         code_point == 0x202F || code_point == 0x205F || code_point == 0x3000 ||
         (newlines && is_newline);
}

// The code point at offset and its length in bytes. A byte that does not
// start a complete UTF-8 sequence reads as one U+FFFD, so callers never
// step past the end of the text.
uint32_t CodePointAt(const std::string &text, size_t offset, size_t &length) {
  const auto byte = [&](size_t k) { return static_cast<uint8_t>(text[k]); };
  const uint8_t lead = byte(offset);
  uint32_t code_point = 0;
  if (lead < 0x80) {
    length = 1;
    return lead;
  } else if ((lead >> 5) == 0x6) {
    length = 2;
    code_point = lead & 0x1F;
  } else if ((lead >> 4) == 0xE) {
    length = 3;
    code_point = lead & 0x0F;
  } else if ((lead >> 3) == 0x1E) {
    length = 4;
    code_point = lead & 0x07;
  } else {
    length = 1;
    return 0xFFFD;
  }
  if (offset + length > text.size()) {
    length = 1;
    return 0xFFFD;
  } else {
  }
  for (size_t k = 1; k < length; ++k) {
    if ((byte(offset + k) >> 6) != 0x2) {
      length = 1;
      return 0xFFFD;
    } else {
    }
    code_point = (code_point << 6) | (byte(offset + k) & 0x3F);
  }
  return code_point;
}

} // namespace

std::vector<float> SlaneyMelFilterBank(int fft_size, int mel_bin_count) {
  const int frequency_bin_count = fft_size / 2 + 1;
  const float linear_step_hz = 200.0f / 3.0f;
  const float min_log_hz = 1000.0f;
  const float min_log_mel = min_log_hz / linear_step_hz;
  const float log_step = std::log(6.4f) / 27.0f;

  std::vector<float> bin_frequencies_hz(frequency_bin_count);
  for (int i = 0; i < frequency_bin_count; ++i) {
    bin_frequencies_hz[i] = static_cast<float>(i) *
                            static_cast<float>(kSampleRate) /
                            static_cast<float>(fft_size);
  }
  const float mel_max =
      HertzToMel(static_cast<float>(kSampleRate) / 2.0f, linear_step_hz,
                 min_log_hz, min_log_mel, log_step);
  std::vector<float> edges_hz(mel_bin_count + 2);
  for (int i = 0; i < mel_bin_count + 2; ++i) {
    edges_hz[i] = MelToHertz(static_cast<float>(i) * mel_max /
                                 static_cast<float>(mel_bin_count + 1),
                             linear_step_hz, min_log_hz, min_log_mel, log_step);
  }
  std::vector<float> filters(
      static_cast<size_t>(frequency_bin_count) * mel_bin_count, 0.0f);
  for (int mel_bin = 0; mel_bin < mel_bin_count; ++mel_bin) {
    const float low = edges_hz[mel_bin];
    const float center = edges_hz[mel_bin + 1];
    const float high = edges_hz[mel_bin + 2];
    const float normalization = 2.0f / (high - low);
    for (int frequency_bin = 0; frequency_bin < frequency_bin_count;
         ++frequency_bin) {
      const float frequency_hz = bin_frequencies_hz[frequency_bin];
      float weight = 0.0f;
      if (low <= frequency_hz && frequency_hz < center) {
        weight = (frequency_hz - low) / (center - low);
      } else if (center <= frequency_hz && frequency_hz <= high) {
        weight = (high - frequency_hz) / (high - center);
      } else {
        weight = 0.0f;
      }
      filters[static_cast<size_t>(frequency_bin) * mel_bin_count + mel_bin] =
          weight * normalization;
    }
  }
  return filters;
}

mx::array Silu(const mx::array &x) {
  static const auto compiled = mx::compile(SiluGraph, true);
  return compiled({x})[0];
}

mx::array Relu(const mx::array &x) {
  static const auto compiled = mx::compile(ReluGraph, true);
  return compiled({x})[0];
}

std::string TrimWhitespace(const std::string &text, bool newlines) {
  size_t begin = 0;
  while (begin < text.size()) {
    size_t length = 0;
    if (!IsWhitespace(CodePointAt(text, begin, length), newlines)) {
      break;
    } else {
    }
    begin += length;
  }
  // Note (khazic): scan forward, remembering where the last kept code point
  // ended.
  size_t kept_end = begin;
  for (size_t offset = begin; offset < text.size();) {
    size_t length = 0;
    if (!IsWhitespace(CodePointAt(text, offset, length), newlines)) {
      kept_end = offset + length;
    } else {
    }
    offset += length;
  }
  return text.substr(begin, kept_end - begin);
}

} // namespace swift_port
