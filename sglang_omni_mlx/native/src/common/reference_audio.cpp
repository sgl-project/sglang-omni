// SPDX-License-Identifier: Apache-2.0
#include "common/reference_audio.h"

#include <algorithm>
#include <cassert>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <string>

namespace omni_common {

namespace {

constexpr uint16_t kWavFormatPcm = 1;
constexpr uint16_t kWavFormatFloat = 3;
constexpr uint16_t kWavFormatExtensible = 0xFFFE;
constexpr int kMaxChannelCount = 254;
// Note (Jiaxin Deng): a header outside these rates, or a rate pair whose
// resampling filter would pass kMaxResampleTaps in all or
// kMaxTapsPerOutputSample per output sample, is refused before any
// allocation, which bounds the work per output sample; every common rate up to
// 192 kHz is inside for the 16, 24 and 44.1 kHz the models read.
constexpr uint32_t kMinFileSampleRate = 1000;
constexpr uint32_t kMaxFileSampleRate = 384000;
constexpr long kMaxResampleTaps = 1 << 22;
constexpr long kMaxTapsPerOutputSample = 1024;
constexpr int kResampleZeroCrossings = 6;
constexpr double kResampleRolloff = 0.99;

uint16_t ReadU16(std::string_view bytes, size_t offset) {
  uint16_t value = 0;
  std::memcpy(&value, bytes.data() + offset, sizeof(value));
  return value;
}

uint32_t ReadU32(std::string_view bytes, size_t offset) {
  uint32_t value = 0;
  std::memcpy(&value, bytes.data() + offset, sizeof(value));
  return value;
}

// One stored sample as float32, scaled as the CUDA servers' WAV reader does.
float SampleValue(const unsigned char *sample, uint16_t format,
                  int bits_per_sample) {
  if (format == kWavFormatFloat && bits_per_sample == 32) {
    float value = 0.0f;
    std::memcpy(&value, sample, sizeof(value));
    return value;
  } else if (format == kWavFormatFloat) {
    double value = 0.0;
    std::memcpy(&value, sample, sizeof(value));
    return static_cast<float>(value);
  } else if (bits_per_sample == 8) {
    return (static_cast<float>(sample[0]) - 128.0f) / 128.0f;
  } else if (bits_per_sample == 16) {
    int16_t value = 0;
    std::memcpy(&value, sample, sizeof(value));
    return static_cast<float>(value) / 32768.0f;
  } else if (bits_per_sample == 24) {
    const int32_t value =
        static_cast<int32_t>((static_cast<uint32_t>(sample[0]) << 8) |
                             (static_cast<uint32_t>(sample[1]) << 16) |
                             (static_cast<uint32_t>(sample[2]) << 24)) >>
        8;
    return static_cast<float>(value) / 8388608.0f;
  } else {
    int32_t value = 0;
    std::memcpy(&value, sample, sizeof(value));
    return static_cast<float>(value) / 2147483648.0f;
  }
}

struct ResampleFilterShape {
  long from;
  long to;
  double base_frequency;
  long width;
  long tap_count;
};

ResampleFilterShape FilterShape(int from_rate, int to_rate) {
  const long divisor = std::gcd(from_rate, to_rate);
  const long from = from_rate / divisor;
  const long to = to_rate / divisor;
  const double base_frequency = std::min(from, to) * kResampleRolloff;
  const long width = static_cast<long>(
      std::ceil(kResampleZeroCrossings * from / base_frequency));
  const long tap_count = 2 * width + from;
  if (tap_count * to > kMaxResampleTaps ||
      tap_count > kMaxTapsPerOutputSample) {
    throw std::invalid_argument("reference WAV sample rate " +
                                std::to_string(from_rate) +
                                " Hz is not supported");
  } else {
  }
  return {from, to, base_frequency, width, tap_count};
}

} // namespace

std::string AudioDataUrlPayload(std::string_view url,
                                const std::string &field) {
  assert(url.starts_with("data:"));
  const size_t comma = url.find(',');
  std::string header(url.substr(0, comma));
  std::transform(header.begin(), header.end(), header.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  if (comma == std::string_view::npos ||
      header.find(";base64") == std::string::npos || comma + 1 == url.size()) {
    throw std::invalid_argument(field +
                                " data URL must include base64 media data");
  } else {
  }
  const std::string_view media_type =
      url.substr(5, url.substr(0, comma).find(';') - 5);
  if (!media_type.empty() && !media_type.starts_with("audio/")) {
    throw std::invalid_argument(field +
                                " data URL must use an audio media type");
  } else {
    return std::string(url.substr(comma + 1));
  }
}

DecodedAudio DecodeAnyWav(std::string_view file_bytes) {
  if (file_bytes.size() < 12 || file_bytes.substr(0, 4) != "RIFF" ||
      file_bytes.substr(8, 4) != "WAVE") {
    throw std::invalid_argument("reference audio must be a RIFF/WAVE file");
  } else {
  }
  size_t offset = 12;
  bool has_format = false;
  uint16_t format = 0;
  int channel_count = 0;
  uint32_t sample_rate = 0;
  int bits_per_sample = 0;
  std::string_view data;
  bool has_data = false;
  while (!has_data && offset + 8 <= file_bytes.size()) {
    const std::string_view chunk_id = file_bytes.substr(offset, 4);
    const uint32_t chunk_size = ReadU32(file_bytes, offset + 4);
    const size_t body_offset = offset + 8;
    const size_t body_size =
        std::min<size_t>(chunk_size, file_bytes.size() - body_offset);
    if (chunk_id == "fmt ") {
      if (body_size < 16) {
        throw std::invalid_argument("reference WAV fmt chunk is truncated");
      } else {
      }
      format = ReadU16(file_bytes, body_offset);
      channel_count = ReadU16(file_bytes, body_offset + 2);
      sample_rate = ReadU32(file_bytes, body_offset + 4);
      bits_per_sample = ReadU16(file_bytes, body_offset + 14);
      if (format == kWavFormatExtensible) {
        if (body_size < 40) {
          throw std::invalid_argument(
              "reference WAV extensible fmt chunk is truncated");
        } else {
        }
        // The subformat GUID starts with the plain format code.
        format = ReadU16(file_bytes, body_offset + 24);
      } else {
      }
      has_format = true;
    } else if (chunk_id == "data") {
      if (!has_format) {
        throw std::invalid_argument(
            "reference WAV data chunk precedes its fmt chunk");
      } else {
      }
      data = file_bytes.substr(body_offset, body_size);
      has_data = true;
    } else {
    }
    offset = body_offset + static_cast<size_t>(chunk_size) + (chunk_size & 1u);
  }
  if (!has_data) {
    throw std::invalid_argument("reference WAV has no data chunk");
  } else if (channel_count == 0 || channel_count > kMaxChannelCount) {
    throw std::invalid_argument(
        "reference WAV has an unsupported channel count");
  } else if (!(format == kWavFormatPcm &&
               (bits_per_sample == 8 || bits_per_sample == 16 ||
                bits_per_sample == 24 || bits_per_sample == 32)) &&
             !(format == kWavFormatFloat &&
               (bits_per_sample == 32 || bits_per_sample == 64))) {
    throw std::invalid_argument(
        "reference WAV must be 8/16/24/32-bit PCM or 32/64-bit float");
  } else if (sample_rate < kMinFileSampleRate ||
             sample_rate > kMaxFileSampleRate) {
    throw std::invalid_argument("reference WAV sample rate is out of range");
  } else {
  }
  const size_t bytes_per_sample = static_cast<size_t>(bits_per_sample) / 8;
  const size_t bytes_per_frame = bytes_per_sample * channel_count;
  const size_t frame_count = data.size() / bytes_per_frame;
  const auto *bytes = reinterpret_cast<const unsigned char *>(data.data());
  DecodedAudio audio;
  audio.sample_rate = static_cast<int>(sample_rate);
  audio.channel_count = channel_count;
  audio.samples.resize(frame_count);
  for (size_t frame = 0; frame < frame_count; ++frame) {
    const unsigned char *frame_bytes = bytes + frame * bytes_per_frame;
    float sum = 0.0f;
    for (int channel = 0; channel < channel_count; ++channel) {
      sum += SampleValue(frame_bytes + channel * bytes_per_sample, format,
                         bits_per_sample);
    }
    audio.samples[frame] = sum / static_cast<float>(channel_count);
  }
  for (const float sample : audio.samples) {
    if (!std::isfinite(sample)) {
      throw std::invalid_argument("reference audio has non-finite samples");
    } else {
    }
  }
  return audio;
}

void CheckResampleRates(int from_rate, int to_rate) {
  FilterShape(from_rate, to_rate);
}

std::vector<float> Resample(const std::vector<float> &samples, int from_rate,
                            int to_rate) {
  if (from_rate == to_rate) {
    return samples;
  } else {
  }
  const auto [from, to, base_frequency, width, tap_count] =
      FilterShape(from_rate, to_rate);
  // One filter per output phase, as torchaudio builds them.
  std::vector<float> kernels(static_cast<size_t>(to) * tap_count);
  for (long phase = 0; phase < to; ++phase) {
    for (long tap = 0; tap < tap_count; ++tap) {
      double t = (-static_cast<double>(phase) / to +
                  static_cast<double>(tap - width) / from) *
                 base_frequency;
      t = std::clamp(t, -static_cast<double>(kResampleZeroCrossings),
                     static_cast<double>(kResampleZeroCrossings));
      const double window =
          std::pow(std::cos(t * M_PI / kResampleZeroCrossings / 2), 2);
      t *= M_PI;
      const double sinc = t == 0.0 ? 1.0 : std::sin(t) / t;
      kernels[phase * tap_count + tap] =
          static_cast<float>(sinc * window * base_frequency / from);
    }
  }
  const long input_length = static_cast<long>(samples.size());
  const long output_length = static_cast<long>(
      std::ceil(static_cast<double>(to) * input_length / from));
  std::vector<float> output(output_length);
  for (long index = 0; index < output_length; ++index) {
    const long frame = index / to;
    const long phase = index % to;
    // The input is zero padded by width before and width + from after.
    const long start = frame * from - width;
    const float *kernel = kernels.data() + phase * tap_count;
    double sum = 0.0;
    for (long tap = std::max(0L, -start);
         tap < tap_count && start + tap < input_length; ++tap) {
      sum += static_cast<double>(samples[start + tap]) * kernel[tap];
    }
    output[index] = static_cast<float>(sum);
  }
  return output;
}

} // namespace omni_common
