// SPDX-License-Identifier: Apache-2.0
#include "cosyvoice3/cosyvoice3_audio.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <string>

namespace cosyvoice3 {

namespace mx = mlx::core;

namespace {

constexpr uint16_t kWavFormatPcm = 1;
constexpr uint16_t kWavFormatFloat = 3;
constexpr uint16_t kWavFormatExtensible = 0xFFFE;
constexpr int kMaxChannelCount = 254;
// Note (Jiaxin Deng): a header outside these rates, or a rate pair whose
// resampling filter would pass kMaxResampleTaps in all or
// kMaxTapsPerOutputSample per output sample, is refused before any
// allocation, which bounds the work 30 s of audio takes to about 1e9
// multiply-adds; every common rate pair, 88.2 kHz the costliest, is inside.
constexpr uint32_t kMinFileSampleRate = 1000;
constexpr uint32_t kMaxFileSampleRate = 384000;
constexpr long kMaxResampleTaps = 1 << 22;
constexpr long kMaxTapsPerOutputSample = 1024;
constexpr int kResampleZeroCrossings = 6;
constexpr double kResampleRolloff = 0.99;

constexpr int kMelSampleRate = 24000;
constexpr int kMelFftSize = 1920;
constexpr int kMelHopSamples = 480;
constexpr int kMelBinCount = 80;
constexpr float kMagnitudeEpsilon = 1e-9f;
constexpr float kMelFloor = 1e-5f;

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

// One stored sample as float32, scaled as the CUDA server's WAV reader does.
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

DecodedAudio DecodeWav(std::string_view file_bytes) {
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
  const double seconds = static_cast<double>(frame_count) / sample_rate;
  if (seconds < kMinReferenceSeconds || seconds > kMaxReferenceSeconds) {
    throw std::invalid_argument(
        "reference audio must be between 0.5 and 30 seconds long");
  } else {
  }
  const auto *bytes = reinterpret_cast<const unsigned char *>(data.data());
  DecodedAudio audio;
  audio.sample_rate = static_cast<int>(sample_rate);
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

std::vector<float> Resample(const std::vector<float> &samples, int from_rate,
                            int to_rate) {
  if (from_rate == to_rate) {
    return samples;
  } else {
  }
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
