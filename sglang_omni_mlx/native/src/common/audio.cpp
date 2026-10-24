// SPDX-License-Identifier: Apache-2.0
#include "common/audio.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <mutex>
#include <stdexcept>

namespace qwen3_asr {

namespace mx = mlx::core;

namespace {

constexpr float kLogMelFloor = 1e-10f;
constexpr float kLogMelDynamicRange = 8.0f;
constexpr float kPcm16FullScale = 32768.0f;
constexpr uint16_t kWavFormatPcm = 1;
constexpr uint16_t kWavFormatFloat = 3;

uint32_t ReadU32(std::string_view bytes, size_t offset) {
  uint32_t value = 0;
  std::memcpy(&value, bytes.data() + offset, sizeof(value));
  return value;
}

uint16_t ReadU16(std::string_view bytes, size_t offset) {
  uint16_t value = 0;
  std::memcpy(&value, bytes.data() + offset, sizeof(value));
  return value;
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

std::vector<float> BuildMelFilterBank(int mel_bin_count) {
  constexpr int frequency_bin_count = kFftSize / 2 + 1;
  const float linear_step_hz = 200.0f / 3.0f;
  const float min_log_hz = 1000.0f;
  const float min_log_mel = min_log_hz / linear_step_hz;
  const float log_step = std::log(6.4f) / 27.0f;

  std::vector<float> bin_frequencies_hz(frequency_bin_count);
  for (int i = 0; i < frequency_bin_count; ++i) {
    bin_frequencies_hz[i] = static_cast<float>(i) *
                            static_cast<float>(kSampleRate) /
                            static_cast<float>(kFftSize);
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

std::vector<float> BuildPeriodicHannWindow() {
  std::vector<float> window(kFftSize);
  const float denominator = static_cast<float>(kFftSize);
  // Note (Jiaxin Deng): Swift's Float.pi, one float step below (float)M_PI.
  const float pi = std::nextafter(static_cast<float>(M_PI), 0.0f);
  for (int n = 0; n < kFftSize; ++n) {
    window[n] =
        0.5f *
        (1.0f - std::cos(2.0f * pi * static_cast<float>(n) / denominator));
  }
  return window;
}

} // namespace

std::vector<float> DecodeWav(std::string_view wav_bytes) {
  if (wav_bytes.size() < 12 || wav_bytes.substr(0, 4) != "RIFF" ||
      wav_bytes.substr(8, 4) != "WAVE") {
    throw std::invalid_argument("audio must be a RIFF/WAVE file");
  }
  size_t offset = 12;
  bool have_format = false;
  uint16_t audio_format = 0;
  uint16_t channel_count = 0;
  uint32_t sample_rate = 0;
  uint16_t bits_per_sample = 0;
  while (offset + 8 <= wav_bytes.size()) {
    const std::string_view chunk_id = wav_bytes.substr(offset, 4);
    const uint32_t chunk_size = ReadU32(wav_bytes, offset + 4);
    const size_t body_offset = offset + 8;
    const size_t body_size =
        std::min<size_t>(chunk_size, wav_bytes.size() - body_offset);
    if (chunk_id == "fmt ") {
      if (body_size < 16) {
        throw std::invalid_argument("WAV fmt chunk is truncated");
      }
      audio_format = ReadU16(wav_bytes, body_offset);
      channel_count = ReadU16(wav_bytes, body_offset + 2);
      sample_rate = ReadU32(wav_bytes, body_offset + 4);
      bits_per_sample = ReadU16(wav_bytes, body_offset + 14);
      have_format = true;
    } else if (chunk_id == "data") {
      if (!have_format || channel_count != 1 || sample_rate != kSampleRate) {
        throw std::invalid_argument("audio must be 16 kHz mono");
      } else if (audio_format == kWavFormatPcm && bits_per_sample == 16) {
        std::vector<float> samples(body_size / 2);
        for (size_t i = 0; i < samples.size(); ++i) {
          int16_t value = 0;
          std::memcpy(&value, wav_bytes.data() + body_offset + 2 * i,
                      sizeof(value));
          samples[i] = static_cast<float>(value) / kPcm16FullScale;
        }
        return samples;
      } else if (audio_format == kWavFormatPcm && bits_per_sample == 24) {
        std::vector<float> samples(body_size / 3);
        for (size_t i = 0; i < samples.size(); ++i) {
          const uint32_t b0 =
              static_cast<uint8_t>(wav_bytes[body_offset + 3 * i]);
          const uint32_t b1 =
              static_cast<uint8_t>(wav_bytes[body_offset + 3 * i + 1]);
          const uint32_t b2 =
              static_cast<uint8_t>(wav_bytes[body_offset + 3 * i + 2]);
          const uint32_t raw = b0 | (b1 << 8) | (b2 << 16);
          const int32_t value = static_cast<int32_t>(raw << 8) >> 8;
          samples[i] = static_cast<float>(value) / 8388608.0f;
        }
        return samples;
      } else if (audio_format == kWavFormatFloat && bits_per_sample == 32) {
        std::vector<float> samples(body_size / 4);
        std::memcpy(samples.data(), wav_bytes.data() + body_offset,
                    samples.size() * 4);
        return samples;
      } else {
        throw std::invalid_argument(
            "audio must be PCM16, PCM24 or float32 WAV");
      }
    } else {
    }
    offset = body_offset + chunk_size + (chunk_size & 1u);
  }
  throw std::invalid_argument("WAV file has no data chunk");
}

WavAudio DecodeWavAudio(std::string_view wav_bytes) {
  if (wav_bytes.size() < 12 || wav_bytes.substr(0, 4) != "RIFF" ||
      wav_bytes.substr(8, 4) != "WAVE") {
    throw std::invalid_argument("audio must be a RIFF/WAVE file");
  }
  size_t offset = 12;
  bool have_format = false;
  uint16_t audio_format = 0;
  uint16_t channel_count = 0;
  uint32_t sample_rate = 0;
  uint16_t bits_per_sample = 0;
  size_t data_offset = 0;
  size_t data_size = 0;
  bool have_data = false;
  while (offset + 8 <= wav_bytes.size()) {
    const std::string_view chunk_id = wav_bytes.substr(offset, 4);
    const uint32_t chunk_size = ReadU32(wav_bytes, offset + 4);
    const size_t body_offset = offset + 8;
    const size_t body_size =
        std::min<size_t>(chunk_size, wav_bytes.size() - body_offset);
    if (chunk_id == "fmt ") {
      if (body_size < 16) {
        throw std::invalid_argument("WAV fmt chunk is truncated");
      }
      audio_format = ReadU16(wav_bytes, body_offset);
      channel_count = ReadU16(wav_bytes, body_offset + 2);
      sample_rate = ReadU32(wav_bytes, body_offset + 4);
      bits_per_sample = ReadU16(wav_bytes, body_offset + 14);
      have_format = true;
    } else if (chunk_id == "data") {
      data_offset = body_offset;
      data_size = body_size;
      have_data = true;
      break;
    }
    offset = body_offset + chunk_size + (chunk_size & 1u);
  }
  if (!have_format) {
    throw std::invalid_argument("WAV file has no fmt chunk");
  }
  if (!have_data) {
    throw std::invalid_argument("WAV file has no data chunk");
  }
  if (audio_format != kWavFormatPcm && audio_format != kWavFormatFloat) {
    throw std::invalid_argument("audio must be PCM or IEEE-float WAV");
  }
  if ((bits_per_sample != 8 && bits_per_sample != 16 && bits_per_sample != 24 &&
       bits_per_sample != 32) ||
      (audio_format == kWavFormatFloat && bits_per_sample != 32)) {
    throw std::invalid_argument("WAV bit depth is not 8, 16, 24 or 32");
  }
  if (channel_count == 0) {
    throw std::invalid_argument("WAV has no channels");
  }
  const uint32_t bytes_per_sample = bits_per_sample / 8;
  const uint32_t size_per_frame = bytes_per_sample * channel_count;
  const size_t frame_count = data_size / size_per_frame;
  std::vector<float> interleaved(frame_count * channel_count);
  for (size_t i = 0; i < interleaved.size(); ++i) {
    if (bits_per_sample == 8) {
      const uint8_t sample = static_cast<uint8_t>(wav_bytes[data_offset + i]);
      interleaved[i] = (static_cast<float>(sample) - 128.0f) / 128.0f;
    } else if (bits_per_sample == 16) {
      const int16_t sample =
          static_cast<int16_t>(ReadU16(wav_bytes, data_offset + 2 * i));
      interleaved[i] = static_cast<float>(sample) / kPcm16FullScale;
    } else if (bits_per_sample == 24) {
      const uint32_t b0 = static_cast<uint8_t>(wav_bytes[data_offset + 3 * i]);
      const uint32_t b1 =
          static_cast<uint8_t>(wav_bytes[data_offset + 3 * i + 1]);
      const uint32_t b2 =
          static_cast<uint8_t>(wav_bytes[data_offset + 3 * i + 2]);
      const uint32_t raw = b0 | (b1 << 8) | (b2 << 16);
      const int32_t sample = static_cast<int32_t>(raw << 8) >> 8;
      interleaved[i] = static_cast<float>(sample) / 8388608.0f;
    } else if (audio_format == kWavFormatFloat) {
      float sample = 0.0f;
      std::memcpy(&sample, wav_bytes.data() + data_offset + 4 * i,
                  sizeof(sample));
      interleaved[i] = sample;
    } else {
      const int32_t sample =
          static_cast<int32_t>(ReadU32(wav_bytes, data_offset + 4 * i));
      interleaved[i] = static_cast<float>(sample) / 2147483648.0f;
    }
  }
  std::vector<float> mono(frame_count);
  if (channel_count == 1) {
    mono = std::move(interleaved);
  } else {
    for (size_t frame = 0; frame < frame_count; ++frame) {
      float sum = 0.0f;
      for (uint16_t channel = 0; channel < channel_count; ++channel) {
        sum += interleaved[frame * channel_count + channel];
      }
      mono[frame] = sum / static_cast<float>(channel_count);
    }
  }
  return WavAudio{std::move(mono), static_cast<int>(sample_rate)};
}

std::string EncodeWav(const std::vector<float> &samples, int sample_rate) {
  const uint32_t data_bytes = static_cast<uint32_t>(samples.size()) * 2;
  std::string out;
  out.reserve(44 + data_bytes);
  const auto push = [&](const void *bytes, size_t count) {
    out.append(reinterpret_cast<const char *>(bytes), count);
  };
  const uint32_t total = 36 + data_bytes;
  const uint16_t format = 1;
  const uint16_t channels = 1;
  const uint16_t bits = 16;
  const uint32_t byte_rate = static_cast<uint32_t>(sample_rate) * 2;
  const uint16_t block_align = 2;
  push("RIFF", 4);
  push(&total, 4);
  push("WAVE", 4);
  push("fmt ", 4);
  const uint32_t fmt_size = 16;
  push(&fmt_size, 4);
  push(&format, 2);
  push(&channels, 2);
  push(&sample_rate, 4);
  push(&byte_rate, 4);
  push(&block_align, 2);
  push(&bits, 2);
  push("data", 4);
  push(&data_bytes, 4);
  for (const float sample : samples) {
    const int16_t value = static_cast<int16_t>(
        std::max(-1.0f, std::min(1.0f, sample)) * 32767.0f);
    push(&value, 2);
  }
  return out;
}

const std::vector<float> &MelFilterBank(int mel_bin_count) {
  static std::mutex mutex;
  static std::map<int, std::vector<float>> filters_by_mel_bin_count;
  const std::lock_guard<std::mutex> lock(mutex);
  auto found = filters_by_mel_bin_count.find(mel_bin_count);
  if (found == filters_by_mel_bin_count.end()) {
    found = filters_by_mel_bin_count
                .emplace(mel_bin_count, BuildMelFilterBank(mel_bin_count))
                .first;
  } else {
  }
  return found->second;
}

const std::vector<float> &PeriodicHannWindow() {
  static const std::vector<float> window = BuildPeriodicHannWindow();
  return window;
}

mx::array LogMel(const std::vector<float> &samples, AudioLayout layout) {
  std::vector<float> padded_samples = samples;
  if (padded_samples.size() < static_cast<size_t>(kFftSize)) {
    // Note (Jiaxin Deng): reflect padding needs more than half a window, so a
    // tail this short (a realtime cut) is zero-filled to one window.
    padded_samples.resize(kFftSize, 0.0f);
  } else {
  }
  const int sample_count = static_cast<int>(padded_samples.size());
  const mx::array audio(padded_samples.data(), {sample_count}, mx::float32);
  constexpr int padding = kFftSize / 2;
  const mx::array head = mx::slice(audio, {padding}, {0}, {-1});
  const mx::array tail =
      mx::slice(audio, {sample_count - 2}, {sample_count - padding - 2}, {-1});
  const mx::array padded = mx::concatenate({head, audio, tail});
  const int frame_count = 1 + (padded.shape(0) - kFftSize) / kHopLength;
  const mx::array frames =
      mx::as_strided(padded, {frame_count, kFftSize}, {kHopLength, 1}, 0);
  const mx::array window(PeriodicHannWindow().data(), {kFftSize}, mx::float32);
  mx::array power =
      mx::square(mx::abs(mx::fft::rfft(mx::multiply(frames, window), 1)));
  if (layout == AudioLayout::kReference) {
    power = mx::slice(power, {0, 0}, {power.shape(0) - 1, power.shape(1)});
  } else {
  }
  const mx::array filters(MelFilterBank().data(),
                          {kFftSize / 2 + 1, kMelBinCount}, mx::float32);
  mx::array log_spectrum = mx::log10(
      mx::maximum(mx::matmul(power, filters), mx::array(kLogMelFloor)));
  log_spectrum =
      mx::maximum(log_spectrum, mx::subtract(mx::max(log_spectrum),
                                             mx::array(kLogMelDynamicRange)));
  return mx::transpose(
      mx::divide(mx::add(log_spectrum, mx::array(4.0f)), mx::array(4.0f)));
}

mx::array WhisperWindowFeatures(const float *samples, int sample_count,
                                int mel_bin_count) {
  // Note (Dayuxiaoshui): the reference pads before the floor, so the floor
  // covers the padded tail too.
  std::vector<float> window_samples(kWhisperWindowSampleCount, 0.0f);
  std::copy(samples,
            samples + std::min(sample_count, kWhisperWindowSampleCount),
            window_samples.begin());
  const mx::array audio(window_samples.data(), {kWhisperWindowSampleCount},
                        mx::float32);
  constexpr int padding = kFftSize / 2;
  const mx::array head = mx::slice(audio, {padding}, {0}, {-1});
  const mx::array tail =
      mx::slice(audio, {kWhisperWindowSampleCount - 2},
                {kWhisperWindowSampleCount - padding - 2}, {-1});
  const mx::array padded = mx::concatenate({head, audio, tail});
  const int frame_count = 1 + (padded.shape(0) - kFftSize) / kHopLength;
  const mx::array frames =
      mx::as_strided(padded, {frame_count, kFftSize}, {kHopLength, 1}, 0);
  // Note (Dayuxiaoshui): Swift's Float.pi is one step below (float)M_PI, and
  // the window must match Voxt's bit for bit.
  const float pi = std::nextafter(static_cast<float>(M_PI), 0.0f);
  const mx::array window = mx::multiply(
      mx::array(0.5f),
      mx::subtract(
          mx::array(1.0f),
          mx::cos(mx::divide(mx::multiply(mx::array(2.0f * pi),
                                          mx::arange(kFftSize, mx::float32)),
                             mx::array(static_cast<float>(kFftSize))))));
  mx::array power = mx::square(mx::abs(
      mx::fft::rfft(mx::multiply(frames, mx::expand_dims(window, 0)), 1)));
  power = mx::slice(power, {0, 0}, {power.shape(0) - 1, power.shape(1)});
  const mx::array filters(MelFilterBank(mel_bin_count).data(),
                          {kFftSize / 2 + 1, mel_bin_count}, mx::float32);
  mx::array log_spectrum = mx::log10(
      mx::maximum(mx::matmul(mx::transpose(filters), mx::transpose(power)),
                  mx::array(kLogMelFloor)));
  log_spectrum =
      mx::maximum(log_spectrum, mx::subtract(mx::max(log_spectrum),
                                             mx::array(kLogMelDynamicRange)));
  return mx::transpose(
      mx::divide(mx::add(log_spectrum, mx::array(4.0f)), mx::array(4.0f)));
}

int ConvOutputFrames(int frame_count) {
  for (int i = 0; i < 3; ++i) {
    frame_count = (frame_count - 1) / 2 + 1;
  }
  return frame_count;
}

namespace {

// Note (Jiaxin Deng): Python floor division; the remainder formula can pass
// negative operands.
int FloorDivide(int numerator, int denominator) {
  const int quotient = numerator / denominator;
  return (numerator % denominator != 0 &&
          ((numerator < 0) != (denominator < 0)))
             ? quotient - 1
             : quotient;
}

int RemainderTokens(int frame_count) {
  const int remainder_frames = frame_count % kChunkFrameCount;
  const int after_first_conv = FloorDivide(remainder_frames - 1, 2) + 1;
  return FloorDivide(FloorDivide(after_first_conv - 1, 2) + 1 - 1, 2) + 1;
}

} // namespace

int ReferenceTokenCount(int frame_count) {
  return RemainderTokens(frame_count) +
         (frame_count / kChunkFrameCount) * kChunkTokenCount;
}

int SwiftTokenCount(int frame_count) {
  const float chunks =
      static_cast<float>(frame_count) / static_cast<float>(kChunkFrameCount);
  return static_cast<int>(
      std::trunc(static_cast<float>(RemainderTokens(frame_count)) +
                 chunks * static_cast<float>(kChunkTokenCount)));
}

int TokenCount(int frame_count, AudioLayout layout) {
  if (layout == AudioLayout::kReference) {
    return ReferenceTokenCount(frame_count);
  } else {
    return SwiftTokenCount(frame_count);
  }
}

bool PeakIsSilent(const float *samples, size_t sample_count,
                  float peak_threshold) {
  for (size_t i = 0; i < sample_count; ++i) {
    if (std::fabs(samples[i]) >= peak_threshold) {
      return false;
    } else {
    }
  }
  return true;
}

} // namespace qwen3_asr
