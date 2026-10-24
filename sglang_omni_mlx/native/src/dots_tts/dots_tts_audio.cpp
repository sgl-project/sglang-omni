// SPDX-License-Identifier: Apache-2.0
#include "dots_tts/dots_tts_audio.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace dots_tts {

namespace {

constexpr float kPi = 3.14159265358979323846f;
constexpr float kKaiserBeta = 14.769656459379492f;
constexpr float kResampleRolloff = 0.95f;
constexpr int kResampleLowpassWidth = 64;
constexpr int kTrimFrameLength = 2048;
constexpr int kTrimHopLength = 512;
constexpr float kOnsetRelDb = 12.0f;
constexpr int kOnsetHopMs = 10;
constexpr int kOnsetSmoothMs = 40;
constexpr int kOnsetRunMs = 60;
constexpr float kOnsetLeadPadMs = 30.0f;
constexpr float kOnsetFadeMs = 10.0f;

double BesselI0(double value) {
  double sum = 1.0;
  double term = 1.0;
  const double squared = value * value;
  for (int index = 1; index < 64; ++index) {
    term *= squared / (4.0 * index * index);
    sum += term;
    if (term < 1e-15 * sum) {
      break;
    }
  }
  return sum;
}

int GreatestCommonDivisor(int first, int second) {
  while (second != 0) {
    const int remainder = first % second;
    first = second;
    second = remainder;
  }
  return first;
}

struct KaiserSincKernel {
  std::vector<float> rows;
  int width = 0;
  int original_rate_reduced = 0;
  int target_rate_reduced = 0;
};

KaiserSincKernel BuildKaiserSincKernel(int original_sample_rate,
                                       int target_sample_rate) {
  const int divisor =
      GreatestCommonDivisor(original_sample_rate, target_sample_rate);
  const int original_rate_reduced = original_sample_rate / divisor;
  const int target_rate_reduced = target_sample_rate / divisor;
  const float base_frequency =
      static_cast<float>(std::min(original_rate_reduced, target_rate_reduced)) *
      kResampleRolloff;
  const int width = static_cast<int>(std::ceil(
      static_cast<float>(kResampleLowpassWidth * original_rate_reduced) /
      base_frequency));
  const int tap_count = 2 * width + original_rate_reduced;
  std::vector<float> rows(static_cast<size_t>(target_rate_reduced) * tap_count,
                          0.0f);
  for (int phase = 0; phase < target_rate_reduced; ++phase) {
    for (int tap = 0; tap < tap_count; ++tap) {
      const float position = (static_cast<float>(-width + tap) /
                                  static_cast<float>(original_rate_reduced) -
                              static_cast<float>(phase) /
                                  static_cast<float>(target_rate_reduced)) *
                             base_frequency;
      const float clipped = std::max(
          -static_cast<float>(kResampleLowpassWidth),
          std::min(position, static_cast<float>(kResampleLowpassWidth)));
      const float normalized =
          std::max(0.0f, 1.0f - (clipped / kResampleLowpassWidth) *
                                    (clipped / kResampleLowpassWidth));
      const float window =
          BesselI0(kKaiserBeta * std::sqrt(normalized)) / BesselI0(kKaiserBeta);
      const float scale = base_frequency / original_rate_reduced;
      const float sinc = std::abs(position) < 1e-9f
                             ? 1.0f
                             : std::sin(position * kPi) / (position * kPi);
      rows[static_cast<size_t>(phase) * tap_count + tap] =
          sinc * window * scale;
    }
  }
  return KaiserSincKernel{std::move(rows), width, original_rate_reduced,
                          target_rate_reduced};
}

} // namespace

std::vector<float> Resample(const std::vector<float> &samples,
                            int original_sample_rate, int target_sample_rate) {
  if (original_sample_rate <= 0 || target_sample_rate <= 0) {
    throw std::invalid_argument("sample rates must be positive");
  }
  if (original_sample_rate == target_sample_rate) {
    return samples;
  }
  const KaiserSincKernel kernel =
      BuildKaiserSincKernel(original_sample_rate, target_sample_rate);
  const int tap_count =
      static_cast<int>(kernel.rows.size() / kernel.target_rate_reduced);
  const int sample_count = static_cast<int>(samples.size());
  std::vector<float> padded_samples(
      static_cast<size_t>(sample_count + 2 * kernel.width +
                          kernel.original_rate_reduced),
      0.0f);
  for (int index = 0; index < sample_count; ++index) {
    padded_samples[static_cast<size_t>(kernel.width + index)] = samples[index];
  }
  const int output_positions =
      (static_cast<int>(padded_samples.size()) - tap_count) /
          kernel.original_rate_reduced +
      1;
  std::vector<float> resampled(
      static_cast<size_t>(output_positions) * kernel.target_rate_reduced, 0.0f);
  for (int output_position = 0; output_position < output_positions;
       ++output_position) {
    for (int tap = 0; tap < tap_count; ++tap) {
      const float sample = padded_samples[static_cast<size_t>(
          output_position * kernel.original_rate_reduced + tap)];
      for (int phase = 0; phase < kernel.target_rate_reduced; ++phase) {
        resampled[static_cast<size_t>(output_position) *
                      kernel.target_rate_reduced +
                  phase] +=
            kernel.rows[static_cast<size_t>(phase) * tap_count + tap] * sample;
      }
    }
  }
  const int target_sample_count = static_cast<int>(
      std::ceil(static_cast<double>(kernel.target_rate_reduced) * sample_count /
                kernel.original_rate_reduced));
  if (target_sample_count < static_cast<int>(resampled.size())) {
    resampled.resize(static_cast<size_t>(target_sample_count));
  }
  return resampled;
}

std::vector<float> TrimSilence(const std::vector<float> &samples,
                               float top_db) {
  if (samples.empty()) {
    return samples;
  }
  const size_t sample_count = samples.size();
  const int frame_count = static_cast<int>(
      1 + (static_cast<int>(sample_count) > kTrimFrameLength
               ? (static_cast<int>(sample_count) - kTrimFrameLength) /
                     kTrimHopLength
               : 0));
  if (frame_count <= 0) {
    return samples;
  }
  float peak = 0.0f;
  for (float sample : samples) {
    peak = std::max(peak, std::abs(sample));
  }
  const float threshold = (peak + 1e-9f) * std::pow(10.0f, -top_db / 20.0f);
  std::vector<bool> nonsilent(frame_count, false);
  for (int frame = 0; frame < frame_count; ++frame) {
    const int start = frame * kTrimHopLength;
    const int end =
        std::min(start + kTrimFrameLength, static_cast<int>(sample_count));
    float sum_squared = 0.0f;
    const int window_sample_count = end - start;
    for (int index = start; index < end; ++index) {
      sum_squared += samples[index] * samples[index];
    }
    const float rms = std::sqrt(
        sum_squared / static_cast<float>(window_sample_count) + 1e-12f);
    nonsilent[frame] = rms > threshold;
  }
  int first_nonsilent_frame = -1;
  int last_nonsilent_frame = -1;
  for (int frame = 0; frame < frame_count; ++frame) {
    if (nonsilent[frame]) {
      first_nonsilent_frame = frame;
      break;
    }
  }
  if (first_nonsilent_frame < 0) {
    return samples;
  }
  for (int frame = frame_count - 1; frame >= 0; --frame) {
    if (nonsilent[frame]) {
      last_nonsilent_frame = frame;
      break;
    }
  }
  const int start = first_nonsilent_frame * kTrimHopLength;
  const int end =
      std::min(static_cast<int>(sample_count),
               (last_nonsilent_frame + 1) * kTrimHopLength + kTrimFrameLength);
  return std::vector<float>(samples.begin() + start, samples.begin() + end);
}

std::vector<float> CleanOnset(const std::vector<float> &samples,
                              int sample_rate) {
  std::vector<float> cleaned_samples = samples;
  if (cleaned_samples.size() <= static_cast<size_t>(sample_rate)) {
    return cleaned_samples;
  }
  const int hop_sample_count = std::max(1, sample_rate * kOnsetHopMs / 1000);
  const int frame_count = std::max(
      1, (static_cast<int>(cleaned_samples.size()) - hop_sample_count) /
             hop_sample_count);
  std::vector<float> rms(frame_count, 0.0f);
  for (int frame = 0; frame < frame_count; ++frame) {
    float sum_squared = 0.0f;
    const int end = std::min(static_cast<int>(cleaned_samples.size()),
                             (frame + 1) * hop_sample_count);
    const int frame_sample_count = end - frame * hop_sample_count;
    for (int index = frame * hop_sample_count; index < end; ++index) {
      sum_squared += cleaned_samples[index] * cleaned_samples[index];
    }
    rms[frame] = std::sqrt(
        sum_squared / static_cast<float>(frame_sample_count) + 1e-12f);
  }
  std::vector<float> decibels(frame_count, 0.0f);
  float maximum_decibels = -1e9f;
  for (int frame = 0; frame < frame_count; ++frame) {
    decibels[frame] = 20.0f * std::log10(rms[frame] + 1e-9f);
    maximum_decibels = std::max(maximum_decibels, decibels[frame]);
  }
  const int smoothing_window = std::max(1, kOnsetSmoothMs / kOnsetHopMs);
  std::vector<float> smoothed_decibels(frame_count, 0.0f);
  for (int frame = 0; frame < frame_count; ++frame) {
    float sum = 0.0f;
    int count = 0;
    for (int neighbor = std::max(0, frame - smoothing_window / 2);
         neighbor <= std::min(frame_count - 1, frame + smoothing_window / 2);
         ++neighbor) {
      sum += decibels[neighbor];
      ++count;
    }
    smoothed_decibels[frame] = sum / static_cast<float>(count);
  }
  const int sustained_frame_count = std::max(1, kOnsetRunMs / kOnsetHopMs);
  const float threshold = maximum_decibels - kOnsetRelDb;
  std::vector<bool> sustained_onset(frame_count, false);
  int run_start = -1;
  for (int frame = 0; frame < frame_count; ++frame) {
    if (smoothed_decibels[frame] > threshold) {
      if (run_start < 0) {
        run_start = frame;
      }
      if (frame - run_start + 1 >= sustained_frame_count) {
        for (int run_frame = run_start; run_frame <= frame; ++run_frame) {
          sustained_onset[run_frame] = true;
        }
      }
    } else {
      run_start = -1;
    }
  }
  int onset_frame = -1;
  for (int frame = 0; frame < frame_count; ++frame) {
    if (sustained_onset[frame]) {
      onset_frame = frame;
      break;
    }
  }
  if (onset_frame < 0) {
    return cleaned_samples;
  }
  const int onset_sample = onset_frame * hop_sample_count;
  const int start =
      std::max(0, onset_sample - static_cast<int>(kOnsetLeadPadMs / 1000.0f *
                                                  sample_rate));
  std::vector<float> output(cleaned_samples.begin() + start,
                            cleaned_samples.end());
  const int fade_sample_count =
      std::min(static_cast<int>(kOnsetFadeMs / 1000.0f * sample_rate),
               static_cast<int>(output.size()) / 2);
  if (fade_sample_count > 0) {
    for (int index = 0; index < fade_sample_count; ++index) {
      output[index] *= static_cast<float>(index) / fade_sample_count;
    }
  }
  return output;
}

} // namespace dots_tts
