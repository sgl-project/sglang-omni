// SPDX-License-Identifier: Apache-2.0
#include "qwen3_asr/qwen3_audio.h"

#include <cmath>
#include <stdexcept>

#include "common/audio.h"

namespace qwen3_asr {

namespace mx = mlx::core;

mx::array LogMel(const std::vector<float> &samples, AudioLayout layout) {
  const mx::array window(omni_common::PeriodicHannWindow().data(),
                         {omni_common::kFftSize}, mx::float32);
  const mx::array filters(
      omni_common::MelFilterBank().data(),
      {omni_common::kFftSize / 2 + 1, omni_common::kMelBinCount}, mx::float32);
  return LogMel(samples, layout, window, filters);
}

mx::array LogMel(const std::vector<float> &samples, AudioLayout layout,
                 const mx::array &window, const mx::array &filters) {
  if (window.shape() != mx::Shape{omni_common::kFftSize} ||
      filters.shape() !=
          mx::Shape{omni_common::kFftSize / 2 + 1, omni_common::kMelBinCount} ||
      window.dtype() != mx::float32 || filters.dtype() != mx::float32) {
    throw std::invalid_argument(
        "log-mel window and filters have incompatible shapes or dtypes");
  } else {
  }
  std::vector<float> padded_samples = samples;
  if (padded_samples.size() < static_cast<size_t>(omni_common::kFftSize)) {
    // Note (Jiaxin Deng): reflect padding needs more than half a window, so a
    // tail this short (a realtime cut) is zero-filled to one window.
    padded_samples.resize(omni_common::kFftSize, 0.0f);
  } else {
  }
  const int sample_count = static_cast<int>(padded_samples.size());
  const mx::array audio(padded_samples.data(), {sample_count}, mx::float32);
  constexpr int padding = omni_common::kFftSize / 2;
  const mx::array head = mx::slice(audio, {padding}, {0}, {-1});
  const mx::array tail =
      mx::slice(audio, {sample_count - 2}, {sample_count - padding - 2}, {-1});
  const mx::array padded = mx::concatenate({head, audio, tail});
  const int frame_count =
      1 + (padded.shape(0) - omni_common::kFftSize) / omni_common::kHopLength;
  const mx::array frames =
      mx::as_strided(padded, {frame_count, omni_common::kFftSize},
                     {omni_common::kHopLength, 1}, 0);
  mx::array power =
      mx::square(mx::abs(mx::fft::rfft(mx::multiply(frames, window), 1)));
  if (layout == AudioLayout::kReference) {
    power = mx::slice(power, {0, 0}, {power.shape(0) - 1, power.shape(1)});
  } else {
  }
  mx::array log_spectrum = mx::log10(mx::maximum(
      mx::matmul(power, filters), mx::array(omni_common::kLogMelFloor)));
  log_spectrum = mx::maximum(
      log_spectrum, mx::subtract(mx::max(log_spectrum),
                                 mx::array(omni_common::kLogMelDynamicRange)));
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

} // namespace qwen3_asr
