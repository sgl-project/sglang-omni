// SPDX-License-Identifier: Apache-2.0
#include "qwen3_asr/qwen3_audio.h"

#include <cmath>

#include "common/audio.h"

namespace qwen3_asr {

namespace mx = mlx::core;

mx::array LogMel(const std::vector<float> &samples, AudioLayout layout) {
  const mx::array window(omni_common::PeriodicHannWindow().data(),
                         {omni_common::kFftSize}, mx::float32);
  const mx::array filters(
      omni_common::MelFilterBank().data(),
      {omni_common::kFftSize / 2 + 1, omni_common::kMelBinCount}, mx::float32);
  return omni_common::LogMel(
      samples, window, filters,
      {.drop_final_frame = layout == AudioLayout::kReference,
       .pad_short_audio = true});
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
