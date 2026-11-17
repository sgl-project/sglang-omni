// SPDX-License-Identifier: Apache-2.0
#include "fish_s2_pro/fish_s2_pro_stream.h"

#include <algorithm>
#include <cstddef>
#include <utility>

namespace fish_s2_pro {

namespace mx = mlx::core;

namespace {

constexpr int kStreamFirstWindowFrames = 40;
constexpr int kStreamFollowupFrames = 45;
constexpr int kStreamOverlapFrames = 20;
constexpr size_t kStreamCrossfadeSamples = 512;

} // namespace

std::vector<float> VocoderStream::Push(const mx::array &frame_codes) {
  frames_.push_back(frame_codes);
  const int total_frames = first_kept_frame_ + static_cast<int>(frames_.size());
  const int next_window_frames =
      next_window_frames_ > 0 ? next_window_frames_ : kStreamFirstWindowFrames;
  if (total_frames < next_window_frames) {
    next_window_frames_ = next_window_frames;
    return {};
  } else {
  }
  std::vector<float> samples = DecodeWindow(false);
  next_window_frames_ = total_frames + kStreamFollowupFrames;
  return samples;
}

std::vector<float> VocoderStream::Finish() {
  const int total_frames = first_kept_frame_ + static_cast<int>(frames_.size());
  if (total_frames <= emitted_frames_) {
    return std::move(pending_tail_);
  } else {
    return DecodeWindow(true);
  }
}

std::vector<float> VocoderStream::DecodeWindow(bool is_final) {
  const int total_frames = first_kept_frame_ + static_cast<int>(frames_.size());
  const int window_start =
      std::max(first_kept_frame_, emitted_frames_ - kStreamOverlapFrames);
  const std::vector<mx::array> window(
      frames_.begin() + (window_start - first_kept_frame_), frames_.end());
  mx::array audio = codec_.Decode(mx::stack(window, 1));
  mx::eval(audio);
  const float *decoded = audio.data<float>();
  const size_t decoded_count = audio.size();
  const size_t overlap_samples =
      static_cast<size_t>(emitted_frames_ - window_start) *
      fish_dac::kFrameSamples;
  std::vector<float> delta;
  // The first window has no tail; each later one crossfades the whole of it.
  const size_t crossfade_count = pending_tail_.size();
  // Note (Jiaxin Deng): torch.linspace's float32 ramp, stepped from each end.
  const float step = 1.0f / static_cast<float>(kStreamCrossfadeSamples - 1);
  for (size_t index = 0; index < crossfade_count; ++index) {
    const float fade_in =
        index < crossfade_count / 2
            ? static_cast<float>(index) * step
            : 1.0f - static_cast<float>(crossfade_count - 1 - index) * step;
    const float refreshed = decoded[overlap_samples - crossfade_count + index];
    delta.push_back(pending_tail_[index] * (1.0f - fade_in) +
                    refreshed * fade_in);
  }
  delta.insert(delta.end(), decoded + overlap_samples, decoded + decoded_count);
  pending_tail_.clear();
  emitted_frames_ = total_frames;
  if (is_final) {
    return delta;
  } else {
  }
  pending_tail_.assign(delta.end() - kStreamCrossfadeSamples, delta.end());
  delta.resize(delta.size() - kStreamCrossfadeSamples);
  const int keep_from = std::max(0, total_frames - kStreamOverlapFrames);
  if (keep_from > first_kept_frame_) {
    frames_.erase(frames_.begin(),
                  frames_.begin() + (keep_from - first_kept_frame_));
    first_kept_frame_ = keep_from;
  } else {
  }
  return delta;
}

} // namespace fish_s2_pro
