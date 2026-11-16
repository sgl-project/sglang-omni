// SPDX-License-Identifier: Apache-2.0
// Fish S2-Pro's streamed speech, decoded on the CUDA vocoder's schedule.
#pragma once

#include <vector>

#include "fish_dac/fish_dac.h"
#include "mlx/mlx.h"

namespace fish_s2_pro {

// Note (Jiaxin Deng): the CUDA vocoder's streaming schedule: a first window
// after 40 frames, then every 45, each redecoding 20 frames of left context
// and crossfading the 512 samples where windows meet.
class VocoderStream {
public:
  explicit VocoderStream(const fish_dac::AudioCodec &codec) : codec_(codec) {}

  // Adds one frame's codes [codebooks]; returns the samples a completed
  // window adds, empty until one does.
  std::vector<float> Push(const mlx::core::array &frame_codes);
  // Decodes every frame not yet returned, with the held crossfade tail.
  std::vector<float> Finish();

private:
  std::vector<float> DecodeWindow(bool is_final);

  const fish_dac::AudioCodec &codec_;
  std::vector<mlx::core::array> frames_;
  int first_kept_frame_ = 0;
  int emitted_frames_ = 0;
  int next_window_frames_ = 0;
  std::vector<float> pending_tail_;
};

} // namespace fish_s2_pro
