// SPDX-License-Identifier: Apache-2.0
// Chatterbox-Turbo's voice encoder: a speaker embedding from overlapping
// 40-band mel windows through three stacked LSTMs and a projection.
#pragma once

#include <vector>

#include "common/layers.h"
#include "mlx/mlx.h"

namespace chatterbox {

class VoiceEncoder {
public:
  explicit VoiceEncoder(const omni_common::Checkpoint &checkpoint);

  // The L2-normalized speaker embedding of 16 kHz samples, float32 [1, 256].
  mlx::core::array Embed(const std::vector<float> &samples_16k) const;

private:
  // L2-normalized embeddings [windows, 256] of mel windows [windows, frames,
  // mel bands].
  mlx::core::array EmbedWindows(const mlx::core::array &mel_windows) const;

  const omni_common::Checkpoint &checkpoint_;
  mlx::core::array mel_filters_;
};

} // namespace chatterbox
