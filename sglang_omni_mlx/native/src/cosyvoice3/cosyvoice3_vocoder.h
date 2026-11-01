// SPDX-License-Identifier: Apache-2.0
// Fun-CosyVoice3's vocoder: the DiT flow that turns speech tokens into an
// 80-bin mel spectrogram by conditional flow matching, and the HiFT generator
// that turns the mel into 24 kHz samples.
#pragma once

#include <functional>
#include <string>
#include <vector>

#include "common/layers.h"
#include "mlx/mlx.h"

namespace cosyvoice3 {

inline constexpr int kMelBinCount = 80;
inline constexpr int kSpeakerEmbeddingSize = 192;
inline constexpr int kMelFramesPerToken = 2;
inline constexpr int kSampleRate = 24000;
// The flow's fixed noise covers this many mel frames, its longest output.
inline constexpr int kMaxMelFrames = 50 * 300;

// Speech tokens to mel frames, given the checkpoint's flow.* weights.
class Flow {
public:
  explicit Flow(const omni_common::Checkpoint &checkpoint);

  // Mel [1, 80, 2 * tokens] of speech tokens [tokens] after a prompt of
  // prompt_tokens with its mel prompt_mel [1, 2 * prompt tokens, 80], in the
  // voice of speaker_embedding [1, 192] (zeros for no speaker); calls
  // on_step before each of its ten steps, which may throw to stop it.
  mlx::core::array Decode(const std::vector<int> &speech_tokens,
                          const std::vector<int> &prompt_tokens,
                          const mlx::core::array &prompt_mel,
                          const mlx::core::array &speaker_embedding,
                          const std::function<void()> &on_step) const;

private:
  // The DiT's velocity [batch, 80, frames] at noisy mel x.
  mlx::core::array Velocity(const mlx::core::array &x,
                            const mlx::core::array &mu,
                            const mlx::core::array &timesteps,
                            const mlx::core::array &speaker_features,
                            const mlx::core::array &mel_conditioning) const;
  mlx::core::array Block(const mlx::core::array &x,
                         const mlx::core::array &timestep_embedding,
                         int block) const;

  const omni_common::Checkpoint &checkpoint_;
  mlx::core::array dtype_probe_;
  mlx::core::array noise_;
  mlx::core::array rope_cos_;
  mlx::core::array rope_sin_;
};

// Mel frames to samples, given the checkpoint's hifigan.* weights.
class Hift {
public:
  explicit Hift(const omni_common::Checkpoint &checkpoint);

  // Float32 samples [480 * frames] of mel [1, 80, frames].
  mlx::core::array Decode(const mlx::core::array &mel) const;

private:
  mlx::core::array HarmonicSource(const mlx::core::array &f0) const;
  mlx::core::array ResBlock(const mlx::core::array &x,
                            const std::string &prefix, int kernel_size) const;
  // Channel-last convolution of x with the weights under prefix, padded by
  // left and right frames.
  mlx::core::array Conv(const mlx::core::array &x, const std::string &prefix,
                        int left, int right, int stride = 1,
                        int dilation = 1) const;

  const omni_common::Checkpoint &checkpoint_;
  mlx::core::array window_;
  mlx::core::array initial_phases_;
};

} // namespace cosyvoice3
