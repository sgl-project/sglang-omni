// SPDX-License-Identifier: Apache-2.0
// Chatterbox-Turbo's HiFT vocoder: an F0-driven harmonic source and a
// mel-conditioned upsampling network whose STFT frames give 24 kHz audio.
#pragma once

#include <string>
#include <vector>

#include "common/layers.h"
#include "mlx/mlx.h"

namespace chatterbox {

class Vocoder {
public:
  explicit Vocoder(const omni_common::Checkpoint &checkpoint);

  // 24 kHz samples [1, samples] for mel [1, 80, frames], float32. Draws the
  // source's phases and noise from MLX's global random keys.
  mlx::core::array Waveform(const mlx::core::array &mel) const;

private:
  // The harmonic source [1, samples, 1] for F0 in Hz [1, samples, 1].
  mlx::core::array Source(const mlx::core::array &f0_hz) const;
  // The source's STFT [1, frames, 2 * bins]: real parts, then imaginary.
  mlx::core::array SourceSpectrum(const mlx::core::array &source) const;
  // Overlap-added, window-normalized and trimmed samples [1, samples] of
  // frames [1, frames, n_fft].
  mlx::core::array OverlapAdd(const mlx::core::array &frames) const;
  // A convolution with weights at prefix on channels-last x [1, steps,
  // channels].
  mlx::core::array Convolution(const mlx::core::array &x,
                               const std::string &prefix, int stride,
                               int padding, int dilation) const;
  mlx::core::array Snake(const mlx::core::array &x,
                         const std::string &prefix) const;
  mlx::core::array ResidualBlock(const mlx::core::array &x,
                                 const std::string &prefix,
                                 int kernel_size) const;

  const omni_common::Checkpoint &checkpoint_;
  // The periodic Hann window over n_fft samples.
  std::vector<float> window_values_;
  mlx::core::array window_;
};

} // namespace chatterbox
