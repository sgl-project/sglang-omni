// SPDX-License-Identifier: Apache-2.0
// Sortformer v2.1 front end (NeMo FilterbankFeatures without normalization), as
// Voxt's Swift MLXAudioVAD computes it for AOSC checkpoints.
#pragma once

#include <vector>

#include "mlx/mlx.h"

namespace sortformer {

struct ProcessorConfig {
  int feature_size = 80;
  int sampling_rate = 16000;
  int hop_length = 160;
  int fft_size = 512;
  int window_length = 400;
  float preemphasis = 0.97f;
};

// Slaney-scale, Slaney-normalized filters [fft_size / 2 + 1, mel_bin_count]
// from 0 Hz to sample_rate / 2, built in float32 in the order of Swift
// melFilters, so the values are bit-identical to Voxt's.
std::vector<float> MelFilterBank(int sample_rate, int fft_size,
                                 int mel_bin_count);

// Symmetric Hann window (denominator size - 1), as Swift hanningWindow.
std::vector<float> SymmetricHannWindow(int size);

// The Hann window zero-padded on both sides to fft_size, as torch.stft centres
// a window shorter than the FFT.
std::vector<float> CenteredWindow(int window_length, int fft_size);

class FeatureExtractor {
public:
  explicit FeatureExtractor(const ProcessorConfig &config);

  // Log-mel features (1, feature_size, frames) in float32, unnormalized and
  // unpadded (the use_aosc path of Swift extractMelFeatures).
  mlx::core::array operator()(const std::vector<float> &samples) const;

  const std::vector<float> &mel_filters() const { return mel_filters_; }
  const std::vector<float> &window() const { return window_; }

private:
  ProcessorConfig config_;
  std::vector<float> mel_filters_;
  std::vector<float> window_;
  mlx::core::array mel_filters_array_;
  mlx::core::array window_array_;
};

} // namespace sortformer
