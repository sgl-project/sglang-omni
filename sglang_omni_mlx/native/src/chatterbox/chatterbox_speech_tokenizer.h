// SPDX-License-Identifier: Apache-2.0
// Chatterbox-Turbo's S3 speech tokenizer (v2, 25 tokens per second): a
// 128-band log-mel, a six-block FSMN attention encoder and a finite scalar
// quantizer; plus the 16 kHz spectrum and Slaney mel bank it shares with the
// voice encoder.
#pragma once

#include <vector>

#include "common/layers.h"
#include "mlx/mlx.h"

namespace chatterbox {

inline constexpr int kSpeechTokenizerSampleRate = 16000;
inline constexpr int kSpectrumFftSize = 400;
inline constexpr int kSpectrumHopSamples = 160;

// The centered, reflect-padded STFT of 16 kHz samples under a symmetric Hann
// window: complex64 [1 + samples / hop, fft size / 2 + 1].
mlx::core::array Spectrum16k(const std::vector<float> &samples_16k);

// Slaney-scale, Slaney-normalized triangular filters from 0 to 8 kHz over the
// spectrum's bins, float32 [mel_count, fft size / 2 + 1].
mlx::core::array SlaneyMelFilters16k(int mel_count);

class SpeechTokenizer {
public:
  explicit SpeechTokenizer(const omni_common::Checkpoint &checkpoint);

  // All speech tokens of up to 30 s of 16 kHz samples, int32 [1, tokens].
  mlx::core::array Encode(const std::vector<float> &samples_16k) const;

private:
  mlx::core::array LogMel(const std::vector<float> &samples_16k) const;
  mlx::core::array Block(const mlx::core::array &x, int layer,
                         const mlx::core::array &attention_mask) const;

  const omni_common::Checkpoint &checkpoint_;
  mlx::core::array mel_filters_;
  // Rotary tables [positions, head dim].
  mlx::core::array rotary_cos_;
  mlx::core::array rotary_sin_;
};

} // namespace chatterbox
