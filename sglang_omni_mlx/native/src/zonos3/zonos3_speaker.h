// SPDX-License-Identifier: Apache-2.0
// ZONOS3-Mini's speaker encoder (ownmect_s2): a Kaldi fbank of 16 kHz audio,
// a MECT-B2 network of grouped convolutions and transformer mixers, and
// attentive statistics pooling into a 192-dimensional embedding.
#pragma once

#include <filesystem>
#include <map>
#include <string>
#include <vector>

#include "mlx/mlx.h"
#include "zonos3/zonos3_weights.h"

namespace zonos3 {

class SpeakerEncoder {
public:
  // Reads encoder_directory's config.json, model.safetensors and
  // frontend.npz; throws std::runtime_error unless it is a MECT-B2 encoder.
  explicit SpeakerEncoder(const std::filesystem::path &encoder_directory);

  int embedding_size() const { return embedding_size_; }
  // The embedding (not L2-normalized) of mono 44.1 kHz samples, of which the
  // first 60 s count. Throws std::invalid_argument for audio under 25 ms at
  // 16 kHz, or when the embedding is not finite.
  std::vector<float> Embed(const std::vector<float> &samples) const;

private:
  // Log mel filterbank [frames, mels] of 16 kHz samples.
  std::vector<float> Fbank(const std::vector<float> &samples,
                           int &frame_count) const;
  mlx::core::array Pointwise(const mlx::core::array &x,
                             const std::string &prefix, bool has_bias) const;
  mlx::core::array Linear(const mlx::core::array &x,
                          const std::string &prefix) const;
  mlx::core::array BatchNorm(const mlx::core::array &x,
                             const std::string &prefix) const;
  mlx::core::array LayerNorm(const mlx::core::array &x,
                             const std::string &prefix) const;
  mlx::core::array Grouped(const mlx::core::array &x, const std::string &prefix,
                           int frequency_stride, int time_stride) const;
  mlx::core::array ResidualBlock(const mlx::core::array &x,
                                 const std::string &prefix,
                                 int frequency_stride, int time_stride) const;
  mlx::core::array Attention(const mlx::core::array &hidden,
                             const std::string &prefix) const;
  mlx::core::array Experts(const mlx::core::array &hidden,
                           const std::string &prefix) const;
  mlx::core::array PositionConvolution(const mlx::core::array &hidden,
                                       const std::string &prefix) const;

  struct BlockShape {
    int residual_count;
    int frequency_stride;
    int time_stride;
  };
  // One MECT block over x [1, F, T, C] with the running flat features
  // [T, 2560]; returns both, updated.
  std::pair<mlx::core::array, mlx::core::array>
  Block(const mlx::core::array &x, const mlx::core::array &flat,
        const BlockShape &shape, const std::string &prefix) const;

  Weights weights_;
  int embedding_size_;
  int mel_count_;
  int frame_length_;
  int frame_shift_;
  int fft_size_;
  int head_count_;
  int position_kernel_;
  double preemphasis_;
  double waveform_scale_;
  double log_floor_;
  float layer_norm_epsilon_;
  float variance_floor_;
  std::vector<double> window_;
  // [mels][fft size / 2 + 1].
  std::vector<std::vector<double>> mel_banks_;
  // [160][475] float64, for 44.1 to 16 kHz.
  mlx::core::array resample_kernel_;
  // Grouped 3x3 convolution weights [out, 3, 3, in per group] by prefix.
  std::map<std::string, mlx::core::array> grouped_;
};

} // namespace zonos3
