// SPDX-License-Identifier: Apache-2.0
// Drama 3's codec (Fish S1 DAC): 44.1 kHz audio to ten codebooks of codes per
// 2048-sample frame, and codes back to audio, streamed frame by frame.
#pragma once

#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "common/layers.h"
#include "mlx/mlx.h"

namespace fishaudio_drama3 {

inline constexpr int kCodecSampleRate = 44100;
inline constexpr int kCodecFrameSamples = 2048;

// A convolution's kernel with its weight norm folded, in MLX layout.
struct ConvolutionWeights {
  mlx::core::array kernel;
  std::optional<mlx::core::array> bias;
};

class AudioCodec {
public:
  explicit AudioCodec(const std::filesystem::path &model_directory);

  // Codes [codebooks, frames] int32 of mono samples at kCodecSampleRate.
  mlx::core::array Encode(const std::vector<float> &samples) const;

private:
  friend class CodecStream;

  static constexpr int kDecoderStrides[] = {8, 8, 4, 2};
  static constexpr int kResidualDilations[] = {1, 3, 9};
  static constexpr int kDownsampleCount = 2;
  static constexpr int kResidualCodebookCount = 9;
  static constexpr int kQuantizerWindow = 128;
  static constexpr const char *kPostModule = "quantizer.post_module";

  static std::string Index(const std::string &prefix, int index);
  static mlx::core::array Snake(const mlx::core::array &x,
                                const mlx::core::array &alpha);
  static mlx::core::array TransformerNorm(const mlx::core::array &x,
                                          const mlx::core::array &weight);
  // Conv1d on NCL input, as PyTorch lays it out.
  static mlx::core::array Convolve(const mlx::core::array &x,
                                   const ConvolutionWeights &weights,
                                   int stride, int dilation, int groups);
  static mlx::core::array ConvolveTransposed(const mlx::core::array &x,
                                             const ConvolutionWeights &weights,
                                             int stride);
  static mlx::core::array SliceTime(const mlx::core::array &x, int begin,
                                    int end);
  // The additive mask of queries [query_begin, total) over keys [0, total)
  // within window positions.
  static mlx::core::array WindowMask(int query_begin, int total, int window,
                                     mlx::core::Dtype dtype);

  // The layers of the transformer under prefix, as the checkpoint stores them.
  int LayerCount(const std::string &prefix) const;
  const ConvolutionWeights &Convolution(const std::string &prefix) const;
  // A whole-sequence causal convolution, NCL in and out.
  mlx::core::array CausalConvolve(const mlx::core::array &x,
                                  const std::string &prefix, int kernel_size,
                                  int stride, int dilation, int groups) const;
  mlx::core::array ResidualUnit(const mlx::core::array &x,
                                const std::string &prefix, int dilation) const;
  mlx::core::array ConvNeXtPointwise(const mlx::core::array &input,
                                     const mlx::core::array &convolved,
                                     const std::string &prefix) const;
  // A transformer layer over x [batch, time, dim], with keys and values
  // (before rotation) prepended from cache when given; returns the new
  // keys and values through cache.
  mlx::core::array TransformerLayer(
      const mlx::core::array &x, const std::string &prefix,
      const mlx::core::array &freqs_cis, const mlx::core::array &mask,
      std::optional<std::pair<mlx::core::array, mlx::core::array>> &cache)
      const;
  // A quantizer's codes [batch, time] of latents [batch, dim, time], and its
  // quantized output [batch, input_dim, time].
  std::pair<mlx::core::array, mlx::core::array>
  Quantize(const mlx::core::array &latents, const std::string &prefix) const;
  mlx::core::array DecodeCode(const mlx::core::array &codes,
                              const std::string &prefix) const;

  omni_common::Checkpoint checkpoint_;
  std::map<std::string, ConvolutionWeights> convolutions_;
};

// Decodes codes into audio as they arrive; concatenating the audio of
// successive pushes matches decoding all codes at once.
class CodecStream {
public:
  explicit CodecStream(const AudioCodec &codec) : codec_(codec) {}
  // Audio [frames * kCodecFrameSamples] of codes [codebooks, frames], lazy.
  mlx::core::array Push(const mlx::core::array &codes);

private:
  mlx::core::array CausalStep(const mlx::core::array &x,
                              const std::string &prefix, int kernel_size,
                              int dilation, int groups);
  mlx::core::array TransposedStep(const mlx::core::array &x,
                                  const std::string &prefix, int stride);
  mlx::core::array ResidualUnitStep(const mlx::core::array &x,
                                    const std::string &prefix, int dilation);
  mlx::core::array TransformerStep(const mlx::core::array &x,
                                   const std::string &prefix);

  const AudioCodec &codec_;
  // Each causal convolution's left context, by its prefix.
  std::map<std::string, mlx::core::array> left_contexts_;
  // Each transposed convolution's tail into the next frames.
  std::map<std::string, mlx::core::array> overlaps_;
  std::vector<std::optional<std::pair<mlx::core::array, mlx::core::array>>>
      transformer_caches_;
  int transformer_position_ = 0;
};

} // namespace fishaudio_drama3
