// SPDX-License-Identifier: Apache-2.0
// Fish S2-Pro's causal DAC codec: 44.1 kHz samples to ten codebooks of codes
// at 2048 samples a frame (reference voices) and back (generated speech), and
// the CUDA vocoder's windowed schedule for streamed speech.
#pragma once

#include <string>
#include <string_view>
#include <vector>

#include "common/layers.h"
#include "mlx/mlx.h"

namespace fish_s2_pro {

// Codes per residual codebook; the semantic codebook has the model's 4096.
inline constexpr int kResidualCodebookSize = 1024;

class AudioCodec {
public:
  explicit AudioCodec(omni_common::WeightMap weights);

  // Mono samples at sample_rate() to int32 codes [codebooks, frames]: the
  // semantic codebook, then the residual ones.
  mlx::core::array Encode(const std::vector<float> &samples) const;
  // Codes [codebooks, frames] to float32 samples [frames * samples_per_frame].
  mlx::core::array Decode(const mlx::core::array &codes) const;
  int samples_per_frame() const { return kSamplesPerFrame; }
  int sample_rate() const { return kSampleRate; }

private:
  static constexpr int kSampleRate = 44100;
  static constexpr int kSamplesPerFrame = 2048;

  // x [batch, channels, length] through a causal convolution.
  mlx::core::array CausalConvolution(const mlx::core::array &x,
                                     const std::string &prefix, int stride,
                                     int dilation, int groups) const;
  mlx::core::array CausalTransposedConvolution(const mlx::core::array &x,
                                               const std::string &prefix,
                                               int stride) const;
  mlx::core::array Snake(const mlx::core::array &x,
                         const std::string &prefix) const;
  mlx::core::array ResidualUnit(const mlx::core::array &x,
                                const std::string &prefix, int dilation) const;
  mlx::core::array ConvNextBlock(const mlx::core::array &x,
                                 const std::string &prefix) const;
  // The causal window-limited transformer over x [batch, channels, frames].
  mlx::core::array Transformer(const mlx::core::array &x,
                               const std::string &prefix, int layer_count,
                               int head_count, int window_frames) const;
  mlx::core::array TransformerRmsNorm(const mlx::core::array &x,
                                      const std::string &prefix) const;
  // One codebook's projected latents [batch, codebook_dim, frames] to the
  // indices of their nearest entries by cosine distance, [batch, frames].
  mlx::core::array NearestCodes(const mlx::core::array &latents,
                                const std::string &prefix) const;
  // A codebook's entries for codes [batch, frames] through its out_proj.
  mlx::core::array QuantizedFeatures(const mlx::core::array &codes,
                                     const std::string &prefix) const;

  omni_common::Checkpoint checkpoint_;
};

// Note (Jiaxin Deng): the CUDA vocoder's streaming schedule: a first window
// after 40 frames, then every 45, each redecoding 20 frames of left context
// and crossfading the 512 samples where windows meet.
class CodecStream {
public:
  explicit CodecStream(const AudioCodec &codec) : codec_(codec) {}

  // Adds one frame's codes [codebooks]; returns the samples a completed
  // window adds, empty until one does.
  std::vector<float> Push(const mlx::core::array &frame_codes);
  // Decodes every frame not yet returned, with the held crossfade tail.
  std::vector<float> Finish();

private:
  std::vector<float> DecodeWindow(bool is_final);

  const AudioCodec &codec_;
  std::vector<mlx::core::array> frames_;
  int first_kept_frame_ = 0;
  int emitted_frames_ = 0;
  int next_window_frames_ = 0;
  std::vector<float> pending_tail_;
};

// Mono PCM16 or float32 WAV at any rate, channels averaged, as float32
// samples at its own rate; throws std::invalid_argument for anything else.
struct DecodedWav {
  std::vector<float> samples;
  int sample_rate = 0;
};
DecodedWav DecodeAnyWav(std::string_view wav_bytes);
// torchaudio's resample (sinc, Hann window, 6 zero crossings, rolloff 0.99);
// throws std::invalid_argument for a pair of rates whose kernel table would
// be too large.
std::vector<float> Resample(const std::vector<float> &samples, int source_rate,
                            int target_rate);

// Samples as 16-bit little-endian PCM, clipped to [-1, 1] and scaled by
// 32767 as the CUDA server's encode_pcm does.
std::string Pcm16Bytes(const std::vector<float> &samples);
// A mono 16-bit WAV file around PCM bytes.
std::string WavBytes(const std::string &pcm, int sample_rate);

} // namespace fish_s2_pro
