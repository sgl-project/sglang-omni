// SPDX-License-Identifier: Apache-2.0
// Voxtral TTS's audio codec decoder (codes to 24 kHz samples) and the
// windowed schedule that decodes frames as generation produces them.
#pragma once

#include <string>
#include <vector>

#include "common/layers.h"
#include "mlx/mlx.h"
#include "nlohmann/json.hpp"

namespace voxtral_tts {

// Codes 0 and 1 are the empty and end-of-audio tokens; codebook entries
// follow them.
constexpr int kSpecialAudioCodeCount = 2;
constexpr int kEndOfAudioCode = 1;

class AudioCodec {
public:
  // weights are the checkpoint's audio_tokenizer.* entries; arguments are
  // config.json's multimodal.audio_tokenizer_args.
  AudioCodec(qwen3_asr::WeightMap weights, const nlohmann::json &arguments);

  // Frames of model codes [frames, 1 + acoustic codebooks] to samples
  // [frames * samples_per_frame] in the decoder's dtype.
  mlx::core::array Decode(const mlx::core::array &codes) const;
  int samples_per_frame() const { return samples_per_frame_; }
  int sample_rate() const { return sample_rate_; }

private:
  enum class BlockKind { kConvolution, kTransposedConvolution, kTransformer };
  struct Block {
    BlockKind kind;
    int kernel_size;
    int stride;
    int layer_count;
    int window_frames;
  };

  static std::vector<Block> BuildBlocks(const nlohmann::json &arguments);
  // The semantic codebook's embedding, and weight-normed kernels folded into
  // MLX convolution weights.
  static qwen3_asr::WeightMap PrepareWeights(qwen3_asr::WeightMap weights,
                                             const std::vector<Block> &blocks);
  mlx::core::array TransformerLayer(const mlx::core::array &x,
                                    const std::string &prefix,
                                    int window_frames) const;

  std::vector<Block> blocks_;
  qwen3_asr::Checkpoint checkpoint_;
  int dimension_;
  int head_count_;
  int head_dim_;
  float norm_eps_;
  float query_key_norm_eps_;
  int acoustic_levels_;
  int samples_per_frame_;
  int sample_rate_;
  mlx::core::array alibi_slopes_;
};

// Note (Minyi Tang): the CUDA vocoder's streaming schedule. A first window
// after 5 frames, then every 25, each with 25 frames of left context, which
// covers the decoder's 21-frame dependency; non-streamed requests use it too,
// so both return the same samples.
class CodecStream {
public:
  explicit CodecStream(const AudioCodec &codec) : codec_(codec) {}

  // Adds one frame's codes [1 + acoustic codebooks]; returns the samples a
  // completed window adds, empty until one does.
  std::vector<float> Push(const mlx::core::array &frame_codes);
  // Decodes every frame not yet returned.
  std::vector<float> Finish();

private:
  std::vector<float> DecodeWindow();

  const AudioCodec &codec_;
  std::vector<mlx::core::array> frames_;
  int emitted_frames_ = 0;
};

// Samples as 16-bit little-endian PCM, clipped to [-1, 1] and scaled by
// 32767 as the CUDA server's encode_pcm does.
std::string Pcm16Bytes(const std::vector<float> &samples);
// A mono 16-bit WAV file around PCM bytes.
std::string WavBytes(const std::string &pcm, int sample_rate);

} // namespace voxtral_tts
