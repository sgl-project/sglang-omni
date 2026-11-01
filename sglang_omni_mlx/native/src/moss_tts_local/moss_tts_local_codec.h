// SPDX-License-Identifier: Apache-2.0
// MOSS-Audio-Tokenizer-v2's decoder (codes to 48 kHz stereo samples) and the
// streaming schedule that decodes frames as generation produces them.
#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "common/layers.h"
#include "mlx/mlx.h"
#include "nlohmann/json.hpp"

namespace moss_tts_local {

class AudioCodec {
public:
  // Loads the decoder half of the tokenizer checkpoint in codec_directory;
  // its transformers run in transformer_dtype, the quantizer in float32.
  AudioCodec(const std::filesystem::path &codec_directory,
             mlx::core::Dtype transformer_dtype);

  int sample_rate() const { return sample_rate_; }
  int channel_count() const { return channel_count_; }
  // Samples per channel one frame of codes decodes to.
  int samples_per_frame() const { return samples_per_frame_; }
  int codebook_count() const { return codebook_count_; }

private:
  friend class CodecStream;

  enum class StageKind { kTransformer, kPatch };
  struct Stage {
    StageKind kind;
    int patch_size = 0;
    int layer_count = 0;
    int head_count = 0;
    int model_dimension = 0;
    // Keys each attention layer looks back over, the query's own included.
    int context_frames = 0;
  };
  // A transformer layer's keys and values so far, at most context - 1 of
  // them, and the position of the next frame.
  struct AttentionCache {
    std::optional<mlx::core::array> keys;
    std::optional<mlx::core::array> values;
    int offset = 0;
  };
  using StageCaches = std::vector<std::vector<AttentionCache>>;

  StageCaches NewCaches() const;
  // Codes [codebooks, frames] to interleaved samples [frames *
  // samples_per_frame * channels] in float32, advancing caches.
  mlx::core::array DecodeStep(const mlx::core::array &codes,
                              StageCaches &caches) const;
  // The quantizer's sum of the first codebooks' entries, [1, 768, frames].
  mlx::core::array DequantizeCodes(const mlx::core::array &codes) const;
  mlx::core::array TransformerStep(const mlx::core::array &x, int stage,
                                   std::vector<AttentionCache> &caches) const;
  mlx::core::array AttentionStep(const mlx::core::array &x,
                                 const std::string &prefix, const Stage &stage,
                                 AttentionCache &cache) const;

  omni_common::Checkpoint checkpoint_;
  std::vector<Stage> stages_;
  mlx::core::Dtype transformer_dtype_;
  int sample_rate_;
  int channel_count_;
  int samples_per_frame_;
  int codebook_count_;
};

// Note (Jiaxin Deng): the CUDA vocoder's streaming schedule: a first chunk
// after first_chunk_frames (5 by default), then every 25. Each transformer
// keeps its causal attention cache across chunks, so the samples do not depend
// on where chunks end; non-streamed requests use the schedule too.
class CodecStream {
public:
  explicit CodecStream(const AudioCodec &codec, int first_chunk_frames = 5);

  // Adds one frame's codes [codebooks]; returns the interleaved samples a
  // completed chunk adds, empty until one does.
  std::vector<float> Push(const mlx::core::array &frame_codes);
  // Decodes every frame not yet returned.
  std::vector<float> Finish();

private:
  std::vector<float> DecodePending();

  const AudioCodec &codec_;
  AudioCodec::StageCaches caches_;
  std::vector<mlx::core::array> pending_frames_;
  const int first_chunk_frames_;
  bool emitted_first_chunk_ = false;
};

// Samples as 16-bit little-endian PCM, clipped to [-1, 1] and scaled by
// 32767 as the CUDA server's encode_pcm does.
std::string Pcm16Bytes(const std::vector<float> &samples);
// A 16-bit WAV file around interleaved PCM bytes.
std::string WavBytes(const std::string &pcm, int sample_rate,
                     int channel_count);

} // namespace moss_tts_local
