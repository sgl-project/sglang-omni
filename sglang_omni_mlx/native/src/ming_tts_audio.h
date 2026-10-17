// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <optional>
#include <vector>

#include "layers.h"

namespace ming_tts {

struct AudioCache {
  std::optional<mlx::core::array> keys, values;
  int offset = 0;
};

class AudioTransformer {
public:
  AudioTransformer(const qwen3_asr::Checkpoint &checkpoint, std::string prefix,
                   const nlohmann::json &config);
  mlx::core::array Forward(mlx::core::array hidden,
                           std::vector<AudioCache> &caches) const;
  std::vector<AudioCache> NewCaches() const;
  int hidden_size() const { return hidden_size_; }

private:
  const qwen3_asr::Checkpoint &checkpoint_;
  std::string prefix_;
  int hidden_size_, layers_, heads_, kv_heads_;
  float epsilon_, rope_theta_;
  std::vector<int> windows_;
};

struct AudioDecoderState {
  std::vector<AudioCache> caches;
  std::optional<mlx::core::array> pending, left, audio_overlap,
      envelope_overlap;
};

class MingAudioDecoder {
public:
  explicit MingAudioDecoder(const std::filesystem::path &directory);
  mlx::core::array DecodeFull(const mlx::core::array &latents) const;
  mlx::core::array DecodeChunk(const mlx::core::array &latents,
                               AudioDecoderState &state, bool last_chunk) const;
  int sample_rate() const { return sample_rate_; }

private:
  MingAudioDecoder(const nlohmann::json &config,
                   const std::filesystem::path &directory);
  mlx::core::array Interpolate(const mlx::core::array &hidden) const;
  mlx::core::array Synthesize(const mlx::core::array &hidden,
                              AudioDecoderState &state, bool streaming,
                              bool last_chunk) const;
  int patch_size_, hop_length_, latent_dim_, sample_rate_;
  qwen3_asr::Checkpoint checkpoint_;
  AudioTransformer transformer_;
};

} // namespace ming_tts
