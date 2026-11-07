// SPDX-License-Identifier: Apache-2.0
// Qwen3-TTS's 12 Hz speech tokenizer decoder (16 codebooks to 24 kHz
// samples) and the chunked schedule that decodes frames as generation
// produces them.
#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "common/layers.h"
#include "mlx/mlx.h"
#include "nlohmann/json.hpp"

namespace qwen3_tts {

inline constexpr int kDefaultFirstChunkFrames = 5;
inline constexpr int kStreamedChunkFrames = 25;

struct SpeechDecoderState {
  int decoded_frames = 0;
  std::vector<std::optional<std::pair<mlx::core::array, mlx::core::array>>>
      attention;
  std::unordered_map<std::string, mlx::core::array> convolution_history;
};

class SpeechDecoder {
public:
  // Reads speech_tokenizer/config.json and its safetensors.
  explicit SpeechDecoder(
      const std::filesystem::path &speech_tokenizer_directory,
      bool incremental_streaming);

  // Codes [frames, codebooks] to float32 samples [frames * samples_per_frame].
  mlx::core::array Decode(const mlx::core::array &codes) const;
  // Keep front-end shapes fixed and upsample real prefixes with a separate
  // state.
  mlx::core::array DecodeFeatures(const mlx::core::array &codes,
                                  SpeechDecoderState *state) const;
  mlx::core::array DecodeSamples(const mlx::core::array &features, int frames,
                                 SpeechDecoderState *state) const;
  std::optional<SpeechDecoderState> NewStreamState() const;
  int samples_per_frame() const { return samples_per_frame_; }
  int sample_rate() const { return sample_rate_; }
  int codebook_count() const { return codebook_count_; }

private:
  SpeechDecoder(const nlohmann::json &config, omni_common::WeightMap weights,
                bool incremental_streaming);
  // Codebook embeddings from their usage counts, and convolution kernels in
  // MLX's layout.
  static omni_common::WeightMap PrepareWeights(omni_common::WeightMap weights);
  mlx::core::array TransformerLayer(const mlx::core::array &x, int layer,
                                    const mlx::core::array &cosines,
                                    const mlx::core::array &sines,
                                    const mlx::core::array &mask,
                                    SpeechDecoderState *state) const;
  mlx::core::array RmsNorm(const mlx::core::array &x,
                           const std::string &prefix) const;
  mlx::core::array Snake(const mlx::core::array &x,
                         const std::string &prefix) const;
  // A causal convolution: left padding of the dilated kernel's reach.
  mlx::core::array CausalConvolution(const mlx::core::array &x,
                                     const std::string &prefix, int dilation,
                                     int groups,
                                     SpeechDecoderState *state) const;
  // A transposed convolution trimmed on the right to stride times its input.
  mlx::core::array TransposedConvolution(const mlx::core::array &x,
                                         const std::string &prefix, int stride,
                                         SpeechDecoderState *state) const;

  omni_common::Checkpoint checkpoint_;
  bool incremental_streaming_;
  int codebook_count_;
  int semantic_codebook_count_;
  int head_count_;
  int head_dim_;
  int transformer_layer_count_;
  float rms_norm_eps_;
  // Each frame attends to itself and the frames before it within this many.
  int sliding_window_frames_;
  std::vector<int> upsampling_ratios_;
  std::vector<int> upsample_rates_;
  int samples_per_frame_;
  int sample_rate_;
  mlx::core::array inverse_frequencies_;
};

// One request owns its codec history and the position of its next audio packet.
// Note (Jiaxin Deng): a whole request decodes in the CUDA vocoder's 300-frame
// chunks and returns every sample at Finish; a streamed one decodes a first
// chunk (5 frames by default), then 25-frame chunks, so its first audio comes
// early and its samples differ only near chunk edges.
class SpeechStream {
public:
  // first_chunk_frames sets a streamed request's first chunk, at most the
  // steady chunk; 0 starts at the steady chunk.
  SpeechStream(const SpeechDecoder &decoder, bool streamed,
               int first_chunk_frames = kDefaultFirstChunkFrames);

  // Adds one frame's codes [codebooks]; returns the samples a completed
  // chunk adds, empty until one does.
  std::vector<float> Push(const mlx::core::array &frame_codes);
  // Decodes every frame not yet returned.
  std::vector<float> Finish();

private:
  std::vector<float> DecodeChunk(int end_frame);
  std::vector<float> DecodeIncremental(int end_frame);
  int ChunkFrames() const;

  const SpeechDecoder &decoder_;
  const bool streamed_;
  const int first_chunk_frames_;
  std::optional<SpeechDecoderState> feature_state_;
  std::optional<SpeechDecoderState> wave_state_;
  std::vector<mlx::core::array> frames_;
  int emitted_frames_ = 0;
  // Frames whose first code is not 0.
  int voiced_frames_ = 0;
  // A whole request's decoded samples, returned by Finish.
  std::vector<float> held_samples_;
};

} // namespace qwen3_tts
