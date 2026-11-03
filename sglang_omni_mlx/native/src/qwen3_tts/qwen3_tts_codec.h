// SPDX-License-Identifier: Apache-2.0
// Qwen3-TTS's 12 Hz speech tokenizer decoder (16 codebooks to 24 kHz
// samples) and the chunked schedule that decodes frames as generation
// produces them.
#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "common/layers.h"
#include "mlx/mlx.h"
#include "nlohmann/json.hpp"

namespace qwen3_tts {

class SpeechDecoder {
public:
  // Reads speech_tokenizer/config.json and its safetensors.
  explicit SpeechDecoder(
      const std::filesystem::path &speech_tokenizer_directory);

  // Codes [frames, codebooks] to float32 samples [frames * samples_per_frame].
  mlx::core::array Decode(const mlx::core::array &codes) const;
  int samples_per_frame() const { return samples_per_frame_; }
  int sample_rate() const { return sample_rate_; }
  int codebook_count() const { return codebook_count_; }

private:
  SpeechDecoder(const nlohmann::json &config, omni_common::WeightMap weights);
  // Codebook embeddings from their usage counts, and convolution kernels in
  // MLX's layout.
  static omni_common::WeightMap PrepareWeights(omni_common::WeightMap weights);
  mlx::core::array TransformerLayer(const mlx::core::array &x, int layer,
                                    const mlx::core::array &cosines,
                                    const mlx::core::array &sines,
                                    const mlx::core::array &mask) const;
  mlx::core::array RmsNorm(const mlx::core::array &x,
                           const std::string &prefix) const;
  mlx::core::array Snake(const mlx::core::array &x,
                         const std::string &prefix) const;
  // A causal convolution: left padding of the dilated kernel's reach.
  mlx::core::array CausalConvolution(const mlx::core::array &x,
                                     const std::string &prefix, int dilation,
                                     int groups) const;
  // A transposed convolution trimmed on the right to stride times its input.
  mlx::core::array TransposedConvolution(const mlx::core::array &x,
                                         const std::string &prefix,
                                         int stride) const;

  omni_common::Checkpoint checkpoint_;
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

// Frames decoded in chunks, each with up to 25 frames of left context.
// Note (Jiaxin Deng): a whole request decodes in the CUDA vocoder's 300-frame
// chunks and returns every sample at Finish; a streamed one decodes a first
// chunk (5 frames by default), then 25-frame chunks, so its first audio comes
// early and its samples differ only near chunk edges.
class SpeechStream {
public:
  // first_chunk_frames sets a streamed request's first chunk, at most the
  // steady chunk; 0 starts at the steady chunk.
  SpeechStream(const SpeechDecoder &decoder, bool streamed,
               int first_chunk_frames = 5);

  // Adds one frame's codes [codebooks]; returns the samples a completed
  // chunk adds, empty until one does.
  std::vector<float> Push(const mlx::core::array &frame_codes);
  // Decodes every frame not yet returned.
  std::vector<float> Finish();

private:
  std::vector<float> DecodeChunk(int end_frame);
  int ChunkFrames() const;

  const SpeechDecoder &decoder_;
  const bool streamed_;
  const int first_chunk_frames_;
  std::vector<mlx::core::array> frames_;
  int emitted_frames_ = 0;
  // Frames whose first code is not 0.
  int voiced_frames_ = 0;
  // A whole request's decoded samples, returned by Finish.
  std::vector<float> held_samples_;
};

} // namespace qwen3_tts
