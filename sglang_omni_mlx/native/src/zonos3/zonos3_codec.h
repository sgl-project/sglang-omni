// SPDX-License-Identifier: Apache-2.0
// The ZONOS3 audio codec's decoder: a semantic and residual codebook sum, five
// causal windowed-attention transformer stages that upsample 15 Hz frames to
// 240 Hz, a ConvNeXt head, a two-resolution iSTFT mixed by a FIR filter, and a
// 44160 to 44100 Hz polyphase resampler. Decode runs a whole utterance;
// CodecStream decodes frames as they come, carrying each layer's state.
#pragma once

#include <filesystem>
#include <memory>
#include <optional>
#include <vector>

#include "mlx/mlx.h"
#include "zonos3/zonos3_istft_stream.h"
#include "zonos3/zonos3_weights.h"

namespace zonos3 {

struct CodecStage {
  int output_size = 0;
  int model_size = 0;
  int head_count = 0;
  int layer_count = 0;
  // Earlier positions a query attends to, itself included.
  int context = 0;
  // Positions each output row unfolds into.
  int patch = 0;
};

class CodecDecoder {
public:
  // Reads codec_directory/config.json and model.safetensors; throws
  // std::runtime_error unless they are the 44.1 kHz, 15 Hz, ten-codebook
  // iSTFT decoder.
  explicit CodecDecoder(const std::filesystem::path &codec_directory);

  int sample_rate() const { return sample_rate_; }
  int codebook_count() const { return codebook_count_; }
  int codebook_size() const { return codebook_size_; }
  // Output samples per frame.
  int hop() const { return hop_; }
  // The gain of headroom_db, or of the codec's own headroom when unset.
  float OutputGain(std::optional<double> headroom_db) const;

  // Float32 samples [frames * hop] of int32 codes [frames, codebooks], before
  // the output gain.
  mlx::core::array Decode(const mlx::core::array &codes) const;

private:
  friend class CodecStream;

  // x [rows, channels] through a causal ConvNeXt block; history holds the
  // kernel - 1 rows before x, zeros when unset.
  mlx::core::array
  ConvNextBlock(const mlx::core::array &x, const std::string &prefix,
                const std::optional<mlx::core::array> &history) const;
  // The latent [frames, latent] of codes [frames, codebooks] before the
  // transformer stages; histories, when given, carry the semantic head's
  // ConvNeXt inputs across calls.
  mlx::core::array
  Latent(const mlx::core::array &codes,
         std::vector<std::optional<mlx::core::array>> *histories) const;
  // The two iSTFT branches' spectra [head rows, fft size + 2] of the head's
  // input [head rows, width]; histories as for Latent.
  std::pair<mlx::core::array, mlx::core::array>
  Head(mlx::core::array x,
       std::vector<std::optional<mlx::core::array>> *histories) const;
  // One whole-utterance transformer layer of a stage.
  mlx::core::array TransformerLayer(const mlx::core::array &x,
                                    const std::string &prefix,
                                    const CodecStage &stage) const;
  // One branch's inverse STFT, centered, frames * head hop samples.
  mlx::core::array Istft(const mlx::core::array &spectrum, int branch) const;

  Weights weights_;
  int sample_rate_;
  int hop_;
  int codebook_count_;
  int codebook_size_;
  int semantic_depth_;
  int head_depth_;
  int head_hop_;
  std::vector<int> fft_sizes_;
  float max_log_magnitude_;
  double headroom_db_;
  float layer_norm_epsilon_;
  float convnext_epsilon_;
  int resample_from_;
  int resample_to_;
  int resample_width_;
  std::vector<CodecStage> stages_;
  std::vector<mlx::core::array> windows_;
  mlx::core::array mixer_;
  mlx::core::array resample_kernel_;
  IstftHeadConfig head_config_;
};

class CodecStream {
public:
  explicit CodecStream(const CodecDecoder &decoder);

  // The samples final after codes [frames, codebooks] int32 follow what came
  // before: 2940 * frames - 1470 of the frames decoded so far in all.
  std::vector<float> Decode(const mlx::core::array &codes);
  // The samples left after the last frame.
  std::vector<float> Flush();

private:
  std::vector<float> DecodeChunk(const mlx::core::array &codes);

  const CodecDecoder &decoder_;
  int frames_ = 0;
  std::vector<std::optional<mlx::core::array>> semantic_histories_;
  std::vector<std::optional<mlx::core::array>> head_histories_;
  // Per stage and layer, the keys and values [heads, positions, head dim] of
  // the last context - 1 positions.
  std::vector<
      std::vector<std::optional<std::pair<mlx::core::array, mlx::core::array>>>>
      caches_;
  IstftHeadStream head_;
  ResamplerStream resampler_;
};

} // namespace zonos3
