// SPDX-License-Identifier: Apache-2.0
// Voxtral audio-code decoder.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "common/layers.h"
#include "mlx/mlx.h"
#include "voxtral_tts/config.h"

namespace voxtral_tts {

class AudioTokenizerDecoder {
public:
  AudioTokenizerDecoder(const AudioTokenizerConfig &config,
                        std::unique_ptr<qwen3_asr::Checkpoint> checkpoint);

  mlx::core::array Decode(const mlx::core::array &audio_codes) const;
  int sampling_rate_hz() const;
  int downsample_factor() const;

private:
  struct TransformerStack {
    int decoder_block_index;
    int layer_count;
    int window_size;
  };

  mlx::core::array CausalConv(const mlx::core::array &hidden_states,
                              const std::string &weight_prefix, int kernel_size,
                              int stride,
                              const std::string &padding_mode) const;
  mlx::core::array CausalConvTranspose(const mlx::core::array &hidden_states,
                                       const std::string &weight_prefix,
                                       int kernel_size, int stride) const;
  mlx::core::array RunTransformerStack(const mlx::core::array &hidden_states,
                                       const TransformerStack &stack) const;

  AudioTokenizerConfig config_;
  std::unique_ptr<qwen3_asr::Checkpoint> checkpoint_;
  mlx::core::array semantic_embedding_;
  std::vector<TransformerStack> stacks_;
  std::vector<int> transposed_convolution_block_indices_;
  int downsample_factor_;
};

} // namespace voxtral_tts
