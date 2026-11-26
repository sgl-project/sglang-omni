// SPDX-License-Identifier: Apache-2.0
// Fun-ASR-Nano on MLX: the Kaldi fbank stacked at a low frame rate, the SAN-M
// encoder SenseVoice shares, a transformer adaptor and the Qwen3 text decoder,
// loaded from the Hugging Face checkpoint as published.
#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "common/layers.h"
#include "common/qwen3_decoder.h"
#include "common/sanm_encoder.h"
#include "mlx/mlx.h"

namespace funasr {

class FunAsrNano {
public:
  explicit FunAsrNano(const std::filesystem::path &model_directory);

  // Stacked fbank frames [frames, input_size] of 16 kHz samples; none for
  // audio shorter than one window.
  mlx::core::array Features(const std::vector<float> &samples) const;
  // Audio token embeddings [ceil(frames / 8), hidden] of features
  // [frames, input_size].
  mlx::core::array EncodeAudio(const mlx::core::array &features) const;
  const omni_common::Qwen3Decoder &decoder() const { return decoder_; }

private:
  FunAsrNano(const nlohmann::json &config,
             const nlohmann::json &feature_extractor,
             const std::filesystem::path &model_directory);
  mlx::core::array AdaptorLayer(const mlx::core::array &x,
                                const std::string &prefix) const;

  omni_common::SanmEncoderConfig encoder_;
  int lfr_m_ = 0;
  int lfr_n_ = 0;
  int hop_length_ = 0;
  int adaptor_layer_count_ = 0;
  int adaptor_head_count_ = 0;
  float adaptor_norm_eps_ = 0.0f;
  std::vector<float> window_;
  mlx::core::array mel_filters_;
  omni_common::Checkpoint checkpoint_;
  omni_common::Qwen3Decoder decoder_;
};

} // namespace funasr
