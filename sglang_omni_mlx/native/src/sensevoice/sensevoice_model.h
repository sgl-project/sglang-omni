// SPDX-License-Identifier: Apache-2.0
// SenseVoice Small on MLX: the Kaldi fbank front end with low frame rate
// stacking and CMVN, the SAN-M encoder, and the CTC head, as Voxt's Swift
// port computes them in float32.
#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "common/layers.h"
#include "common/sanm_encoder.h"
#include "mlx/mlx.h"

namespace sensevoice {

struct SenseVoiceConfig {
  int vocabulary_size = 0;
  omni_common::SanmEncoderConfig encoder;
  int sample_rate = 0;
  int mel_count = 0;
  int frame_length_ms = 0;
  int frame_shift_ms = 0;
  // Low frame rate stacking: lfr_m frames every lfr_n.
  int lfr_m = 0;
  int lfr_n = 0;
};

class SenseVoiceModel {
public:
  // Reads config.json, model.safetensors and the CMVN statistics, from
  // am.mvn or else the config.
  explicit SenseVoiceModel(const std::filesystem::path &model_directory);

  // Stacked, normalized fbank frames [frames, input_size] of 16 kHz samples;
  // no frames for audio shorter than one 25 ms window.
  mlx::core::array Features(const std::vector<float> &samples) const;
  // CTC logits [4 + frames, vocabulary] of features [frames,
  // input_size]: the language, emotion, event and text normalization query
  // frames first, then the speech.
  mlx::core::array Logits(const mlx::core::array &features, int language_id,
                          bool use_itn) const;

  const SenseVoiceConfig &config() const { return config_; }

private:
  SenseVoiceModel(const nlohmann::json &config,
                  const std::filesystem::path &model_directory);

  SenseVoiceConfig config_;
  omni_common::Checkpoint checkpoint_;
  std::vector<float> window_;
  mlx::core::array mel_filters_;
  std::optional<mlx::core::array> cmvn_means_;
  std::optional<mlx::core::array> cmvn_inverse_stddevs_;
};

} // namespace sensevoice
