// SPDX-License-Identifier: Apache-2.0
// SenseVoice Small on MLX: the Kaldi fbank front end with low frame rate
// stacking and CMVN, the SAN-M encoder, and the CTC head, as Voxt's Swift
// port computes them in float32.
#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "layers.h"
#include "mlx/mlx.h"

namespace sensevoice {

struct SenseVoiceConfig {
  int vocabulary_size = 0;
  // Width of the stacked features and of the first encoder layer's input.
  int input_size = 0;
  int model_width = 0;
  int head_count = 0;
  int feed_forward_width = 0;
  // The first layer plus the main stack, then the stack after after_norm.
  int block_count = 0;
  int tp_block_count = 0;
  int kernel_size = 0;
  int sanm_shift = 0;
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
  // Log probabilities [4 + frames, vocabulary] of features [frames,
  // input_size]: the language, emotion, event and text normalization query
  // frames first, then the speech.
  mlx::core::array LogProbabilities(const mlx::core::array &features,
                                    int language_id, bool use_itn) const;

  const SenseVoiceConfig &config() const { return config_; }

private:
  SenseVoiceModel(const nlohmann::json &config,
                  const std::filesystem::path &model_directory);
  mlx::core::array Fbank(const std::vector<float> &samples) const;
  mlx::core::array EncoderLayer(const mlx::core::array &x,
                                const std::string &prefix, bool residual) const;

  SenseVoiceConfig config_;
  qwen3_asr::Checkpoint checkpoint_;
  std::vector<float> window_;
  mlx::core::array mel_filters_;
  std::optional<mlx::core::array> cmvn_means_;
  std::optional<mlx::core::array> cmvn_inverse_stddevs_;
};

} // namespace sensevoice
