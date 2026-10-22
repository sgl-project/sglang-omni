// SPDX-License-Identifier: Apache-2.0
// Voxtral TTS checkpoint configuration.
#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "nlohmann/json.hpp"

namespace voxtral_tts {

struct TextConfig {
  int dimension;
  int layer_count;
  int head_dimension;
  int hidden_dimension;
  int head_count;
  int kv_head_count;
  int vocabulary_size;
  float rope_theta;
  float norm_epsilon;
};

struct AcousticConfig {
  int input_dimension;
  int dimension;
  int layer_count;
  int head_dimension;
  int hidden_dimension;
  int head_count;
  int kv_head_count;
  bool use_biases;
  float norm_epsilon;
};

struct AudioModelConfig {
  int semantic_codebook_size;
  int acoustic_codebook_size;
  int acoustic_codebook_count;
  AcousticConfig transformer;
  int audio_token_id;
};

struct AudioTokenizerConfig {
  int sampling_rate_hz;
  int patch_size;
  int patch_projection_kernel_size;
  int semantic_codebook_size;
  int semantic_dimension;
  int acoustic_codebook_size;
  int acoustic_dimension;
  bool causal;
  int sliding_window_size;
  bool half_attention_window_on_downsample;
  std::vector<int> encoder_convolution_strides;
  int dimension;
  int hidden_dimension;
  int head_dimension;
  int head_count;
  int kv_head_count;
  float qk_norm_epsilon;
  bool qk_norm;
  bool use_biases;
  float norm_epsilon;
  bool layer_scale;
  std::optional<float> layer_scale_initial;
  std::vector<int> transformer_lengths;
  std::vector<int> convolution_kernels;
  std::vector<int> convolution_strides;
};

struct VoxtralConfig {
  TextConfig text;
  AudioModelConfig audio;
  AudioTokenizerConfig audio_tokenizer;
  std::unordered_map<std::string, int> voices;
  std::filesystem::path checkpoint_directory;

  static VoxtralConfig Load(const std::filesystem::path &checkpoint_directory);
};

} // namespace voxtral_tts
