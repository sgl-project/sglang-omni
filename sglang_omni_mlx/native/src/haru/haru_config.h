// SPDX-License-Identifier: Apache-2.0
// Haru-4B configuration: the structs of config.json's "codec", "head" and
// "lm" sections, and their derived values.
#pragma once

#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace haru {

inline constexpr const char *kLinear = "linear_attention";
inline constexpr const char *kFull = "full_attention";

// VibeVoice acoustic codec: 24 kHz waveform <-> 64-dim latents at 7.5 Hz.
struct CodecConfig {
  int latent_dim = 64;
  int base_channels = 32;
  std::vector<int> ratios = {8, 5, 5, 4, 2, 2};      // upsampling order
  std::vector<int> encoder_depths = {3, 3, 3, 3, 3, 3, 8};
  std::vector<int> decoder_depths = {8, 3, 3, 3, 3, 3, 3};
  float norm_eps = 1e-5f;
  float latent_std = 0.5f;

  int frame_samples() const {
    int samples = 1;
    for (int r : ratios) samples *= r;
    return samples; // 3200
  }
};

// MeanFlow head: LM hidden state -> next audio latent.
struct HeadConfig {
  int hidden_size = 2560;
  int latent_dim = 64;
  int num_layers = 4;
  float ffn_ratio = 3.0f;
  float norm_eps = 1e-5f;
  float interval_embed = 7.0f;
  int num_steps = 1;
  float guidance_scale = 1.5f;
  std::pair<float, float> guidance_range = {1.0f, 2.0f};

  int ffn_dim() const { return static_cast<int>(hidden_size * ffn_ratio); }
};

// Qwen3.5 text backbone: Gated DeltaNet layers with periodic full attention.
struct TextConfig {
  int vocab_size = 0;
  int hidden_size = 0;
  int intermediate_size = 0;
  int num_attention_heads = 0;
  int num_key_value_heads = 0;
  int head_dim = 0;
  int linear_num_key_heads = 0;
  int linear_num_value_heads = 0;
  int linear_key_head_dim = 0;
  int linear_value_head_dim = 0;
  int linear_conv_kernel_dim = 0;
  std::vector<std::string> layer_types; // per layer, kLinear or kFull
  float rms_norm_eps = 1e-6f;
  float rope_theta = 10'000'000.0f;
  float partial_rotary_factor = 0.25f;

  int key_dim() const { return linear_num_key_heads * linear_key_head_dim; }
  int value_dim() const { return linear_num_value_heads * linear_value_head_dim; }
  // Width of the Gated DeltaNet convolution: queries, keys and values.
  int conv_dim() const { return 2 * key_dim() + value_dim(); }
  int num_layers() const { return static_cast<int>(layer_types.size()); }
};

struct HaruConfig {
  int sample_rate = 24'000;
  int frame_samples = 3200;
  int latent_dim = 64;
  int pad_token_id = 248076;
  int eop_token_id = 248071;
  TextConfig lm;
  CodecConfig codec;
  HeadConfig head;
  int eos_token_id = -1; // -1 when the checkpoint has none

  float frame_rate() const { return static_cast<float>(sample_rate) / frame_samples; }
  int hidden_size() const { return lm.hidden_size; }

  static HaruConfig FromFile(const std::filesystem::path &path);
  static HaruConfig FromDict(const nlohmann::json &params);
};

} // namespace haru
