// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team.
// AuK's Qwen2.5-Omni text and reference-audio conditioning.
#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "layers.h"
#include "tokenizer.h"

namespace auk {

class Conditioner {
public:
  // Original thinker weights, with or without the thinker. prefix. Convolution
  // weights retain the checkpoint's [output, input, kernel] layout on disk.
  Conditioner(const std::filesystem::path &conditioner_directory,
              mlx::core::Dtype dtype);
  int TextLayerCount() const { return config_.text_layer_count; }
  int TextHiddenChannels() const { return config_.hidden_size; }

  // All text layers, including embeddings and the final normalized state:
  // [layers + 1, tokens, hidden]. Reference audio is mono at 16 kHz.
  mlx::core::array Encode(const std::string &instruction,
                          const std::optional<std::vector<float>>
                              &reference_audio = std::nullopt) const;

  std::vector<int> Tokenize(const std::string &instruction,
                            int audio_frame_count = 0) const;
  // Explicit processor outputs for parity checks; features contain only valid
  // mel frames, [128, frames], for at most one reference recording.
  mlx::core::array EncodeInputs(
      const std::vector<int> &token_ids, const std::vector<int> &attention_mask,
      const std::optional<mlx::core::array> &features = std::nullopt) const;
  mlx::core::array EncodeAudio(const mlx::core::array &features) const;

private:
  struct Config {
    int hidden_size;
    int vocabulary_size;
    int text_layer_count;
    int text_head_count;
    int text_kv_head_count;
    int text_intermediate_size;
    int max_tokens;
    float rms_norm_epsilon;
    float rope_theta;
    int sliding_window;
    std::vector<bool> sliding_layers;
    int audio_hidden_size;
    int audio_layer_count;
    int audio_head_count;
    int audio_intermediate_size;
    int audio_window_frames;
    int max_audio_samples;
    int audio_token_id;
    int image_token_id;
    int video_token_id;
    static Config Read(const std::filesystem::path &directory);
  };

  mlx::core::array TextLayer(const mlx::core::array &hidden, int layer,
                             const mlx::core::array &mask,
                             const mlx::core::array &cosine,
                             const mlx::core::array &sine) const;

  Config config_;
  qwen3_asr::Checkpoint checkpoint_;
  qwen3_asr::Tokenizer tokenizer_;
};

} // namespace auk
