// SPDX-License-Identifier: Apache-2.0
// MOSS-Transcribe-Diarize on MLX: Whisper encoder, frame-merging adaptor and
// Qwen3 text decoder, loaded from the checkpoint as published.
#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "layers.h"
#include "mlx/mlx.h"
#include "qwen3_decoder.h"

namespace moss {

struct WhisperEncoderConfig {
  int num_mel_bins = 0;
  int encoder_layers = 0;
  int encoder_attention_heads = 0;
};

class MossTranscribeDiarize {
public:
  explicit MossTranscribeDiarize(const std::filesystem::path &model_directory);

  // Window features [windows, frames, mel_bins] to audio token embeddings
  // [tokens, hidden], window_token_counts[i] tokens from window i.
  // Note (Dayuxiaoshui): the counts follow each window's own samples, so a
  // short last window contributes fewer tokens than a full one.
  mlx::core::array
  EncodeAudio(const mlx::core::array &window_features,
              const std::vector<int> &window_token_counts) const;
  const qwen3_asr::Qwen3Decoder &decoder() const { return decoder_; }

  int mel_bin_count() const { return whisper_.num_mel_bins; }
  int audio_token_id() const { return audio_token_id_; }
  int audio_merge_size() const { return audio_merge_size_; }

private:
  MossTranscribeDiarize(const nlohmann::json &config,
                        const std::filesystem::path &model_directory);
  mlx::core::array Conv1d(const mlx::core::array &x, const std::string &prefix,
                          int stride) const;
  mlx::core::array WhisperEncoderLayer(const mlx::core::array &x,
                                       int layer) const;

  WhisperEncoderConfig whisper_;
  float adaptor_norm_eps_ = 0.0f;
  int audio_token_id_ = 0;
  int audio_merge_size_ = 0;
  qwen3_asr::Checkpoint checkpoint_;
  qwen3_asr::Qwen3Decoder decoder_;
};

} // namespace moss
