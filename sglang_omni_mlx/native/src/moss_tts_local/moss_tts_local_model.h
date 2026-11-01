// SPDX-License-Identifier: Apache-2.0
// MOSS-TTS-Local v1.5: a Qwen3 backbone over the prompt rows and each
// generated frame, a one-layer local transformer that decides whether audio
// continues and samples the frame's codebooks in order, and the codec that
// turns frames into samples.
#pragma once

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "common/layers.h"
#include "common/tokenizer.h"
#include "mlx/mlx.h"
#include "moss_tts_local/moss_tts_local_codec.h"
#include "qwen3_asr/qwen3_asr_transcriber.h"
#include "qwen3_asr/qwen3_decoder.h"

namespace moss_tts_local {

// One head's sampling: temperature 0 samples greedily; top_k <= 0 and
// top_p 1 keep every candidate.
struct SamplingParameters {
  double temperature = 1.0;
  double top_p = 1.0;
  int top_k = 0;
  double repetition_penalty = 1.0;
};

struct GenerationParameters {
  int max_frames = 4096;
  // The continue-or-stop decision, and every audio codebook.
  SamplingParameters text;
  SamplingParameters audio;
  uint64_t seed = 0;
};

// The prompt fields the checkpoint's user template takes; empty ones render
// as None.
struct SpeechPrompt {
  std::string text;
  std::optional<std::string> instruction;
  std::optional<int> token_count;
  std::optional<std::string> language;
};

// The checkpoint's user and assistant turns around a request's fields, as
// prompt rows [rows, 1 + codebooks]: a text id, then the audio pad code in
// every codebook.
class PromptEncoder {
public:
  explicit PromptEncoder(const std::filesystem::path &model_directory);

  std::vector<std::vector<int>> Rows(const SpeechPrompt &prompt) const;

private:
  omni_common::Tokenizer tokenizer_;
  int codebook_count_;
  int audio_pad_code_;
  int im_start_id_;
  int im_end_id_;
  int audio_start_id_;
};

class MossTtsLocalModel {
public:
  // quantization_bits 4 or 8 quantizes the backbone and local transformer at
  // load, in groups of 64; 16 keeps the checkpoint's bfloat16.
  MossTtsLocalModel(const std::filesystem::path &model_directory,
                    const std::filesystem::path &codec_directory,
                    int quantization_bits);

  const AudioCodec &codec() const { return codec_; }
  const PromptEncoder &prompt_encoder() const { return prompt_encoder_; }

  // Generates up to max_frames frames after the prompt rows, passing each
  // frame's evaluated int32 codes [codebooks] to on_frame. Stops when the
  // local transformer samples the end of audio; throws TranscriptionCancelled
  // once cancel is set.
  qwen3_asr::FinishReason
  Generate(const std::vector<std::vector<int>> &prompt_rows,
           const GenerationParameters &parameters,
           const std::atomic<bool> &cancel,
           const std::function<void(const mlx::core::array &)> &on_frame) const;

private:
  struct LoadedWeights {
    omni_common::WeightMap weights;
    nlohmann::json quantization;
  };
  MossTtsLocalModel(const std::filesystem::path &model_directory,
                    const std::filesystem::path &codec_directory,
                    const nlohmann::json &config, LoadedWeights weights);
  static LoadedWeights LoadWeights(const std::filesystem::path &directory,
                                   int quantization_bits);

  // Rows [1, rows, 1 + codebooks] to the backbone's input embeddings: the
  // text embedding plus every codebook's embedding where it is not padding.
  mlx::core::array EmbedRows(const mlx::core::array &rows) const;
  // The local transformer over its inputs [1, length, hidden]; returns the
  // last position's hidden state [1, hidden].
  mlx::core::array LocalLast(const mlx::core::array &inputs) const;
  mlx::core::array LocalAttention(const mlx::core::array &x,
                                  const std::string &prefix) const;

  omni_common::Checkpoint checkpoint_;
  AudioCodec codec_;
  PromptEncoder prompt_encoder_;
  qwen3_asr::Qwen3Decoder backbone_;
  float backbone_norm_eps_;
  int local_layer_count_;
  int local_head_count_;
  float local_norm_eps_;
  float local_rope_base_;
  int codebook_count_;
  int audio_pad_code_;
  int audio_end_id_;
  int audio_assistant_slot_id_;
};

} // namespace moss_tts_local
