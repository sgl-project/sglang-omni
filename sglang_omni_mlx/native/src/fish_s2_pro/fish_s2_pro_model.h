// SPDX-License-Identifier: Apache-2.0
// Fish S2-Pro: a dual autoregressive TTS model. The slow transformer reads the
// chat prompt and each generated frame and samples the frame's semantic code;
// the fast transformer then picks the frame's residual codes greedily, and the
// DAC codec turns frames into 44.1 kHz samples.
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <string>
#include <vector>

#include "common/layers.h"
#include "common/tokenizer.h"
#include "common/transcription.h"
#include "fish_dac/fish_dac.h"
#include "mlx/mlx.h"
#include "qwen3_asr/qwen3_decoder.h"

namespace fish_s2_pro {

// Codes per residual codebook; the semantic codebook has the model's 4096.
inline constexpr int kResidualCodebookSize = 1024;
// The buffers MLX keeps for reuse during a request; the worker clears them once
// idle, so an idle server holds only the model.
inline constexpr size_t kCacheLimitBytes = 2ULL * 1024 * 1024 * 1024;
// Note (Jiaxin Deng): the CUDA server's default (stages.py) and its engine's
// cap on max_new_tokens, which it applies without an error.
inline constexpr int kDefaultMaxFrames = 1024;
inline constexpr int kMaxFrames = 2048;
inline constexpr uint32_t kSeedMask = 0x7FFFFFFF;

// The CUDA server's sampling of the semantic code; seed is the request's
// seed already masked to 31 bits.
struct SamplingOptions {
  float temperature = 0.8f;
  float top_p = 0.8f;
  // -1 keeps the sampler's full 30 candidates.
  int top_k = 30;
  float repetition_penalty = 1.1f;
  uint32_t seed = 0;
};

// A voice to clone: its transcript (may be empty) and its codes
// [codebooks, frames].
struct VoiceReference {
  std::string text;
  mlx::core::array codes;
};

// Prompt ids and, for the reference speech's positions, the codes their
// embeddings mix in, [codebooks, audio positions].
struct SpeechPrompt {
  std::vector<int> ids;
  std::vector<bool> is_audio;
  std::vector<mlx::core::array> audio_codes;
};

class FishS2ProModel {
public:
  explicit FishS2ProModel(const std::filesystem::path &model_directory);

  const fish_dac::AudioCodec &codec() const { return codec_; }
  int codebook_count() const { return codebook_count_; }
  // Codes per codebook the prompt's audio positions take.
  int codebook_size() const { return codebook_size_; }
  // The positions the slow transformer takes, max_seq_len in config.json.
  int max_positions() const { return max_positions_; }

  // The CUDA server's chat prompt: a system turn with the references'
  // transcripts and speech when there are any, the text as speaker 0's user
  // turn, then the assistant's voice turn.
  SpeechPrompt BuildPrompt(const std::string &text,
                           const std::vector<VoiceReference> &references) const;

  // Generates up to max_frames frames, or as many as fit in max_positions()
  // after the prompt, passing each frame's evaluated int32 codes [codebooks] to
  // on_frame. Stops at <|im_end|>; throws TranscriptionCancelled once cancel is
  // set, and std::invalid_argument for a prompt that leaves no position.
  omni_common::FinishReason
  Generate(const SpeechPrompt &prompt, int max_frames,
           const SamplingOptions &sampling, const std::atomic<bool> &cancel,
           const std::function<void(const mlx::core::array &)> &on_frame) const;

private:
  struct TransformerConfig {
    std::string prefix;
    int layer_count;
    int head_count;
    int key_value_head_count;
    int head_dim;
    float norm_eps;
    float rope_base;
    bool has_query_key_norm;
  };

  FishS2ProModel(const std::filesystem::path &model_directory,
                 const nlohmann::json &config);

  // Rows of prompt ids [1, length] with each audio position's codebook
  // embeddings mixed in; codebook_rows [codebooks, length] are 0 elsewhere.
  mlx::core::array Embed(const mlx::core::array &ids,
                         const mlx::core::array &codebook_rows,
                         const mlx::core::array &is_audio) const;
  // Runs the layers of config over x, appending to caches; returns the
  // final-normed hidden state of the last position, [1, dim].
  mlx::core::array Forward(const mlx::core::array &x,
                           const TransformerConfig &config,
                           std::vector<qwen3_asr::KVCache> &caches) const;
  // The residual codes after semantic_code, chosen greedily from hidden; caches
  // are the fast transformer's, reused from frame to frame.
  std::vector<mlx::core::array>
  ResidualCodes(const mlx::core::array &hidden,
                const mlx::core::array &semantic_code,
                std::vector<qwen3_asr::KVCache> &caches) const;
  // The semantic token for candidate logits (semantic codes, then
  // <|im_end|>), as the CUDA server's sampler draws it.
  int SampleSemanticToken(const float *candidate_logits,
                          const std::vector<int> &history, int step,
                          const SamplingOptions &sampling) const;

  omni_common::Checkpoint checkpoint_;
  fish_dac::AudioCodec codec_;
  omni_common::Tokenizer tokenizer_;
  TransformerConfig slow_;
  TransformerConfig fast_;
  int codebook_count_;
  int codebook_size_;
  int max_positions_;
  int semantic_begin_id_;
  int semantic_end_id_;
  int im_end_id_;
};

} // namespace fish_s2_pro
