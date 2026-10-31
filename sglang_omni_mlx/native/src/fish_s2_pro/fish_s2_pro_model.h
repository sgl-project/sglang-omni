// SPDX-License-Identifier: Apache-2.0
// Fish S2-Pro: a dual autoregressive TTS model. The slow transformer reads the
// chat prompt and each generated frame and samples the frame's semantic code;
// the fast transformer then picks the frame's residual codes greedily, and the
// DAC codec turns frames into 44.1 kHz samples.
#pragma once

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/layers.h"
#include "common/tokenizer.h"
#include "fish_s2_pro/fish_s2_pro_codec.h"
#include "mlx/mlx.h"
#include "qwen3_asr/qwen3_asr_transcriber.h"
#include "qwen3_asr/qwen3_decoder.h"

namespace fish_s2_pro {

// A prompt and its frames longer than the model's positions: a request
// error, found only once the references are encoded.
class PromptTooLong : public std::invalid_argument {
public:
  using std::invalid_argument::invalid_argument;
};

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

  const AudioCodec &codec() const { return codec_; }
  int codebook_count() const { return codebook_count_; }
  // Codes per codebook the prompt's audio positions take.
  int codebook_size() const { return codebook_size_; }
  // The positions the slow transformer's rotary table covers.
  int max_positions() const { return max_positions_; }

  // The CUDA server's chat prompt: a system turn with the references'
  // transcripts and speech when there are any, the text as speaker 0's user
  // turn, then the assistant's voice turn.
  SpeechPrompt BuildPrompt(const std::string &text,
                           const std::vector<VoiceReference> &references) const;

  // Generates up to max_frames frames, passing each frame's evaluated int32
  // codes [codebooks] to on_frame. Stops at <|im_end|>; throws
  // TranscriptionCancelled once cancel is set, and PromptTooLong when the
  // prompt and max_frames outrun max_positions().
  qwen3_asr::FinishReason
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
    bool has_query_key_norm;
  };
  struct RotaryTable {
    mlx::core::array cosines;
    mlx::core::array sines;
  };

  FishS2ProModel(const std::filesystem::path &model_directory,
                 const nlohmann::json &config);

  static RotaryTable BuildRotaryTable(int positions, int head_dim, float base);
  // Rows of prompt ids [1, length] with each audio position's codebook
  // embeddings mixed in; codebook_rows [codebooks, length] are 0 elsewhere.
  mlx::core::array Embed(const mlx::core::array &ids,
                         const mlx::core::array &codebook_rows,
                         const mlx::core::array &is_audio) const;
  // Runs the layers of config over x, appending to caches; returns the
  // final-normed hidden state of the last position, [1, dim].
  mlx::core::array Forward(const mlx::core::array &x,
                           const TransformerConfig &config,
                           const RotaryTable &rotary,
                           std::vector<qwen3_asr::KVCache> &caches) const;
  // The residual codes after semantic_code, chosen greedily from hidden.
  std::vector<mlx::core::array>
  ResidualCodes(const mlx::core::array &hidden,
                const mlx::core::array &semantic_code) const;
  // The semantic token for candidate logits (semantic codes, then
  // <|im_end|>), as the CUDA server's sampler draws it.
  int SampleSemanticToken(const float *candidate_logits,
                          const std::vector<int> &history, int step,
                          const SamplingOptions &sampling) const;

  omni_common::Checkpoint checkpoint_;
  AudioCodec codec_;
  omni_common::Tokenizer tokenizer_;
  TransformerConfig slow_;
  TransformerConfig fast_;
  RotaryTable slow_rotary_;
  RotaryTable fast_rotary_;
  int codebook_count_;
  int codebook_size_;
  int max_positions_;
  int semantic_begin_id_;
  int semantic_end_id_;
  int im_end_id_;
};

} // namespace fish_s2_pro
