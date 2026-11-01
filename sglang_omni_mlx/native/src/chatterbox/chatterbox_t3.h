// SPDX-License-Identifier: Apache-2.0
// Chatterbox-Turbo's T3: a GPT-2 medium decoder over the speaker embedding,
// the prompt's speech tokens and the text, sampling S3 speech tokens.
#pragma once

#include <atomic>
#include <functional>
#include <vector>

#include "common/layers.h"
#include "mlx/mlx.h"
#include "qwen3_asr/qwen3_asr_transcriber.h"
#include "qwen3_asr/qwen3_decoder.h"

namespace chatterbox {

// S3 speech tokens are 0 to 6560; T3 adds the start and stop tokens.
inline constexpr int kSpeechTokenCount = 6561;
inline constexpr int kStartSpeechToken = 6561;
inline constexpr int kStopSpeechToken = 6562;
// The most prompt speech tokens T3 reads from a voice.
inline constexpr int kPromptSpeechTokenLimit = 375;

struct SamplingOptions {
  float temperature = 0.8f;
  int top_k = 1000;
  float top_p = 0.95f;
  float repetition_penalty = 1.2f;
};

// What T3 conditions on: the voice encoder's speaker embedding [1, 256] and
// the voice's prompt speech tokens, int32 [1, tokens].
struct T3Voice {
  mlx::core::array speaker_embedding;
  mlx::core::array prompt_speech_tokens;
};

class T3 {
public:
  explicit T3(const omni_common::Checkpoint &checkpoint);

  // Samples up to max_tokens speech tokens for text_ids with MLX's global
  // random keys, passing each one before the stop token to on_token. Throws
  // TranscriptionCancelled once cancel is set.
  qwen3_asr::FinishReason
  Generate(const T3Voice &voice, const std::vector<int> &text_ids,
           const SamplingOptions &options, int max_tokens,
           const std::atomic<bool> &cancel,
           const std::function<void(int)> &on_token) const;

private:
  // Speech-head logits [1, speech tokens + 2] of the last of embeddings
  // [1, positions, 1024].
  mlx::core::array LastLogits(const mlx::core::array &embeddings,
                              std::vector<qwen3_asr::KVCache> &caches) const;
  mlx::core::array Block(const mlx::core::array &x, int layer,
                         qwen3_asr::KVCache &cache) const;
  // A token [1] drawn from logits [1, vocabulary] after the penalty on the
  // seen tokens (when given), temperature, top-k and top-p.
  mlx::core::array Sample(mlx::core::array logits,
                          const std::vector<float> *seen_tokens,
                          const SamplingOptions &options) const;

  const omni_common::Checkpoint &checkpoint_;
  // sqrt(2 / pi) as the tanh GELU computes it.
  mlx::core::array gelu_scale_;
};

} // namespace chatterbox
