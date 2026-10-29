// SPDX-License-Identifier: Apache-2.0
// Fish Audio Drama 3: a Qwen3.5-MoE backbone samples each frame's main token,
// a temporal codebook tower samples its other codebooks, and the codec turns
// frames into 44.1 kHz audio.
#pragma once

#include <atomic>
#include <filesystem>
#include <functional>
#include <list>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "common/layers.h"
#include "common/tokenizer.h"
#include "fishaudio_drama3/fishaudio_drama3_codec.h"
#include "fishaudio_drama3/fishaudio_drama3_prompt.h"
#include "mlx/mlx.h"
#include "qwen3_asr/qwen3_asr_transcriber.h"
#include "qwen3_asr/qwen3_decoder.h"

namespace fishaudio_drama3 {

// Frames per chunk of text when a request sets no cap.
inline constexpr int kDefaultMaxFrames = 1024;
inline constexpr float kDefaultTemperature = 1.0f;
inline constexpr float kDefaultTopP = 0.9f;

struct Sampling {
  float temperature = kDefaultTemperature;
  // None keeps every candidate of the main token; the codebook heads keep at
  // most 30.
  std::optional<int> top_k;
  float top_p = kDefaultTopP;
};

struct SynthesisResult {
  qwen3_asr::FinishReason finish_reason = qwen3_asr::FinishReason::kStop;
  // Prompt rows of every chunk, and the frames generated for them.
  int prompt_row_count = 0;
  int frame_count = 0;
};

// A reference: mono samples at kCodecSampleRate or codes [codebooks][frames],
// and its transcript.
struct ReferenceInput {
  std::string text;
  std::vector<float> samples;
  std::optional<std::vector<std::vector<int32_t>>> codes;
};

// One layer's state: a gated delta layer's convolution tail and recurrent
// state, or an attention layer's keys and values.
struct LayerCache {
  std::optional<mlx::core::array> convolution_state;
  std::optional<mlx::core::array> recurrent_state;
  qwen3_asr::KVCache attention;
};

struct ModelCaches {
  std::vector<LayerCache> backbone;
  std::vector<qwen3_asr::KVCache> tower;
};

// Encoded references and, once built, the caches of the prompt's system block.
struct Voice {
  std::vector<SpeechTurn> references;
  std::optional<ModelCaches> prefix;
};

class Drama3Model {
public:
  explicit Drama3Model(const std::filesystem::path &model_directory);

  const omni_common::Tokenizer &tokenizer() const { return tokenizer_; }
  const AudioCodec &codec() const { return codec_; }
  int codebook_count() const { return codebook_count_; }
  int codebook_size() const { return codebook_size_; }
  int semantic_begin() const { return semantic_begin_; }

  // The voice of the references; the few most recent are kept by content, so
  // a repeated voice skips encoding and the system block's prefill. Throws
  // std::invalid_argument for codes that are not a [codebooks][frames] grid in
  // range.
  std::shared_ptr<Voice>
  GetVoice(const std::vector<ReferenceInput> &references);
  // Speech for text in voice, chunk by chunk, each chunk capped at
  // max_frames frames. on_audio receives the audio as it is decoded: every
  // 7 frames (0.33 s) when streaming, else at each chunk's end; on_frame
  // receives each frame's evaluated int32 codes [codebooks] first. Throws
  // TranscriptionCancelled once cancel is set.
  SynthesisResult
  Synthesize(const std::string &text, Voice &voice, const Sampling &sampling,
             int max_frames, bool stream, const std::atomic<bool> &cancel,
             const std::function<void(const mlx::core::array &)> &on_frame,
             const std::function<void(std::vector<float>)> &on_audio);

private:
  // x in float32, RMS-normed by the float32 gain at prefix, back in x's dtype.
  mlx::core::array Fp32Norm(const mlx::core::array &x,
                            const std::string &prefix, float eps) const;
  mlx::core::array Embed(const mlx::core::array &frames) const;
  // The final-normed hidden states [batch, time, dim] of embeddings.
  mlx::core::array Backbone(const mlx::core::array &embeddings,
                            std::vector<LayerCache> &caches) const;
  mlx::core::array GatedDeltaLayer(const mlx::core::array &x,
                                   const std::string &prefix,
                                   LayerCache &cache) const;
  mlx::core::array AttentionLayer(const mlx::core::array &x,
                                  const std::string &prefix,
                                  qwen3_asr::KVCache &cache) const;
  mlx::core::array MixtureOfExperts(const mlx::core::array &x,
                                    const std::string &prefix) const;
  mlx::core::array
  TowerLayer(const mlx::core::array &hidden, const mlx::core::array &codes,
             const mlx::core::array &is_semantic, int layer,
             qwen3_asr::KVCache &cache,
             const std::optional<mlx::core::array> &unseen_embedding) const;
  // The tower's shared layers over hidden, given the input frame's codebooks
  // the backbone input left out.
  mlx::core::array Enrich(mlx::core::array hidden,
                          const mlx::core::array &semantic_codes,
                          const mlx::core::array &is_semantic,
                          const mlx::core::array &frames,
                          std::vector<qwen3_asr::KVCache> &caches) const;
  mlx::core::array EmbeddingSum(const mlx::core::array &frames, int first,
                                int end) const;
  mlx::core::array IsSemantic(const mlx::core::array &tokens) const;
  // Logits [rows, codebook_size] of residual codebook index + 1.
  mlx::core::array CodebookHead(const mlx::core::array &hidden,
                                int index) const;
  // Feeds rows [start, stop) into both caches; returns row stop - 1's
  // backbone hidden state [1, 1, dim]. Throws TranscriptionCancelled between
  // blocks of rows once cancel is set.
  mlx::core::array Prefill(const mlx::core::array &frames, ModelCaches &caches,
                           int start, int stop,
                           const std::atomic<bool> &cancel) const;

  struct Step {
    mlx::core::array frame;
    mlx::core::array hidden;
    mlx::core::array history;
    mlx::core::array main_token;
  };
  // One frame, all lazy: the frame [1, 1, 1 + codebooks], the next hidden
  // state, the main-token history and the main token [1].
  Step DecodeStep(const mlx::core::array &hidden, const mlx::core::array &frame,
                  const mlx::core::array &history, ModelCaches &caches,
                  const Sampling &sampling) const;
  // The codes [codebooks] of each frame of one chunk until the end of speech;
  // returns whether it ended there rather than at max_frames.
  bool GenerateChunk(
      const PromptFrames &prompt, Voice &voice, const Sampling &sampling,
      int max_frames, const std::atomic<bool> &cancel,
      const std::function<void(const mlx::core::array &)> &on_frame) const;

  nlohmann::json config_;
  omni_common::Checkpoint checkpoint_;
  omni_common::Tokenizer tokenizer_;
  AudioCodec codec_;
  int layer_count_;
  float rms_norm_eps_;
  int attention_head_count_;
  int key_value_head_count_;
  int head_dim_;
  int rotary_dims_;
  float rope_theta_;
  int linear_key_head_count_;
  int linear_value_head_count_;
  int linear_key_head_dim_;
  int linear_value_head_dim_;
  int convolution_kernel_size_;
  int expert_count_;
  int experts_per_token_;
  int semantic_begin_;
  int semantic_end_;
  int end_of_speech_id_;
  int codebook_count_;
  int codebook_size_;
  int backbone_codebook_count_;
  int shared_tower_layer_count_;
  int tower_head_count_;
  int tower_key_value_head_count_;
  int tower_head_dim_;
  int tower_rotary_dims_;
  float tower_rope_base_;
  float tower_norm_eps_;
  // Ids a step may emit: the semantic band, then the end of speech.
  mlx::core::array candidate_ids_;
  // Most recently used first, by the SHA-256 of the references.
  std::list<std::pair<std::string, std::shared_ptr<Voice>>> voices_;
};

// x sliced to [begin, end) along its last axis, or along axis.
mlx::core::array SliceLast(const mlx::core::array &x, int begin, int end);
mlx::core::array SliceAxis(const mlx::core::array &x, int axis, int begin,
                           int end);

} // namespace fishaudio_drama3
