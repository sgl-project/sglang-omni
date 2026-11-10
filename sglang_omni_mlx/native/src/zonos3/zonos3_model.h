// SPDX-License-Identifier: Apache-2.0
// ZONOS3-Mini: a hybrid gated delta net / global attention backbone with a
// latent mixture of experts samples ten codebooks per 15 Hz row in a delay
// pattern, and the codec turns the aligned frames into 44.1 kHz audio. A
// voice is a 192-dimensional speaker embedding: a bundled one, or one the
// speaker encoder computes from reference audio.
#pragma once

#include <atomic>
#include <filesystem>
#include <functional>
#include <list>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "mlx/mlx.h"
#include "nlohmann/json.hpp"
#include "qwen3_asr/qwen3_asr_transcriber.h"
#include "zonos3/zonos3_codec.h"
#include "zonos3/zonos3_speaker.h"
#include "zonos3/zonos3_text.h"
#include "zonos3/zonos3_weights.h"

namespace zonos3 {

inline constexpr double kDefaultTemperature = 0.75;
inline constexpr int kDefaultTopK = 80;
inline constexpr double kDefaultTopP = 0.69;
inline constexpr double kDefaultRepetitionPenalty = 2.4;
inline constexpr int kDefaultRepetitionWindow = 96;
inline constexpr int kDefaultRepetitionCodebooks = 6;
// Delayed rows a request generates when it sets no budget.
inline constexpr int kDefaultMaxRows = 2000;
inline constexpr int kDefaultSilencePrefillFrames = 3;
inline constexpr int kMaxSilencePrefillFrames = 30;
inline constexpr double kDefaultStreamingIntervalSeconds = 0.5;
inline constexpr const char *kDefaultVoice = "jett";
// The voice name that generates without a speaker row.
inline constexpr const char *kNoVoice = "none";
inline constexpr int kSampleRate = 44100;

// The backbone section of config.json, checked at load.
struct ModelConfig {
  int layer_count = 0;
  int max_rows = 0;
  int hidden_size = 0;
  int head_dimension = 0;
  int head_count = 0;
  int key_value_head_count = 0;
  int expert_count = 0;
  int experts_per_row = 0;
  int codebook_count = 0;
  int codebook_size = 0;
  int audio_vocabulary_size = 0;
  int end_of_audio_id = 0;
  int audio_pad_id = 0;
  // The text column's pad id, one past the last text token.
  int text_pad_id = 0;
  int delta_head_count = 0;
  int delta_head_dimension = 0;
  int delta_kernel_size = 0;
  int speaker_embedding_size = 0;
  std::vector<int> condition_bucket_counts;
  std::vector<std::string> condition_names;
  std::map<std::string, std::vector<double>> condition_edges;
  float rope_theta = 0;
  float norm_epsilon = 0;
  float query_key_norm_epsilon = 0;
  float logit_softcap = 0;
  // Every gated_delta_period-th layer is global attention.
  int gated_delta_period = 0;
  std::vector<int32_t> silence_frame;

  bool IsGatedDelta(int layer) const {
    return (layer + 1) % gated_delta_period != 0;
  }
};

struct Sampling {
  // Knobs stay double, as the reference's Python floats; each becomes float32
  // where it meets an array.
  double temperature = kDefaultTemperature;
  int top_k = kDefaultTopK;
  double top_p = kDefaultTopP;
  double min_p = 0.0;
  double repetition_penalty = kDefaultRepetitionPenalty;
  int repetition_window = kDefaultRepetitionWindow;
  int repetition_codebooks = kDefaultRepetitionCodebooks;
  // Rows before which no codebook may end; codebook c may end from row
  // min_frames + c on.
  int min_frames = 0;
  // Unset draws from MLX's global generator; set, row r draws with the key of
  // seed + r.
  std::optional<uint64_t> seed;
};

struct SynthesisRequest {
  std::string text;
  bool normalize_text = true;
  std::string language = "en";
  // No embedding generates without a speaker row.
  std::optional<std::vector<float>> speaker_embedding;
  // Speaker classifier-free guidance; 1 turns it off.
  double speaker_guidance_scale = 1.0;
  // One bucket id per condition; a condition's bucket count means unknown.
  std::vector<int> condition_buckets;
  Sampling sampling;
  // Delayed rows to generate at most; the model context caps it.
  int max_rows = kDefaultMaxRows;
  int silence_prefill_frames = kDefaultSilencePrefillFrames;
  bool stream = false;
  double streaming_interval_seconds = kDefaultStreamingIntervalSeconds;
  // Output headroom in dB; unset takes the codec's.
  std::optional<double> headroom_db;
};

struct SynthesisResult {
  qwen3_asr::FinishReason finish_reason = qwen3_asr::FinishReason::kStop;
  int prompt_row_count = 0;
  // Delayed rows generated.
  int generated_row_count = 0;
  int frame_count = 0;
};

// The prompt a request lowers to, and the rows it replaces.
struct Prompt {
  // Rows [rows][codebooks + 1]: audio pad and text columns.
  std::vector<std::vector<int32_t>> rows;
  std::optional<int> speaker_row;
  // (row, bucket) per condition, in condition order.
  std::vector<std::pair<int, int>> condition_rows;
  // Leading silence rows that open the audio stream.
  int silence_row_count = 0;
};

// One sequence's decoding state: the next position, each global attention
// layer's keys and values [kv heads, capacity, head dim], and each gated delta
// layer's recurrent state [heads, value dim, key dim] and convolution history
// [channels, kernel - 1].
struct BackboneState {
  int position = 0;
  std::vector<std::optional<mlx::core::array>> keys;
  std::vector<std::optional<mlx::core::array>> values;
  std::vector<std::optional<mlx::core::array>> recurrent;
  std::vector<std::optional<mlx::core::array>> convolution;
};

class Zonos3Model {
public:
  explicit Zonos3Model(const std::filesystem::path &model_directory);

  const ModelConfig &config() const { return config_; }
  const TextFrontend &text() const { return text_; }
  const CodecDecoder &codec() const { return codec_; }
  // Bundled voice names, sorted.
  std::vector<std::string> VoiceNames() const;
  bool HasVoice(const std::string &name) const;
  // A bundled voice's embedding; throws std::invalid_argument for an unknown
  // name.
  std::vector<float> VoiceEmbedding(const std::string &name) const;
  // The embedding of mono 44.1 kHz reference audio, at most 60 s of it; the
  // most recent few are kept by content. Throws std::invalid_argument for
  // audio the encoder cannot read.
  std::vector<float> ReferenceEmbedding(const std::vector<float> &samples);

  // One bucket id per condition from a request's condition_buckets or
  // condition_values (at most one of them), each keyed by condition name;
  // a missing or null entry is unknown. Throws std::invalid_argument for an
  // unknown name or an out-of-range value.
  std::vector<int> ConditionIds(
      const std::optional<std::map<std::string, nlohmann::json>> &buckets,
      const std::optional<std::map<std::string, nlohmann::json>> &values) const;
  // The condition ids of a speaking rate speed times the training default.
  std::vector<int> SpeedConditionIds(double speed) const;
  // The prompt rows of request; throws std::invalid_argument when it leaves
  // the context no room.
  Prompt BuildPrompt(const SynthesisRequest &request) const;
  // Logits [rows, codebooks, audio vocabulary] of rows [rows, codebooks + 1]
  // after state, soft-capped; state advances past them.
  mlx::core::array
  Forward(const mlx::core::array &rows, BackboneState &state,
          const std::optional<std::pair<int, std::vector<float>>> &speaker_row,
          const std::vector<std::pair<int, int>> &condition_rows) const;
  // One delayed row [codebooks] of int32 codes drawn from logits [codebooks,
  // vocabulary] after history [rows, codebooks].
  mlx::core::array SampleRow(const mlx::core::array &logits, int step,
                             const mlx::core::array &history,
                             const Sampling &sampling,
                             bool has_audio_prefix) const;

  // Speech for request. on_rows receives every generated delayed row's int32
  // codes [codebooks], evaluated; on_audio receives the 44.1 kHz audio as it
  // is decoded: every streaming interval when streaming, else at the end.
  // Throws TranscriptionCancelled once cancel is set.
  SynthesisResult
  Synthesize(const SynthesisRequest &request, const std::atomic<bool> &cancel,
             const std::function<void(const mlx::core::array &)> &on_rows,
             const std::function<void(std::vector<float>)> &on_audio) const;

private:
  mlx::core::array GlobalAttention(const mlx::core::array &x, int layer,
                                   BackboneState &state) const;
  mlx::core::array GatedDeltaNet(const mlx::core::array &x, int layer,
                                 BackboneState &state) const;
  mlx::core::array FeedForward(const mlx::core::array &x, int layer) const;

  ModelConfig config_;
  Weights weights_;
  TextFrontend text_;
  CodecDecoder codec_;
  std::filesystem::path model_directory_;
  // Voice name to its embedding archive under voices/.
  std::map<std::string, std::string> voice_files_;
  std::optional<SpeakerEncoder> speaker_encoder_;
  // Most recently used first, by the SHA-256 of the samples.
  std::list<std::pair<std::string, std::vector<float>>> reference_embeddings_;
};

} // namespace zonos3
