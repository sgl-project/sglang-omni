// SPDX-License-Identifier: Apache-2.0
// Qwen3-ASR transcription: prompt, greedy decoding and stop rules.
#pragma once

#include <atomic>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/audio.h"
#include "common/tokenizer.h"
#include "nlohmann/json.hpp"
#include "qwen3_asr/dflash_draft.h"
#include "qwen3_asr/qwen3_asr_model.h"

namespace qwen3_asr {

enum class FinishReason { kStop, kLength };

const char *FinishReasonName(FinishReason reason);

struct TranscriptionOptions {
  std::optional<std::string> language;
  std::optional<std::string> context;
  std::optional<int> max_new_tokens;
  bool stop_at_end_of_text = false;
  // Stops decoding at a greedy token loop and keeps one copy of its unit; the
  // HTTP server turns it on when a request leaves it out.
  bool stop_on_token_loop = false;
  // Divides the logits of tokens already generated (multiplies them when
  // negative); 1 turns it off.
  float repetition_penalty = 1.0f;
  AudioLayout layout = AudioLayout::kReference;
  // Realtime refreshes continue from text already shown, as ids and text.
  std::vector<int> prefix_token_ids;
  std::string prefix_text;
  // Tokens likely to come next (a realtime refresh's previous text after the
  // prefix), checked in the prompt's pass and kept while greedy decoding
  // agrees. Decoding with the speculative draft leaves them out.
  std::vector<int> guessed_token_ids;
};

struct SpeakerSegment {
  double start_seconds = 0.0;
  double end_seconds = 0.0;
  // Empty when the output carried no speaker segments.
  std::string speaker;
  std::string text;
  // Note (Dayuxiaoshui): SenseVoice's language, emotion and audio event of
  // the segment's audio; none for other models.
  std::optional<std::string> language;
  std::optional<std::string> emotion;
  std::optional<std::string> event;
};

struct TranscriptionResult {
  std::string text;
  std::optional<std::string> language;
  int generated_token_count = 0;
  FinishReason finish_reason = FinishReason::kLength;
  // Timestamped speaker segments, for models that produce them.
  std::vector<SpeakerSegment> segments;
  // Rounds of draft verification; 0 when decoding without a draft.
  int verify_round_count = 0;
};

class TranscriptionCancelled : public std::runtime_error {
public:
  TranscriptionCancelled() : std::runtime_error("transcription cancelled") {}
};

// Segments as the HTTP and realtime APIs send them: start, end, speaker, text,
// and language, emotion and event when set.
nlohmann::ordered_json
SpeakerSegmentsJson(const std::vector<SpeakerSegment> &segments);

// The canonical prompt name for a Qwen3-ASR language code or name. Like
// Voxt's Swift port, a name outside the table is used as given; blank is none.
std::optional<std::string> NormalizeLanguage(const std::string &language);

// Text already shown, split where a realtime refresh continues from it.
struct RetainedText {
  std::vector<int> prefix_ids;
  std::string prefix_text;
  // The tokens after the prefix, which may still change.
  std::vector<int> rest_ids;
};

// What a realtime segment's refreshes keep for the next: the encoder rows of
// the attention windows whose audio is complete, and the decoder's keys and
// values through them. The next refresh's audio extends this one's; a
// refresh whose mel differs over those windows (a louder frame raises the
// log-mel floor) starts over.
struct RefreshReuse {
  // The windows' mel frames, [mel_bins, frames].
  std::optional<mlx::core::array> mel;
  // Their encoder rows, [rows, hidden].
  std::optional<mlx::core::array> audio_rows;
  // The prompt's tokens before the audio.
  std::vector<int> head_ids;
  // Keys and values of head_ids and audio_rows.
  std::vector<KVCache> caches;
};

struct BatchRequest {
  const std::vector<float> *samples;
  const TranscriptionOptions *options;
  const std::atomic<bool> *cancel;
};

// A finished request's result, or the exception that ended it.
struct BatchOutcome {
  std::optional<TranscriptionResult> result;
  std::exception_ptr error;
};

class Qwen3ASRTranscriber {
public:
  // With a DFlash draft directory, decoding verifies the draft's proposals
  // instead of one token per step; the transcript stays greedy.
  explicit Qwen3ASRTranscriber(
      const std::filesystem::path &model_directory,
      const std::filesystem::path &draft_directory = {},
      int draft_quantization_bits = 0);

  std::vector<int> PromptIds(int audio_token_count,
                             const TranscriptionOptions &options) const;
  // Text already shown minus its last tokens, cut back to whole characters.
  RetainedText RetainedPrefix(const std::string &text,
                              int rollback_token_count) const;
  // With reuse, a realtime refresh: the audio windows reuse holds are not
  // encoded or prefilled again, and reuse then holds this refresh's for the
  // next. It decodes without the speculative draft.
  TranscriptionResult Transcribe(const std::vector<float> &samples,
                                 const TranscriptionOptions &options,
                                 const std::atomic<bool> &cancel,
                                 RefreshReuse *reuse = nullptr) const;
  // Whether realtime refreshes can reuse: the speculative draft keeps context
  // of its own, which RefreshReuse does not hold.
  bool CanReuseRefreshes() const { return !draft_.has_value(); }
  // Whether the request can decode together with others: the batched loop
  // has neither the draft nor the repetition penalty.
  bool CanBatch(const TranscriptionOptions &options) const {
    return !draft_.has_value() && options.repetition_penalty == 1.0f;
  }
  // Transcribes several requests, decoding them together. Greedy output
  // matches Transcribe's. on_finished gets each request's index and outcome as
  // soon as that request finishes, once per request.
  using BatchFinished = std::function<void(size_t, BatchOutcome)>;
  void TranscribeBatch(const std::vector<BatchRequest> &requests,
                       const BatchFinished &on_finished) const;

private:
  struct Prompt {
    mlx::core::array embeddings;
    int max_new_tokens;
  };
  // Mel, audio encoder and prompt embeddings [1, length, hidden], plus the
  // token budget.
  Prompt BuildPrompt(const std::vector<float> &samples,
                     const TranscriptionOptions &options) const;
  struct RefreshPrompt {
    // The prompt after the positions held has keys and values for.
    mlx::core::array embeddings;
    // The cache positions the next refresh can reuse.
    int kept_length;
  };
  // A refresh's prompt over what held can reuse, after which held holds this
  // refresh's complete windows; held starts over when it cannot be reused.
  RefreshPrompt BuildRefreshPrompt(const std::vector<float> &samples,
                                   const TranscriptionOptions &options,
                                   RefreshReuse &held) const;
  // The token budget: the request's, else one that grows with the audio.
  static int MaxNewTokens(size_t sample_count,
                          const TranscriptionOptions &options);
  bool IsStopToken(int token_id, const TranscriptionOptions &options) const;
  // The result for a finished output; a stop on a token loop keeps one copy
  // of the looping unit.
  TranscriptionResult FinishedResult(const std::vector<int> &output_ids,
                                     const TranscriptionOptions &options,
                                     FinishReason finish_reason,
                                     bool stopped_on_token_loop,
                                     int verify_round_count = 0) const;

  struct BatchRow;
  // Prefills every request; one that fails is reported finished with its
  // error and gets no row.
  std::vector<BatchRow> PrefillRows(const std::vector<BatchRequest> &requests,
                                    const BatchFinished &on_finished) const;
  static std::vector<BatchedKVCache>
  StackCaches(const std::vector<BatchRow> &rows);
  // Decodes the rows together until each has finished or been cancelled,
  // reporting each as it does.
  void DecodeRows(const std::vector<BatchRequest> &requests,
                  std::vector<BatchRow> &rows,
                  const BatchFinished &on_finished) const;
  // Appends the token to the row; true once the row's result is final.
  bool AcceptToken(BatchRow &row, int token_id,
                   const TranscriptionOptions &options,
                   BatchOutcome &outcome) const;
  std::pair<std::string, std::optional<std::string>>
  SplitOutput(const std::vector<int> &output_ids,
              const TranscriptionOptions &options) const;

  Qwen3ASR model_;
  omni_common::Tokenizer tokenizer_;
  int audio_pad_id_;
  int end_of_text_id_;
  int im_end_id_;
  std::vector<int> asr_text_ids_;
  std::optional<DFlashDraft> draft_;
};

// One request the worker may decode together with others. Called alone it
// transcribes serially; the worker finds it through
// std::function::target<BatchableTranscription>().
struct BatchableTranscription {
  const Qwen3ASRTranscriber *transcriber;
  std::vector<float> samples;
  TranscriptionOptions options;

  TranscriptionResult operator()(const std::atomic<bool> &cancel) const {
    return transcriber->Transcribe(samples, options, cancel);
  }
};

} // namespace qwen3_asr
