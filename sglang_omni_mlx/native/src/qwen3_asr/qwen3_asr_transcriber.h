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
#include "qwen3_asr/qwen3_asr_model.h"

namespace qwen3_asr {

enum class FinishReason { kStop, kLength };

const char *FinishReasonName(FinishReason reason);

struct TranscriptionOptions {
  std::optional<std::string> language;
  std::optional<std::string> context;
  std::optional<int> max_new_tokens;
  bool stop_at_end_of_text = false;
  bool stop_on_token_loop = false;
  AudioLayout layout = AudioLayout::kReference;
  // Realtime refreshes continue from text already shown, as ids and text.
  std::vector<int> prefix_token_ids;
  std::string prefix_text;
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
  explicit Qwen3ASRTranscriber(const std::filesystem::path &model_directory);

  std::vector<int> PromptIds(int audio_token_count,
                             const TranscriptionOptions &options) const;
  // Text already shown minus its last tokens, cut back to whole characters.
  std::pair<std::vector<int>, std::string>
  RetainedPrefix(const std::string &text, int rollback_token_count) const;
  TranscriptionResult Transcribe(const std::vector<float> &samples,
                                 const TranscriptionOptions &options,
                                 const std::atomic<bool> &cancel) const;
  // Transcribes several requests, decoding them together; one outcome per
  // request, in order. Greedy output matches Transcribe's.
  std::vector<BatchOutcome>
  TranscribeBatch(const std::vector<BatchRequest> &requests) const;

private:
  struct Prompt {
    mlx::core::array embeddings;
    int max_new_tokens;
  };
  // Mel, audio encoder and prompt embeddings [1, length, hidden], plus the
  // token budget.
  Prompt BuildPrompt(const std::vector<float> &samples,
                     const TranscriptionOptions &options) const;
  bool IsStopToken(int token_id, const TranscriptionOptions &options) const;
  TranscriptionResult FinishedResult(const std::vector<int> &output_ids,
                                     const TranscriptionOptions &options,
                                     FinishReason finish_reason) const;

  struct BatchRow;
  // Prefills every request; outcomes holds the error of any that fail.
  std::vector<BatchRow> PrefillRows(const std::vector<BatchRequest> &requests,
                                    std::vector<BatchOutcome> &outcomes) const;
  static std::vector<BatchedKVCache>
  StackCaches(const std::vector<BatchRow> &rows, int extra_tokens);
  // Decodes the rows together until each has finished or been cancelled.
  void DecodeRows(const std::vector<BatchRequest> &requests,
                  std::vector<BatchRow> &rows,
                  std::vector<BatchOutcome> &outcomes) const;
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
