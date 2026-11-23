// SPDX-License-Identifier: Apache-2.0
// MOSS-Transcribe-Diarize transcription: prompt with time markers, greedy
// decoding, and timestamped speaker segments, as Voxt's Swift port runs it.
#pragma once

#include <atomic>
#include <exception>
#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "common/dflash_draft.h"
#include "common/tokenizer.h"
#include "common/transcription.h"
#include "common/worker.h"
#include "moss/moss_model.h"

namespace moss {

struct MossOptions {
  // The task instruction; blank is the model's default diarization prompt.
  std::optional<std::string> prompt;
  // Note (Dayuxiaoshui): unset or zero is Voxt's final-pass budget for the
  // audio's duration, applied per chunk.
  std::optional<int> max_new_tokens;
  // Note (Dayuxiaoshui): decoding always stops at <|im_end|>; these add the
  // checks Voxt's Swift port always applied.
  bool stop_at_end_of_text = false;
  bool stop_on_token_loop = false;
  // Realtime: decode the samples as one window that starts this far into the
  // session, without chunking or padding.
  std::optional<double> window_offset_seconds;
};

// One request of a batch: its audio, options and cancel flag.
struct BatchRequest {
  const std::vector<float> *samples;
  const MossOptions *options;
  const std::atomic<bool> *cancel;
};

// A finished request's result, or the exception that ended it.
struct BatchOutcome {
  std::optional<omni_common::TranscriptionResult> result;
  std::exception_ptr error;
};

class MossTranscriber {
public:
  MossTranscriber(const std::filesystem::path &model_directory,
                  const std::filesystem::path &draft_directory,
                  int draft_quantization_bits);

  omni_common::TranscriptionResult
  Transcribe(const std::vector<float> &samples, const MossOptions &options,
             const std::atomic<bool> &cancel) const;
  std::vector<int> PromptIds(int audio_token_count,
                             const std::optional<std::string> &prompt) const;
  // Whether the request can decode together with others: one chunk, without
  // the speculative draft.
  bool CanBatch(const std::vector<float> &samples,
                const MossOptions &options) const;
  // Transcribes several requests, decoding them together; on_finished gets
  // each request's index and outcome as soon as that request finishes.
  using BatchFinished = std::function<void(size_t, BatchOutcome)>;
  void TranscribeBatch(const std::vector<BatchRequest> &requests,
                       const BatchFinished &on_finished) const;

private:
  // Generated ids of a chunk and why decoding stopped.
  struct ChunkOutput {
    std::vector<int> output_ids;
    omni_common::FinishReason finish_reason =
        omni_common::FinishReason::kLength;
  };
  // Audio encoder and prompt embeddings [1, length, hidden] of one chunk.
  mlx::core::array
  PromptEmbeddings(const std::vector<float> &samples,
                   const std::optional<std::string> &prompt) const;
  // The chunked prefill of the embeddings and the first greedy token.
  omni_common::PrefilledSequence Prefill(const mlx::core::array &embeddings,
                                         const std::atomic<bool> &cancel) const;
  // Appends the token to the output; true once the chunk is finished.
  bool Commit(ChunkOutput &output, int token_id, const MossOptions &options,
              int max_new_tokens) const;
  omni_common::TranscriptionResult
  ChunkResult(const ChunkOutput &output, int sample_count,
              double offset_seconds,
              const omni_common::DraftStatistics &draft_statistics) const;
  omni_common::TranscriptionResult
  TranscribeChunk(const std::vector<float> &samples, const MossOptions &options,
                  int max_new_tokens, double offset_seconds,
                  const std::atomic<bool> &cancel) const;
  std::vector<int> AudioSpanIds(int audio_token_count) const;

  MossTranscribeDiarize model_;
  std::optional<omni_common::DFlashDraft> draft_;
  omni_common::Tokenizer tokenizer_;
  std::vector<int> digit_token_ids_;
  int end_of_text_id_ = 0;
  int im_end_id_ = 0;
  float audio_tokens_per_second_ = 12.5f;
  int time_marker_every_seconds_ = 5;
  bool enable_time_marker_ = true;
};

// One request the worker may decode together with others. Called alone it
// transcribes serially; BatchableTranscriptionRunner finds it through
// std::function::target<BatchableTranscription>().
struct BatchableTranscription {
  const MossTranscriber *transcriber;
  std::vector<float> samples;
  MossOptions options;

  omni_common::TranscriptionResult
  operator()(const std::atomic<bool> &cancel) const {
    return transcriber->Transcribe(samples, options, cancel);
  }
};

// The worker's batching of consecutive BatchableTranscription requests,
// decoded together by their transcriber's TranscribeBatch.
omni_server::BatchRunner BatchableTranscriptionRunner();

// Throws std::invalid_argument for a prompt with more than one audio
// placeholder, so the server can reject it before decoding.
void ValidatePrompt(const std::optional<std::string> &prompt);

// Timestamp tags [12.34] shifted by offset_seconds, printed with 2 decimals.
std::string OffsetTimestampTags(const std::string &text, double offset_seconds);
// [start][Sxx]text[end] segments; one unlabelled segment spanning the audio
// when none parse.
std::vector<omni_common::SpeakerSegment> ParseSegments(const std::string &text,
                                                       double duration_seconds,
                                                       double offset_seconds);

} // namespace moss
