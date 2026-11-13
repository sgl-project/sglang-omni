// SPDX-License-Identifier: Apache-2.0
// Parakeet TDT transcription with greedy decoding and timed sentences.
#pragma once

#include <atomic>
#include <filesystem>
#include <string>
#include <vector>

#include "parakeet/parakeet_model.h"
#include "qwen3_asr/qwen3_asr_transcriber.h"

namespace parakeet {

struct ParakeetOptions {
  // Whole-recording mode (1199 s or more) cuts at quiet boundaries within a
  // 30 s limit, reserving at least 10 s for the final tail when split.
  // Recovery padding uses only spare encoder capacity; no source is dropped.
  // Automatic decoding retries audible blank tails at most twice. Explicit
  // lengths use fixed windows with 1 s overlap and no recovery.
  double chunk_duration_seconds = 1200.0;
};

// One emitted token: its id, its text with the word marker read as a space,
// and its time.
struct AlignedToken {
  int id = 0;
  std::string text;
  double start_seconds = 0.0;
  double duration_seconds = 0.0;
  // A space that no combining mark attaches to, for sentence closing.
  bool has_standalone_space = false;

  double end_seconds() const { return start_seconds + duration_seconds; }
};

class ParakeetTranscriber {
public:
  explicit ParakeetTranscriber(const std::filesystem::path &model_directory);

  // The text, trimmed as Swift trims it, and one segment per sentence.
  qwen3_asr::TranscriptionResult
  Transcribe(const std::vector<float> &samples, const ParakeetOptions &options,
             const std::atomic<bool> &cancel) const;

private:
  // The tokens greedy TDT decoding emits on one chunk, in sentence order.
  std::vector<AlignedToken> DecodeChunk(const std::vector<float> &samples,
                                        const std::atomic<bool> &cancel) const;

  std::vector<AlignedToken>
  DecodeFinalChunk(const std::vector<float> &samples,
                   const std::atomic<bool> &cancel) const;

  ParakeetModel model_;
  std::vector<std::string> token_texts_;
  std::vector<bool> standalone_spaces_;
  std::vector<bool> special_tokens_;
};

} // namespace parakeet
