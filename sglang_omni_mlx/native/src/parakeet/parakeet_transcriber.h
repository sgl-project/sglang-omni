// SPDX-License-Identifier: Apache-2.0
// Parakeet TDT transcription with greedy decoding and timed sentences.
#pragma once

#include <atomic>
#include <cstddef>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "common/transcription.h"
#include "parakeet/parakeet_model.h"

namespace parakeet {

// Note (Mengwei Liu): each process serves one model; the server and the
// multi-file CLI bound freed buffers during requests, not only when idle.
inline constexpr size_t kBufferCacheLimitBytes = 256ULL << 20;

struct ParakeetOptions {
  // Fixed windows overlapping by 1 s; unset cuts automatic chunks at quiet
  // boundaries.
  std::optional<double> window_seconds;
};

// The options for a requested chunk_duration: 2 to 300 seconds for fixed
// windows, or 1199 or more (Voxt's whole recording) for automatic chunks.
// Throws std::invalid_argument for anything else.
ParakeetOptions ChunkDurationOptions(double chunk_duration_seconds);

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
  omni_common::TranscriptionResult
  Transcribe(const std::vector<float> &samples, const ParakeetOptions &options,
             const std::atomic<bool> &cancel) const;

private:
  // The tokens greedy TDT decoding emits on one chunk, in sentence order.
  std::vector<AlignedToken> DecodeChunk(const std::vector<float> &samples,
                                        const std::atomic<bool> &cancel) const;

  // DecodeChunk, then bounded retries of an audible tail it left blank;
  // samples hold at most max_chunk_samples.
  std::vector<AlignedToken>
  DecodeWithBlankTailRecovery(const std::vector<float> &samples,
                              size_t max_chunk_samples,
                              const std::atomic<bool> &cancel) const;

  ParakeetModel model_;
  std::vector<std::string> token_texts_;
  std::vector<bool> standalone_spaces_;
  std::vector<bool> special_tokens_;
};

} // namespace parakeet
