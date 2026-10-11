// SPDX-License-Identifier: Apache-2.0
// Parakeet TDT transcription as Voxt's Swift path runs it: the audio decoded
// in overlapping chunks, each with greedy TDT decoding, the chunks' tokens
// merged on their overlap, and the tokens grouped into timed sentences.
#pragma once

#include <atomic>
#include <filesystem>
#include <string>
#include <vector>

#include "parakeet_model.h"
#include "transcriber.h"

namespace parakeet {

struct ParakeetOptions {
  // The chunk length Voxt asks for; the Swift port decodes 5 s chunks for
  // 1199 s or more and at least 0.5 s otherwise, each overlapping the last
  // by 1 s.
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

  ParakeetModel model_;
  std::vector<std::string> token_texts_;
  std::vector<bool> standalone_spaces_;
  std::vector<bool> special_tokens_;
};

} // namespace parakeet
