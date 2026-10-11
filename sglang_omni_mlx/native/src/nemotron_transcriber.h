// SPDX-License-Identifier: Apache-2.0
// Nemotron 3.5 ASR transcription as Voxt's Swift port runs it: one
// cache-aware streaming loop for both the Final pass and live sessions,
// greedy RNN-T decoding, and the vocabulary's text with sentence segments.
#pragma once

#include <atomic>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "nemotron_model.h"
#include "sortformer_features.h"
#include "transcriber.h"

namespace nemotron {

struct NemotronOptions {
  // A prompt dictionary key such as "en-US"; unknown or none is the model's
  // default language.
  std::optional<std::string> language;
  // Encoder frames per chunk, 80 ms each; none is the native chunk.
  std::optional<int> chunk_frames;
};

class NemotronTranscriber;

// One audio stream decoded as it arrives: audio is encoded in whole chunks
// of mel frames that later audio can no longer change, and Finish flushes
// the rest, so the text equals one Transcribe of all the audio.
class NemotronStream {
public:
  NemotronStream(const NemotronTranscriber &transcriber,
                 const NemotronOptions &options);

  void Append(const std::vector<float> &samples,
              const std::atomic<bool> &cancel);
  // Appends the last samples and decodes everything left, tail included.
  void Finish(const std::vector<float> &samples,
              const std::atomic<bool> &cancel);

  // The transcript so far, trimmed as Swift trims it.
  std::string Text() const;
  qwen3_asr::TranscriptionResult Result() const;

private:
  struct Token {
    std::string text;
    // A space that no combining mark attaches to, for sentence closing.
    bool has_standalone_space = false;
    double start_seconds = 0.0;
  };

  void Decode(int mel_frame_limit, bool flush, const std::atomic<bool> &cancel);
  void DecodeChunk(const mlx::core::array &prompted);

  const NemotronTranscriber &transcriber_;
  std::optional<std::string> language_;
  int prompt_index_ = 0;
  int chunk_frames_ = 0;
  // The audio from sample_offset_ on; earlier samples feed no mel frame that
  // is still to be decoded, so they are dropped.
  std::vector<float> samples_;
  int sample_offset_ = 0;

  // Encoder state between chunks.
  std::vector<LayerCache> layer_caches_;
  std::optional<mlx::core::array> mel_cache_;
  int emitted_frames_ = 0;
  int consumed_mel_frames_ = 0;

  // Greedy RNN-T state between chunks.
  int last_token_ = 0;
  std::optional<LstmState> prediction_state_;
  int decoded_frames_ = 0;
  std::vector<Token> tokens_;
};

class NemotronTranscriber {
public:
  explicit NemotronTranscriber(const std::filesystem::path &model_directory);

  // The whole recording through the streaming loop; with no chunk_frames,
  // at the native chunk size, as Voxt's Swift port decodes its Final pass.
  // The audio goes in one chunk at a time, so memory stays flat with length.
  qwen3_asr::TranscriptionResult
  Transcribe(const std::vector<float> &samples, const NemotronOptions &options,
             const std::atomic<bool> &cancel) const;

  const NemotronModel &model() const { return model_; }
  const sortformer::FeatureExtractor &features() const { return features_; }
  const sortformer::ProcessorConfig &front_end() const { return front_end_; }
  // Seconds of audio per encoder frame.
  double frame_seconds() const;
  bool IsSpecialToken(int token_id) const;
  // The token's text, its word marker read as a space.
  const std::string &TokenText(int token_id) const;
  bool HasStandaloneSpace(int token_id) const;

private:
  NemotronModel model_;
  sortformer::ProcessorConfig front_end_;
  sortformer::FeatureExtractor features_;
  std::vector<std::string> token_texts_;
  std::vector<bool> standalone_spaces_;
  std::vector<bool> special_tokens_;
};

} // namespace nemotron
