// SPDX-License-Identifier: Apache-2.0
// Qwen3-ASR transcription: prompt, greedy decoding and stop rules.
#pragma once

#include <atomic>
#include <cstdint>
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

bool IsUnicodeSpace(uint32_t code_point);
std::vector<uint32_t> CodePoints(const std::string &text);
// Strips every Unicode space from both ends.
std::string StripUnicodeWhitespace(const std::string &text);
// Greedy loop guard: the last 24 tokens use at most 3 distinct ids.
bool IsTokenLoop(const std::vector<int> &output_ids);

// The canonical prompt name for a Qwen3-ASR language code or name. Like
// Voxt's Swift port, a name outside the table is used as given; blank is none.
std::optional<std::string> NormalizeLanguage(const std::string &language);

class StreamingEncoder;

class Qwen3ASRTranscriber {
public:
  explicit Qwen3ASRTranscriber(const std::filesystem::path &model_directory);

  std::vector<int> PromptIds(int audio_token_count,
                             const TranscriptionOptions &options) const;
  // Text already shown minus its last tokens, cut back to whole characters.
  std::pair<std::vector<int>, std::string>
  RetainedPrefix(const std::string &text, int rollback_token_count) const;
  // With an encoder, samples extend the segment it was last updated with and
  // its encoded windows are reused; it must use options.layout.
  TranscriptionResult Transcribe(const std::vector<float> &samples,
                                 const TranscriptionOptions &options,
                                 const std::atomic<bool> &cancel,
                                 StreamingEncoder *encoder = nullptr) const;

private:
  std::pair<std::string, std::optional<std::string>>
  SplitOutput(const std::vector<int> &output_ids,
              const TranscriptionOptions &options) const;

  Qwen3ASR model_;
  Tokenizer tokenizer_;
  int audio_pad_id_;
  int end_of_text_id_;
  int im_end_id_;
  std::vector<int> asr_text_ids_;
  friend StreamingEncoder;
};

// Caches each full encoder window of a growing segment until the segment's
// log-mel max changes. A window is cached only once all its mel frames are
// final, none reaching into the end padding.
class StreamingEncoder {
public:
  StreamingEncoder(const Qwen3ASRTranscriber &transcriber, AudioLayout layout);
  // Takes the segment so far, which only grows; caches full settled windows
  // and re-encodes them when the segment's log-mel max changes.
  void Update(const std::vector<float> &segment_samples);
  // [audio_tokens, output_dim] of the last segment; nullopt before any.
  std::optional<mlx::core::array> Output() const;
  int FrameCount() const { return frame_count_; }

private:
  const Qwen3ASRTranscriber &transcriber_;
  AudioLayout layout_;
  int window_frame_count_;
  int frame_count_ = 0;
  int cached_frame_count_ = 0;
  std::optional<mlx::core::array> cached_rows_; // encoded full windows
  std::optional<float> cached_max_;             // log-mel max they used
  std::optional<mlx::core::array> tail_mel_;    // frames after them
};

} // namespace qwen3_asr
