// SPDX-License-Identifier: Apache-2.0
// Qwen3-ASR transcription: prompt, greedy decoding and stop rules.
#pragma once

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <functional>
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
  std::pair<std::vector<int>, std::string>
  RetainedPrefix(const std::string &text, int rollback_token_count) const;
  TranscriptionResult Transcribe(const std::vector<float> &samples,
                                 const TranscriptionOptions &options,
                                 const std::atomic<bool> &cancel) const;

private:
  // Greedy decoding with the draft: each round the target verifies the block
  // the draft proposed and commits the accepted tokens plus one more. Tokens
  // go to commit until it reports the output complete; returns the rounds.
  int DecodeWithDraft(const mlx::core::array &embeddings,
                      const std::function<bool(int)> &commit,
                      const std::atomic<bool> &cancel) const;
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

} // namespace qwen3_asr
