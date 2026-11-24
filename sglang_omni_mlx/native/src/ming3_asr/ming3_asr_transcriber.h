#pragma once

#include <atomic>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "common/dflash_draft.h"
#include "common/tokenizer.h"
#include "common/transcription.h"
#include "ming3_asr/ming3_asr_model.h"

namespace ming3_asr {

struct GreedyDecoding {
  std::vector<int> token_ids;
  omni_common::FinishReason finish_reason = omni_common::FinishReason::kLength;
  omni_common::DraftStatistics draft_statistics;
};

// Greedy decoding after the prompt embeddings [1, token_count, hidden_size]:
// a stop token ends the output unemitted, and a positive no_repeat_ngram_size
// bans any token that would repeat an ngram of that many. A DSpark draft
// leaves the tokens as they are and verifies a block of them per target pass;
// under DraftPolicy::kAdaptive it backs off to plain decoding as Qwen3-ASR's
// DFlash decoding does, when plain decoding measures faster.
GreedyDecoding DecodeGreedy(const MingModel &model,
                            const omni_common::DFlashDraft *draft,
                            const mlx::core::array &embeddings,
                            const std::vector<int> &stop_token_ids,
                            int max_new_tokens, int no_repeat_ngram_size,
                            omni_common::DraftPolicy draft_policy,
                            const std::atomic<bool> &cancel);

struct MingOptions {
  std::optional<std::string> language;
  std::optional<std::string> prompt;
  std::optional<std::string> hotwords;
  int max_new_tokens = 256;
  int no_repeat_ngram_size = 10;
};

struct MingRequest {
  std::vector<float> samples;
  std::vector<int> prefix_token_ids;
  std::vector<int> suffix_token_ids;
  std::optional<std::string> language;
  int max_new_tokens;
  int no_repeat_ngram_size;
};

class MingTranscriber {
public:
  // An empty draft_directory decodes without a draft; draft_quantization_bits
  // 0 keeps the draft's weights as stored.
  MingTranscriber(const std::filesystem::path &model_directory,
                  const std::filesystem::path &draft_directory,
                  int draft_quantization_bits,
                  omni_common::DraftPolicy draft_policy);

  MingRequest Prepare(std::vector<float> samples,
                      const MingOptions &options) const;

  omni_common::TranscriptionResult
  Transcribe(const MingRequest &request, const std::atomic<bool> &cancel) const;

private:
  struct AudioTokenLayout {
    std::string start_token;
    int end_token_id;
  };

  static AudioTokenLayout
  ResolveAudioTokens(const omni_common::Tokenizer &tokenizer,
                     int audio_token_id);

  MingModel model_;
  omni_common::Tokenizer tokenizer_;
  AudioTokenLayout audio_tokens_;
  std::vector<int> stop_token_ids_;
  std::optional<omni_common::DFlashDraft> draft_;
  omni_common::DraftPolicy draft_policy_;
};

} // namespace ming3_asr
