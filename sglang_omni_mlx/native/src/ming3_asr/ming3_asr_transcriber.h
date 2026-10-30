#pragma once

#include <atomic>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "common/tokenizer.h"
#include "ming3_asr/ming3_asr_model.h"
#include "qwen3_asr/qwen3_asr_transcriber.h"

namespace ming3_asr {

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
  explicit MingTranscriber(const std::filesystem::path &model_directory);

  MingRequest Prepare(std::vector<float> samples,
                      const MingOptions &options) const;

  qwen3_asr::TranscriptionResult
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
};

} // namespace ming3_asr
