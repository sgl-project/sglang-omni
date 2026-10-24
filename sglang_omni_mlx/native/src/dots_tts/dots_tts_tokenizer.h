// SPDX-License-Identifier: Apache-2.0
// dots.tts tokenizer: the special audio-span tokens and the generation schedule
// (prompt, text, language tag, and the span run) the backbone consumes.
#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "common/tokenizer.h"

namespace dots_tts {

class DotsTokenizer {
public:
  explicit DotsTokenizer(const std::filesystem::path &model_directory);

  std::vector<int> Encode(const std::string &text) const;

  std::vector<int> BuildGenerationSchedule(
      const std::string &text, const std::optional<std::string> &prompt_text,
      const std::optional<std::string> &language, int max_audio_tokens) const;

  int audio_gen_span_id() const { return audio_gen_span_id_; }
  int audio_comp_span_id() const { return audio_comp_span_id_; }

private:
  qwen3_asr::Tokenizer tokenizer_;
  int audio_gen_start_id_ = 0;
  int audio_gen_span_id_ = 0;
  int audio_comp_span_id_ = 0;
};

} // namespace dots_tts
