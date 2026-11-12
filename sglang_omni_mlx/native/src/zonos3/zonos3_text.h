// SPDX-License-Identifier: Apache-2.0
// ZONOS3-Mini's text column: the reference's English number, date, money and
// unit normalization, then SuperBPE token ids shifted past the special ids
// and wrapped in begin and end of text.
#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "common/tokenizer.h"
#include "nlohmann/json.hpp"

namespace zonos3 {

// text with numbers, dates, times, money, percentages, units, fractions,
// ordinals, phone numbers and acronyms spelled out in English words, and
// whitespace collapsed; any other text is returned unchanged.
std::string NormalizeEnglishText(const std::string &text);

class TextFrontend {
public:
  // Reads the tokenizer the spec names under model_directory; throws
  // std::runtime_error unless it is the SuperBPE layout the backbone was
  // trained with (offset 4, begin 2, end 3) with the listed SHA-256.
  TextFrontend(const std::filesystem::path &model_directory,
               const nlohmann::json &tokenizer_spec, int text_pad_id);

  // text normalized as the reference does for language ("en", "en-US" and
  // "en_GB" alike); other languages and blank text are returned unchanged.
  static std::string Normalize(const std::string &text,
                               const std::string &language);
  // Begin of text, the shifted SuperBPE ids of text, end of text.
  std::vector<int> Encode(const std::string &text) const;

private:
  omni_common::Tokenizer tokenizer_;
};

} // namespace zonos3
