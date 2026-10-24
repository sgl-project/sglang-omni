// SPDX-License-Identifier: Apache-2.0
#include "dots_tts/dots_tts_tokenizer.h"

#include <algorithm>
#include <cctype>
#include <map>
#include <stdexcept>

#include "common/tokenizer.h"

namespace dots_tts {

namespace {

constexpr const char *kAudioGenStartToken = "<|audio_gen_start|>";
constexpr const char *kAudioGenSpanToken = "<|audio_gen_span|>";
constexpr const char *kAudioCompSpanToken = "<|audio_comp_span|>";
constexpr const char *kTtsTextPrefix = "[文本]";
constexpr const char *kTtsAudioPrefix = "[文本对应语音]";

int Utf8Width(const std::string &text, size_t offset) {
  const unsigned char lead = static_cast<unsigned char>(text[offset]);
  if (lead < 0x80) {
    return 1;
  }
  if ((lead & 0xE0) == 0xC0) {
    return 2;
  }
  if ((lead & 0xF0) == 0xE0) {
    return 3;
  }
  if ((lead & 0xF8) == 0xF0) {
    return 4;
  }
  return 1;
}

std::string Trimmed(const std::string &text) {
  size_t begin = 0;
  while (begin < text.size()) {
    const unsigned char c = static_cast<unsigned char>(text[begin]);
    const bool space = std::isspace(c) != 0 ||
                       (c == 0xC2 && begin + 1 < text.size() &&
                        static_cast<unsigned char>(text[begin + 1]) == 0xA0) ||
                       (c == 0xE3 && begin + 2 < text.size() &&
                        static_cast<unsigned char>(text[begin + 1]) == 0x80 &&
                        static_cast<unsigned char>(text[begin + 2]) == 0x80);
    if (!space) {
      break;
    }
    begin += Utf8Width(text, begin);
  }
  size_t end = begin;
  for (size_t i = begin; i < text.size();) {
    const unsigned char c = static_cast<unsigned char>(text[i]);
    const bool space = std::isspace(c) != 0 ||
                       (c == 0xC2 && i + 1 < text.size() &&
                        static_cast<unsigned char>(text[i + 1]) == 0xA0) ||
                       (c == 0xE3 && i + 2 < text.size() &&
                        static_cast<unsigned char>(text[i + 1]) == 0x80 &&
                        static_cast<unsigned char>(text[i + 2]) == 0x80);
    i += Utf8Width(text, i);
    if (!space) {
      end = i;
    }
  }
  return text.substr(begin, end - begin);
}

std::string ToLowerAscii(std::string input) {
  for (char &ch : input) {
    ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  }
  return input;
}

std::string ToUpperAscii(std::string input) {
  for (char &ch : input) {
    ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
  }
  return input;
}

const std::map<std::string, std::string> kLanguageNameCodes = {
    {"chinese", "ZH"},  {"mandarin", "ZH"}, {"中文", "ZH"},
    {"english", "EN"},  {"英语", "EN"},     {"cantonese", "YUE"},
    {"粤语", "YUE"},    {"广东话", "YUE"},  {"zh-yue", "YUE"},
    {"japanese", "JA"}, {"日语", "JA"},     {"korean", "KO"},
    {"韩语", "KO"},     {"french", "FR"},   {"法语", "FR"},
    {"german", "DE"},   {"德语", "DE"},     {"spanish", "ES"},
    {"西班牙语", "ES"}, {"russian", "RU"},  {"俄语", "RU"},
};

const std::map<std::string, std::string> kThreeLetterMacros = {
    {"zho", "ZH"}, {"chi", "ZH"}, {"cmn", "ZH"}, {"eng", "EN"}, {"yue", "YUE"},
    {"jpn", "JA"}, {"kor", "KO"}, {"fra", "FR"}, {"deu", "DE"}, {"ger", "DE"},
    {"spa", "ES"}, {"rus", "RU"}, {"ita", "IT"}, {"por", "PT"}, {"tha", "TH"},
    {"vie", "VI"}, {"ind", "ID"}, {"ara", "AR"}, {"hin", "HI"}, {"tur", "TR"},
    {"ukr", "UK"}, {"nld", "NL"}, {"dut", "NL"}, {"swe", "SV"}, {"pol", "PL"},
    {"ell", "EL"}, {"gre", "EL"}, {"heb", "HE"}, {"fin", "FI"}, {"nor", "NO"},
    {"dan", "DA"}, {"ces", "CS"}, {"cze", "CS"}, {"slk", "SK"}, {"slo", "SK"},
    {"hun", "HU"}, {"ron", "RO"}, {"rum", "RO"}, {"bul", "BG"}, {"hrv", "HR"},
    {"srp", "SR"}, {"slv", "SL"}, {"lit", "LT"}, {"lav", "LV"}, {"est", "ET"},
    {"fas", "FA"}, {"per", "FA"}, {"urd", "UR"}, {"ben", "BN"}, {"tam", "TA"},
    {"tel", "TE"}, {"mal", "ML"}, {"afr", "AF"}, {"swa", "SW"},
};

std::optional<std::string> NormalizeLanguage(const std::string &raw) {
  const std::string stripped = Trimmed(raw);
  if (stripped.empty()) {
    return std::nullopt;
  }
  if (ToLowerAscii(stripped) == "none") {
    return std::nullopt;
  }
  if (stripped.rfind("口音:", 0) == 0) {
    return stripped;
  }
  const std::string lower = ToLowerAscii(stripped);
  if (lower == "auto" || lower == "auto_detect") {
    throw std::invalid_argument(
        "dots.tts language auto-detection is unavailable in the native "
        "runtime; pass an explicit language code");
  }
  const auto name = kLanguageNameCodes.find(lower);
  if (name != kLanguageNameCodes.end()) {
    return name->second;
  }

  std::string primary = lower;
  std::string suffix;
  const size_t separator = lower.find_first_of("-_");
  if (separator != std::string::npos) {
    primary = lower.substr(0, separator);
    suffix = lower.substr(separator + 1);
  }
  std::optional<std::string> code;
  const bool primary_matches =
      (primary.size() == 2 || primary.size() == 3) &&
      std::all_of(primary.begin(), primary.end(), [](char ch) {
        return std::isalpha(static_cast<unsigned char>(ch)) != 0;
      });
  const bool suffix_matches =
      suffix.empty() ||
      (suffix.size() <= 8 &&
       std::all_of(suffix.begin(), suffix.end(), [](char ch) {
         return std::isalnum(static_cast<unsigned char>(ch)) != 0;
       }));
  if (primary_matches && suffix_matches) {
    if (primary.size() == 2) {
      code = ToUpperAscii(primary);
    } else {
      const auto macro = kThreeLetterMacros.find(primary);
      if (macro != kThreeLetterMacros.end()) {
        code = macro->second;
      }
    }
  }
  if (!code.has_value()) {
    throw std::invalid_argument("unsupported dots.tts language " + raw);
  }
  return code;
}

std::string AttachLanguageTag(std::string text, const std::string &code) {
  if (text.empty()) {
    return text;
  }
  const std::string tag_code = code == "YUE" ? "口音:粤语" : code;
  const std::string tag = "[" + tag_code + "]";
  if (text.rfind(tag, 0) == 0) {
    return text;
  }
  return tag + text;
}

} // namespace

DotsTokenizer::DotsTokenizer(const std::filesystem::path &model_directory)
    : tokenizer_(model_directory / "tokenizer") {
  const auto require_id = [&](const char *token) {
    const int id = tokenizer_.AddedTokenId(token);
    if (id < 0) {
      throw std::invalid_argument(
          std::string("dots.tts tokenizer is missing ") + token);
    }
    return id;
  };
  audio_gen_start_id_ = require_id(kAudioGenStartToken);
  audio_gen_span_id_ = require_id(kAudioGenSpanToken);
  audio_comp_span_id_ = require_id(kAudioCompSpanToken);
}

std::vector<int> DotsTokenizer::Encode(const std::string &text) const {
  return tokenizer_.Encode(text);
}

std::vector<int> DotsTokenizer::BuildGenerationSchedule(
    const std::string &text, const std::optional<std::string> &prompt_text,
    const std::optional<std::string> &language, int max_audio_tokens) const {
  if (max_audio_tokens <= 0) {
    throw std::invalid_argument(
        "dots.tts max_audio_tokens must be positive for generation");
  }
  const std::string prompt = Trimmed(prompt_text.value_or(""));
  std::string combined = prompt.empty() ? std::string() : prompt + "\n";
  combined += Trimmed(text);
  if (language.has_value()) {
    const std::optional<std::string> code = NormalizeLanguage(*language);
    if (code.has_value()) {
      combined = AttachLanguageTag(std::move(combined), *code);
    }
  }

  std::vector<int> ids;
  const auto append_encoded = [&](const std::string &t) {
    const std::vector<int> encoded = Encode(t);
    ids.insert(ids.end(), encoded.begin(), encoded.end());
  };
  append_encoded(kTtsTextPrefix);
  append_encoded(combined);
  append_encoded(kTtsAudioPrefix);
  ids.push_back(audio_gen_start_id_);
  ids.insert(ids.end(), max_audio_tokens, audio_gen_span_id_);
  return ids;
}

} // namespace dots_tts