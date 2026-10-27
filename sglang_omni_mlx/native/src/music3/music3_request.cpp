// SPDX-License-Identifier: Apache-2.0
#include "music3/music3_model.h"

#define PCRE2_CODE_UNIT_WIDTH 8
#include <CoreFoundation/CoreFoundation.h>
#include <pcre2.h>

#include <cmath>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>

namespace music3 {

namespace {

constexpr size_t kMaximumPromptFieldBytes = 64 * 1024;

struct PromptMatch {
  size_t begin;
  size_t end;
  std::string capture;
};

class PromptPattern {
public:
  explicit PromptPattern(const char *pattern) {
    int error = 0;
    PCRE2_SIZE offset = 0;
    code_.reset(pcre2_compile(reinterpret_cast<PCRE2_SPTR>(pattern),
                              PCRE2_ZERO_TERMINATED, PCRE2_UTF | PCRE2_UCP,
                              &error, &offset, nullptr));
    if (!code_) {
      throw std::runtime_error("cannot compile Music3 prompt pattern");
    } else {
    }
  }

  std::optional<PromptMatch> Search(const std::string &text,
                                    size_t offset = 0) const {
    std::unique_ptr<pcre2_match_data, decltype(&pcre2_match_data_free)> match(
        pcre2_match_data_create_from_pattern(code_.get(), nullptr),
        pcre2_match_data_free);
    const int result =
        pcre2_match(code_.get(), reinterpret_cast<PCRE2_SPTR>(text.data()),
                    text.size(), offset, 0, match.get(), nullptr);
    if (result == PCRE2_ERROR_NOMATCH) {
      return std::nullopt;
    } else if (result < 0) {
      throw std::invalid_argument("Music3 prompt pattern match failed");
    } else {
      const auto *positions = pcre2_get_ovector_pointer(match.get());
      const int capture = result > 1 ? 2 : 0;
      return PromptMatch{
          positions[0], positions[1],
          text.substr(positions[capture],
                      positions[capture + 1] - positions[capture])};
    }
  }

  std::string Replace(const std::string &text,
                      const std::string &replacement) const {
    std::string output(text.size() + 1, '\0');
    PCRE2_SIZE bytes = output.size();
    const auto substitute = [&]() {
      return pcre2_substitute(
          code_.get(), reinterpret_cast<PCRE2_SPTR>(text.data()), text.size(),
          0, PCRE2_SUBSTITUTE_GLOBAL | PCRE2_SUBSTITUTE_OVERFLOW_LENGTH,
          nullptr, nullptr, reinterpret_cast<PCRE2_SPTR>(replacement.data()),
          replacement.size(), reinterpret_cast<PCRE2_UCHAR *>(output.data()),
          &bytes);
    };
    int result = substitute();
    if (result == PCRE2_ERROR_NOMEMORY) {
      output.resize(bytes);
      result = substitute();
    } else {
    }
    if (result < 0) {
      throw std::invalid_argument("Music3 prompt substitution failed");
    } else {
      output.resize(bytes);
      return output;
    }
  }

private:
  std::unique_ptr<pcre2_code, decltype(&pcre2_code_free)> code_{
      nullptr, pcre2_code_free};
};

} // namespace

std::string BuildPrompt(const std::string &caption, const std::string &lyrics) {
  if (caption.size() > kMaximumPromptFieldBytes ||
      lyrics.size() > kMaximumPromptFieldBytes) {
    throw std::invalid_argument(
        "input and instructions must each be at most 64 KiB");
  } else {
  }
  // Note (Akazaakane): Python whitespace includes four controls outside Unicode
  // White_Space.
  static const PromptPattern trim(R"(^[\s\x1c-\x1f]+|[\s\x1c-\x1f]+$)");
  static const PromptPattern whitespace(R"([\s\x1c-\x1f]+)");
  static const PromptPattern line_break(
      R"(\r\n|[\n\r\v\f\x1c-\x1e\x{85}\x{2028}\x{2029}])");
  static const PromptPattern special(R"(<\|([^|]*)\|>)");
  static const PromptPattern heading(
      R"(^[\s\x1c-\x1f]{0,3}#{1,6}[\s\x1c-\x1f]+)");
  static const PromptPattern bullet(R"(^[\s\x1c-\x1f]*[*+-][\s\x1c-\x1f]+)");
  static const PromptPattern asterisk_bullet(
      R"(^[\s\x1c-\x1f]*\*[\s\x1c-\x1f]+)");
  static const PromptPattern bold(R"(\*\*([^*]+)\*\*)");
  static const PromptPattern italic(R"((?<!\*)\*([^*\n]+)\*(?!\*))");
  static const PromptPattern trailing_whitespace(R"([\s\x1c-\x1f]+$)");
  static const PromptPattern horizontal_rule(
      R"((?m)^[\s\x1c-\x1f]*[-*_]{3,}[\s\x1c-\x1f]*$)");
  static const PromptPattern inline_bullet("• ");
  static const PromptPattern indentation("    ");
  static const PromptPattern blank_lines("\n{2,}");
  static const PromptPattern leading_tags(R"(^[ \t]*((?:\[[^\]]+\][ \t]*)+))");
  static const PromptPattern tag_space(R"(\] )");
  static const PromptPattern space_tag(R"( \[)");
  static const PromptPattern lyric_break(" \\^ ");
  static const PromptPattern tag(R"(\[([^\]]+)\])");
  std::string cleaned;
  size_t offset = 0;
  while (const auto match = special.Search(caption, offset)) {
    cleaned += caption.substr(offset, match->begin - offset);
    const std::string inner = trim.Replace(match->capture, "");
    if (const auto separator = whitespace.Search(inner)) {
      cleaned += inner.substr(0, separator->begin) + " is " +
                 inner.substr(separator->end);
    } else {
      cleaned += inner;
    }
    offset = match->end;
  }
  cleaned += caption.substr(offset);
  std::istringstream lines(line_break.Replace(cleaned, "\n"));
  cleaned.clear();
  std::string line;
  bool first_line = true;
  while (std::getline(lines, line)) {
    line = heading.Replace(line, "");
    line = bullet.Replace(line, "");
    line = asterisk_bullet.Replace(line, "");
    while (line.find("**") != std::string::npos) {
      const std::string replaced = bold.Replace(line, "$1");
      if (replaced == line) {
        break;
      } else {
        line = replaced;
      }
    }
    line = italic.Replace(line, "$1");
    line = trailing_whitespace.Replace(line, "");
    cleaned += (first_line ? "" : "\n") + line;
    first_line = false;
  }
  cleaned = horizontal_rule.Replace(cleaned, "");
  cleaned = inline_bullet.Replace(cleaned, "");
  cleaned = indentation.Replace(cleaned, "");
  cleaned = blank_lines.Replace(cleaned, "\n");
  std::istringstream lyric_lines(lyrics);
  std::string normalized;
  first_line = true;
  while (std::getline(lyric_lines, line)) {
    if (const auto tags = leading_tags.Search(line)) {
      line = trim.Replace(tags->capture, "");
    } else {
    }
    normalized += (first_line ? "" : "\n") + line;
    first_line = false;
  }
  if (!lyrics.empty() && lyrics.back() == '\n') {
    normalized += '\n';
  } else {
  }
  normalized = tag_space.Replace(normalized, "]\n");
  normalized = space_tag.Replace(normalized, "\n[");
  normalized = lyric_break.Replace(normalized, "\n");
  std::string lowered;
  offset = 0;
  while (const auto match = tag.Search(normalized, offset)) {
    lowered += normalized.substr(offset, match->begin - offset);
    const std::string &tag_text = match->capture;
    CFStringRef source = CFStringCreateWithBytes(
        kCFAllocatorDefault, reinterpret_cast<const UInt8 *>(tag_text.data()),
        static_cast<CFIndex>(tag_text.size()), kCFStringEncodingUTF8, false);
    if (source == nullptr) {
      throw std::invalid_argument("section tags must contain valid UTF-8");
    } else {
    }
    CFMutableStringRef lowercase =
        CFStringCreateMutableCopy(kCFAllocatorDefault, 0, source);
    CFRelease(source);
    // Note (Akazaakane): a null locale keeps tag casing independent of the
    // user's locale.
    CFStringLowercase(lowercase, nullptr);
    const CFRange characters = CFRangeMake(0, CFStringGetLength(lowercase));
    CFIndex byte_count = 0;
    CFStringGetBytes(lowercase, characters, kCFStringEncodingUTF8, 0, false,
                     nullptr, 0, &byte_count);
    std::string content(static_cast<size_t>(byte_count), '\0');
    CFStringGetBytes(lowercase, characters, kCFStringEncodingUTF8, 0, false,
                     reinterpret_cast<UInt8 *>(content.data()), byte_count,
                     nullptr);
    CFRelease(lowercase);
    lowered += "[" + content + "]";
    offset = match->end;
  }
  lowered += normalized.substr(offset);
  return "<|im_start|><|caption_start|>" + cleaned +
         "<|caption_end|><|lyrics_start|>[start]\n" + lowered +
         "<|lyrics_end|><|im_end|><|audio_start|>";
}

Request ParseRequest(const nlohmann::json &body,
                     const omni_common::Tokenizer &tokenizer,
                     const std::string &model_name, int context_tokens) {
  if (!body.is_object()) {
    throw std::invalid_argument("request must be a JSON object");
  } else {
  }
  const std::string lyrics = body.at("input").get<std::string>();
  const std::string caption = body.at("instructions").get<std::string>();
  if (lyrics.find_first_not_of(" \t\r\n") == std::string::npos ||
      caption.find_first_not_of(" \t\r\n") == std::string::npos) {
    throw std::invalid_argument(
        "input and instructions must be nonempty strings");
  } else if (!model_name.empty() && body.contains("model") &&
             body.at("model").get<std::string>() != model_name) {
    throw std::invalid_argument("model does not match the served model");
  } else if (body.value("stream", false)) {
    throw std::invalid_argument("Music3 supports non-streaming output only");
  } else if (body.contains("speed") &&
             (!body.at("speed").is_number() ||
              body.at("speed").get<double>() != 1.0)) {
    throw std::invalid_argument("Music3 supports only speed=1");
  } else {
  }
  Request request;
  const auto integer = [&](const std::string &name, int fallback, int maximum) {
    if (!body.contains(name)) {
      return fallback;
    } else if (!body.at(name).is_number_integer() ||
               body.at(name).get<double>() < 1 ||
               body.at(name).get<double>() > maximum) {
      throw std::invalid_argument(name + " must be an integer in [1, " +
                                  std::to_string(maximum) + "]");
    } else {
      return body.at(name).get<int>();
    }
  };
  request.max_frames =
      integer("max_new_tokens", request.max_frames, kMaximumFrames);
  request.dit_steps = integer("dit_steps", request.dit_steps, 100);
  if (body.contains("seed")) {
    const auto &seed = body.at("seed");
    if (!seed.is_number_integer() || seed.get<double>() < 0 ||
        seed.get<uint64_t>() > std::numeric_limits<int64_t>::max()) {
      throw std::invalid_argument("seed must be an integer in [0, 2**63)");
    } else {
      request.seed = seed.get<uint64_t>();
    }
  } else {
  }
  if (body.contains("dit_cfg_scale")) {
    if (!body.at("dit_cfg_scale").is_number()) {
      throw std::invalid_argument("dit_cfg_scale must be numeric");
    } else {
      request.dit_guidance = body.at("dit_cfg_scale").get<float>();
    }
  } else {
  }
  if (!std::isfinite(request.dit_guidance) || request.dit_guidance < 0) {
    throw std::invalid_argument("dit_cfg_scale must be finite and nonnegative");
  } else {
  }
  const std::string format = body.value("response_format", std::string("wav"));
  if (format != "wav" && format != "pcm") {
    throw std::invalid_argument("response_format must be wav or pcm");
  } else {
    request.pcm = format == "pcm";
  }
  request.token_ids = tokenizer.Encode(BuildPrompt(caption, lyrics));
  if (request.token_ids.size() > kMaximumPromptTokens ||
      request.token_ids.size() + request.max_frames + 1 >
          static_cast<size_t>(context_tokens)) {
    throw std::invalid_argument(
        "Music3 prompt or requested generation exceeds the context limit");
  } else {
  }
  return request;
}

} // namespace music3
