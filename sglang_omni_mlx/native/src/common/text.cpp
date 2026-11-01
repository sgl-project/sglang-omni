// SPDX-License-Identifier: Apache-2.0
#include "common/text.h"

#include <set>

namespace omni_common {

namespace {

constexpr size_t kTokenLoopWindow = 24;
constexpr size_t kTokenLoopMaxDistinct = 3;

} // namespace

bool IsUnicodeSpace(uint32_t code_point) {
  return code_point == ' ' || (code_point >= 0x09 && code_point <= 0x0D) ||
         (code_point >= 0x1C && code_point <= 0x1F) || code_point == 0x85 ||
         code_point == 0xA0 || code_point == 0x1680 ||
         (code_point >= 0x2000 && code_point <= 0x200A) ||
         code_point == 0x2028 || code_point == 0x2029 || code_point == 0x202F ||
         code_point == 0x205F || code_point == 0x3000;
}

// Note (Jiaxin Deng): assumes valid UTF-8, the only kind the tokenizer emits.
std::vector<uint32_t> CodePoints(const std::string &text) {
  std::vector<uint32_t> code_points;
  size_t i = 0;
  while (i < text.size()) {
    const auto byte = [&](size_t k) { return static_cast<uint8_t>(text[k]); };
    const uint8_t lead = byte(i);
    uint32_t cp = lead;
    size_t length = 1;
    if (lead >= 0xF0 && i + 3 < text.size()) {
      cp = ((lead & 0x07) << 18) | ((byte(i + 1) & 0x3F) << 12) |
           ((byte(i + 2) & 0x3F) << 6) | (byte(i + 3) & 0x3F);
      length = 4;
    } else if (lead >= 0xE0 && i + 2 < text.size()) {
      cp = ((lead & 0x0F) << 12) | ((byte(i + 1) & 0x3F) << 6) |
           (byte(i + 2) & 0x3F);
      length = 3;
    } else if (lead >= 0xC0 && i + 1 < text.size()) {
      cp = ((lead & 0x1F) << 6) | (byte(i + 1) & 0x3F);
      length = 2;
    } else {
    }
    code_points.push_back(cp);
    i += length;
  }
  return code_points;
}

std::string StripUnicodeWhitespace(const std::string &text) {
  const std::vector<uint32_t> code_points = CodePoints(text);
  size_t begin_cp = 0;
  size_t end_cp = code_points.size();
  while (begin_cp < end_cp && IsUnicodeSpace(code_points[begin_cp]))
    ++begin_cp;
  while (end_cp > begin_cp && IsUnicodeSpace(code_points[end_cp - 1]))
    --end_cp;
  size_t byte = 0;
  size_t begin_byte = 0;
  size_t end_byte = 0;
  for (size_t cp = 0; cp <= code_points.size(); ++cp) {
    if (cp == begin_cp)
      begin_byte = byte;
    if (cp == end_cp)
      end_byte = byte;
    if (cp == code_points.size())
      break;
    const uint8_t lead = static_cast<uint8_t>(text[byte]);
    byte += lead >= 0xF0 ? 4 : lead >= 0xE0 ? 3 : lead >= 0xC0 ? 2 : 1;
  }
  return text.substr(begin_byte, end_byte - begin_byte);
}

bool IsTokenLoop(const std::vector<int> &output_ids) {
  if (output_ids.size() < kTokenLoopWindow) {
    return false;
  } else {
  }
  const std::set<int> distinct(output_ids.end() - kTokenLoopWindow,
                               output_ids.end());
  return distinct.size() <= kTokenLoopMaxDistinct;
}

std::optional<std::vector<uint8_t>> DecodeBase64(const std::string &text) {
  // Note (Jiaxin Deng): matches Python's b64decode(validate=False), which
  // drops characters outside the alphabet but still checks the padding.
  std::string clean;
  for (const char c : text) {
    if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
        (c >= '0' && c <= '9') || c == '+' || c == '/' || c == '=') {
      clean.push_back(c);
    } else {
    }
  }
  size_t padding = 0;
  while (!clean.empty() && clean.back() == '=') {
    clean.pop_back();
    ++padding;
  }
  const size_t remainder = clean.size() % 4;
  if (clean.find('=') != std::string::npos || remainder == 1 ||
      (remainder != 0 && padding < 4 - remainder)) {
    return std::nullopt;
  } else {
  }
  const auto value = [](char c) -> uint32_t {
    if (c >= 'A' && c <= 'Z')
      return c - 'A';
    if (c >= 'a' && c <= 'z')
      return c - 'a' + 26;
    if (c >= '0' && c <= '9')
      return c - '0' + 52;
    return c == '+' ? 62 : 63;
  };
  std::vector<uint8_t> bytes;
  uint32_t buffer = 0;
  int bits = 0;
  for (const char c : clean) {
    buffer = (buffer << 6) | value(c);
    bits += 6;
    if (bits >= 8) {
      bits -= 8;
      bytes.push_back(static_cast<uint8_t>((buffer >> bits) & 0xFF));
    } else {
    }
  }
  return bytes;
}

} // namespace omni_common
