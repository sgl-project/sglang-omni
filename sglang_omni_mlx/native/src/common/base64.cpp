// SPDX-License-Identifier: Apache-2.0
#include "common/base64.h"

#include <stdexcept>

namespace omni_common {

constexpr bool IsBase64(char c) {
  return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
         (c >= '0' && c <= '9') || c == '+' || c == '/' || c == '=';
}

uint32_t Base64Value(char c) {
  if (c >= 'A' && c <= 'Z')
    return static_cast<uint32_t>(c - 'A');
  if (c >= 'a' && c <= 'z')
    return static_cast<uint32_t>(c - 'a' + 26);
  if (c >= '0' && c <= '9')
    return static_cast<uint32_t>(c - '0' + 52);
  return c == '+' ? 62 : 63;
}

std::vector<uint8_t> DecodeBase64(const std::string &input, bool strict) {
  std::string clean;
  clean.reserve(input.size());
  for (const char c : input) {
    if (IsBase64(c)) {
      clean.push_back(c);
    } else if (strict) {
      throw std::invalid_argument(
          "base64 input has a character outside the alphabet");
    }
  }
  // note (guozhihao-224): the padding is checked like Python's b64decode:
  // no '=' in the middle, and the final quantum is not a lone byte.
  size_t padding = 0;
  while (!clean.empty() && clean.back() == '=') {
    clean.pop_back();
    ++padding;
  }
  const size_t remainder = clean.size() % 4;
  if (clean.find('=') != std::string::npos || remainder == 1 ||
      (remainder != 0 && padding < 4 - remainder)) {
    throw std::invalid_argument("base64 input has invalid padding");
  }
  const size_t expected_padding = remainder == 0 ? 0 : 4 - remainder;
  if (strict && padding != expected_padding) {
    throw std::invalid_argument("base64 input has invalid padding");
  }
  if (strict && !clean.empty() && remainder != 0) {
    // note (guozhihao-224): canonical encodings zero the unused low bits of
    // the final quantum; reject leftover bits like Python b64decode(validate).
    const size_t last = clean.size() - 1;
    const uint32_t leftover_bits = (6 * remainder) % 8;
    if ((Base64Value(clean[last]) & ((1u << leftover_bits) - 1)) != 0) {
      throw std::invalid_argument("base64 input is not canonical");
    }
  }
  std::vector<uint8_t> bytes;
  bytes.reserve(clean.size() * 3 / 4);
  uint32_t buffer = 0;
  int bits = 0;
  for (const char c : clean) {
    buffer = (buffer << 6) | Base64Value(c);
    bits += 6;
    if (bits >= 8) {
      bits -= 8;
      bytes.push_back(static_cast<uint8_t>((buffer >> bits) & 0xFF));
    }
  }
  return bytes;
}

} // namespace omni_common

namespace qwen3_asr {

std::vector<uint8_t> DecodeBase64(const std::string &input, bool strict) {
  return omni_common::DecodeBase64(input, strict);
}

} // namespace qwen3_asr
