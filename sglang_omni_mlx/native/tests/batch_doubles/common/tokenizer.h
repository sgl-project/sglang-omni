// SPDX-License-Identifier: Apache-2.0
// In-memory tokenizer for the MOSS batch timing regression.
#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace omni_common {

class Tokenizer {
public:
  explicit Tokenizer(const std::filesystem::path &) {}

  std::vector<int> Encode(const std::string &) const { return {3}; }
  std::string Decode(const std::vector<int> &, bool) const { return ""; }

  int AddedTokenId(const std::string &content) const {
    if (content == "<|endoftext|>") {
      return 0;
    } else if (content == "<|im_end|>") {
      return 1;
    } else {
      return -1;
    }
  }
};

} // namespace omni_common
