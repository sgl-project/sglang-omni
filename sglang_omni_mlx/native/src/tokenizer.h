// SPDX-License-Identifier: Apache-2.0
// The checkpoint's Qwen2 byte-level BPE tokenizer, with its added tokens.
#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace qwen3_asr {

class Tokenizer {
public:
  // Reads vocab.json, merges.txt and tokenizer_config.json from the checkpoint.
  explicit Tokenizer(const std::filesystem::path &model_directory);
  ~Tokenizer();
  Tokenizer(const Tokenizer &) = delete;
  Tokenizer &operator=(const Tokenizer &) = delete;

  std::vector<int> Encode(const std::string &text) const;
  // Lossy UTF-8 decoding: invalid bytes become U+FFFD, as Hugging Face does.
  std::string Decode(const std::vector<int> &ids,
                     bool skip_special_tokens) const;
  // -1 when the token is not an added token.
  int AddedTokenId(const std::string &content) const;

private:
  struct AddedToken {
    std::string content;
    int id;
    bool special;
  };
  struct Pcre2Regex;

  void EncodeOrdinary(const std::string &text, std::vector<int> &ids) const;
  void EncodeWord(const std::string &byte_level_word,
                  std::vector<int> &ids) const;

  std::unordered_map<std::string, int> vocabulary_;
  std::vector<std::string> tokens_by_id_;
  std::unordered_map<std::string, int> merge_ranks_;
  std::vector<AddedToken> added_tokens_;
  std::unordered_map<int, size_t> added_token_index_by_id_;
  std::unique_ptr<Pcre2Regex> split_pattern_;
};

} // namespace qwen3_asr
