// SPDX-License-Identifier: Apache-2.0
// The checkpoint's byte-level BPE tokenizer, with Qwen2's or GPT-2's split,
// and its added tokens.
#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "nlohmann/json_fwd.hpp"

namespace omni_common {

// The regular expression that splits vocab.json/merges.txt text into words.
enum class SplitPattern { kQwen2, kGpt2 };

class Tokenizer {
public:
  // Reads byte-level BPE from vocab.json/merges.txt or tokenizer.json.
  explicit Tokenizer(const std::filesystem::path &model_directory,
                     SplitPattern split_pattern = SplitPattern::kQwen2);
  static Tokenizer FromJson(const nlohmann::json &tokenizer);
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
  struct Serialized {};
  Tokenizer(const nlohmann::json &tokenizer, Serialized);
  void LoadJson(const nlohmann::json &tokenizer);

  struct AddedToken {
    std::string content;
    int id;
    bool special;
  };
  struct Pcre2Regex;

  void Initialize(const std::string &split_pattern);
  void EncodeOrdinary(const std::string &text, std::vector<int> &ids) const;
  void EncodeWord(const std::string &byte_level_word,
                  std::vector<int> &ids) const;

  std::unordered_map<std::string, int> vocabulary_;
  std::vector<std::string> tokens_by_id_;
  std::unordered_map<std::string, int> merge_ranks_;
  std::vector<AddedToken> added_tokens_;
  std::unordered_map<int, size_t> added_token_index_by_id_;
  std::unique_ptr<Pcre2Regex> split_pattern_;
  // Qwen2 and tokenizer.json text is NFC-normalized before the split; GPT-2
  // text is not.
  bool normalizes_nfc_ = true;
};

} // namespace omni_common
