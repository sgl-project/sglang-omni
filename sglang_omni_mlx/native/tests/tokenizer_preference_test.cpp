// SPDX-License-Identifier: Apache-2.0
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../src/common/tokenizer.cpp"

namespace {

void Require(bool condition, const std::string &message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

} // namespace

int main() {
  const std::filesystem::path directory =
      std::filesystem::temp_directory_path() / "moss-tokenizer-preference";
  std::filesystem::remove_all(directory);
  std::filesystem::create_directories(directory);
  std::ofstream(directory / "vocab.json") << R"({"hello": 1, "world": 3})";
  std::ofstream(directory / "merges.txt") << "#version: 0.2\n";
  std::ofstream(directory / "tokenizer_config.json") << "{}";

  nlohmann::json tokenizer;
  tokenizer["model"] = {
      {"type", "BPE"},
      {"vocab", {{"hello", 2}, {"world", 3}}},
      {"merges", nlohmann::json::array()}};
  tokenizer["normalizer"]["type"] = "NFC";
  nlohmann::json added = nlohmann::json::object();
  added["id"] = 99;
  added["content"] = "<|endoftext|>";
  added["single_word"] = false;
  added["lstrip"] = false;
  added["rstrip"] = false;
  added["normalized"] = false;
  added["special"] = true;
  tokenizer["added_tokens"].push_back(added);
  nlohmann::json split = {
      {"type", "Split"},
      {"behavior", "Isolated"},
      {"invert", false}};
  split["pattern"]["Regex"] = omni_common::kQwen2SplitPattern;
  nlohmann::json byte_level = nlohmann::json::object();
  byte_level["type"] = "ByteLevel";
  byte_level["add_prefix_space"] = false;
  byte_level["use_regex"] = false;
  tokenizer["pre_tokenizer"]["type"] = "Sequence";
  tokenizer["pre_tokenizer"]["pretokenizers"].push_back(split);
  tokenizer["pre_tokenizer"]["pretokenizers"].push_back(byte_level);
  std::ofstream(directory / "tokenizer.json") << tokenizer.dump(2);

  omni_common::Tokenizer preferred(directory);
  const std::vector<int> expected{2};
  Require(preferred.Decode(expected, true) == "hello",
          "the constructor must prefer tokenizer.json when it exists");
  Require(omni_common::Tokenizer::FromJson(tokenizer).Decode(expected, true) == "hello",
          "the preferred and direct tokenizer.json paths must agree");
  return 0;
}
