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
  } else {
  }
}

const std::filesystem::path kRoot =
    std::filesystem::temp_directory_path() / "moss-tokenizer-preference";

std::filesystem::path FreshDirectory(const std::string &name) {
  const std::filesystem::path directory = kRoot / name;
  std::filesystem::remove_all(directory);
  std::filesystem::create_directories(directory);
  return directory;
}

// The legacy vocab/merges/config triplet every fallback test shares: a merge
// ("h"+"e" -> "he"), an NFC case (é byte-levels to "Ã©"), and one special
// added token.
void WriteLegacyFiles(const std::filesystem::path &directory) {
  std::ofstream(directory / "vocab.json")
      << R"({"h": 0, "e": 1, "he": 2, "Ã": 3, "©": 4, "Ã©": 5})";
  std::ofstream(directory / "merges.txt")
      << "#version: 0.2\nh e\n\xC3\x83 \xC2\xA9\n";
  std::ofstream(directory / "tokenizer_config.json")
      << R"({"added_tokens_decoder": {"99": {"content": "<|endoftext|>",)"
         R"( "special": true}}})";
}

nlohmann::json AddedToken(int id, const std::string &content, bool special) {
  return {{"id", id},          {"content", content}, {"single_word", false},
          {"lstrip", false},   {"rstrip", false},    {"normalized", false},
          {"special", special}};
}

nlohmann::json SplitPreTokenizer(const std::string &pattern) {
  nlohmann::json split = {
      {"type", "Split"}, {"behavior", "Isolated"}, {"invert", false}};
  split["pattern"]["Regex"] = pattern;
  return {{"type", "Sequence"},
          {"pretokenizers",
           {split,
            {{"type", "ByteLevel"},
             {"add_prefix_space", false},
             {"use_regex", false}}}}};
}

// The same model as WriteLegacyFiles, in tokenizer.json form.
nlohmann::json EquivalentTokenizerJson() {
  // All two-element initializer lists become objects in nlohmann::json, so
  // the merge pairs are built explicitly.
  nlohmann::json merges = nlohmann::json::array();
  merges.push_back({"h", "e"});
  merges.push_back({"Ã", "©"});
  nlohmann::json tokenizer;
  tokenizer["model"] = {
      {"type", "BPE"},
      {"vocab", {{"h", 0}, {"e", 1}, {"he", 2}, {"Ã", 3}, {"©", 4}, {"Ã©", 5}}},
      {"merges", merges}};
  tokenizer["normalizer"] = {{"type", "NFC"}};
  tokenizer["pre_tokenizer"] =
      SplitPreTokenizer(omni_common::kQwen2SplitPattern);
  tokenizer["added_tokens"] = {AddedToken(99, "<|endoftext|>", true)};
  return tokenizer;
}

void RequireSameBehaviour(const omni_common::Tokenizer &first,
                          const omni_common::Tokenizer &second,
                          const std::string &context) {
  const std::vector<std::string> texts{
      "he", "h", "\u00e9", "e\u0301", "<|endoftext|>", "he<|endoftext|>"};
  for (const std::string &text : texts) {
    Require(first.Encode(text) == second.Encode(text),
            context + ": Encode mismatch for " + text);
  }
  for (const bool skip : {true, false}) {
    Require(first.Decode({2, 99}, skip) == second.Decode({2, 99}, skip),
            context + ": Decode mismatch");
  }
  Require(first.AddedTokenId("<|endoftext|>") ==
              second.AddedTokenId("<|endoftext|>"),
          context + ": AddedTokenId mismatch");
  Require(first.AddedTokenId("<|nope|>") == second.AddedTokenId("<|nope|>"),
          context + ": AddedTokenId miss mismatch");
}

void TestPrefersTokenizerJson() {
  const std::filesystem::path directory = FreshDirectory("preferred");
  std::ofstream(directory / "vocab.json") << R"({"hello": 1, "world": 3})";
  std::ofstream(directory / "merges.txt") << "#version: 0.2\n";
  std::ofstream(directory / "tokenizer_config.json") << "{}";
  std::ofstream(directory / "tokenizer.json")
      << EquivalentTokenizerJson().dump(2);

  const omni_common::Tokenizer preferred(directory);
  Require(preferred.Decode({2}, true) == "he",
          "the constructor must prefer tokenizer.json when it exists");
  Require(preferred.AddedTokenId("<|endoftext|>") == 99,
          "the preferred path must load the added tokens");
}

void TestJsonAndLegacyAgree() {
  const std::filesystem::path json_directory = FreshDirectory("json-only");
  std::ofstream(json_directory / "tokenizer.json")
      << EquivalentTokenizerJson().dump(2);
  const std::filesystem::path legacy_directory = FreshDirectory("legacy-only");
  WriteLegacyFiles(legacy_directory);

  const omni_common::Tokenizer from_json(json_directory);
  const omni_common::Tokenizer from_legacy(legacy_directory);
  RequireSameBehaviour(from_json, from_legacy, "tokenizer.json vs legacy");
  const omni_common::Tokenizer direct =
      omni_common::Tokenizer::FromJson(EquivalentTokenizerJson());
  RequireSameBehaviour(direct, from_legacy, "FromJson vs legacy");
  Require(from_json.Encode("e\u0301") == from_json.Encode("\u00e9"),
          "NFC normalization must apply on the tokenizer.json path");
}

void TestWhisperShapeFallsBack() {
  const std::filesystem::path directory = FreshDirectory("whisper-shape");
  WriteLegacyFiles(directory);
  nlohmann::json whisper;
  whisper["model"] = {{"type", "BPE"},
                      {"vocab", {{"h", 0}}},
                      {"merges", nlohmann::json::array()}};
  whisper["normalizer"] = nullptr;
  whisper["pre_tokenizer"] = {
      {"type", "ByteLevel"}, {"add_prefix_space", false}, {"use_regex", true}};
  whisper["added_tokens"] = [] {
    nlohmann::json token = AddedToken(50257, "<|0.00|>", false);
    token["normalized"] = true;
    return nlohmann::json::array({token});
  }();
  std::ofstream(directory / "tokenizer.json") << whisper.dump(2);

  const std::filesystem::path legacy_directory =
      FreshDirectory("whisper-legacy");
  WriteLegacyFiles(legacy_directory);
  const omni_common::Tokenizer fallback(directory);
  const omni_common::Tokenizer legacy(legacy_directory);
  RequireSameBehaviour(fallback, legacy, "Whisper-shape fallback");
}

void TestTruncatedJsonFallsBack() {
  const std::filesystem::path directory = FreshDirectory("truncated");
  WriteLegacyFiles(directory);
  std::ofstream(directory / "tokenizer.json") << R"({"model": {"type": "BPE")";

  const omni_common::Tokenizer fallback(directory);
  Require(fallback.Encode("he") == std::vector<int>{2},
          "a truncated tokenizer.json must fall back to the legacy files");
}

void TestPartialJsonStateIsReset() {
  {
    const std::filesystem::path directory = FreshDirectory("poison-vocab");
    WriteLegacyFiles(directory);
    nlohmann::json poisoned;
    poisoned["model"] = {
        {"type", "BPE"}, {"vocab", {{"h", 50}}}, {"merges", {{"h"}}}};
    poisoned["normalizer"] = {{"type", "NFC"}};
    poisoned["pre_tokenizer"] =
        SplitPreTokenizer(omni_common::kQwen2SplitPattern);
    poisoned["added_tokens"] = nlohmann::json::array();
    std::ofstream(directory / "tokenizer.json") << poisoned.dump(2);

    const omni_common::Tokenizer fallback(directory);
    Require(fallback.Encode("h") == std::vector<int>{0},
            "vocabulary entries from a failed tokenizer.json must not leak");
  }
  {
    const std::filesystem::path directory = FreshDirectory("poison-added");
    WriteLegacyFiles(directory);
    nlohmann::json poisoned;
    poisoned["model"] = {{"type", "BPE"},
                         {"vocab", {{"h", 50}}},
                         {"merges", nlohmann::json::array()}};
    poisoned["normalizer"] = {{"type", "NFC"}};
    poisoned["pre_tokenizer"] =
        SplitPreTokenizer(omni_common::kQwen2SplitPattern);
    nlohmann::json bad = AddedToken(43, "<|y|>", false);
    bad["normalized"] = true;
    poisoned["added_tokens"] = {AddedToken(42, "<|x|>", false), bad};
    std::ofstream(directory / "tokenizer.json") << poisoned.dump(2);

    const omni_common::Tokenizer fallback(directory);
    Require(fallback.AddedTokenId("<|x|>") == -1,
            "added tokens from a failed tokenizer.json must not leak");
  }
  {
    const std::filesystem::path directory = FreshDirectory("poison-merges");
    std::ofstream(directory / "vocab.json")
        << R"({"a": 0, "b": 1, "c": 2, "ab": 3, "bc": 4})";
    std::ofstream(directory / "merges.txt") << "#version: 0.2\na b\nb c\n";
    std::ofstream(directory / "tokenizer_config.json")
        << R"({"added_tokens_decoder": {}})";
    nlohmann::json poisoned;
    nlohmann::json merges = nlohmann::json::array();
    // Two valid merges in the reverse of the legacy rank order, then a
    // malformed one: any leaked rank would reorder the legacy BPE.
    merges.push_back({"b", "c"});
    merges.push_back({"a", "b"});
    merges.push_back({"a"});
    poisoned["model"] = {
        {"type", "BPE"},
        {"vocab", {{"a", 0}, {"b", 1}, {"c", 2}, {"ab", 3}, {"bc", 4}}},
        {"merges", merges}};
    poisoned["normalizer"] = {{"type", "NFC"}};
    poisoned["pre_tokenizer"] =
        SplitPreTokenizer(omni_common::kQwen2SplitPattern);
    poisoned["added_tokens"] = nlohmann::json::array();
    std::ofstream(directory / "tokenizer.json") << poisoned.dump(2);

    const omni_common::Tokenizer fallback(directory);
    Require(fallback.Encode("abc") == std::vector<int>{3, 2},
            "merge ranks from a failed tokenizer.json must not leak");
  }
  {
    const std::filesystem::path directory =
        FreshDirectory("poison-added-index");
    WriteLegacyFiles(directory);
    nlohmann::json poisoned;
    poisoned["model"] = {{"type", "BPE"},
                         {"vocab", {{"h", 0}}},
                         {"merges", nlohmann::json::array()}};
    poisoned["normalizer"] = {{"type", "NFC"}};
    poisoned["pre_tokenizer"] =
        SplitPreTokenizer(omni_common::kQwen2SplitPattern);
    // A valid added token whose id the legacy files never use, then a
    // malformed one: a leaked id index would alias the legacy added token.
    nlohmann::json bad = AddedToken(43, "<|y|>", false);
    bad["normalized"] = true;
    poisoned["added_tokens"] = {AddedToken(42, "<|poison|>", false), bad};
    std::ofstream(directory / "tokenizer.json") << poisoned.dump(2);

    const omni_common::Tokenizer fallback(directory);
    Require(fallback.Decode({42}, false).empty(),
            "added-token ids from a failed tokenizer.json must not leak");
  }
  {
    const std::filesystem::path directory = FreshDirectory("poison-pattern");
    WriteLegacyFiles(directory);
    nlohmann::json poisoned;
    poisoned["model"] = {{"type", "BPE"},
                         {"vocab", {{"h", 50}}},
                         {"merges", nlohmann::json::array()}};
    poisoned["normalizer"] = {{"type", "NFC"}};
    poisoned["pre_tokenizer"] = SplitPreTokenizer("([a-");
    poisoned["added_tokens"] = nlohmann::json::array();
    std::ofstream(directory / "tokenizer.json") << poisoned.dump(2);

    const omni_common::Tokenizer fallback(directory);
    Require(fallback.Decode({50}, false).empty(),
            "id tables from a failed tokenizer.json must not leak");
    Require(fallback.Encode("h") == std::vector<int>{0},
            "the legacy split pattern must be compiled after a fallback");
  }
}

void TestCorruptJsonOnlyKeepsTheError() {
  const std::filesystem::path directory = FreshDirectory("corrupt-only");
  std::ofstream(directory / "tokenizer.json") << "not json at all";
  bool threw = false;
  try {
    omni_common::Tokenizer tokenizer(directory);
  } catch (const std::exception &) {
    threw = true;
  }
  Require(threw,
          "a corrupt tokenizer.json without legacy files must still throw");
}

} // namespace

int main() {
  TestPrefersTokenizerJson();
  TestJsonAndLegacyAgree();
  TestWhisperShapeFallsBack();
  TestTruncatedJsonFallsBack();
  TestPartialJsonStateIsReset();
  TestCorruptJsonOnlyKeepsTheError();
  std::filesystem::remove_all(kRoot);
  return 0;
}
