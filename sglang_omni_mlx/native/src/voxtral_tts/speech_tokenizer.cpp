// SPDX-License-Identifier: Apache-2.0
#include "voxtral_tts/speech_tokenizer.h"

#define PCRE2_CODE_UNIT_WIDTH 8
#include <pcre2.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <utility>

#include "common/layers.h"

namespace voxtral_tts {

namespace {

std::string DecodeBase64(const std::string &encoded) {
  static constexpr std::string_view alphabet =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string decoded;
  uint32_t accumulator = 0;
  int bit_count = -8;
  for (const unsigned char character : encoded) {
    if (character == '=') {
      break;
    } else {
    }
    const size_t value = alphabet.find(static_cast<char>(character));
    if (value == std::string_view::npos) {
      throw std::invalid_argument("invalid base64 token in tekken.json");
    } else {
    }
    accumulator = (accumulator << 6) | static_cast<uint32_t>(value);
    bit_count += 6;
    if (bit_count >= 0) {
      decoded.push_back(static_cast<char>((accumulator >> bit_count) & 0xff));
      bit_count -= 8;
      accumulator &= (1U << (bit_count + 8)) - 1;
    } else {
    }
  }
  return decoded;
}

} // namespace

struct SpeechTokenizer::Regex {
  pcre2_code *code = nullptr;

  ~Regex() {
    if (code != nullptr) {
      pcre2_code_free(code);
    } else {
    }
  }
};

SpeechTokenizer::SpeechTokenizer(
    const std::filesystem::path &checkpoint_directory)
    : split_pattern_(std::make_unique<Regex>()) {
  const nlohmann::json tokenizer =
      qwen3_asr::ReadJson(checkpoint_directory / "tekken.json");
  const nlohmann::json &tokenizer_config = tokenizer.at("config");
  const std::string pattern = tokenizer_config.at("pattern").get<std::string>();
  const int vocabulary_size =
      tokenizer_config.at("default_vocab_size").get<int>();
  special_token_count_ =
      tokenizer_config.at("default_num_special_tokens").get<int>();
  if (special_token_count_ < 1 || vocabulary_size <= special_token_count_) {
    throw std::invalid_argument("invalid Tekken vocabulary sizes");
  } else {
  }
  int error_code = 0;
  PCRE2_SIZE error_offset = 0;
  split_pattern_->code = pcre2_compile(
      reinterpret_cast<PCRE2_SPTR>(pattern.data()), pattern.size(),
      PCRE2_UTF | PCRE2_UCP, &error_code, &error_offset, nullptr);
  if (split_pattern_->code == nullptr) {
    throw std::invalid_argument("invalid Tekken split pattern at byte " +
                                std::to_string(error_offset));
  } else {
  }

  const int regular_token_count = vocabulary_size - special_token_count_;
  const nlohmann::json &vocabulary = tokenizer.at("vocab");
  if (vocabulary.size() < static_cast<size_t>(regular_token_count)) {
    throw std::invalid_argument("Tekken vocabulary is shorter than its config");
  } else {
  }
  for (int token_rank = 0; token_rank < regular_token_count; ++token_rank) {
    const nlohmann::json &token = vocabulary.at(token_rank);
    if (token.at("rank").get<int>() != token_rank) {
      throw std::invalid_argument("Tekken vocabulary ranks are not contiguous");
    } else {
    }
    const std::string token_bytes =
        DecodeBase64(token.at("token_bytes").get<std::string>());
    if (!byte_ranks_.emplace(token_bytes, token_rank).second) {
      throw std::invalid_argument("Tekken vocabulary contains duplicate bytes");
    } else {
    }
  }

  for (const nlohmann::json &special_token : tokenizer.at("special_tokens")) {
    const int token_id = special_token.at("rank").get<int>();
    const std::string token_text =
        special_token.at("token_str").get<std::string>();
    if (token_id < 0 || token_id >= special_token_count_) {
      continue;
    } else {
    }
    if (token_text == "<s>") {
      beginning_of_sequence_id_ = token_id;
    } else if (token_text == "[BEGIN_AUDIO]") {
      begin_audio_id_ = token_id;
    } else if (token_text == "[AUDIO]") {
      audio_id_ = token_id;
    } else if (token_text == "[NEXT_AUDIO_TEXT]") {
      text_to_audio_id_ = token_id;
    } else if (token_text == "[REPEAT_AUDIO_TEXT]") {
      audio_to_text_id_ = token_id;
    } else {
    }
  }
  if (beginning_of_sequence_id_ < 0 || begin_audio_id_ < 0 || audio_id_ < 0 ||
      text_to_audio_id_ < 0 || audio_to_text_id_ < 0) {
    throw std::invalid_argument("Tekken tokenizer is missing speech tokens");
  } else {
  }

  const nlohmann::json &audio_config = tokenizer.at("audio");
  const nlohmann::json &voice_counts =
      audio_config.at("voice_num_audio_tokens");
  for (const auto &[voice, count] : voice_counts.items()) {
    const int audio_token_count = count.get<int>();
    if (audio_token_count < 1) {
      throw std::invalid_argument(
          "Tekken voice placeholder count must be positive");
    } else {
    }
    voice_audio_token_counts_.emplace(voice, audio_token_count);
  }
}

SpeechTokenizer::~SpeechTokenizer() = default;

std::vector<int> SpeechTokenizer::EncodePiece(const std::string &piece) const {
  std::vector<std::string> byte_pieces;
  byte_pieces.reserve(piece.size());
  for (const char byte : piece) {
    byte_pieces.emplace_back(1, byte);
  }
  while (byte_pieces.size() > 1) {
    int lowest_rank = std::numeric_limits<int>::max();
    size_t merge_index = byte_pieces.size();
    std::string merged_piece;
    for (size_t index = 0; index + 1 < byte_pieces.size(); ++index) {
      const std::string candidate = byte_pieces[index] + byte_pieces[index + 1];
      const auto rank = byte_ranks_.find(candidate);
      if (rank != byte_ranks_.end() && rank->second < lowest_rank) {
        lowest_rank = rank->second;
        merge_index = index;
        merged_piece = candidate;
      } else {
      }
    }
    if (merge_index == byte_pieces.size()) {
      break;
    } else {
    }
    byte_pieces[merge_index] = std::move(merged_piece);
    byte_pieces.erase(byte_pieces.begin() +
                      static_cast<std::ptrdiff_t>(merge_index + 1));
  }
  std::vector<int> token_ids;
  token_ids.reserve(byte_pieces.size());
  for (const std::string &byte_piece : byte_pieces) {
    const auto token = byte_ranks_.find(byte_piece);
    if (token == byte_ranks_.end()) {
      throw std::runtime_error("Tekken vocabulary is missing a byte token");
    } else {
    }
    token_ids.push_back(token->second + special_token_count_);
  }
  return token_ids;
}

std::vector<int> SpeechTokenizer::EncodeText(const std::string &text) const {
  struct MatchData {
    pcre2_match_data *value;
    ~MatchData() { pcre2_match_data_free(value); }
  } match_data{
      pcre2_match_data_create_from_pattern(split_pattern_->code, nullptr)};
  if (match_data.value == nullptr) {
    throw std::runtime_error("cannot allocate Tekken regex match data");
  } else {
  }
  std::vector<int> token_ids;
  size_t text_offset = 0;
  while (text_offset < text.size()) {
    const int match_result = pcre2_match(
        split_pattern_->code, reinterpret_cast<PCRE2_SPTR>(text.data()),
        text.size(), text_offset, 0, match_data.value, nullptr);
    if (match_result < 0) {
      throw std::invalid_argument("Tekken regex could not tokenize input text");
    } else {
    }
    PCRE2_SIZE *match = pcre2_get_ovector_pointer(match_data.value);
    if (match[0] != text_offset || match[1] <= text_offset) {
      throw std::invalid_argument(
          "Tekken regex produced a noncontiguous match");
    } else {
    }
    std::vector<int> piece_token_ids =
        EncodePiece(text.substr(text_offset, match[1] - text_offset));
    token_ids.insert(token_ids.end(), piece_token_ids.begin(),
                     piece_token_ids.end());
    text_offset = match[1];
  }
  return token_ids;
}

std::vector<int>
SpeechTokenizer::EncodeSpeechRequest(const std::string &text,
                                     const std::string &voice) const {
  if (text.empty() ||
      std::all_of(text.begin(), text.end(), [](unsigned char character) {
        return std::isspace(character) != 0;
      })) {
    throw std::invalid_argument("Voxtral TTS input must contain text");
  } else {
  }
  const auto voice_count = voice_audio_token_counts_.find(voice);
  if (voice_count == voice_audio_token_counts_.end()) {
    throw std::invalid_argument("unknown Voxtral voice: " + voice);
  } else {
  }
  std::vector<int> token_ids = {beginning_of_sequence_id_, begin_audio_id_};
  token_ids.insert(token_ids.end(), voice_count->second, audio_id_);
  token_ids.push_back(text_to_audio_id_);
  const std::vector<int> text_token_ids = EncodeText(text);
  token_ids.insert(token_ids.end(), text_token_ids.begin(),
                   text_token_ids.end());
  token_ids.push_back(audio_to_text_id_);
  token_ids.push_back(begin_audio_id_);
  return token_ids;
}

} // namespace voxtral_tts
