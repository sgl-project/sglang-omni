// SPDX-License-Identifier: Apache-2.0
#include "voxtral_tts/voxtral_tts_tokenizer.h"

#define PCRE2_CODE_UNIT_WIDTH 8
#include <pcre2.h>

#include <limits>
#include <stdexcept>

#include "common/layers.h"
#include "qwen3_asr/realtime.h"

namespace voxtral_tts {

namespace {

constexpr int kNoRank = std::numeric_limits<int>::max();

int SpecialTokenId(const nlohmann::json &special_tokens,
                   const std::string &token_text) {
  for (const auto &token : special_tokens) {
    if (token.at("token_str").get<std::string>() == token_text) {
      return token.at("rank").get<int>();
    } else {
    }
  }
  throw std::runtime_error("tekken.json has no special token " + token_text);
}

} // namespace

struct SpeechTokenizer::Pcre2Regex {
  pcre2_code *code = nullptr;
  ~Pcre2Regex() {
    if (code != nullptr) {
      pcre2_code_free(code);
    } else {
    }
  }
};

SpeechTokenizer::SpeechTokenizer(const std::filesystem::path &model_directory)
    : split_pattern_(std::make_unique<Pcre2Regex>()) {
  const nlohmann::json tekken =
      qwen3_asr::ReadJson(model_directory / "tekken.json");
  const nlohmann::json &config = tekken.at("config");
  special_token_count_ = config.at("default_num_special_tokens").get<int>();
  // Note (Minyi Tang): tekken.json lists more ranks than the model uses;
  // mistral-common keeps the vocabulary size minus the special tokens.
  const int ordinary_token_count =
      config.at("default_vocab_size").get<int>() - special_token_count_;
  const nlohmann::json &vocabulary = tekken.at("vocab");
  if (static_cast<int>(vocabulary.size()) < ordinary_token_count) {
    throw std::runtime_error("tekken.json lists fewer tokens than its "
                             "vocabulary size");
  } else {
  }
  ranks_.reserve(ordinary_token_count);
  for (int rank = 0; rank < ordinary_token_count; ++rank) {
    const auto bytes = qwen3_asr::DecodeBase64(
        vocabulary[rank].at("token_bytes").get<std::string>());
    if (!bytes.has_value() || vocabulary[rank].at("rank").get<int>() != rank) {
      throw std::runtime_error("tekken.json has a malformed token at rank " +
                               std::to_string(rank));
    } else {
    }
    ranks_.emplace(std::string(bytes->begin(), bytes->end()), rank);
  }
  const nlohmann::json &special_tokens = tekken.at("special_tokens");
  begin_of_sequence_token_id_ = SpecialTokenId(special_tokens, "<s>");
  begin_audio_token_id_ = SpecialTokenId(special_tokens, "[BEGIN_AUDIO]");
  audio_token_id_ = SpecialTokenId(special_tokens, "[AUDIO]");
  text_to_audio_token_id_ = SpecialTokenId(special_tokens, "[NEXT_AUDIO_TEXT]");
  audio_to_text_token_id_ =
      SpecialTokenId(special_tokens, "[REPEAT_AUDIO_TEXT]");
  for (const auto &[voice, frame_count] :
       tekken.at("audio").at("voice_num_audio_tokens").items()) {
    voice_frame_counts_.emplace(voice, frame_count.get<int>());
  }

  const std::string pattern = config.at("pattern").get<std::string>();
  int error_code = 0;
  PCRE2_SIZE error_offset = 0;
  split_pattern_->code = pcre2_compile(
      reinterpret_cast<PCRE2_SPTR>(pattern.data()), pattern.size(),
      PCRE2_UTF | PCRE2_UCP, &error_code, &error_offset, nullptr);
  if (split_pattern_->code == nullptr) {
    throw std::runtime_error("cannot compile the Tekken split pattern");
  } else {
  }
}

SpeechTokenizer::~SpeechTokenizer() = default;

std::vector<int> SpeechTokenizer::Encode(const std::string &text) const {
  std::vector<int> ids;
  pcre2_match_data *match_data =
      pcre2_match_data_create_from_pattern(split_pattern_->code, nullptr);
  const auto subject = reinterpret_cast<PCRE2_SPTR>(text.data());
  size_t offset = 0;
  int result = 0;
  while (offset < text.size()) {
    result = pcre2_match(split_pattern_->code, subject, text.size(), offset, 0,
                         match_data, nullptr);
    if (result < 0) {
      break;
    } else {
    }
    const PCRE2_SIZE *ovector = pcre2_get_ovector_pointer(match_data);
    // Note (Minyi Tang): every alternative of the pattern consumes a
    // character, so a match always advances.
    EncodePiece(text.substr(ovector[0], ovector[1] - ovector[0]), ids);
    offset = ovector[1];
  }
  pcre2_match_data_free(match_data);
  if (result < 0 && result != PCRE2_ERROR_NOMATCH) {
    throw std::invalid_argument("input must be UTF-8 text");
  } else {
  }
  return ids;
}

void SpeechTokenizer::EncodePiece(const std::string &piece,
                                  std::vector<int> &ids) const {
  const auto whole = ranks_.find(piece);
  if (whole != ranks_.end()) {
    ids.push_back(whole->second + special_token_count_);
    return;
  } else {
  }
  // Merges the adjacent pair with the lowest rank, the leftmost on a tie,
  // until no pair has one. Each part keeps its start and the rank of it
  // merged with the next part.
  struct Part {
    size_t start;
    int rank;
  };
  const auto rank_of = [&](size_t begin, size_t end) {
    const auto found = ranks_.find(piece.substr(begin, end - begin));
    return found == ranks_.end() ? kNoRank : found->second;
  };
  std::vector<Part> parts;
  parts.reserve(piece.size() + 1);
  for (size_t i = 0; i + 1 < piece.size(); ++i) {
    parts.push_back({i, rank_of(i, i + 2)});
  }
  parts.push_back({piece.size() - 1, kNoRank});
  parts.push_back({piece.size(), kNoRank});
  while (true) {
    int best_rank = kNoRank;
    size_t best_index = 0;
    for (size_t i = 0; i + 1 < parts.size(); ++i) {
      if (parts[i].rank < best_rank) {
        best_rank = parts[i].rank;
        best_index = i;
      } else {
      }
    }
    if (best_rank == kNoRank) {
      break;
    } else {
    }
    const auto merged_rank = [&](size_t index) {
      return index + 3 < parts.size()
                 ? rank_of(parts[index].start, parts[index + 3].start)
                 : kNoRank;
    };
    if (best_index > 0) {
      parts[best_index - 1].rank = merged_rank(best_index - 1);
    } else {
    }
    parts[best_index].rank = merged_rank(best_index);
    parts.erase(parts.begin() + best_index + 1);
  }
  for (size_t i = 0; i + 1 < parts.size(); ++i) {
    ids.push_back(ranks_.at(piece.substr(parts[i].start,
                                         parts[i + 1].start - parts[i].start)) +
                  special_token_count_);
  }
}

std::vector<int>
SpeechTokenizer::EncodeSpeechRequest(const std::string &text,
                                     const std::string &voice) const {
  const auto voice_frames = voice_frame_counts_.find(voice);
  if (voice_frames == voice_frame_counts_.end()) {
    throw std::invalid_argument("unknown voice " + voice);
  } else {
  }
  std::vector<int> ids = {begin_of_sequence_token_id_, begin_audio_token_id_};
  ids.insert(ids.end(), voice_frames->second, audio_token_id_);
  ids.push_back(text_to_audio_token_id_);
  const std::vector<int> text_ids = Encode(text);
  ids.insert(ids.end(), text_ids.begin(), text_ids.end());
  ids.push_back(audio_to_text_token_id_);
  ids.push_back(begin_audio_token_id_);
  return ids;
}

} // namespace voxtral_tts
