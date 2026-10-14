// SPDX-License-Identifier: Apache-2.0
#include "tokenizer.h"

#define PCRE2_CODE_UNIT_WIDTH 8
#include <CoreFoundation/CoreFoundation.h>
#include <pcre2.h>

#include <algorithm>
#include <array>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>

#include "nlohmann/json.hpp"

namespace qwen3_asr {

namespace {

// Note (Jiaxin Deng): Qwen2's pre-tokenization split, verbatim from the
// checkpoint's tokenizer.
constexpr const char *kQwen2SplitPattern =
    R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N})"
    R"(| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)";

std::string ReadFile(const std::filesystem::path &path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) {
    throw std::runtime_error("cannot read " + path.string());
  } else {
  }
  std::ostringstream contents;
  contents << stream.rdbuf();
  return contents.str();
}

void AppendUtf8(uint32_t code_point, std::string &out) {
  if (code_point < 0x80) {
    out.push_back(static_cast<char>(code_point));
  } else if (code_point < 0x800) {
    out.push_back(static_cast<char>(0xC0 | (code_point >> 6)));
    out.push_back(static_cast<char>(0x80 | (code_point & 0x3F)));
  } else if (code_point < 0x10000) {
    out.push_back(static_cast<char>(0xE0 | (code_point >> 12)));
    out.push_back(static_cast<char>(0x80 | ((code_point >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (code_point & 0x3F)));
  } else {
    out.push_back(static_cast<char>(0xF0 | (code_point >> 18)));
    out.push_back(static_cast<char>(0x80 | ((code_point >> 12) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | ((code_point >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (code_point & 0x3F)));
  }
}

// Note (Jiaxin Deng): GPT-2's byte-to-unicode table; printable bytes map to
// themselves, the rest to code points from 256 upward.
struct ByteLevelTables {
  std::array<std::string, 256> byte_to_text;
  std::unordered_map<uint32_t, uint8_t> code_point_to_byte;

  ByteLevelTables() {
    std::array<uint32_t, 256> code_points{};
    std::array<bool, 256> printable{};
    for (int b = '!'; b <= '~'; ++b)
      printable[b] = true;
    for (int b = 0xA1; b <= 0xAC; ++b)
      printable[b] = true;
    for (int b = 0xAE; b <= 0xFF; ++b)
      printable[b] = true;
    uint32_t next = 256;
    for (int b = 0; b < 256; ++b) {
      code_points[b] = printable[b] ? static_cast<uint32_t>(b) : next++;
    }
    for (int b = 0; b < 256; ++b) {
      AppendUtf8(code_points[b], byte_to_text[b]);
      code_point_to_byte[code_points[b]] = static_cast<uint8_t>(b);
    }
  }
};

const ByteLevelTables &Tables() {
  static const ByteLevelTables tables;
  return tables;
}

// Note (Jiaxin Deng): assumes valid UTF-8; advances offset.
uint32_t NextCodePoint(const std::string &text, size_t &offset) {
  const auto byte = [&](size_t i) { return static_cast<uint8_t>(text[i]); };
  const uint8_t lead = byte(offset);
  if (lead < 0x80) {
    offset += 1;
    return lead;
  } else if ((lead >> 5) == 0x6) {
    const uint32_t cp = ((lead & 0x1F) << 6) | (byte(offset + 1) & 0x3F);
    offset += 2;
    return cp;
  } else if ((lead >> 4) == 0xE) {
    const uint32_t cp = ((lead & 0x0F) << 12) |
                        ((byte(offset + 1) & 0x3F) << 6) |
                        (byte(offset + 2) & 0x3F);
    offset += 3;
    return cp;
  } else {
    const uint32_t cp =
        ((lead & 0x07) << 18) | ((byte(offset + 1) & 0x3F) << 12) |
        ((byte(offset + 2) & 0x3F) << 6) | (byte(offset + 3) & 0x3F);
    offset += 4;
    return cp;
  }
}

// Note (Jiaxin Deng): NFC, as the tokenizer's normalizer applies to ordinary
// text.
std::string NormalizeNfc(const std::string &text) {
  if (std::all_of(text.begin(), text.end(),
                  [](char c) { return static_cast<uint8_t>(c) < 0x80; })) {
    return text;
  } else {
  }
  CFStringRef source = CFStringCreateWithBytes(
      kCFAllocatorDefault, reinterpret_cast<const UInt8 *>(text.data()),
      static_cast<CFIndex>(text.size()), kCFStringEncodingUTF8, false);
  if (source == nullptr) {
    return text;
  } else {
  }
  CFMutableStringRef normalized =
      CFStringCreateMutableCopy(kCFAllocatorDefault, 0, source);
  CFRelease(source);
  CFStringNormalize(normalized, kCFStringNormalizationFormC);
  const CFIndex length = CFStringGetLength(normalized);
  CFIndex byte_count = 0;
  CFStringGetBytes(normalized, CFRangeMake(0, length), kCFStringEncodingUTF8, 0,
                   false, nullptr, 0, &byte_count);
  std::string out(static_cast<size_t>(byte_count), '\0');
  CFStringGetBytes(normalized, CFRangeMake(0, length), kCFStringEncodingUTF8, 0,
                   false, reinterpret_cast<UInt8 *>(out.data()), byte_count,
                   nullptr);
  CFRelease(normalized);
  return out;
}

// Note (Jiaxin Deng): matches Rust's String::from_utf8_lossy, one U+FFFD per
// maximal invalid subpart.
std::string FromUtf8Lossy(const std::string &bytes) {
  std::string out;
  out.reserve(bytes.size());
  size_t i = 0;
  const auto byte = [&](size_t k) { return static_cast<uint8_t>(bytes[k]); };
  while (i < bytes.size()) {
    const uint8_t lead = byte(i);
    if (lead < 0x80) {
      out.push_back(static_cast<char>(lead));
      ++i;
      continue;
    } else {
    }
    size_t need = 0;
    uint8_t low = 0x80;
    uint8_t high = 0xBF;
    if (lead >= 0xC2 && lead <= 0xDF) {
      need = 1;
    } else if (lead >= 0xE0 && lead <= 0xEF) {
      need = 2;
      low = lead == 0xE0 ? 0xA0 : 0x80;
      high = lead == 0xED ? 0x9F : 0xBF;
    } else if (lead >= 0xF0 && lead <= 0xF4) {
      need = 3;
      low = lead == 0xF0 ? 0x90 : 0x80;
      high = lead == 0xF4 ? 0x8F : 0xBF;
    } else {
      out += "\xEF\xBF\xBD";
      ++i;
      continue;
    }
    size_t consumed = 1;
    bool valid = true;
    for (size_t k = 0; k < need; ++k) {
      if (i + consumed >= bytes.size()) {
        valid = false;
        break;
      } else {
      }
      const uint8_t continuation = byte(i + consumed);
      const uint8_t allowed_low = k == 0 ? low : 0x80;
      const uint8_t allowed_high = k == 0 ? high : 0xBF;
      if (continuation < allowed_low || continuation > allowed_high) {
        valid = false;
        break;
      } else {
      }
      ++consumed;
    }
    if (valid) {
      out.append(bytes, i, consumed);
    } else {
      out += "\xEF\xBF\xBD";
    }
    i += consumed;
  }
  return out;
}

} // namespace

struct Tokenizer::Pcre2Regex {
  pcre2_code *code = nullptr;
  ~Pcre2Regex() {
    if (code != nullptr) {
      pcre2_code_free(code);
    } else {
    }
  }
};

Tokenizer::Tokenizer(const std::filesystem::path &model_directory)
    : split_pattern_(std::make_unique<Pcre2Regex>()) {
  const nlohmann::json vocabulary =
      nlohmann::json::parse(ReadFile(model_directory / "vocab.json"));
  for (const auto &[token, id] : vocabulary.items()) {
    vocabulary_.emplace(token, id.get<int>());
  }
  std::istringstream merges(ReadFile(model_directory / "merges.txt"));
  std::string line;
  int rank = 0;
  while (std::getline(merges, line)) {
    if (line.rfind("#version", 0) == 0 || line.empty()) {
      continue;
    } else {
    }
    merge_ranks_.emplace(line, rank++);
  }
  const nlohmann::json config = nlohmann::json::parse(
      ReadFile(model_directory / "tokenizer_config.json"));
  if (config.contains("added_tokens_decoder")) {
    for (const auto &[id_text, token] :
         config.at("added_tokens_decoder").items()) {
      const int id = std::stoi(id_text);
      added_token_index_by_id_[id] = added_tokens_.size();
      added_tokens_.push_back({token.at("content").get<std::string>(), id,
                               token.at("special").get<bool>()});
    }
  } else {
    // Note (Dayuxiaoshui): newer checkpoints, MOSS-Transcribe-Diarize's among
    // them, list their added tokens only in tokenizer.json.
    const nlohmann::json tokenizer =
        nlohmann::json::parse(ReadFile(model_directory / "tokenizer.json"));
    for (const auto &token : tokenizer.at("added_tokens")) {
      const int id = token.at("id").get<int>();
      added_token_index_by_id_[id] = added_tokens_.size();
      added_tokens_.push_back({token.at("content").get<std::string>(), id,
                               token.at("special").get<bool>()});
    }
  }
  size_t largest_id = 0;
  for (const auto &[token, id] : vocabulary_)
    largest_id = std::max(largest_id, static_cast<size_t>(id));
  for (const auto &added : added_tokens_)
    largest_id = std::max(largest_id, static_cast<size_t>(added.id));
  tokens_by_id_.resize(largest_id + 1);
  for (const auto &[token, id] : vocabulary_)
    tokens_by_id_[id] = token;

  int error_code = 0;
  PCRE2_SIZE error_offset = 0;
  split_pattern_->code = pcre2_compile(
      reinterpret_cast<PCRE2_SPTR>(kQwen2SplitPattern), PCRE2_ZERO_TERMINATED,
      PCRE2_UTF | PCRE2_UCP, &error_code, &error_offset, nullptr);
  if (split_pattern_->code == nullptr) {
    throw std::runtime_error("cannot compile the Qwen2 split pattern");
  } else {
  }
}

Tokenizer::~Tokenizer() = default;

int Tokenizer::AddedTokenId(const std::string &content) const {
  for (const auto &added : added_tokens_) {
    if (added.content == content) {
      return added.id;
    } else {
    }
  }
  return -1;
}

std::vector<int> Tokenizer::Encode(const std::string &text) const {
  // Note (Jiaxin Deng): added tokens match on the raw text first (leftmost,
  // longest); only the text between them is normalized and BPE-encoded.
  std::vector<int> ids;
  size_t ordinary_start = 0;
  size_t position = 0;
  while (position < text.size()) {
    const AddedToken *match = nullptr;
    for (const auto &added : added_tokens_) {
      if (text.compare(position, added.content.size(), added.content) == 0 &&
          (match == nullptr || added.content.size() > match->content.size())) {
        match = &added;
      } else {
      }
    }
    if (match != nullptr) {
      EncodeOrdinary(text.substr(ordinary_start, position - ordinary_start),
                     ids);
      ids.push_back(match->id);
      position += match->content.size();
      ordinary_start = position;
    } else {
      ++position;
    }
  }
  EncodeOrdinary(text.substr(ordinary_start), ids);
  return ids;
}

void Tokenizer::EncodeOrdinary(const std::string &raw_text,
                               std::vector<int> &ids) const {
  if (raw_text.empty()) {
    return;
  } else {
  }
  const std::string text = NormalizeNfc(raw_text);
  pcre2_match_data *match_data =
      pcre2_match_data_create_from_pattern(split_pattern_->code, nullptr);
  const auto subject = reinterpret_cast<PCRE2_SPTR>(text.data());
  size_t offset = 0;
  size_t previous_end = 0;
  const auto encode_piece = [&](size_t begin, size_t end) {
    std::string byte_level;
    for (size_t i = begin; i < end; ++i) {
      byte_level += Tables().byte_to_text[static_cast<uint8_t>(text[i])];
    }
    EncodeWord(byte_level, ids);
  };
  while (offset <= text.size()) {
    const int result = pcre2_match(split_pattern_->code, subject, text.size(),
                                   offset, 0, match_data, nullptr);
    if (result < 0) {
      break;
    } else {
    }
    const PCRE2_SIZE *ovector = pcre2_get_ovector_pointer(match_data);
    const size_t match_begin = ovector[0];
    const size_t match_end = ovector[1];
    if (match_end == match_begin) {
      // Note (Jiaxin Deng): the pattern never matches empty text; guard
      // against a stall anyway.
      offset = match_end + 1;
      continue;
    } else {
    }
    if (match_begin > previous_end) {
      encode_piece(previous_end, match_begin);
    } else {
    }
    encode_piece(match_begin, match_end);
    previous_end = match_end;
    offset = match_end;
  }
  if (previous_end < text.size()) {
    encode_piece(previous_end, text.size());
  } else {
  }
  pcre2_match_data_free(match_data);
}

void Tokenizer::EncodeWord(const std::string &byte_level_word,
                           std::vector<int> &ids) const {
  std::vector<std::string> symbols;
  size_t offset = 0;
  while (offset < byte_level_word.size()) {
    const size_t start = offset;
    NextCodePoint(byte_level_word, offset);
    symbols.push_back(byte_level_word.substr(start, offset - start));
  }
  while (symbols.size() > 1) {
    int best_rank = std::numeric_limits<int>::max();
    size_t best_index = 0;
    for (size_t i = 0; i + 1 < symbols.size(); ++i) {
      const auto rank = merge_ranks_.find(symbols[i] + " " + symbols[i + 1]);
      if (rank != merge_ranks_.end() && rank->second < best_rank) {
        best_rank = rank->second;
        best_index = i;
      } else {
      }
    }
    if (best_rank == std::numeric_limits<int>::max()) {
      break;
    } else {
    }
    const std::string left = symbols[best_index];
    const std::string right = symbols[best_index + 1];
    std::vector<std::string> merged;
    merged.reserve(symbols.size());
    for (size_t i = 0; i < symbols.size(); ++i) {
      if (i + 1 < symbols.size() && symbols[i] == left &&
          symbols[i + 1] == right) {
        merged.push_back(left + right);
        ++i;
      } else {
        merged.push_back(symbols[i]);
      }
    }
    symbols.swap(merged);
  }
  for (const auto &symbol : symbols) {
    const auto id = vocabulary_.find(symbol);
    if (id != vocabulary_.end()) {
      ids.push_back(id->second);
    } else {
    }
  }
}

std::string Tokenizer::Decode(const std::vector<int> &ids,
                              bool skip_special_tokens) const {
  std::string bytes;
  for (const int id : ids) {
    const auto added = added_token_index_by_id_.find(id);
    std::string token;
    if (added != added_token_index_by_id_.end()) {
      const AddedToken &added_token = added_tokens_[added->second];
      if (skip_special_tokens && added_token.special) {
        continue;
      } else {
      }
      token = added_token.content;
    } else if (id >= 0 && static_cast<size_t>(id) < tokens_by_id_.size()) {
      token = tokens_by_id_[id];
    } else {
      continue;
    }
    // Note (Jiaxin Deng): a token whose characters do not all map back to
    // bytes keeps its own UTF-8.
    std::string token_bytes;
    bool all_mapped = true;
    size_t offset = 0;
    while (offset < token.size()) {
      const uint32_t code_point = NextCodePoint(token, offset);
      const auto mapped = Tables().code_point_to_byte.find(code_point);
      if (mapped == Tables().code_point_to_byte.end()) {
        all_mapped = false;
        break;
      } else {
        token_bytes.push_back(static_cast<char>(mapped->second));
      }
    }
    bytes += all_mapped ? token_bytes : token;
  }
  return FromUtf8Lossy(bytes);
}

} // namespace qwen3_asr
