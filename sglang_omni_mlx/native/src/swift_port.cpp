// SPDX-License-Identifier: Apache-2.0
#include "swift_port.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string_view>

namespace swift_port {

namespace {

namespace mx = mlx::core;

constexpr int kSampleRate = 16000;
// SentencePiece piece types that decoding skips.
constexpr int kControlPiece = 3;
constexpr int kUnusedPiece = 5;
constexpr const char *kWordMarker = "\xE2\x96\x81"; // U+2581

// A protobuf message read field by field.
class ProtobufReader {
public:
  explicit ProtobufReader(std::string_view bytes) : bytes_(bytes) {}

  bool AtEnd() const { return offset_ >= bytes_.size(); }
  uint64_t Varint() {
    uint64_t value = 0;
    for (int shift = 0; offset_ < bytes_.size() && shift < 64; shift += 7) {
      const uint8_t byte = static_cast<uint8_t>(bytes_[offset_++]);
      value |= static_cast<uint64_t>(byte & 0x7F) << shift;
      if ((byte & 0x80) == 0) {
        return value;
      } else {
      }
    }
    throw std::runtime_error("SentencePiece model has a malformed varint");
  }
  std::string_view Bytes(size_t size) {
    if (offset_ + size > bytes_.size()) {
      throw std::runtime_error("SentencePiece model has a truncated field");
    } else {
    }
    const std::string_view field = bytes_.substr(offset_, size);
    offset_ += size;
    return field;
  }
  void Skip(uint64_t wire_type) {
    if (wire_type == 0) {
      Varint();
    } else if (wire_type == 1) {
      Bytes(8);
    } else if (wire_type == 2) {
      Bytes(Varint());
    } else if (wire_type == 5) {
      Bytes(4);
    } else {
      throw std::runtime_error(
          "SentencePiece model has an unsupported wire type");
    }
  }

private:
  std::string_view bytes_;
  size_t offset_ = 0;
};

// Note (khazic): strict UTF-8, as the Swift port's String(bytes:encoding:)
// accepts it.
bool IsValidUtf8(const std::string &bytes) {
  size_t i = 0;
  const auto byte = [&](size_t k) { return static_cast<uint8_t>(bytes[k]); };
  while (i < bytes.size()) {
    const uint8_t lead = byte(i);
    size_t need = 0;
    uint8_t low = 0x80;
    uint8_t high = 0xBF;
    if (lead < 0x80) {
      need = 0;
    } else if (lead >= 0xC2 && lead <= 0xDF) {
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
      return false;
    }
    if (i + need >= bytes.size() && need > 0) {
      return false;
    } else {
    }
    for (size_t k = 1; k <= need; ++k) {
      const uint8_t continuation = byte(i + k);
      if (continuation < (k == 1 ? low : 0x80) ||
          continuation > (k == 1 ? high : 0xBF)) {
        return false;
      } else {
      }
    }
    i += need + 1;
  }
  return true;
}

std::vector<mx::array> SiluGraph(const std::vector<mx::array> &inputs) {
  return {mx::multiply(inputs[0], mx::sigmoid(inputs[0]))};
}

std::vector<mx::array> ReluGraph(const std::vector<mx::array> &inputs) {
  return {mx::maximum(inputs[0], mx::array(0.0f, inputs[0].dtype()))};
}

float HertzToMel(float frequency_hz, float linear_step_hz, float min_log_hz,
                 float min_log_mel, float log_step) {
  if (frequency_hz < min_log_hz) {
    return frequency_hz / linear_step_hz;
  } else {
    return min_log_mel + std::log(frequency_hz / min_log_hz) / log_step;
  }
}

float MelToHertz(float mel, float linear_step_hz, float min_log_hz,
                 float min_log_mel, float log_step) {
  if (mel < min_log_mel) {
    return linear_step_hz * mel;
  } else {
    return min_log_hz * std::exp(log_step * (mel - min_log_mel));
  }
}

bool IsWhitespace(uint32_t code_point, bool newlines) {
  const bool is_newline = (code_point >= 0x0A && code_point <= 0x0D) ||
                          code_point == 0x85 || code_point == 0x2028 ||
                          code_point == 0x2029;
  return code_point == 0x09 || code_point == 0x20 || code_point == 0xA0 ||
         code_point == 0x1680 ||
         (code_point >= 0x2000 && code_point <= 0x200A) ||
         code_point == 0x202F || code_point == 0x205F || code_point == 0x3000 ||
         (newlines && is_newline);
}

// The code point at offset and its length in bytes. A byte that does not
// start a complete UTF-8 sequence reads as one U+FFFD, so callers never
// step past the end of the text.
uint32_t CodePointAt(const std::string &text, size_t offset, size_t &length) {
  const auto byte = [&](size_t k) { return static_cast<uint8_t>(text[k]); };
  const uint8_t lead = byte(offset);
  uint32_t code_point = 0;
  if (lead < 0x80) {
    length = 1;
    return lead;
  } else if ((lead >> 5) == 0x6) {
    length = 2;
    code_point = lead & 0x1F;
  } else if ((lead >> 4) == 0xE) {
    length = 3;
    code_point = lead & 0x0F;
  } else if ((lead >> 3) == 0x1E) {
    length = 4;
    code_point = lead & 0x07;
  } else {
    length = 1;
    return 0xFFFD;
  }
  if (offset + length > text.size()) {
    length = 1;
    return 0xFFFD;
  } else {
  }
  for (size_t k = 1; k < length; ++k) {
    if ((byte(offset + k) >> 6) != 0x2) {
      length = 1;
      return 0xFFFD;
    } else {
    }
    code_point = (code_point << 6) | (byte(offset + k) & 0x3F);
  }
  return code_point;
}

} // namespace

std::vector<float> SlaneyMelFilterBank(int fft_size, int mel_bin_count) {
  const int frequency_bin_count = fft_size / 2 + 1;
  const float linear_step_hz = 200.0f / 3.0f;
  const float min_log_hz = 1000.0f;
  const float min_log_mel = min_log_hz / linear_step_hz;
  const float log_step = std::log(6.4f) / 27.0f;

  std::vector<float> bin_frequencies_hz(frequency_bin_count);
  for (int i = 0; i < frequency_bin_count; ++i) {
    bin_frequencies_hz[i] = static_cast<float>(i) *
                            static_cast<float>(kSampleRate) /
                            static_cast<float>(fft_size);
  }
  const float mel_max =
      HertzToMel(static_cast<float>(kSampleRate) / 2.0f, linear_step_hz,
                 min_log_hz, min_log_mel, log_step);
  std::vector<float> edges_hz(mel_bin_count + 2);
  for (int i = 0; i < mel_bin_count + 2; ++i) {
    edges_hz[i] = MelToHertz(static_cast<float>(i) * mel_max /
                                 static_cast<float>(mel_bin_count + 1),
                             linear_step_hz, min_log_hz, min_log_mel, log_step);
  }
  std::vector<float> filters(
      static_cast<size_t>(frequency_bin_count) * mel_bin_count, 0.0f);
  for (int mel_bin = 0; mel_bin < mel_bin_count; ++mel_bin) {
    const float low = edges_hz[mel_bin];
    const float center = edges_hz[mel_bin + 1];
    const float high = edges_hz[mel_bin + 2];
    const float normalization = 2.0f / (high - low);
    for (int frequency_bin = 0; frequency_bin < frequency_bin_count;
         ++frequency_bin) {
      const float frequency_hz = bin_frequencies_hz[frequency_bin];
      float weight = 0.0f;
      if (low <= frequency_hz && frequency_hz < center) {
        weight = (frequency_hz - low) / (center - low);
      } else if (center <= frequency_hz && frequency_hz <= high) {
        weight = (high - frequency_hz) / (high - center);
      } else {
        weight = 0.0f;
      }
      filters[static_cast<size_t>(frequency_bin) * mel_bin_count + mel_bin] =
          weight * normalization;
    }
  }
  return filters;
}

mx::array Silu(const mx::array &x) {
  static const auto compiled = mx::compile(SiluGraph, true);
  return compiled({x})[0];
}

mx::array Relu(const mx::array &x) {
  static const auto compiled = mx::compile(ReluGraph, true);
  return compiled({x})[0];
}

std::string TrimWhitespace(const std::string &text, bool newlines) {
  size_t begin = 0;
  while (begin < text.size()) {
    size_t length = 0;
    if (!IsWhitespace(CodePointAt(text, begin, length), newlines)) {
      break;
    } else {
    }
    begin += length;
  }
  // Note (khazic): scan forward, remembering where the last kept code point
  // ended.
  size_t kept_end = begin;
  for (size_t offset = begin; offset < text.size();) {
    size_t length = 0;
    if (!IsWhitespace(CodePointAt(text, offset, length), newlines)) {
      kept_end = offset + length;
    } else {
    }
    offset += length;
  }
  return text.substr(begin, kept_end - begin);
}

SentencePieceVocabulary::SentencePieceVocabulary(
    const std::filesystem::path &model_file) {
  std::ifstream stream(model_file, std::ios::binary);
  if (!stream) {
    throw std::runtime_error("cannot read " + model_file.string());
  } else {
  }
  std::ostringstream contents;
  contents << stream.rdbuf();
  const std::string model_proto = contents.str();
  ProtobufReader model_reader(model_proto);
  while (!model_reader.AtEnd()) {
    const uint64_t key = model_reader.Varint();
    if ((key >> 3) == 1 && (key & 7) == 2) {
      ProtobufReader piece_reader(model_reader.Bytes(model_reader.Varint()));
      std::optional<std::string> piece;
      int type = 1;
      while (!piece_reader.AtEnd()) {
        const uint64_t field = piece_reader.Varint();
        if ((field >> 3) == 1 && (field & 7) == 2) {
          piece = std::string(piece_reader.Bytes(piece_reader.Varint()));
        } else if ((field >> 3) == 3 && (field & 7) == 0) {
          type = static_cast<int>(piece_reader.Varint());
        } else {
          piece_reader.Skip(field & 7);
        }
      }
      // Note (khazic): a piece without text is dropped, and later ids shift
      // down.
      if (piece.has_value()) {
        skipped_.push_back(type == kControlPiece || type == kUnusedPiece);
        pieces_.push_back(*piece);
      } else {
      }
    } else {
      model_reader.Skip(key & 7);
    }
  }
}

std::string
SentencePieceVocabulary::Decode(const std::vector<int> &ids,
                                const std::set<int> &dropped) const {
  std::string text;
  std::string pending_bytes;
  // Note (khazic): byte pieces gather into a run, kept only when it is valid
  // UTF-8.
  const auto flush = [&]() {
    if (!pending_bytes.empty() && IsValidUtf8(pending_bytes)) {
      text += pending_bytes;
    } else {
    }
    pending_bytes.clear();
  };
  for (const int id : ids) {
    if (id < 0 || id >= static_cast<int>(pieces_.size()) || skipped_[id] ||
        dropped.count(id) > 0) {
      continue;
    } else {
    }
    const std::string &piece = pieces_[id];
    if (piece.size() == 6 && piece.compare(0, 3, "<0x") == 0 &&
        piece.back() == '>') {
      const std::string hex = piece.substr(3, 2);
      if (std::all_of(hex.begin(), hex.end(),
                      [](unsigned char c) { return std::isxdigit(c) != 0; })) {
        pending_bytes.push_back(static_cast<char>(std::stoi(hex, nullptr, 16)));
      } else {
      }
      continue;
    } else {
    }
    flush();
    text += piece;
  }
  flush();
  // Note (khazic): the word boundary marker U+2581 becomes a space.
  const std::string marker = kWordMarker;
  std::string spaced;
  size_t start = 0;
  for (size_t found = text.find(marker); found != std::string::npos;
       found = text.find(marker, start)) {
    spaced.append(text, start, found - start);
    spaced += ' ';
    start = found + marker.size();
  }
  spaced.append(text, start, std::string::npos);
  return spaced;
}

} // namespace swift_port
