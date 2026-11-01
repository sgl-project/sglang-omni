// SPDX-License-Identifier: Apache-2.0
#include "fishaudio_drama3/fishaudio_drama3_text.h"

#define PCRE2_CODE_UNIT_WIDTH 8
#include <pcre2.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <optional>
#include <set>
#include <stdexcept>
#include <string_view>

#include "common/text.h"

namespace fishaudio_drama3 {

namespace {

// Written only by the prompt builder; caller text holding them would inject
// conversation structure or audio rows.
constexpr std::array<std::string_view, 6> kStructuralTokens = {
    "<|im_start|>", "<|im_end|>",     "<|voice|>",
    "<|text|>",     "<|interleave|>", "<|semantic:"};
constexpr std::string_view kSpeakerTagStart = "<|speaker:";
// The tokenizer has <|speaker:0|> to <|speaker:15|>.
constexpr int kMaxSpeakerId = 15;
constexpr size_t kChunkBytes = 300;
constexpr size_t kChunkSpeakers = 3;
// Spans that are never cut: special tokens, paired tags and [bracketed] cues.
// Note (Yitong Li): word characters are letters, numbers and underscores, as
// for Python's str patterns; PCRE2's Unicode \w would also take marks.
constexpr const char *kUncuttablePattern =
    R"(<\|[^|>]*\|>|<([A-Za-z][\p{L}\p{N}_-]*))"
    R"((?:(?<=[\p{L}\p{N}_])(?![\p{L}\p{N}_])|(?<![\p{L}\p{N}_])(?=[\p{L}\p{N}_])))"
    R"([^>]*>.*?</\1>|\[[^\[\]]*\])";
// Code points Python's str.isdigit accepts (Unicode 15.0 Numeric_Type Digit
// or Decimal), as inclusive ranges.
constexpr std::pair<char32_t, char32_t> kDigitRanges[] = {
    {0x30, 0x39},       {0xB2, 0xB3},       {0xB9, 0xB9},
    {0x660, 0x669},     {0x6F0, 0x6F9},     {0x7C0, 0x7C9},
    {0x966, 0x96F},     {0x9E6, 0x9EF},     {0xA66, 0xA6F},
    {0xAE6, 0xAEF},     {0xB66, 0xB6F},     {0xBE6, 0xBEF},
    {0xC66, 0xC6F},     {0xCE6, 0xCEF},     {0xD66, 0xD6F},
    {0xDE6, 0xDEF},     {0xE50, 0xE59},     {0xED0, 0xED9},
    {0xF20, 0xF29},     {0x1040, 0x1049},   {0x1090, 0x1099},
    {0x1369, 0x1371},   {0x17E0, 0x17E9},   {0x1810, 0x1819},
    {0x1946, 0x194F},   {0x19D0, 0x19DA},   {0x1A80, 0x1A89},
    {0x1A90, 0x1A99},   {0x1B50, 0x1B59},   {0x1BB0, 0x1BB9},
    {0x1C40, 0x1C49},   {0x1C50, 0x1C59},   {0x2070, 0x2070},
    {0x2074, 0x2079},   {0x2080, 0x2089},   {0x2460, 0x2468},
    {0x2474, 0x247C},   {0x2488, 0x2490},   {0x24EA, 0x24EA},
    {0x24F5, 0x24FD},   {0x24FF, 0x24FF},   {0x2776, 0x277E},
    {0x2780, 0x2788},   {0x278A, 0x2792},   {0xA620, 0xA629},
    {0xA8D0, 0xA8D9},   {0xA900, 0xA909},   {0xA9D0, 0xA9D9},
    {0xA9F0, 0xA9F9},   {0xAA50, 0xAA59},   {0xABF0, 0xABF9},
    {0xFF10, 0xFF19},   {0x104A0, 0x104A9}, {0x10A40, 0x10A43},
    {0x10D30, 0x10D39}, {0x10E60, 0x10E68}, {0x11052, 0x1105A},
    {0x11066, 0x1106F}, {0x110F0, 0x110F9}, {0x11136, 0x1113F},
    {0x111D0, 0x111D9}, {0x112F0, 0x112F9}, {0x11450, 0x11459},
    {0x114D0, 0x114D9}, {0x11650, 0x11659}, {0x116C0, 0x116C9},
    {0x11730, 0x11739}, {0x118E0, 0x118E9}, {0x11950, 0x11959},
    {0x11C50, 0x11C59}, {0x11D50, 0x11D59}, {0x11DA0, 0x11DA9},
    {0x11F50, 0x11F59}, {0x16A60, 0x16A69}, {0x16AC0, 0x16AC9},
    {0x16B50, 0x16B59}, {0x1D7CE, 0x1D7FF}, {0x1E140, 0x1E149},
    {0x1E2F0, 0x1E2F9}, {0x1E4F0, 0x1E4F9}, {0x1E950, 0x1E959},
    {0x1F100, 0x1F10A}, {0x1FBF0, 0x1FBF9}};

using CodePointText = std::u32string;

// Unicode White_Space; unlike omni_common::IsUnicodeSpace, U+001C..U+001F are
// not white space here.
bool IsWhiteSpace(char32_t code_point) {
  return code_point == ' ' || (code_point >= 0x09 && code_point <= 0x0D) ||
         code_point == 0x85 || code_point == 0xA0 || code_point == 0x1680 ||
         (code_point >= 0x2000 && code_point <= 0x200A) ||
         code_point == 0x2028 || code_point == 0x2029 || code_point == 0x202F ||
         code_point == 0x205F || code_point == 0x3000;
}

bool IsLineBreak(char32_t code_point) {
  return code_point == '\n' || code_point == '\r' || code_point == 0x0B ||
         code_point == 0x0C || code_point == 0x85 || code_point == 0x2028 ||
         code_point == 0x2029;
}

bool IsSentenceEnd(char32_t code_point) {
  return code_point == '.' || code_point == '!' || code_point == '?' ||
         code_point == ';' || code_point == U'。' || code_point == U'！' ||
         code_point == U'？' || code_point == U'；' || code_point == U'…' ||
         code_point == '\n';
}

bool IsClauseMark(char32_t code_point) {
  return code_point == ',' || code_point == ':' || code_point == U'，' ||
         code_point == U'：' || code_point == U'、';
}

bool IsSpace(char32_t code_point) { return code_point == ' '; }

using MarkSet = bool (*)(char32_t);

CodePointText Decode(const std::string &text) {
  const std::vector<uint32_t> code_points = omni_common::CodePoints(text);
  return CodePointText(code_points.begin(), code_points.end());
}

std::string Encode(const CodePointText &text) {
  std::string bytes;
  for (const char32_t code_point : text) {
    omni_common::AppendUtf8(code_point, bytes);
  }
  return bytes;
}

size_t ByteCount(const CodePointText &text) {
  size_t byte_count = 0;
  for (const char32_t code_point : text) {
    byte_count += code_point < 0x80      ? 1
                  : code_point < 0x800   ? 2
                  : code_point < 0x10000 ? 3
                                         : 4;
  }
  return byte_count;
}

CodePointText StripWhiteSpace(const CodePointText &text) {
  size_t begin = 0;
  size_t end = text.size();
  while (begin < end && IsWhiteSpace(text[begin]))
    ++begin;
  while (end > begin && IsWhiteSpace(text[end - 1]))
    --end;
  return text.substr(begin, end - begin);
}

size_t LeadingWhiteSpace(const CodePointText &text) {
  size_t count = 0;
  while (count < text.size() && IsWhiteSpace(text[count]))
    ++count;
  return count;
}

// The length of the speaker tag <|speaker:N|> at position, with N's digits.
struct SpeakerTag {
  size_t length;
  CodePointText number;
};

std::optional<SpeakerTag> MatchSpeakerTag(const CodePointText &text,
                                          size_t position) {
  const CodePointText start(kSpeakerTagStart.begin(), kSpeakerTagStart.end());
  if (text.compare(position, start.size(), start) != 0) {
    return std::nullopt;
  } else {
  }
  size_t end = position + start.size();
  while (end < text.size() && text[end] >= '0' && text[end] <= '9')
    ++end;
  const size_t digits_begin = position + start.size();
  if (end == digits_begin || text.compare(end, 2, U"|>") != 0) {
    return std::nullopt;
  } else {
    return SpeakerTag{end + 2 - position,
                      text.substr(digits_begin, end - digits_begin)};
  }
}

// The first speaker tag at or after position.
std::optional<std::pair<size_t, SpeakerTag>>
FindSpeakerTag(const CodePointText &text, size_t position) {
  for (; position < text.size(); ++position) {
    if (const std::optional<SpeakerTag> tag = MatchSpeakerTag(text, position)) {
      return std::make_pair(position, *tag);
    } else {
    }
  }
  return std::nullopt;
}

bool IsDigit(char32_t code_point) {
  const auto *range = std::upper_bound(
      std::begin(kDigitRanges), std::end(kDigitRanges), code_point,
      [](char32_t value, const std::pair<char32_t, char32_t> &digits) {
        return value < digits.first;
      });
  return range != std::begin(kDigitRanges) && code_point <= (range - 1)->second;
}

// Replaces each white space run holding a line break with one space, or with
// nothing at the edges of the text.
CodePointText FlattenLines(const CodePointText &text) {
  CodePointText flattened;
  size_t position = 0;
  while (position < text.size()) {
    if (!IsWhiteSpace(text[position])) {
      flattened.push_back(text[position]);
      ++position;
      continue;
    } else {
    }
    size_t end = position;
    while (end < text.size() && IsWhiteSpace(text[end]))
      ++end;
    const CodePointText run = text.substr(position, end - position);
    if (std::none_of(run.begin(), run.end(), IsLineBreak)) {
      flattened += run;
    } else if (position == 0 || end == text.size()) {
    } else {
      flattened.push_back(' ');
    }
    position = end;
  }
  return flattened;
}

// Cuts after every mark, except a period between two digits (3.14).
std::vector<CodePointText> CutAfter(const CodePointText &text, MarkSet marks) {
  std::vector<CodePointText> parts;
  size_t start = 0;
  for (size_t i = 0; i < text.size(); ++i) {
    const bool is_decimal_point = text[i] == '.' && i > 0 &&
                                  i + 1 < text.size() && IsDigit(text[i - 1]) &&
                                  IsDigit(text[i + 1]);
    if (marks(text[i]) && !is_decimal_point) {
      parts.push_back(text.substr(start, i + 1 - start));
      start = i + 1;
    } else {
    }
  }
  if (start < text.size()) {
    parts.push_back(text.substr(start));
  } else {
  }
  return parts;
}

// Pieces of at most limit bytes, cut at the first kind of mark that helps.
std::vector<CodePointText> Fit(const CodePointText &text, size_t limit,
                               const std::vector<MarkSet> &marks,
                               size_t mark_index) {
  if (ByteCount(text) <= limit) {
    return {text};
  } else if (mark_index < marks.size()) {
    std::vector<CodePointText> pieces;
    for (const CodePointText &part : CutAfter(text, marks[mark_index])) {
      for (CodePointText &piece : Fit(part, limit, marks, mark_index + 1)) {
        pieces.push_back(std::move(piece));
      }
    }
    return pieces;
  } else {
    std::vector<CodePointText> pieces;
    CodePointText current;
    for (const char32_t code_point : text) {
      if (!current.empty() &&
          ByteCount(current) + ByteCount(CodePointText(1, code_point)) >
              limit) {
        pieces.push_back(current);
        current.clear();
      } else {
      }
      current.push_back(code_point);
    }
    pieces.push_back(current);
    return pieces;
  }
}

// Free text cut into pieces, with every uncuttable span kept whole.
std::vector<CodePointText> Pieces(const CodePointText &text, size_t limit) {
  static pcre2_code *const uncuttable = [] {
    int error_code = 0;
    PCRE2_SIZE error_offset = 0;
    pcre2_code *code = pcre2_compile(
        reinterpret_cast<PCRE2_SPTR>(kUncuttablePattern), PCRE2_ZERO_TERMINATED,
        PCRE2_UTF | PCRE2_UCP | PCRE2_DOTALL, &error_code, &error_offset,
        nullptr);
    if (code == nullptr) {
      throw std::runtime_error("cannot compile the uncuttable span pattern");
    } else {
    }
    return code;
  }();
  // Byte offsets of the UTF-8 text, mapped back to code points.
  const std::string bytes = Encode(text);
  std::vector<size_t> code_point_at_byte(bytes.size() + 1, 0);
  size_t byte_offset = 0;
  for (size_t i = 0; i < text.size(); ++i) {
    code_point_at_byte[byte_offset] = i;
    byte_offset += ByteCount(CodePointText(1, text[i]));
  }
  code_point_at_byte[bytes.size()] = text.size();

  std::vector<CodePointText> pieces;
  // Free text is cut after sentence ends, then fit at clause marks and spaces.
  const auto add_free = [&](size_t begin, size_t end) {
    for (const CodePointText &sentence :
         CutAfter(text.substr(begin, end - begin), IsSentenceEnd)) {
      for (CodePointText &piece :
           Fit(sentence, limit, {IsClauseMark, IsSpace}, 0)) {
        pieces.push_back(std::move(piece));
      }
    }
  };
  pcre2_match_data *match_data =
      pcre2_match_data_create_from_pattern(uncuttable, nullptr);
  size_t last = 0;
  size_t search_byte = 0;
  while (search_byte <= bytes.size()) {
    const int result =
        pcre2_match(uncuttable, reinterpret_cast<PCRE2_SPTR>(bytes.data()),
                    bytes.size(), search_byte, 0, match_data, nullptr);
    if (result == PCRE2_ERROR_NOMATCH) {
      break;
    } else if (result < 0) {
      pcre2_match_data_free(match_data);
      throw std::invalid_argument("text has spans too nested to split");
    } else {
    }
    const PCRE2_SIZE *ovector = pcre2_get_ovector_pointer(match_data);
    const size_t match_begin = code_point_at_byte[ovector[0]];
    const size_t match_end = code_point_at_byte[ovector[1]];
    add_free(last, match_begin);
    pieces.push_back(text.substr(match_begin, match_end - match_begin));
    last = match_end;
    search_byte = ovector[1];
  }
  pcre2_match_data_free(match_data);
  add_free(last, text.size());
  return pieces;
}

// Pieces packed into chunks of about limit bytes; an uncuttable span longer
// than limit becomes a chunk of its own.
std::vector<CodePointText> SplitByLength(const CodePointText &text,
                                         size_t limit) {
  if (StripWhiteSpace(text).empty()) {
    return {};
  } else if (ByteCount(text) <= limit) {
    return {text};
  } else {
  }
  std::vector<CodePointText> chunks;
  CodePointText current;
  for (const CodePointText &piece : Pieces(text, limit)) {
    if (!current.empty() && ByteCount(current) + ByteCount(piece) > limit) {
      chunks.push_back(current);
      current.clear();
    } else {
    }
    current += piece;
  }
  chunks.push_back(current);
  std::vector<CodePointText> stripped;
  for (const CodePointText &chunk : chunks) {
    CodePointText trimmed = StripWhiteSpace(chunk);
    if (!trimmed.empty()) {
      stripped.push_back(std::move(trimmed));
    } else {
    }
  }
  return stripped;
}

} // namespace

void ValidateText(const std::string &text, const std::string &location) {
  const CodePointText code_points = Decode(text);
  if (StripWhiteSpace(code_points).empty()) {
    throw std::invalid_argument(location + ": text is empty");
  } else {
  }
  for (const std::string_view token : kStructuralTokens) {
    if (text.find(token) != std::string::npos) {
      throw std::invalid_argument(location + ": text must not contain " +
                                  std::string(token));
    } else {
    }
  }
  const CodePointText speaker_prefix = U"<|speaker";
  for (size_t position = code_points.find(speaker_prefix);
       position != CodePointText::npos;
       position = code_points.find(speaker_prefix, position + 1)) {
    const std::optional<SpeakerTag> tag =
        MatchSpeakerTag(code_points, position);
    const bool is_valid =
        tag.has_value() &&
        (tag->number.size() == 1 ||
         (tag->number.size() == 2 && tag->number[0] != '0')) &&
        std::stoi(Encode(tag->number)) <= kMaxSpeakerId;
    if (!is_valid) {
      throw std::invalid_argument(
          location + ": speaker tags run from <|speaker:0|> to <|speaker:" +
          std::to_string(kMaxSpeakerId) + "|>");
    } else {
    }
  }
}

std::string WithSpeakerTag(const std::string &text) {
  const CodePointText code_points = Decode(text);
  if (MatchSpeakerTag(code_points, LeadingWhiteSpace(code_points))) {
    return text;
  } else {
    return "<|speaker:0|> " + text;
  }
}

std::vector<std::string> SplitText(const std::string &text) {
  const CodePointText flattened = FlattenLines(Decode(text));
  std::vector<std::string> chunks;
  if (!FindSpeakerTag(flattened, 0)) {
    for (const CodePointText &chunk : SplitByLength(flattened, kChunkBytes)) {
      chunks.push_back(Encode(chunk));
    }
    return chunks;
  } else {
  }

  // Each speaker's turn, starting with speaker 0's untagged lead.
  std::vector<std::pair<CodePointText, CodePointText>> turns;
  CodePointText speaker = U"0";
  size_t body_begin = 0;
  while (true) {
    const auto found = FindSpeakerTag(flattened, body_begin);
    const size_t body_end = found ? found->first : flattened.size();
    turns.emplace_back(speaker,
                       flattened.substr(body_begin, body_end - body_begin));
    if (!found) {
      break;
    } else {
      speaker = found->second.number;
      body_begin = found->first + found->second.length;
    }
  }

  std::vector<std::vector<CodePointText>> chunk_pieces;
  std::set<CodePointText> chunk_speakers;
  size_t chunk_bytes = 0;
  for (const auto &[turn_speaker, body] : turns) {
    for (const CodePointText &piece : SplitByLength(body, kChunkBytes)) {
      const CodePointText tagged =
          U"<|speaker:" + turn_speaker + U"|> " + piece;
      const bool is_new_speaker = chunk_speakers.count(turn_speaker) == 0;
      if (!chunk_pieces.empty() &&
          ((is_new_speaker &&
            chunk_speakers.size() >= static_cast<size_t>(kChunkSpeakers)) ||
           chunk_bytes + ByteCount(tagged) > kChunkBytes)) {
        chunk_pieces.emplace_back();
        chunk_speakers.clear();
        chunk_bytes = 0;
      } else if (chunk_pieces.empty()) {
        chunk_pieces.emplace_back();
      } else {
      }
      chunk_pieces.back().push_back(tagged);
      chunk_speakers.insert(turn_speaker);
      chunk_bytes += ByteCount(tagged);
    }
  }
  for (const std::vector<CodePointText> &pieces : chunk_pieces) {
    CodePointText joined;
    for (size_t i = 0; i < pieces.size(); ++i) {
      joined += (i == 0 ? U"" : U"\n") + pieces[i];
    }
    chunks.push_back(Encode(joined));
  }
  return chunks;
}

std::string ReferenceTranscript(const std::string &text) {
  const CodePointText tagged = Decode(WithSpeakerTag(text));
  const size_t leading = LeadingWhiteSpace(tagged);
  const size_t tag_length = MatchSpeakerTag(tagged, leading)->length;
  return Encode(tagged.substr(leading, tag_length) + tagged.substr(0, leading) +
                tagged.substr(leading + tag_length));
}

} // namespace fishaudio_drama3
