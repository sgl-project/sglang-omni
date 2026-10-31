// Copyright (c) 2022 and onwards The McBopomofo Authors.
//
// Permission is hereby granted, free of charge, to any person
// obtaining a copy of this software and associated documentation
// files (the "Software"), to deal in the Software without
// restriction, including without limitation the rights to use,
// copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following
// conditions:
//
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
// OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
// HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
// WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
// OTHER DEALINGS IN THE SOFTWARE.

#include "ParselessPhraseDB.h"

#include <cassert>
#include <cstring>
#include <string>
#include <vector>

#ifdef ENABLE_EXPERIMENTAL_SIMD_SUPPORT_NEON
#if defined(__ARM_NEON)
#include <arm_neon.h>

#include <cstdint>
#else
#error ARM NEON support required
#endif
#endif

namespace McBopomofo {

namespace {

#ifdef ENABLE_EXPERIMENTAL_SIMD_SUPPORT_NEON

int FirstNonZeroLane16(uint8x16_t value) {
  // value must be a comparison mask whose lanes are either 0x00 or 0xff.
  // Taking the maximum across the reversed masked lane indices locates the
  // first matching lane and avoids a scalar loop.
  alignas(16) static constexpr uint8_t kReverseLaneIndices[16] = {
      16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1,
  };
  const uint8x16_t laneIndices = vld1q_u8(kReverseLaneIndices);
  return 16 - static_cast<int>(vmaxvq_u8(vandq_u8(value, laneIndices)));
}

int LastNonZeroLane16(uint8x16_t value) {
  // value must be a comparison mask whose lanes are either 0x00 or 0xff.
  // Taking the maximum across the masked lane indices locates the last matching
  // lane and avoids a scalar loop that compilers may expand into up to 16 umov
  // and cbnz branch pairs.
  alignas(16) static constexpr uint8_t kLaneIndices[16] = {
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
  };
  const uint8x16_t laneIndices = vld1q_u8(kLaneIndices);
  return static_cast<int>(vmaxvq_u8(vandq_u8(value, laneIndices))) - 1;
}

#endif

const char* FindNextCharacter(const char* position, const char* end,
                              char character) {
  const char* cursor = position;

#ifdef ENABLE_EXPERIMENTAL_SIMD_SUPPORT_NEON
  const uint8x16_t characters = vdupq_n_u8(static_cast<uint8_t>(character));
  while (end - cursor >= 16) {
    const uint8x16_t block = vld1q_u8(reinterpret_cast<const uint8_t*>(cursor));
    const int positionInBlock = FirstNonZeroLane16(vceqq_u8(block, characters));
    if (positionInBlock != 16) {
      return cursor + positionInBlock;
    }
    cursor += 16;
  }
#endif

  while (cursor != end && *cursor != character) {
    ++cursor;
  }
  return cursor;
}

const char* FindLineStart(const char* begin, const char* position) {
  const char* cursor = position;

#ifdef ENABLE_EXPERIMENTAL_SIMD_SUPPORT_NEON
  const uint8x16_t linefeeds = vdupq_n_u8(static_cast<uint8_t>('\n'));
  while (cursor - begin >= 16) {
    const char* blockStart = cursor - 16;
    const uint8x16_t block =
        vld1q_u8(reinterpret_cast<const uint8_t*>(blockStart));
    const int positionInBlock = LastNonZeroLane16(vceqq_u8(block, linefeeds));
    if (positionInBlock >= 0) {
      return blockStart + positionInBlock + 1;
    }
    cursor = blockStart;
  }
#endif

  while (cursor != begin) {
    --cursor;
    if (*cursor == '\n') {
      return cursor + 1;
    }
  }
  return begin;
}

}  // namespace

bool ParselessPhraseDB::ValidatePragma(const char* buf, size_t length) {
  if (length < SORTED_PRAGMA_HEADER.length()) {
    return false;
  }

  std::string_view header(buf, SORTED_PRAGMA_HEADER.length());
  return header == SORTED_PRAGMA_HEADER;
}

std::unique_ptr<ParselessPhraseDB> ParselessPhraseDB::CreateValidatedDB(
    const char* buf, size_t length) {
  if (buf == nullptr || length == 0) {
    return nullptr;
  }

  if (!ValidatePragma(buf, length)) {
    return nullptr;
  }

  return std::make_unique<ParselessPhraseDB>(buf, length,
                                             /*validate_pragma=*/true);
}

ParselessPhraseDB::ParselessPhraseDB(const char* buf, size_t length,
                                     bool validate_pragma)
    : begin_(buf), end_(buf + length) {
  assert(buf != nullptr);
  assert(length > 0);

  if (validate_pragma) {
    if (ValidatePragma(buf, length)) {
      begin_ += SORTED_PRAGMA_HEADER.length();
    } else {
      // Header invalid; makes the db no-op.
      end_ = begin_;
    }
  }
}

std::vector<std::string_view> ParselessPhraseDB::findRows(
    const std::string_view& key) const {
  std::vector<std::string_view> rows;

  const char* ptr = findFirstMatchingLine(key);
  if (ptr == nullptr) {
    return rows;
  }

  while (ptr + key.length() <= end_ &&
         memcmp(ptr, key.data(), key.length()) == 0) {
    const char* eol = FindNextCharacter(ptr, end_, '\n');

    rows.emplace_back(ptr, eol - ptr);
    if (eol == end_) {
      break;
    }

    ptr = ++eol;
  }

  return rows;
}

// Implements a binary search that returns the pointer to the first matching
// row. In its core it's just a standard binary search, but we use backtracking
// to locate the line start. We also check the previous line to see if the
// current line is actually the first matching line: if the previous line is
// less to the key and the current line starts exactly with the key, then
// the current line is the first matching line.
const char* ParselessPhraseDB::findFirstMatchingLine(
    const std::string_view& key) const {
  if (key.empty()) {
    return begin_;
  }

  const char* top = begin_;
  const char* bottom = end_;

  while (top < bottom) {
    const char* mid = top + ((bottom - top) / 2);
    const char* ptr = FindLineStart(begin_, mid);

    const char* prev = nullptr;
    if (ptr != begin_) {
      prev = ptr - 1;
    }

    // ptr is now in the "current" line we're interested in.
    if (ptr + key.length() > end_) {
      // not enough data to compare at this point, bail.
      break;
    }

    int current_cmp = memcmp(ptr, key.data(), key.length());

    if (current_cmp > 0) {
      bottom = mid - 1;
      continue;
    }

    if (current_cmp < 0) {
      top = mid + 1;
      continue;
    }

    if (!prev) {
      return ptr;
    }

    // Move the prev so that it reaches the previous line.
    prev = FindLineStart(begin_, prev);

    int prev_cmp = memcmp(prev, key.data(), key.length());

    // This is the first occurrence.
    if (prev_cmp < 0 && current_cmp == 0) {
      return ptr;
    }

    // This is not, which means ptr is "larger" than the keyData.
    bottom = mid - 1;
  }

  return nullptr;
}

std::vector<std::string> ParselessPhraseDB::reverseFindRows(
    const std::string_view& value) const {
  std::vector<std::string> rows;

  const char* recordBegin = begin_;

  while (recordBegin < end_) {
    const char* ptr = recordBegin;

    // skip over the key to find the field separator
    ptr = FindNextCharacter(ptr, end_, ' ');
    // skip over the field separator. there should be just one, but loop just in
    // case.
    while (ptr < end_ && *ptr == ' ') {
      ++ptr;
    }

    // now walk to the end of this record
    const char* recordEnd = FindNextCharacter(ptr, end_, '\n');

    if (ptr + value.length() < end_ &&
        memcmp(ptr, value.data(), value.length()) == 0) {
      // prefix match, add entire record to return value
      rows.emplace_back(recordBegin, recordEnd - recordBegin);
    }

    // skip over to the next line start
    recordBegin = recordEnd;
    while (recordBegin < end_ && *recordBegin == '\n') {
      ++recordBegin;
    }
  }

  return rows;
}

}  // namespace McBopomofo
