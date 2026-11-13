// SPDX-License-Identifier: Apache-2.0
// Model-independent text helpers: UTF-8 code points, Unicode whitespace,
// greedy token-loop detection and collapse, and base64.
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace omni_common {

bool IsUnicodeSpace(uint32_t code_point);
std::vector<uint32_t> CodePoints(const std::string &text);
// Appends code_point to out as UTF-8.
void AppendUtf8(uint32_t code_point, std::string &out);
// Code points of UTF-8 text, counted without decoding it.
size_t CodePointCount(const std::string &text);
// Strips every Unicode space from both ends.
std::string StripUnicodeWhitespace(const std::string &text);
// Greedy loop guard: the last 24 tokens use at most 3 distinct ids.
bool IsTokenLoop(const std::vector<int> &output_ids);
// The ids up to the first copy of a looping tail's unit: the shortest unit
// that repeats at least twice over the last 12 or more tokens. Unchanged when
// no such unit exists.
std::vector<int> CollapseTokenLoop(const std::vector<int> &output_ids);
// The bytes of base64 text; none when its padding is invalid.
std::optional<std::vector<uint8_t>> DecodeBase64(const std::string &text);

} // namespace omni_common
