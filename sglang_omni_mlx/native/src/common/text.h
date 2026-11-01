// SPDX-License-Identifier: Apache-2.0
// Model-independent text helpers: UTF-8 code points, Unicode whitespace,
// greedy token-loop detection and base64.
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace omni_common {

bool IsUnicodeSpace(uint32_t code_point);
std::vector<uint32_t> CodePoints(const std::string &text);
// Strips every Unicode space from both ends.
std::string StripUnicodeWhitespace(const std::string &text);
// Greedy loop guard: the last 24 tokens use at most 3 distinct ids.
bool IsTokenLoop(const std::vector<int> &output_ids);
// The bytes of base64 text; none when its padding is invalid.
std::optional<std::vector<uint8_t>> DecodeBase64(const std::string &text);

} // namespace omni_common
