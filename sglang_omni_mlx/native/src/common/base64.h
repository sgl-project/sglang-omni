// SPDX-License-Identifier: Apache-2.0
// Shared RFC 4648 base64 decoding: strict (reject out-of-alphabet bytes) and
// lenient (drop them, as Python's b64decode(validate=False) does) modes.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace omni_common {

std::vector<uint8_t> DecodeBase64(const std::string &input, bool strict);

} // namespace omni_common

namespace qwen3_asr {

std::vector<uint8_t> DecodeBase64(const std::string &input, bool strict);

} // namespace qwen3_asr
