// SPDX-License-Identifier: Apache-2.0
// multipart/form-data parsing for the transcription endpoint.
#pragma once

#include <map>
#include <optional>
#include <string>

namespace qwen3_asr {

struct FormField {
  std::string value;
  // Set for file parts only.
  std::optional<std::string> filename;
};

// Parses a multipart/form-data body; nullopt when the content type or body
// is not multipart. Repeated names keep the last part, as the Python server
// does.
std::optional<std::map<std::string, FormField>>
ParseMultipartForm(const std::string &content_type, const std::string &body);

// "1", "true", "yes" or "on", ignoring case and surrounding whitespace.
bool FormFlag(const std::optional<std::string> &value);

} // namespace qwen3_asr
