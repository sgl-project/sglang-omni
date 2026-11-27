// SPDX-License-Identifier: Apache-2.0
// multipart/form-data parsing for the transcription endpoint.
#pragma once

#include <map>
#include <optional>
#include <string>

namespace omni_common {

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

std::string NormalizeFormToken(const std::string &value);
// An omitted flag is false. An unrecognized value is nullopt.
std::optional<bool> ParseFormFlag(const std::optional<std::string> &value);
// An omitted or unrecognized flag is false.
bool FormFlag(const std::optional<std::string> &value);

} // namespace omni_common
