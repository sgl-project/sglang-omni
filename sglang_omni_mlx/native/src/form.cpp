// SPDX-License-Identifier: Apache-2.0
#include "form.h"

#include <algorithm>
#include <cctype>

namespace qwen3_asr {

namespace {

std::string Lowercase(std::string text) {
  std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return text;
}

std::string Trim(const std::string &text) {
  size_t begin = 0;
  size_t end = text.size();
  while (begin < end && std::isspace(static_cast<unsigned char>(text[begin])))
    ++begin;
  while (end > begin && std::isspace(static_cast<unsigned char>(text[end - 1])))
    --end;
  return text.substr(begin, end - begin);
}

std::optional<std::string> HeaderParameter(const std::string &header,
                                           const std::string &parameter) {
  const std::string lowered = Lowercase(header);
  size_t position = 0;
  while ((position = lowered.find(parameter + "=", position)) !=
         std::string::npos) {
    const bool at_boundary = position == 0 || lowered[position - 1] == ';' ||
                             lowered[position - 1] == ' ' ||
                             lowered[position - 1] == '\t';
    if (!at_boundary) {
      position += parameter.size();
      continue;
    } else {
    }
    size_t start = position + parameter.size() + 1;
    if (start < header.size() && header[start] == '"') {
      const size_t end = header.find('"', start + 1);
      return header.substr(start + 1, end == std::string::npos
                                          ? std::string::npos
                                          : end - start - 1);
    } else {
      const size_t end = header.find(';', start);
      return Trim(header.substr(
          start, end == std::string::npos ? std::string::npos : end - start));
    }
  }
  return std::nullopt;
}

} // namespace

std::optional<std::map<std::string, FormField>>
ParseMultipartForm(const std::string &content_type, const std::string &body) {
  if (Lowercase(content_type).rfind("multipart/form-data", 0) != 0) {
    return std::nullopt;
  } else {
  }
  const std::optional<std::string> boundary =
      HeaderParameter(content_type, "boundary");
  if (!boundary.has_value() || boundary->empty()) {
    return std::nullopt;
  } else {
  }
  const std::string delimiter = "--" + *boundary;
  std::map<std::string, FormField> fields;
  size_t position = body.find(delimiter);
  if (position == std::string::npos) {
    return std::nullopt;
  } else {
  }
  while (true) {
    position += delimiter.size();
    if (body.compare(position, 2, "--") == 0) {
      break;
    } else {
    }
    if (body.compare(position, 2, "\r\n") == 0) {
      position += 2;
    } else {
      return std::nullopt;
    }
    const size_t headers_end = body.find("\r\n\r\n", position);
    if (headers_end == std::string::npos) {
      return std::nullopt;
    } else {
    }
    const std::string headers = body.substr(position, headers_end - position);
    const size_t content_start = headers_end + 4;
    const size_t next = body.find("\r\n" + delimiter, content_start);
    if (next == std::string::npos) {
      return std::nullopt;
    } else {
    }
    std::optional<std::string> name;
    std::optional<std::string> filename;
    size_t line_start = 0;
    while (line_start <= headers.size()) {
      size_t line_end = headers.find("\r\n", line_start);
      if (line_end == std::string::npos)
        line_end = headers.size();
      const std::string line =
          headers.substr(line_start, line_end - line_start);
      if (Lowercase(line).rfind("content-disposition:", 0) == 0) {
        name = HeaderParameter(line, "name");
        filename = HeaderParameter(line, "filename");
      } else {
      }
      line_start = line_end + 2;
    }
    if (name.has_value()) {
      fields[*name] = {body.substr(content_start, next - content_start),
                       filename};
    } else {
    }
    position = next + 2;
  }
  return fields;
}

bool FormFlag(const std::optional<std::string> &value) {
  if (!value.has_value()) {
    return false;
  } else {
  }
  const std::string normalized = Lowercase(Trim(*value));
  return normalized == "1" || normalized == "true" || normalized == "yes" ||
         normalized == "on";
}

} // namespace qwen3_asr
