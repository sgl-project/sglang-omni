// SPDX-License-Identifier: Apache-2.0
#include "common/http.h"

#include <cstring>
#include <stdexcept>

namespace omni_server {

void WriteResponse(mg_connection *connection, int status,
                   const std::string &reason, const std::string &content_type,
                   const std::string &body) {
  mg_printf(connection,
            "HTTP/1.1 %d %s\r\nContent-Type: %s\r\nContent-Length: "
            "%zu\r\nConnection: close\r\n\r\n",
            status, reason.c_str(), content_type.c_str(), body.size());
  mg_write(connection, body.data(), body.size());
}

int WriteJson(mg_connection *connection, int status, const Json &body) {
  WriteResponse(connection, status,
                mg_get_response_code_text(connection, status),
                "application/json", body.dump());
  return status;
}

int BadRequest(mg_connection *connection, const std::string &detail) {
  return WriteJson(connection, 400, {{"detail", detail}});
}

std::string ReadBody(mg_connection *connection, size_t maximum_bytes) {
  std::string body;
  char buffer[65536];
  int read = 0;
  while ((read = mg_read(connection, buffer, sizeof(buffer))) > 0) {
    body.append(buffer, static_cast<size_t>(read));
    if (maximum_bytes != 0 && body.size() > maximum_bytes) {
      throw std::length_error("request body exceeds size limit");
    } else {
    }
  }
  return body;
}

bool IsPost(mg_connection *connection) {
  return std::strcmp(mg_get_request_info(connection)->request_method, "POST") ==
         0;
}

std::optional<std::string>
Field(const std::map<std::string, qwen3_asr::FormField> &form,
      const std::string &name) {
  const auto found = form.find(name);
  if (found == form.end()) {
    return std::nullopt;
  } else {
    return found->second.value;
  }
}

} // namespace omni_server
