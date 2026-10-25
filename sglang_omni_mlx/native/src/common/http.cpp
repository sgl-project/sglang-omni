// SPDX-License-Identifier: Apache-2.0
#include "common/http.h"

#include <cstring>

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

std::string ReadBody(mg_connection *connection) {
  std::string body;
  char buffer[65536];
  int read = 0;
  while ((read = mg_read(connection, buffer, sizeof(buffer))) > 0) {
    body.append(buffer, static_cast<size_t>(read));
  }
  return body;
}

std::optional<std::string> ReadBody(mg_connection *connection,
                                    size_t max_bytes) {
  const auto too_large = [&]() -> std::optional<std::string> {
    WriteJson(connection, 413,
              {{"detail", "request body must be at most " +
                              std::to_string(max_bytes) + " bytes"}});
    return std::nullopt;
  };
  // Note (Minyi Tang): civetweb reports -1 for a body without a length, such
  // as a chunked one, which is then cut off as it is read.
  const long long content_length =
      mg_get_request_info(connection)->content_length;
  if (content_length > 0 &&
      static_cast<unsigned long long>(content_length) > max_bytes) {
    return too_large();
  } else {
  }
  std::string body;
  char buffer[65536];
  int read = 0;
  while ((read = mg_read(connection, buffer, sizeof(buffer))) > 0) {
    if (static_cast<size_t>(read) > max_bytes - body.size()) {
      return too_large();
    } else {
    }
    body.append(buffer, static_cast<size_t>(read));
  }
  return body;
}

bool IsPost(mg_connection *connection) {
  return std::strcmp(mg_get_request_info(connection)->request_method, "POST") ==
         0;
}

std::optional<std::string>
Field(const std::map<std::string, omni_common::FormField> &form,
      const std::string &name) {
  const auto found = form.find(name);
  if (found == form.end()) {
    return std::nullopt;
  } else {
    return found->second.value;
  }
}

} // namespace omni_server
