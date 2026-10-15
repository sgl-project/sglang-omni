// SPDX-License-Identifier: Apache-2.0
// HTTP helpers shared by the native server's services (civetweb).
#pragma once

#include <map>
#include <optional>
#include <string>

#include "civetweb.h"
#include "form.h"
#include "nlohmann/json.hpp"

namespace omni_server {

using Json = nlohmann::ordered_json;

void WriteResponse(mg_connection *connection, int status,
                   const std::string &reason, const std::string &content_type,
                   const std::string &body);
int WriteJson(mg_connection *connection, int status, const Json &body);
int BadRequest(mg_connection *connection, const std::string &detail);
std::string ReadBody(mg_connection *connection);
bool IsPost(mg_connection *connection);
std::optional<std::string>
Field(const std::map<std::string, qwen3_asr::FormField> &form,
      const std::string &name);

} // namespace omni_server
