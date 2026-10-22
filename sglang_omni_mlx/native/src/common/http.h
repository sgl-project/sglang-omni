// SPDX-License-Identifier: Apache-2.0
// HTTP helpers shared by the native server's services (civetweb).
#pragma once

#include <functional>
#include <map>
#include <optional>
#include <string>

#include "civetweb.h"
#include "common/form.h"
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

struct HttpServer {
  mg_context *context;
  int port;
};

HttpServer StartHttpServer(const std::string &host, int requested_port);

class InfoRoutes {
public:
  InfoRoutes(std::string model_name,
             std::function<std::map<std::string, int>()> request_states);
  void Register(mg_context *context);

private:
  static int HandleHealth(mg_connection *connection, void *data);
  static int HandleModels(mg_connection *connection, void *data);

  std::string model_name_;
  std::function<std::map<std::string, int>()> request_states_;
};

} // namespace omni_server
