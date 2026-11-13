// SPDX-License-Identifier: Apache-2.0
// HTTP helpers shared by the native server's services (civetweb).
#pragma once

#include <chrono>
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
// The request's body, or nullopt once it is longer than max_bytes, which is
// then answered with 413 without reading further; civetweb drops the rest of
// the body after the handler returns, so the client still gets the answer.
std::optional<std::string> ReadBody(mg_connection *connection,
                                    size_t max_bytes);
bool IsPost(mg_connection *connection);

// As often as the CUDA server checks whether a speech client disconnected.
constexpr auto kClientPollInterval = std::chrono::milliseconds(50);
// The status civetweb logs for a request whose client left before its answer.
constexpr int kClientLeftStatus = 499;
// The socket civetweb accepted for connection's client, found by the client's
// address and port since civetweb does not expose it, or -1. Never throws.
int ClientSocket(const mg_connection *connection);
// Whether the client of client_socket, which may be -1, closed it.
bool ClientLeft(int client_socket);
std::optional<std::string>
Field(const std::map<std::string, omni_common::FormField> &form,
      const std::string &name);

} // namespace omni_server
