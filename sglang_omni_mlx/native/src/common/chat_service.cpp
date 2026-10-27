// SPDX-License-Identifier: Apache-2.0
#include "common/chat_service.h"

#include <chrono>
#include <iostream>

#include "common/http.h"

namespace chat_service {
namespace {
using omni_server::Json;

Json ErrorBody(int status, const std::string &message) {
  return {{"error",
           {{"message", message},
            {"type", status == 400 || status == 405 ? "invalid_request_error"
                     : status == 429                ? "rate_limit_error"
                                                    : "server_error"},
            {"param", nullptr},
            {"code", nullptr}}}};
}

void WriteEvent(mg_connection *connection, const std::string &event) {
  if (mg_write(connection, event.data(), event.size()) != int(event.size())) {
    throw Cancelled();
  } else {
  }
}
} // namespace

ChatService::ChatService(std::unique_ptr<ServedModel> model,
                         size_t max_body_bytes)
    : model_(std::move(model)), max_body_bytes_(max_body_bytes) {}

void ChatService::Register(mg_context *context, const std::string &model_name) {
  model_name_ = model_name;
  mg_set_request_handler(
      context, "/v1/chat/completions$",
      [](mg_connection *connection, void *service) {
        return static_cast<ChatService *>(service)->Handle(connection);
      },
      this);
}

std::map<std::string, int> ChatService::RequestStates() const {
  return running_.load() ? std::map<std::string, int>{{"running", 1}}
                         : std::map<std::string, int>{};
}

void ChatService::CancelAll() { stopping_.store(true); }

int ChatService::Handle(mg_connection *connection) {
  if (!omni_server::IsPost(connection)) {
    return omni_server::WriteJson(connection, 405,
                                  ErrorBody(405, "POST required"));
  } else {
  }
  const auto body = omni_server::ReadBody(connection, max_body_bytes_);
  if (!body.has_value()) {
    return 413;
  } else {
  }
  bool streaming_started = false;
  const auto fail = [&](int status, const std::string &message) {
    const auto error = ErrorBody(status, message);
    if (streaming_started) {
      const auto event = "data: " + error.dump() + "\n\n";
      mg_write(connection, event.data(), event.size());
      return 200;
    } else {
      return omni_server::WriteJson(connection, status, error);
    }
  };
  try {
    const auto request = nlohmann::json::parse(*body);
    const bool stream = request.value("stream", false);
    bool include_usage = false;
    if (request.contains("stream_options") &&
        !request.at("stream_options").is_null()) {
      include_usage =
          request.at("stream_options").value("include_usage", false);
    } else {
    }
    auto generate = model_->Prepare(request, model_name_);
    std::unique_lock lock(inference_, std::try_to_lock);
    if (!lock.owns_lock()) {
      return fail(429, "model busy");
    } else if (stopping_.load()) {
      return fail(503, "server stopping");
    } else {
    }
    struct RunningGuard {
      std::atomic<bool> &running;
      explicit RunningGuard(std::atomic<bool> &running) : running(running) {
        running.store(true);
      }
      ~RunningGuard() { running.store(false); }
    } running(running_);
    const int client_socket = omni_server::ClientSocket(connection);
    const Cancellation cancelled = [&] {
      return stopping_.load() || omni_server::ClientLeft(client_socket);
    };
    const auto now = std::chrono::system_clock::now().time_since_epoch();
    const auto created =
        std::chrono::duration_cast<std::chrono::seconds>(now).count();
    const auto id =
        "chatcmpl-" +
        std::to_string(
            std::chrono::duration_cast<std::chrono::nanoseconds>(now).count()) +
        "-" + std::to_string(++request_count_);
    const auto payload = [&](const Json &choices) {
      return Json{{"id", id},
                  {"object", "chat.completion.chunk"},
                  {"created", created},
                  {"model", model_name_},
                  {"choices", choices}};
    };
    const auto emit = [&](const Chunk &chunk) {
      if (cancelled()) {
        throw Cancelled();
      } else if (!stream) {
        return;
      } else {
      }
      if (!streaming_started) {
        if (mg_printf(connection,
                      "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
                      "Cache-Control: no-cache\r\nConnection: close\r\n\r\n") <=
            0) {
          throw Cancelled();
        } else {
          streaming_started = true;
        }
        const auto role =
            payload(Json::array({{{"index", 0},
                                  {"delta", {{"role", "assistant"}}},
                                  {"finish_reason", nullptr}}}));
        WriteEvent(connection, "data: " + role.dump() + "\n\n");
      } else {
      }
      Json delta = Json::object();
      if (!chunk.text.empty()) {
        delta["content"] = chunk.text;
      } else {
      }
      if (!chunk.audio_base64.empty()) {
        delta["audio"] = {{"data", chunk.audio_base64}};
      } else {
      }
      if (!delta.empty()) {
        const auto output = payload(Json::array(
            {{{"index", 0}, {"delta", delta}, {"finish_reason", nullptr}}}));
        WriteEvent(connection, "data: " + output.dump() + "\n\n");
      } else {
      }
    };
    const auto result = generate(emit, cancelled);
    if (cancelled()) {
      throw Cancelled();
    } else {
    }
    const Json usage = {
        {"prompt_tokens", result.prompt_tokens},
        {"completion_tokens", result.completion_tokens},
        {"total_tokens", result.prompt_tokens + result.completion_tokens}};
    if (stream) {
      emit({});
      auto final =
          payload(Json::array({{{"index", 0},
                                {"delta", Json::object()},
                                {"finish_reason", result.finish_reason}}}));
      if (include_usage) {
        final["usage"] = usage;
      } else {
      }
      WriteEvent(connection, "data: " + final.dump() + "\n\n");
      WriteEvent(connection, "data: [DONE]\n\n");
      return 200;
    } else {
      Json message = {{"role", "assistant"}, {"content", result.text}};
      if (!result.audio_base64.empty()) {
        message["audio"] = {{"data", result.audio_base64}};
      } else {
      }
      return omni_server::WriteJson(
          connection, 200,
          {{"id", id},
           {"object", "chat.completion"},
           {"created", created},
           {"model", model_name_},
           {"choices",
            Json::array({{{"index", 0},
                          {"message", message},
                          {"finish_reason", result.finish_reason}}})},
           {"usage", usage}});
    }
  } catch (const Cancelled &) {
    return fail(503, "generation cancelled");
  } catch (const nlohmann::json::exception &) {
    return fail(400, "invalid JSON, missing field or incorrect field type");
  } catch (const std::invalid_argument &error) {
    return fail(400, error.what());
  } catch (const std::exception &error) {
    std::cerr << "chat generation failed: " << error.what() << '\n';
    return fail(500, "generation failed");
  }
}
} // namespace chat_service
