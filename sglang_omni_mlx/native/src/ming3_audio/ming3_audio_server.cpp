// SPDX-License-Identifier: Apache-2.0
#include <atomic>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "common/audio.h"
#include "common/http.h"
#include "common/service.h"
#include "common/text.h"
#include "ming3_audio/ming3_audio_model.h"

namespace ming3_audio {
namespace {
using omni_server::Json;

constexpr size_t kMaxRequestBodyBytes = 4 * 1024 * 1024;

Json ErrorBody(int status, const std::string &message) {
  return {{"error",
           {{"message", message},
            {"type", status == 400 || status == 405 ? "invalid_request_error"
                     : status == 429                ? "rate_limit_error"
                                                    : "server_error"},
            {"param", nullptr},
            {"code", nullptr}}}};
}

ChatRequest ParseRequest(const nlohmann::json &request,
                         const std::string &model_name) {
  if (request.value("stream", false) ||
      request.value("temperature", 0.0) != 0.0 ||
      request.value("modalities", std::vector<std::string>{"text"}) !=
          std::vector<std::string>{"text"}) {
    throw std::invalid_argument(
        "only non-streaming text output with temperature=0 is supported");
  } else if (request.contains("model") && request.at("model") != model_name) {
    throw std::invalid_argument("model does not match the served model");
  } else {
  }
  for (const auto *field :
       {"tools", "tool_choice", "stop", "audio", "audios", "top_p", "top_k",
        "presence_penalty", "frequency_penalty", "response_format"}) {
    if (request.contains(field)) {
      throw std::invalid_argument(std::string(field) + " is not supported");
    } else {
    }
  }
  ChatRequest result;
  const auto &limit =
      request.contains("max_completion_tokens")
          ? request.at("max_completion_tokens")
          : request.value("max_tokens", nlohmann::json(result.max_new_tokens));
  if (!limit.is_number_integer() || limit < 1 || limit > 1024) {
    throw std::invalid_argument("max_tokens must be an integer from 1 to 1024");
  } else {
    result.max_new_tokens = limit.get<int>();
  }
  const auto &messages = request.at("messages");
  if (!messages.is_array() || messages.empty() || messages.size() > 2) {
    throw std::invalid_argument(
        "expected one user message and an optional leading system message");
  } else {
  }
  if (messages.size() == 2) {
    if (messages[0].at("role") != "system") {
      throw std::invalid_argument("the first message must be system");
    } else {
      result.system_prompt = messages[0].at("content").get<std::string>();
    }
  } else {
  }
  const auto &user = messages.back();
  if (user.at("role") != "user" || !user.at("content").is_array()) {
    throw std::invalid_argument(
        "user content must contain text and input_audio");
  } else {
  }
  bool has_audio = false;
  for (const auto &part : user.at("content")) {
    if (part.at("type") == "text") {
      (has_audio ? result.after_audio : result.before_audio) +=
          part.at("text").get<std::string>();
    } else if (part.at("type") == "input_audio" && !has_audio) {
      const auto &audio = part.at("input_audio");
      if (audio.at("format") != "wav") {
        throw std::invalid_argument("input_audio.format must be wav");
      } else {
      }
      const auto bytes =
          omni_common::DecodeBase64(audio.at("data").get<std::string>());
      if (!bytes.has_value()) {
        throw std::invalid_argument("invalid input_audio base64");
      } else {
        result.samples = omni_common::DecodeWav(std::string_view(
            reinterpret_cast<const char *>(bytes->data()), bytes->size()));
        has_audio = true;
      }
    } else {
      throw std::invalid_argument(
          "expected text parts and exactly one input_audio");
    }
  }
  if (!has_audio) {
    throw std::invalid_argument("exactly one input_audio is required");
  } else {
    return result;
  }
}

class Ming3AudioService : public omni_server::Service {
public:
  explicit Ming3AudioService(const std::filesystem::path &directory)
      : model_(directory) {}

  void Register(mg_context *context, const std::string &model_name) override {
    model_name_ = model_name;
    mg_set_request_handler(
        context, "/v1/chat/completions$",
        [](mg_connection *connection, void *service) {
          return static_cast<Ming3AudioService *>(service)->Handle(connection);
        },
        this);
  }

  std::map<std::string, int> RequestStates() const override {
    return running_.load() ? std::map<std::string, int>{{"running", 1}}
                           : std::map<std::string, int>{};
  }

  void CancelAll() override { stopping_.store(true); }

private:
  int Handle(mg_connection *connection);

  Model model_;
  std::string model_name_;
  std::mutex inference_;
  std::atomic<bool> running_{false};
  std::atomic<bool> stopping_{false};
  std::atomic<uint64_t> request_count_{0};
};

int Ming3AudioService::Handle(mg_connection *connection) {
  if (!omni_server::IsPost(connection)) {
    return omni_server::WriteJson(connection, 405,
                                  ErrorBody(405, "POST required"));
  } else {
  }
  const auto body = omni_server::ReadBody(connection, kMaxRequestBodyBytes);
  if (!body.has_value()) {
    return 413;
  } else {
  }
  const auto fail = [&](int status, const std::string &message) {
    return omni_server::WriteJson(connection, status,
                                  ErrorBody(status, message));
  };
  try {
    const auto request =
        ParseRequest(nlohmann::json::parse(*body), model_name_);
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
    const auto result = model_.Generate(request, cancelled);
    if (cancelled()) {
      throw Cancelled();
    } else {
    }
    return omni_server::WriteJson(
        connection, 200,
        {{"id", id},
         {"object", "chat.completion"},
         {"created", created},
         {"model", model_name_},
         {"choices",
          Json::array(
              {{{"index", 0},
                {"message", {{"role", "assistant"}, {"content", result.text}}},
                {"finish_reason", result.finish_reason}}})},
         {"usage",
          {{"prompt_tokens", result.prompt_tokens},
           {"completion_tokens", result.completion_tokens},
           {"total_tokens",
            result.prompt_tokens + result.completion_tokens}}}});
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
} // namespace
} // namespace ming3_audio

int main(int argc, char **argv) {
  return omni_server::Serve(
      argc, argv,
      {{"ming3_audio",
        [](const std::filesystem::path &directory) {
          return std::make_unique<ming3_audio::Ming3AudioService>(directory);
        },
        {},
        {}}});
}
