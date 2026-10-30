// SPDX-License-Identifier: Apache-2.0
#include "common/audio.h"
#include "common/chat_service.h"
#include "common/text.h"
#include "ming3_audio/ming3_audio_model.h"

namespace ming3_audio {
namespace {
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

class ServedModel : public chat_service::ServedModel {
public:
  explicit ServedModel(const std::filesystem::path &directory)
      : model_(directory) {}

  chat_service::Generation
  Prepare(const nlohmann::json &body,
          const std::string &model_name) const override {
    const auto request = ParseRequest(body, model_name);
    return [this, request](const chat_service::ChunkEmitter &,
                           const chat_service::Cancellation &cancelled) {
      const auto result = model_.Generate(request, cancelled);
      return chat_service::Result{result.text,
                                  {},
                                  result.finish_reason,
                                  result.prompt_tokens,
                                  result.completion_tokens};
    };
  }

private:
  Model model_;
};
} // namespace
} // namespace ming3_audio

int main(int argc, char **argv) {
  return omni_server::Serve(
      argc, argv,
      {{"ming3-audio",
        [](const std::filesystem::path &directory) {
          return std::make_unique<chat_service::ChatService>(
              std::make_unique<ming3_audio::ServedModel>(directory),
              4 * 1024 * 1024);
        },
        {},
        {}}});
}
