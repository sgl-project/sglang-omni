// SPDX-License-Identifier: Apache-2.0
#include "qwen3_omni/serving.h"
#include "mlx/version.h"
#include <atomic>
#include <chrono>
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/avutil.h>
}

namespace qwen3_omni {
Arguments parse_arguments(int count, char **values, bool server) {
  Arguments arguments;
  bool explicit_port = false;
  for (int i = 1; i < count; ++i) {
    std::string flag = values[i];
    if (flag == "--version") {
      arguments.version = true;
      continue;
    } else if (flag == "--supervised" && server) {
      arguments.supervised = true;
      continue;
    } else {
    }
    if (++i >= count) {
      throw std::invalid_argument("missing value for " + flag);
    } else {
    }
    std::string value = values[i];
    if (flag == "--model-path" || (flag == "--model-directory" && server)) {
      arguments.model_path = value;
    } else if (flag == "--model-name") {
      arguments.model_name = value;
    } else if (flag == "--allowed-local-media-path") {
      arguments.allowed_paths.push_back(std::filesystem::canonical(value));
    } else if (flag == "--request" && !server) {
      arguments.requests.push_back(value);
    } else if (flag == "--host" && server) {
      arguments.host = value;
    } else if (flag == "--port" && server) {
      size_t consumed = 0;
      arguments.port = std::stoi(value, &consumed);
      if (consumed != value.size() || arguments.port < 0 ||
          arguments.port > 65535) {
        throw std::invalid_argument("invalid port");
      } else {
        explicit_port = true;
      }
    } else if (flag == "--model-kind" && server) {
      if (value != "qwen3_omni") {
        throw std::invalid_argument("this binary only serves qwen3_omni");
      } else {
      }
    } else {
      throw std::invalid_argument("unknown option: " + flag);
    }
  }
  if (arguments.supervised && !explicit_port) {
    arguments.port = 0;
  } else {
  }
  if (!arguments.version && (arguments.model_path.empty() ||
                             (!server && arguments.requests.empty()))) {
    throw std::invalid_argument("required: --model-path DIRECTORY and, for "
                                "generation, --request JSON_FILE");
  } else {
    return arguments;
  }
}
Json version_json() {
  return {{"runtime", "qwen3_omni_cpp"},
          {"mlx_version", mx::version()},
          {"ffmpeg_version", av_version_info()},
          {"ffmpeg_license", avcodec_license()},
          {"version_source", "linked mlx::core::version()"}};
}
Completion make_completion(const std::string &model) {
  static std::atomic<uint64_t> sequence{0};
  auto now = std::chrono::system_clock::now().time_since_epoch();
  auto nanoseconds =
      std::chrono::duration_cast<std::chrono::nanoseconds>(now).count();
  return {"chatcmpl-" + std::to_string(nanoseconds) + "-" +
              std::to_string(sequence++),
          std::chrono::duration_cast<std::chrono::seconds>(now).count(),
          model,
          "",
          {},
          "",
          0,
          0};
}
void Completion::append(const GenerationChunk &chunk) {
  text += chunk.text;
  if (!chunk.audio.empty()) {
    if (chunk.audio.size() < 44 ||
        audio_pcm.size() + chunk.audio.size() - 44 > 128 * 1024 * 1024) {
      throw std::runtime_error("invalid or excessive generated audio");
    } else {
      audio_pcm.insert(audio_pcm.end(), chunk.audio.begin() + 44,
                       chunk.audio.end());
    }
  } else {
  }
  if (!chunk.finish_reason.empty()) {
    finish_reason = chunk.finish_reason;
    prompt_tokens = chunk.prompt_tokens;
    completion_tokens = chunk.completion_tokens;
  } else {
  }
}
Json Completion::json() const {
  Json message = {{"role", "assistant"}, {"content", text}};
  if (!audio_pcm.empty()) {
    auto wave = encode_wav({});
    auto patch_integer = [&](int offset, uint32_t value) {
      for (int i = 0; i < 4; ++i) {
        wave[offset + i] = uint8_t(value >> (8 * i));
      }
    };
    patch_integer(4, 36 + audio_pcm.size());
    patch_integer(40, audio_pcm.size());
    wave.insert(wave.end(), audio_pcm.begin(), audio_pcm.end());
    message["audio"] = {{"data", encode_base64(wave)}};
  } else {
  }
  return {{"id", id},
          {"created", created},
          {"model", model},
          {"object", "chat.completion"},
          {"choices",
           {{{"index", 0},
             {"message", message},
             {"finish_reason", finish_reason}}}},
          {"usage",
           {{"prompt_tokens", prompt_tokens},
            {"completion_tokens", completion_tokens},
            {"total_tokens", prompt_tokens + completion_tokens}}}};
}
Json streaming_payload(const GenerationChunk &chunk,
                       const Completion &completion, bool include_usage) {
  Json delta = Json::object();
  if (!chunk.text.empty()) {
    delta["content"] = chunk.text;
  } else {
  }
  if (!chunk.audio.empty()) {
    delta["audio"] = {{"data", encode_base64(chunk.audio)}};
  } else {
  }
  Json result = {{"id", completion.id},
                 {"created", completion.created},
                 {"model", completion.model},
                 {"object", "chat.completion.chunk"},
                 {"choices",
                  {{{"index", 0},
                    {"delta", delta},
                    {"finish_reason", chunk.finish_reason.empty()
                                          ? Json(nullptr)
                                          : Json(chunk.finish_reason)}}}}};
  if (!chunk.finish_reason.empty() && include_usage) {
    result["usage"] = {
        {"prompt_tokens", chunk.prompt_tokens},
        {"completion_tokens", chunk.completion_tokens},
        {"total_tokens", chunk.prompt_tokens + chunk.completion_tokens}};
  } else {
  }
  return result;
}
} // namespace qwen3_omni
