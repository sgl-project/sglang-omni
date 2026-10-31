// SPDX-License-Identifier: Apache-2.0
#include "qwen3_omni/request.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace qwen3_omni {
namespace {
void fields(const Json &object, const std::set<std::string> &allowed) {
  if (!object.is_object()) {
    throw std::invalid_argument("expected JSON object");
  } else {
  }
  for (const auto &[name, value] : object.items()) {
    if (!allowed.contains(name)) {
      throw std::invalid_argument("unsupported field: " + name);
    } else {
    }
  }
}
std::string string_value(const Json &object, const std::string &name,
                         const std::string &fallback) {
  if (!object.contains(name)) {
    return fallback;
  } else if (!object.at(name).is_string()) {
    throw std::invalid_argument(name + " must be a string");
  } else {
    return object.at(name);
  }
}
bool boolean(const Json &object, const std::string &name,
             bool fallback = false) {
  if (!object.contains(name)) {
    return fallback;
  } else if (!object.at(name).is_boolean()) {
    throw std::invalid_argument(name + " must be boolean");
  } else {
    return object.at(name);
  }
}
std::string media_url(const Json &raw, bool image = false) {
  std::string source;
  if (raw.is_string()) {
    source = raw;
  } else {
    fields(raw, image ? std::set<std::string>{"url", "detail"}
                      : std::set<std::string>{"url"});
    source = string_value(raw, "url", "");
    if (image && string_value(raw, "detail", "auto") != "auto") {
      throw std::invalid_argument("only image detail auto is supported");
    } else {
    }
  }
  if (source.empty()) {
    throw std::invalid_argument("media requires nonempty URL");
  } else {
    return source;
  }
}
} // namespace
ChatRequest
parse_request(const Json &root, const std::string &model_name,
              const std::vector<std::filesystem::path> &allowed_paths) {
  fields(root, {"model",
                "messages",
                "modalities",
                "stream",
                "stream_options",
                "temperature",
                "top_p",
                "top_k",
                "repetition_penalty",
                "seed",
                "max_tokens",
                "max_completion_tokens",
                "talker_temperature",
                "talker_top_p",
                "talker_top_k",
                "talker_repetition_penalty",
                "talker_max_new_tokens",
                "audio",
                "images",
                "audios",
                "videos",
                "video_fps",
                "video_max_frames",
                "video_min_pixels",
                "video_max_pixels",
                "video_total_pixels",
                "use_audio_in_video"});
  auto number = [&](const std::string &name, double fallback, double minimum,
                    double maximum, bool integer = false) {
    if (!root.contains(name) ||
        (root.at(name).is_null() &&
         (name == "seed" || name == "max_completion_tokens"))) {
      return fallback;
    } else if (!root.at(name).is_number()) {
      throw std::invalid_argument(name + " must be numeric");
    } else {
      double value = root.at(name).get<double>();
      if (!std::isfinite(value) || value < minimum || value > maximum ||
          (integer && std::floor(value) != value)) {
        throw std::invalid_argument(name + " is out of range");
      } else {
        return value;
      }
    }
  };
  ChatRequest request;
  request.model = string_value(root, "model", model_name);
  if (request.model != model_name) {
    throw std::invalid_argument("unknown model: " + request.model);
  } else {
  }
  request.stream = boolean(root, "stream");
  auto modalities = root.value("modalities", Json::array({"text"}));
  if (modalities != Json::array({"text"}) &&
      modalities != Json::array({"text", "audio"})) {
    throw std::invalid_argument("unsupported output modalities");
  } else {
    request.speech = modalities.size() == 2;
  }
  request.include_usage = true;
  if (root.contains("stream_options") && !root.at("stream_options").is_null()) {
    fields(root.at("stream_options"), {"include_usage"});
    request.include_usage =
        boolean(root.at("stream_options"), "include_usage", true);
  } else {
  }
  request.voice = "chelsie";
  if (root.contains("audio")) {
    const auto &audio = root.at("audio");
    fields(audio, {"voice", "format"});
    request.voice = string_value(audio, "voice", "chelsie");
    if (string_value(audio, "format", "wav") != "wav") {
      throw std::invalid_argument("only WAV output is supported");
    } else {
    }
  } else {
  }
  std::transform(
      request.voice.begin(), request.voice.end(), request.voice.begin(),
      [](unsigned char character) { return std::tolower(character); });
  if (request.voice != "chelsie" && request.voice != "ethan" &&
      request.voice != "aiden") {
    throw std::invalid_argument("unknown voice");
  } else {
  }
  const double minimum_positive = std::numeric_limits<float>::denorm_min();
  request.sampling = {
      float(number("temperature", 0, 0, 10)),
      float(number("top_p", 1, minimum_positive, 1)),
      int(number("top_k", -1, -1, INT32_MAX, true)),
      float(number("repetition_penalty", 1, minimum_positive, 10))};
  request.talker_sampling = {
      float(number("talker_temperature", .9, 0, 10)),
      float(number("talker_top_p", .9, minimum_positive, 1)),
      int(number("talker_top_k", 50, -1, INT32_MAX, true)),
      float(number("talker_repetition_penalty", 1, minimum_positive, 10))};
  request.max_tokens =
      number("max_completion_tokens", number("max_tokens", 256, 1, 4096, true),
             1, 4096, true);
  request.max_speech_frames =
      number("talker_max_new_tokens", 1024, 1, 4096, true);
  if (root.contains("seed") && !root.at("seed").is_null()) {
    request.seed = number("seed", 0, 0, UINT32_MAX, true);
  } else {
  }
  request.video_fps = number("video_fps", 1, minimum_positive, 30);
  request.video_max_frames = number("video_max_frames", 32, 2, 128, true);
  request.video_min_pixels =
      number("video_min_pixels", 3136, 3136, 1003520, true);
  request.video_max_pixels =
      number("video_max_pixels", 200704, 3136, 1003520, true);
  request.video_total_pixels =
      number("video_total_pixels", 128 * 200704, 3136, 128 * 1003520, true);
  request.use_audio_in_video = boolean(root, "use_audio_in_video");
  if (request.video_min_pixels > request.video_max_pixels) {
    throw std::invalid_argument("invalid video pixel bounds");
  } else {
  }
  for (auto [name, kind] : std::vector<std::pair<std::string, ContentKind>>{
           {"images", ContentKind::Image},
           {"videos", ContentKind::Video},
           {"audios", ContentKind::Audio}}) {
    if (root.contains(name)) {
      const auto &sources = root.at(name);
      if (!sources.is_array() || sources.size() > 8) {
        throw std::invalid_argument(name + " must contain at most eight URLs");
      } else {
      }
      for (const auto &source : sources) {
        if (!source.is_string() || source.get<std::string>().empty()) {
          throw std::invalid_argument("invalid media URL");
        } else {
          request.extra_media.push_back({kind, source.get<std::string>()});
        }
      }
    } else {
    }
  }
  if (!root.contains("messages") || !root.at("messages").is_array() ||
      root.at("messages").empty() || root.at("messages").size() > 128) {
    throw std::invalid_argument("expected 1-128 messages");
  } else {
  }
  int user_messages = 0;
  size_t text_bytes = 0;
  std::vector<ContentPart> all_media = request.extra_media;
  for (const auto &message : root.at("messages")) {
    fields(message, {"role", "content"});
    auto role = string_value(message, "role", "");
    if (role != "user" && role != "assistant" && role != "system") {
      throw std::invalid_argument("unsupported role");
    } else {
      user_messages += role == "user";
    }
    ChatMessage decoded{role, {}};
    if (!message.contains("content")) {
      throw std::invalid_argument("missing content");
    } else if (message.at("content").is_string()) {
      decoded.content.push_back({ContentKind::Text, message.at("content")});
    } else if (message.at("content").is_null()) {
      decoded.content.push_back({ContentKind::Text, ""});
    } else if (message.at("content").is_array()) {
      for (const auto &part : message.at("content")) {
        auto type = string_value(part, "type", "");
        if (type == "text" || type == "input_text") {
          fields(part, {"type", "text"});
          if (!part.contains("text")) {
            throw std::invalid_argument("missing text");
          } else {
            decoded.content.push_back(
                {ContentKind::Text, string_value(part, "text", "")});
          }
        } else if (type == "image_url" || type == "input_image" ||
                   type == "audio_url" || type == "video_url") {
          std::string name = type == "input_image" ? "image_url" : type;
          fields(part, {"type", name});
          if (!part.contains(name)) {
            throw std::invalid_argument("missing media URL");
          } else {
            auto kind = name == "image_url"   ? ContentKind::Image
                        : name == "video_url" ? ContentKind::Video
                                              : ContentKind::Audio;
            decoded.content.push_back(
                {kind, media_url(part.at(name), kind == ContentKind::Image)});
          }
        } else if (type == "input_audio") {
          fields(part, {"type", "input_audio"});
          if (!part.contains("input_audio")) {
            throw std::invalid_argument("missing input_audio");
          } else {
          }
          const auto &audio = part.at("input_audio");
          fields(audio, {"data", "format"});
          auto format = string_value(audio, "format", "");
          auto encoded = string_value(audio, "data", "");
          if ((format != "wav" && format != "mp3" && format != "flac" &&
               format != "ogg") ||
              encoded.empty() || encoded.size() > 48000000) {
            throw std::invalid_argument("invalid input_audio");
          } else {
            decoded.content.push_back(
                {ContentKind::Audio,
                 "data:audio/" + format + ";base64," + encoded});
          }
        } else {
          throw std::invalid_argument("unsupported content type");
        }
      }
    } else {
      throw std::invalid_argument("invalid message content");
    }
    for (const auto &part : decoded.content) {
      if (part.kind == ContentKind::Text) {
        text_bytes += part.content.size();
      } else {
        all_media.push_back(part);
      }
    }
    request.messages.push_back(std::move(decoded));
  }
  if (user_messages == 0 || text_bytes > 131072 || all_media.size() > 24 ||
      (request.use_audio_in_video &&
       std::none_of(all_media.begin(), all_media.end(), [](const auto &part) {
         return part.kind == ContentKind::Video;
       }))) {
    throw std::invalid_argument(
        "text/media bounds exceeded or missing user/video");
  } else {
  }
  for (const auto &part : all_media) {
    validate_media(part.content, allowed_paths);
  }
  return request;
}
} // namespace qwen3_omni
