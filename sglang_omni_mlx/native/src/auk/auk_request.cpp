// SPDX-License-Identifier: Apache-2.0
#include "auk/auk_request.h"

#include <algorithm>
#include <cmath>
#include <set>
#include <stdexcept>
#include <string_view>

#include <civetweb.h>
#include <nlohmann/json.hpp>

#include "common/audio.h"

namespace auk {

namespace {

using Json = nlohmann::json;
constexpr size_t kMaximumTextBytes = 8192;

const Json &ObjectField(const Json &request, const std::string &name) {
  static const Json empty = Json::object();
  if (!request.contains(name) || request.at(name).is_null()) {
    return empty;
  } else if (!request.at(name).is_object()) {
    throw std::invalid_argument(name + " must be an object");
  } else {
    return request.at(name);
  }
}

std::string TextField(const Json &request, const std::string &name,
                      const std::string &fallback = "") {
  if (!request.contains(name) || request.at(name).is_null()) {
    return fallback;
  } else if (!request.at(name).is_string()) {
    throw std::invalid_argument(name + " must be a string");
  } else {
    return request.at(name).get<std::string>();
  }
}

std::string PromptTextField(const Json &request, const std::string &name,
                            const std::string &fallback = "") {
  const std::string text = TextField(request, name, fallback);
  const size_t marker = text.find("<|");
  if (marker != std::string::npos &&
      text.find("|>", marker + 2) != std::string::npos) {
    throw std::invalid_argument(
        name + " must not contain <|...|> special-token markers");
  } else {
    return text;
  }
}

std::string Trim(const std::string &text) {
  const size_t start = text.find_first_not_of(" \t\r\n");
  return start == std::string::npos
             ? ""
             : text.substr(start, text.find_last_not_of(" \t\r\n") - start + 1);
}

} // namespace

ParsedRequest ParseRequest(const Json &request, bool raw_instruction,
                           float max_seconds, const std::string &model_name) {
  if (!request.is_object()) {
    throw std::invalid_argument("request must be a JSON object");
  } else {
  }
  static const std::set<std::string> speech_fields{
      "model",    "input",  "instructions",    "voice",
      "speed",    "stream", "response_format", "ref_audio",
      "ref_text", "seed",   "gen_seconds",     "stage_params"};
  static const std::set<std::string> generate_fields{
      "model",       "prompt", "metadata",          "stage_params",
      "gen_seconds", "seed",   "output_modalities", "return_logprob",
      "stream"};
  const auto &fields = raw_instruction ? generate_fields : speech_fields;
  for (const auto &[name, value] : request.items()) {
    if (!fields.contains(name)) {
      throw std::invalid_argument("unsupported request field: " + name);
    } else {
    }
  }
  if (request.value("stream", false)) {
    throw std::invalid_argument("AuK does not support streaming responses");
  } else if (request.contains("model") &&
             TextField(request, "model") != model_name) {
    throw std::invalid_argument("model does not match the loaded model");
  } else {
  }
  const std::string text =
      PromptTextField(request, raw_instruction ? "prompt" : "input");
  if (text.find_first_not_of(" \t\r\n") == std::string::npos ||
      text.size() > kMaximumTextBytes) {
    throw std::invalid_argument(
        "input or prompt must contain 1 to 8192 bytes of text");
  } else {
  }
  const Json &metadata = ObjectField(request, "metadata");
  const Json &parameters =
      raw_instruction ? ObjectField(metadata, "tts_params") : request;
  if (raw_instruction) {
    for (const auto &[name, value] : metadata.items()) {
      if (name != "tts_params") {
        throw std::invalid_argument("only metadata.tts_params is supported");
      } else {
      }
    }
    for (const auto &[name, value] : parameters.items()) {
      if (name != "ref_audio" && name != "ref_text" && name != "seed" &&
          name != "gen_seconds") {
        throw std::invalid_argument("unsupported tts_params field: " + name);
      } else {
      }
    }
  } else {
  }
  const std::string reference_text = PromptTextField(parameters, "ref_text");
  const Json &stages = ObjectField(request, "stage_params");
  const Json &engine = ObjectField(stages, "auk_engine");
  for (const auto &[name, value] : stages.items()) {
    if (name != "auk_engine") {
      throw std::invalid_argument("only stage_params.auk_engine is supported");
    } else {
    }
  }
  for (const auto &[name, value] : engine.items()) {
    if (name != "gen_seconds") {
      throw std::invalid_argument(
          "only auk_engine.gen_seconds is request-configurable");
    } else {
    }
  }
  ParsedRequest parsed{};
  const std::string reference = TextField(parameters, "ref_audio");
  if (!reference.empty()) {
    constexpr std::string_view prefix = "data:audio/wav;base64,";
    if (!reference.starts_with(prefix)) {
      throw std::invalid_argument(
          "ref_audio must be a data:audio/wav;base64 URI");
    } else {
    }
    const std::string_view encoded =
        std::string_view(reference).substr(prefix.size());
    const size_t padding = encoded.find('=');
    if (encoded.size() % 4 != 0 ||
        (padding != std::string::npos &&
         (padding < encoded.size() - 2 ||
          encoded.find_first_not_of('=', padding) != std::string::npos))) {
      throw std::invalid_argument("ref_audio contains invalid base64 padding");
    } else {
    }
    std::string bytes(encoded.size(), '\0');
    size_t length = bytes.size();
    if (mg_base64_decode(encoded.data(), encoded.size(),
                         reinterpret_cast<unsigned char *>(bytes.data()),
                         &length) != -1) {
      throw std::invalid_argument("ref_audio contains invalid base64");
    } else {
    }
    bytes.resize(length - 1);
    auto audio = qwen3_asr::DecodeAudioWav(bytes);
    if (audio.samples.size() <
            static_cast<size_t>((audio.sample_rate * 3 + 99) / 100) ||
        audio.samples.size() >
            static_cast<size_t>(audio.sample_rate * max_seconds)) {
      throw std::invalid_argument(
          "ref_audio must contain at least 30 ms and fit --max-seconds");
    } else {
    }
    parsed.synthesis.reference_audio = std::move(audio.samples);
    parsed.synthesis.reference_sample_rate = audio.sample_rate;
  } else {
  }
  if (raw_instruction) {
    if (request.value("return_logprob", false) ||
        request.value("output_modalities", Json::array({"audio"})) !=
            Json::array({"audio"})) {
      throw std::invalid_argument(
          "generate supports audio output without logprobs");
    } else {
    }
    parsed.synthesis.instruction = Trim(text);
  } else {
    if (TextField(request, "voice", "default") != "default" ||
        request.value("speed", 1.0) != 1.0) {
      throw std::invalid_argument(
          "only voice=default and speed=1 are supported");
    } else {
    }
    const std::string description = Trim(
        PromptTextField(request, "instructions", "A clear, natural voice."));
    if (description.size() > kMaximumTextBytes) {
      throw std::invalid_argument(
          "instructions must contain at most 8192 bytes");
    } else if (!parsed.synthesis.reference_audio.empty()) {
      parsed.synthesis.instruction =
          "Say the following with the same voice: \"" + text + "\"";
    } else {
      parsed.synthesis.instruction =
          "Generate speech based on the following description: \"" +
          (description.empty() ? "A clear, natural voice." : description) +
          "\". The content to speak is: \"" + text + "\".";
    }
  }
  const Json duration = engine.value(
      "gen_seconds",
      parameters.value("gen_seconds", request.value("gen_seconds", Json())));
  const double reference_seconds =
      static_cast<double>(parsed.synthesis.reference_audio.size()) /
      parsed.synthesis.reference_sample_rate;
  if (!duration.is_null()) {
    if (!duration.is_number() || !std::isfinite(duration.get<double>()) ||
        duration.get<double>() <= 0) {
      throw std::invalid_argument(
          "gen_seconds must be a positive finite number");
    } else {
      parsed.synthesis.seconds = duration.get<double>();
    }
  } else if (raw_instruction) {
    parsed.synthesis.seconds = reference_seconds > 0 ? reference_seconds : 5.0;
    parsed.synthesis.match_reference_duration = reference_seconds > 0;
  } else {
    if (reference_seconds <= 0 || reference_text.empty()) {
      throw std::invalid_argument(
          "speech requires gen_seconds or ref_audio with ref_text");
    } else {
      parsed.synthesis.seconds =
          reference_seconds * text.size() / reference_text.size();
    }
  }
  parsed.synthesis.seconds =
      std::min(parsed.synthesis.seconds, static_cast<double>(max_seconds));
  const Json seed = parameters.value("seed", request.value("seed", Json()));
  if (!seed.is_null()) {
    if (!seed.is_number_integer() || seed < 0) {
      throw std::invalid_argument("seed must be a nonnegative integer");
    } else {
      parsed.synthesis.seed = seed.get<uint64_t>();
    }
  } else {
  }
  const std::string format = TextField(request, "response_format", "wav");
  if (format != "wav" && format != "pcm") {
    throw std::invalid_argument("response_format must be wav or pcm");
  } else {
    parsed.pcm = format == "pcm";
  }
  return parsed;
}

} // namespace auk
