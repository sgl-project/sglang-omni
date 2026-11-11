// SPDX-License-Identifier: Apache-2.0
#include "auk/auk_request.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <set>
#include <stdexcept>
#include <string_view>

#include "common/audio.h"
#include "common/text.h"
#include "qwen3_asr/speech_service.h"

namespace auk {

namespace {

using speech_service::FieldOr;
using speech_service::Json;
constexpr size_t kMaximumTextBytes = 8192;

Json ObjectField(const Json &request, const char *name) {
  const Json value = FieldOr(request, name, Json::object());
  if (!value.is_object()) {
    throw std::invalid_argument(std::string(name) + " must be an object");
  } else {
    return value;
  }
}

std::string TextField(const Json &request, const char *name,
                      const std::string &fallback = "") {
  const Json value = FieldOr(request, name, Json(fallback));
  if (!value.is_string()) {
    throw std::invalid_argument(std::string(name) + " must be a string");
  } else {
    return value.get<std::string>();
  }
}

std::string PromptTextField(const Json &request, const char *name,
                            const std::string &fallback = "") {
  const std::string text = TextField(request, name, fallback);
  const size_t marker = text.find("<|");
  if (marker != std::string::npos &&
      text.find("|>", marker + 2) != std::string::npos) {
    throw std::invalid_argument(
        std::string(name) + " must not contain <|...|> special-token markers");
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

SynthesisRequest ParseRequest(const Json &request, bool raw_instruction,
                              float max_seconds) {
  static const std::set<std::string> speech_fields{
      "model",  "input",         "instructions",    "voice",     "speed",
      "stream", "stream_format", "response_format", "ref_audio", "ref_text",
      "seed",   "gen_seconds",   "stage_params"};
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
  if (FieldOr(request, "stream", Json(false)) == Json(true)) {
    throw std::invalid_argument("AuK does not support streaming responses");
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
  const Json metadata = ObjectField(request, "metadata");
  const Json tts_parameters = ObjectField(metadata, "tts_params");
  const Json &parameters = raw_instruction ? tts_parameters : request;
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
  const Json stages = ObjectField(request, "stage_params");
  const Json engine = ObjectField(stages, "auk_engine");
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
  SynthesisRequest synthesis{};
  const std::string reference = TextField(parameters, "ref_audio");
  if (!reference.empty()) {
    constexpr std::string_view prefix = "data:audio/wav;base64,";
    if (!reference.starts_with(prefix)) {
      throw std::invalid_argument(
          "ref_audio must be a data:audio/wav;base64 URI");
    } else {
    }
    const auto bytes =
        omni_common::DecodeBase64(reference.substr(prefix.size()));
    if (!bytes) {
      throw std::invalid_argument("ref_audio contains invalid base64");
    } else {
    }
    auto audio = omni_common::DecodeAudioWav(
        {reinterpret_cast<const char *>(bytes->data()), bytes->size()});
    // The conditioner needs three mel frames, 30 ms at 16 kHz.
    if (audio.samples.size() <
            static_cast<size_t>(
                (static_cast<int64_t>(audio.sample_rate) * 3 + 99) / 100) ||
        audio.samples.size() >
            static_cast<size_t>(audio.sample_rate * max_seconds)) {
      throw std::invalid_argument(
          "ref_audio must contain at least 30 ms and fit --max-seconds");
    } else if (!std::all_of(
                   audio.samples.begin(), audio.samples.end(),
                   [](float sample) { return std::isfinite(sample); })) {
      throw std::invalid_argument("ref_audio samples must be finite");
    } else {
    }
    synthesis.reference_audio = std::move(audio.samples);
    synthesis.reference_sample_rate = audio.sample_rate;
  } else {
  }
  if (raw_instruction) {
    if (FieldOr(request, "return_logprob", Json(false)) != Json(false) ||
        FieldOr(request, "output_modalities", Json::array({"audio"})) !=
            Json::array({"audio"})) {
      throw std::invalid_argument(
          "generate supports audio output without logprobs");
    } else {
    }
    synthesis.instruction = Trim(text);
  } else {
    const Json speed = FieldOr(request, "speed", Json(1.0));
    if (TextField(request, "voice", "default") != "default" ||
        !speed.is_number() || speed.get<double>() != 1.0) {
      throw std::invalid_argument(
          "only voice=default and speed=1 are supported");
    } else {
    }
    const std::string description = Trim(
        PromptTextField(request, "instructions", "A clear, natural voice."));
    if (description.size() > kMaximumTextBytes) {
      throw std::invalid_argument(
          "instructions must contain at most 8192 bytes");
    } else if (!synthesis.reference_audio.empty()) {
      synthesis.instruction =
          "Say the following with the same voice: \"" + text + "\"";
    } else {
      synthesis.instruction =
          "Generate speech based on the following description: \"" +
          (description.empty() ? "A clear, natural voice." : description) +
          "\". The content to speak is: \"" + text + "\".";
    }
  }
  const Json duration = engine.value(
      "gen_seconds",
      parameters.value("gen_seconds", request.value("gen_seconds", Json())));
  const double reference_seconds =
      static_cast<double>(synthesis.reference_audio.size()) /
      synthesis.reference_sample_rate;
  if (!duration.is_null()) {
    if (!duration.is_number() || !std::isfinite(duration.get<double>()) ||
        duration.get<double>() <= 0) {
      throw std::invalid_argument(
          "gen_seconds must be a positive finite number");
    } else {
      synthesis.seconds = duration.get<double>();
    }
  } else if (raw_instruction) {
    synthesis.seconds =
        reference_seconds > 0 ? reference_seconds : kDefaultSeconds;
    synthesis.match_reference_duration = reference_seconds > 0;
  } else {
    if (reference_seconds <= 0 || reference_text.empty()) {
      throw std::invalid_argument(
          "speech requires gen_seconds or ref_audio with ref_text");
    } else {
      synthesis.seconds =
          reference_seconds * text.size() / reference_text.size();
    }
  }
  synthesis.seconds =
      std::min(synthesis.seconds, static_cast<double>(max_seconds));
  const Json seed = parameters.value("seed", request.value("seed", Json()));
  if (!seed.is_null()) {
    if (!seed.is_number_integer() || seed < 0) {
      throw std::invalid_argument("seed must be a nonnegative integer");
    } else {
      synthesis.seed = seed.get<uint64_t>();
    }
  } else {
  }
  return synthesis;
}

} // namespace auk
