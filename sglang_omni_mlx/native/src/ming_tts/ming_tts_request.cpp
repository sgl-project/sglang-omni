// SPDX-License-Identifier: Apache-2.0
#include "ming_tts/ming_tts_request.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <string_view>

#include "common/reference_audio.h"
#include "common/text.h"
#include "qwen3_asr/speech_service.h"

namespace ming_tts {

namespace {

using speech_service::FieldOr;
using speech_service::Json;
using speech_service::PresentFieldOr;

// CUDA's MING_TTS_DEFAULT_MAX_DECODE_STEPS and its serving cap.
constexpr int kDefaultMaxPatches = 200;
constexpr int kMaxPatches = 256;
constexpr float kDefaultCfg = 2.0f;
constexpr float kMinCfg = 1e-5f;
constexpr float kDefaultSigma = 0.25f;
constexpr double kMinReferenceSeconds = 0.1;
constexpr double kMaxReferenceSeconds = 30;
constexpr const char *kEngineStage = "tts_engine";
// The logits sampling fields CUDA rejects when given, at the top level and
// among the engine's stage parameters.
constexpr std::array<const char *, 3> kLogitsSamplingFields = {
    "repetition_penalty", "top_k", "top_p"};
constexpr std::array<const char *, 4> kStageLogitsSamplingFields = {
    "do_sample", "repetition_penalty", "top_k", "top_p"};
constexpr std::array<const char *, 2> kDurationFields = {"duration_tokens",
                                                         "token_count"};
constexpr std::array<std::string_view, 4> kDefaultTaskTypes = {"", "default",
                                                               "tts", "speech"};

// As CUDA's str(value).strip().lower().
std::string Normalized(const std::string &text) {
  std::string lowered;
  for (const char character : omni_common::StripUnicodeWhitespace(text)) {
    lowered.push_back(
        static_cast<char>(std::tolower(static_cast<unsigned char>(character))));
  }
  return lowered;
}

// A field typed str | None: nullopt when it is missing or null.
std::optional<std::string> OptionalString(const Json &body, const char *name) {
  const Json value = FieldOr(body, name, Json());
  if (value.is_null()) {
    return std::nullopt;
  } else if (!value.is_string()) {
    throw std::invalid_argument(std::string(name) + " must be a string");
  } else {
    return value.get<std::string>();
  }
}

// As CUDA's optional_text: stripped, and nullopt when blank.
std::optional<std::string> OptionalText(const Json &body, const char *name) {
  const std::optional<std::string> value = OptionalString(body, name);
  if (!value.has_value()) {
    return std::nullopt;
  } else {
  }
  const std::string stripped = omni_common::StripUnicodeWhitespace(*value);
  return stripped.empty() ? std::nullopt : std::optional(stripped);
}

// As CUDA's has_non_empty_value: given, and not false or empty.
bool HasValue(const Json &value) {
  if (value.is_null() || (value.is_boolean() && !value.get<bool>())) {
    return false;
  } else if (value.is_string()) {
    return !omni_common::StripUnicodeWhitespace(value.get<std::string>())
                .empty();
  } else if (value.is_array() || value.is_object()) {
    return !value.empty();
  } else {
    return true;
  }
}

// The first of names that source gives, as CUDA's first_present; null when
// none is.
Json FirstGiven(const Json &source, std::initializer_list<const char *> names) {
  for (const char *name : names) {
    const Json value = FieldOr(source, name, Json());
    if (!value.is_null()) {
      return value;
    } else {
    }
  }
  return Json();
}

// A finite float32, or fallback for null.
float FiniteNumber(const Json &value, const char *name, float fallback) {
  if (value.is_null()) {
    return fallback;
  } else if (!value.is_number() || !std::isfinite(value.get<float>())) {
    throw std::invalid_argument(std::string("Ming-Omni-TTS ") + name +
                                " must be a finite number");
  } else {
    return value.get<float>();
  }
}

// A base64 mono WAV data URL at the rates the reference encoder takes.
ReferenceAudio DecodeReferenceAudio(const std::string &url) {
  if (!url.starts_with("data:")) {
    throw std::invalid_argument(
        "ref_audio must be a base64 audio data URL on this server");
  } else {
  }
  const std::optional<std::vector<uint8_t>> bytes = omni_common::DecodeBase64(
      omni_common::AudioDataUrlPayload(url, "ref_audio"));
  if (!bytes.has_value()) {
    throw std::invalid_argument("ref_audio must be valid base64");
  } else {
  }
  try {
    const omni_common::DecodedAudio decoded =
        omni_common::DecodeAnyWav(std::string_view(
            reinterpret_cast<const char *>(bytes->data()), bytes->size()));
    const double seconds =
        static_cast<double>(decoded.samples.size()) / decoded.sample_rate;
    if (seconds < kMinReferenceSeconds || seconds > kMaxReferenceSeconds) {
      throw std::invalid_argument(
          "reference audio must be between 0.1 and 30 seconds long");
    } else if (decoded.channel_count != 1) {
      throw std::invalid_argument(
          "Ming-Omni-TTS currently supports only mono reference audio");
    } else {
    }
    // Note (XXXXRT666): each encoder's rate is resampled from the file's, as
    // CUDA's load_reference_waveform does.
    return {omni_common::Resample(decoded.samples, decoded.sample_rate,
                                  kSpeakerSampleRate),
            omni_common::Resample(decoded.samples, decoded.sample_rate,
                                  kAudioSampleRate)};
  } catch (const std::invalid_argument &error) {
    throw std::invalid_argument(std::string("ref_audio: ") + error.what());
  }
}

} // namespace

SpeechRequest ParseSpeechRequest(const Json &body) {
  SpeechRequest request;
  const Json input = FieldOr(body, "input", Json());
  if (!input.is_string()) {
    throw std::invalid_argument("input must be a string");
  } else {
  }
  request.text = omni_common::StripUnicodeWhitespace(input.get<std::string>());
  if (request.text.empty()) {
    throw std::invalid_argument("Ming-Omni-TTS requires non-empty input text");
  } else {
  }

  const Json references = FieldOr(body, "references", Json::array());
  if (!references.is_array()) {
    throw std::invalid_argument("references must be a list");
  } else if (!references.empty()) {
    // Note (XXXXRT666): CUDA's references name local files, which a client
    // of this server cannot share.
    throw std::invalid_argument(
        "references are not supported on this server; give ref_audio and "
        "ref_text");
  } else {
  }
  std::optional<std::string> reference_url = OptionalString(body, "ref_audio");
  if (reference_url.has_value()) {
    reference_url = omni_common::StripUnicodeWhitespace(*reference_url);
    if (reference_url->empty()) {
      reference_url.reset();
    } else {
    }
  } else {
  }
  request.reference_text = OptionalText(body, "ref_text");
  if (reference_url.has_value() && !request.reference_text.has_value()) {
    throw std::invalid_argument(
        "Ming-Omni-TTS currently supports only one reference audio with "
        "non-empty reference text");
  } else if (!reference_url.has_value() && request.reference_text.has_value()) {
    throw std::invalid_argument(
        "Ming-Omni-TTS reference text requires reference audio");
  } else {
  }

  for (const char *field : kDurationFields) {
    if (!FieldOr(body, field, Json()).is_null()) {
      throw std::invalid_argument(std::string("Ming-Omni-TTS field '") + field +
                                  "' is currently unsupported");
    } else {
    }
  }
  // Note (XXXXRT666): as CUDA's AliasChoices("voice", "speaker"), voice wins
  // whenever it is given, even as null.
  const Json voice = PresentFieldOr(
      body, "voice", PresentFieldOr(body, "speaker", Json("default")));
  if (!voice.is_string()) {
    throw std::invalid_argument("voice must be a string");
  } else if (const std::string name = Normalized(voice.get<std::string>());
             name != "" && name != "default") {
    throw std::invalid_argument(
        "Ming-Omni-TTS currently supports only the default voice; got " +
        voice.dump());
  } else {
  }
  if (const std::optional<std::string> language =
          OptionalString(body, "language");
      language.has_value() && Normalized(*language) != "" &&
      Normalized(*language) != "auto") {
    throw std::invalid_argument(
        "Ming-Omni-TTS language selection is currently unsupported; got " +
        Json(*language).dump());
  } else {
  }
  if (OptionalText(body, "instructions").has_value()) {
    throw std::invalid_argument(
        "Ming-Omni-TTS instructions are currently unsupported");
  } else {
  }
  if (const std::optional<std::string> task_type =
          OptionalString(body, "task_type");
      task_type.has_value() &&
      std::find(kDefaultTaskTypes.begin(), kDefaultTaskTypes.end(),
                Normalized(*task_type)) == kDefaultTaskTypes.end()) {
    throw std::invalid_argument(
        "Ming-Omni-TTS currently supports only default text-to-speech; got "
        "task_type=" +
        Json(*task_type).dump());
  } else {
  }
  const Json speed = PresentFieldOr(body, "speed", Json(1.0));
  if (!speed.is_number() || speed.get<double>() != 1.0) {
    throw std::invalid_argument(
        "Ming-Omni-TTS speed control is currently unsupported; got " +
        speed.dump());
  } else {
  }
  if (!FieldOr(body, "initial_codec_chunk_frames", Json()).is_null()) {
    throw std::invalid_argument(
        "Ming-Omni-TTS does not support initial_codec_chunk_frames");
  } else {
  }

  // Note (XXXXRT666): CreateSpeechRequest types these even when a stage
  // parameter overrides them.
  if (const Json value = FieldOr(body, "max_new_tokens", Json());
      !value.is_null() && !value.is_number_integer()) {
    throw std::invalid_argument("max_new_tokens must be an integer");
  } else if (const Json value = FieldOr(body, "temperature", Json());
             !value.is_null() && !value.is_number()) {
    throw std::invalid_argument("temperature must be a number");
  } else {
  }
  const Json stages = FieldOr(body, "stage_params", Json::object());
  if (!stages.is_object()) {
    throw std::invalid_argument("stage_params must be an object");
  } else {
  }
  for (const auto &[stage, parameters] : stages.items()) {
    if (!parameters.is_object()) {
      throw std::invalid_argument("stage_params." + stage +
                                  " must be an object");
    } else {
    }
  }
  const Json engine = FieldOr(stages, kEngineStage, Json::object());
  for (const char *field : kLogitsSamplingFields) {
    if (!FieldOr(body, field, Json()).is_null()) {
      throw std::invalid_argument(
          std::string("Ming-Omni-TTS does not use logits sampling field '") +
          field + "'");
    } else {
    }
  }
  for (const char *field : kStageLogitsSamplingFields) {
    if (HasValue(FieldOr(engine, field, Json()))) {
      throw std::invalid_argument(
          std::string("Ming-Omni-TTS does not use logits sampling field '") +
          field + "'");
    } else {
    }
  }
  if (!FieldOr(engine, "seed", Json()).is_null() ||
      !FieldOr(body, "seed", Json()).is_null()) {
    throw std::invalid_argument("Ming-Omni-TTS seed is currently unsupported "
                                "because FlowLoss sampling is unseeded");
  } else {
  }

  Json max_patches =
      FirstGiven(engine, {"max_decode_steps", "ming_max_decode_steps"});
  if (max_patches.is_null()) {
    max_patches = FieldOr(body, "max_new_tokens", Json(kDefaultMaxPatches));
  } else {
  }
  if (!max_patches.is_number_integer()) {
    throw std::invalid_argument(
        "Ming-Omni-TTS max_decode_steps must be an integer");
  } else if (max_patches.get<int64_t>() <= 0) {
    throw std::invalid_argument(
        "Ming-Omni-TTS max_decode_steps must be > 0, got " +
        max_patches.dump());
  } else if (max_patches.get<int64_t>() > kMaxPatches) {
    throw std::invalid_argument(
        "Ming-Omni-TTS max_decode_steps exceeds serving cap: " +
        max_patches.dump() + " > " + std::to_string(kMaxPatches));
  } else {
  }
  request.max_patches = max_patches.get<int>();
  Json temperature = FirstGiven(engine, {"temperature", "ming_temperature"});
  if (temperature.is_null()) {
    temperature = FieldOr(body, "temperature", Json());
  } else {
  }
  request.temperature = FiniteNumber(temperature, "temperature", 0.0f);
  request.cfg =
      FiniteNumber(FieldOr(engine, "cfg", Json()), "cfg", kDefaultCfg);
  request.sigma =
      FiniteNumber(FieldOr(engine, "sigma", Json()), "sigma", kDefaultSigma);
  if (request.cfg < kMinCfg || request.cfg == 1.0f) {
    throw std::invalid_argument(
        "Ming-Omni-TTS currently supports only guided-branch cfg values; "
        "cfg must be >= 1e-5 and not equal to 1.0");
  } else if (request.sigma < 0) {
    throw std::invalid_argument("Ming-Omni-TTS sigma must be >= 0");
  } else if (request.temperature < 0) {
    throw std::invalid_argument("Ming-Omni-TTS temperature must be >= 0");
  } else {
  }

  if (reference_url.has_value()) {
    request.reference_audio = DecodeReferenceAudio(*reference_url);
  } else {
  }
  return request;
}

} // namespace ming_tts
