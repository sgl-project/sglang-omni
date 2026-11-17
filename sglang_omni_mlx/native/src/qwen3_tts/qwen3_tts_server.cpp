// SPDX-License-Identifier: Apache-2.0
// Native Qwen3-TTS CustomVoice server: POST /v1/audio/speech with the CUDA
// server's request fields, answering a WAV or PCM file, or 16-bit PCM
// streamed as it is decoded, and Voxt's supervisor protocol.
//
//   qwen3_tts_server --model-path DIR [--model-name NAME] [--host H]
//       [--port P] [--max-queued-requests N] [--request-timeout-s S]
//   qwen3_tts_server --supervised --model-kind qwen3_tts
//       --model-directory DIR [--max-queued-requests N]
//       [--request-timeout-s S]
//
// Request fields: input, model (the served name, when given), voice (or
// speaker; "default" is Vivian), language, instructions, response_format (wav
// or pcm), stream (pcm only), speed, max_new_tokens (audio frames, 12.5 a
// second), temperature, top_p, top_k, repetition_penalty, seed, and the
// talker and subtalker settings of stage_params.tts_engine. One request runs
// at a time and N more (2 by default) may wait; the rest get 503. A request
// not done S seconds (600 by default) after it arrives is cancelled, with 504
// or an early end to its stream, and so is a request whose client
// disconnects. A request body over 1 MiB gets 413.
#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/asr_service.h"
#include "common/generation.h"
#include "common/speech_service.h"
#include "common/text.h"
#include "qwen3_tts/qwen3_tts_model.h"

namespace {

using speech_service::FieldOr;
using speech_service::Json;
using speech_service::PresentFieldOr;

constexpr const char *kDefaultSpeaker = "vivian";
constexpr size_t kMaxInputCharacters = 4096;
// Far above the largest valid request: kMaxInputCharacters characters, each
// JSON-escaped as a 12-byte surrogate pair, take 48 KiB.
constexpr size_t kMaxRequestBodyBytes = 1024 * 1024;
// Note (Jiaxin Deng): the CUDA server's default when a request sets none; its
// talker's context is 8192 positions.
constexpr int kDefaultAudioFrames = 2048;
constexpr int kTalkerContextPositions = 8192;
constexpr size_t kMaxInstructionCharacters = 4096;
constexpr int kDefaultFirstChunkFrames = 5;
constexpr double kMinSpeed = 0.25;
constexpr double kMaxSpeed = 4.0;
// Note (Jiaxin Deng): the languages the CUDA server's request validation
// names, matched case-insensitively.
constexpr std::array<const char *, 11> kLanguages = {
    "Auto",     "Chinese", "English",    "French",  "German", "Italian",
    "Japanese", "Korean",  "Portuguese", "Russian", "Spanish"};
// CUDA's raw type checks, each over fields given and not null.
constexpr std::array<const char *, 10> kStringFields = {
    "model",     "input",    "voice",        "speaker",   "response_format",
    "task_type", "language", "instructions", "ref_audio", "ref_text"};
constexpr std::array<const char *, 6> kIntegerFields = {
    "max_new_tokens", "initial_codec_chunk_frames",
    "token_count",    "duration_tokens",
    "seed",           "top_k"};
constexpr std::array<const char *, 4> kNumberFields = {
    "speed", "temperature", "top_p", "repetition_penalty"};
constexpr std::array<const char *, 4> kBooleanFields = {
    "stream", "x_vector_only_mode", "stream_codec_output",
    "suppress_bootstrap_silence"};

struct SpeechRequest {
  qwen3_tts::SpeechPrompt prompt;
  qwen3_tts::SamplingConfig sampling;
  std::optional<std::string> model;
  std::string response_format;
  bool stream = false;
  double speed = 1.0;
  int max_frames = kDefaultAudioFrames;
  // A streamed request's first chunk; 0 starts at the steady chunk.
  int first_chunk_frames = kDefaultFirstChunkFrames;
};

// Note (Jiaxin Deng): CUDA checks each field's JSON type before its model
// validation, in this order.
void CheckFieldTypes(const Json &body) {
  for (const char *field : kStringFields) {
    if (!FieldOr(body, field, Json()).is_null() &&
        !body.at(field).is_string()) {
      throw std::invalid_argument(std::string(field) + " must be a string");
    } else {
    }
  }
  for (const char *field : kIntegerFields) {
    if (!FieldOr(body, field, Json()).is_null() &&
        !body.at(field).is_number_integer()) {
      throw std::invalid_argument(std::string(field) + " must be an integer");
    } else {
    }
  }
  for (const char *field : kNumberFields) {
    if (!FieldOr(body, field, Json()).is_null() &&
        !body.at(field).is_number()) {
      throw std::invalid_argument(std::string(field) + " must be a number");
    } else {
    }
  }
  for (const char *field : kBooleanFields) {
    if (!FieldOr(body, field, Json()).is_null() &&
        !body.at(field).is_boolean()) {
      throw std::invalid_argument(std::string(field) + " must be a boolean");
    } else {
    }
  }
  for (const char *field : {"voice", "response_format"}) {
    if (body.contains(field) && body.at(field).is_null()) {
      throw std::invalid_argument(std::string(field) + " must be a string");
    } else {
    }
  }
  if (body.contains("speed") && body.at("speed").is_null()) {
    throw std::invalid_argument("speed must be a number");
  } else if (body.contains("stream") && body.at("stream").is_null()) {
    throw std::invalid_argument("stream must be a boolean");
  } else {
  }
}

// A sampling setting: stage_params.tts_engine's when given, else the
// request's, else fallback.
Json SamplingField(const Json &body, const Json &engine, const char *name,
                   const char *engine_name, Json fallback) {
  const Json overridden = FieldOr(engine, engine_name, Json());
  return overridden.is_null() ? FieldOr(body, name, fallback) : overridden;
}

double NumberSetting(const Json &value, const std::string &name) {
  if (!value.is_number() || value.is_boolean()) {
    throw std::invalid_argument(name + " must be a number");
  } else {
  }
  return value.get<double>();
}

// Note (Jiaxin Deng): the bounds the CUDA engine's sampling parameters
// verify, with top_k -1 for no limit.
qwen3_tts::SamplerConfig SamplerSettings(const Json &do_sample,
                                         const Json &temperature,
                                         const Json &top_p, const Json &top_k,
                                         const std::string &prefix) {
  qwen3_tts::SamplerConfig sampler;
  if (!do_sample.is_boolean()) {
    throw std::invalid_argument(prefix + "do_sample must be a boolean");
  } else {
  }
  sampler.temperature =
      static_cast<float>(NumberSetting(temperature, prefix + "temperature"));
  sampler.top_p = static_cast<float>(NumberSetting(top_p, prefix + "top_p"));
  if (!top_k.is_number_integer()) {
    throw std::invalid_argument(prefix + "top_k must be an integer");
  } else {
  }
  const int64_t top_k_value = top_k.get<int64_t>();
  if (!(sampler.temperature >= 0.0f)) {
    throw std::invalid_argument(prefix + "temperature must be non-negative");
  } else if (!(sampler.top_p > 0.0f && sampler.top_p <= 1.0f)) {
    throw std::invalid_argument(prefix + "top_p must be in (0, 1]");
  } else if (top_k_value != -1 && top_k_value < 1) {
    throw std::invalid_argument(prefix +
                                "top_k must be -1 (disable) or at least 1");
  } else {
  }
  sampler.top_k = top_k_value == -1 || top_k_value > INT32_MAX
                      ? 0
                      : static_cast<int>(top_k_value);
  if (!do_sample.get<bool>()) {
    sampler.temperature = 0.0f;
  } else {
  }
  return sampler;
}

// The Unicode code points of UTF-8 text.
SpeechRequest ParseRequest(const Json &body,
                           const qwen3_tts::Qwen3TtsModel &model) {
  if (!body.is_object()) {
    throw std::invalid_argument("speech request body must be a JSON object");
  } else {
  }
  CheckFieldTypes(body);
  const Json stream_format =
      PresentFieldOr(body, "stream_format", Json("audio"));
  if (stream_format != "audio") {
    throw std::invalid_argument("stream_format must be audio");
  } else {
  }
  const Json references = FieldOr(body, "references", Json());
  if (!references.is_null() && !references.is_array()) {
    throw std::invalid_argument("references must be a list");
  } else {
  }
  const Json stage_params = FieldOr(body, "stage_params", Json::object());
  if (!stage_params.is_object()) {
    throw std::invalid_argument("stage_params must be an object");
  } else {
  }
  for (const auto &[stage, settings] : stage_params.items()) {
    if (!settings.is_object()) {
      throw std::invalid_argument("stage_params." + stage +
                                  " must be an object");
    } else {
    }
  }
  const Json input = FieldOr(body, "input", Json());
  if (!input.is_string() ||
      omni_common::StripUnicodeWhitespace(input.get<std::string>()).empty()) {
    throw std::invalid_argument("input must be a non-empty string");
  } else {
  }
  const std::string text = input.get<std::string>();
  if (omni_common::CodePointCount(text) > kMaxInputCharacters) {
    throw std::invalid_argument("input must be at most " +
                                std::to_string(kMaxInputCharacters) +
                                " characters");
  } else {
  }

  SpeechRequest request;
  const Json model_name = FieldOr(body, "model", Json());
  if (model_name.is_string()) {
    request.model = model_name.get<std::string>();
  } else {
  }
  // Note (Minyi Tang): as CUDA's response_format.strip().lower().
  for (const char character : omni_common::StripUnicodeWhitespace(
           PresentFieldOr(body, "response_format", Json("wav"))
               .get<std::string>())) {
    request.response_format.push_back(
        static_cast<char>(std::tolower(static_cast<unsigned char>(character))));
  }
  if (request.response_format != "pcm" && request.response_format != "wav") {
    throw std::invalid_argument("response_format must be one of: pcm, wav");
  } else {
  }
  request.stream = PresentFieldOr(body, "stream", Json(false)).get<bool>();
  if (request.stream && request.response_format != "pcm") {
    throw std::invalid_argument("stream=true requires response_format='pcm'");
  } else {
  }
  request.speed = PresentFieldOr(body, "speed", Json(1.0)).get<double>();
  if (!(request.speed >= kMinSpeed && request.speed <= kMaxSpeed)) {
    throw std::invalid_argument("speed must be between 0.25 and 4.0");
  } else {
  }

  const Json task_type = FieldOr(body, "task_type", Json());
  if (task_type.is_string()) {
    std::string normalized;
    for (const char character :
         omni_common::StripUnicodeWhitespace(task_type.get<std::string>())) {
      if (character != '_' && character != '-') {
        normalized.push_back(static_cast<char>(
            std::tolower(static_cast<unsigned char>(character))));
      } else {
      }
    }
    if (normalized != "base" && normalized != "customvoice" &&
        normalized != "voicedesign") {
      throw std::invalid_argument(
          "task_type must be one of: Base, CustomVoice, VoiceDesign");
    } else if (normalized != "customvoice") {
      throw std::invalid_argument("task_type must be one of: CustomVoice");
    } else {
    }
  } else {
  }
  for (const char *field : {"ref_audio", "ref_text", "x_vector_only_mode"}) {
    if (!FieldOr(body, field, Json()).is_null()) {
      throw std::invalid_argument(std::string(field) +
                                  " is not supported by this model");
    } else {
    }
  }
  if (!references.is_null() && !references.empty()) {
    throw std::invalid_argument("references are not supported by this model");
  } else {
  }
  // Note (Minyi Tang): as CUDA's AliasChoices("voice", "speaker"), voice
  // wins whenever it is given.
  const Json voice_field = PresentFieldOr(
      body, "voice", PresentFieldOr(body, "speaker", Json("default")));
  if (!voice_field.is_string()) {
    throw std::invalid_argument("voice must be a string");
  } else {
  }
  const std::string voice = voice_field.get<std::string>();
  std::string speaker =
      omni_common::AsciiLowercase(omni_common::StripUnicodeWhitespace(voice));
  if (speaker.empty() || speaker == "default") {
    speaker = kDefaultSpeaker;
  } else if (model.speaker_ids().count(speaker) == 0) {
    std::string supported = "default";
    for (const std::string &name : model.speaker_names()) {
      supported += ", " + name;
    }
    throw std::invalid_argument("Unknown voice '" + voice +
                                "'. Supported voices: " + supported);
  } else {
  }
  std::string language = "auto";
  const Json language_field = FieldOr(body, "language", Json());
  if (language_field.is_string()) {
    language = omni_common::AsciiLowercase(
        omni_common::StripUnicodeWhitespace(language_field.get<std::string>()));
    const bool known = std::any_of(
        kLanguages.begin(), kLanguages.end(), [&](const char *name) {
          return omni_common::AsciiLowercase(name) == language;
        });
    if (!known) {
      std::string supported;
      for (const char *name : kLanguages) {
        supported += (supported.empty() ? "" : ", ") + std::string(name);
      }
      throw std::invalid_argument("language must be one of: " + supported);
    } else {
    }
  } else {
  }

  const Json engine = FieldOr(stage_params, "tts_engine", Json::object());
  const Json max_new_tokens =
      SamplingField(body, engine, "max_new_tokens", "max_new_tokens",
                    Json(kDefaultAudioFrames));
  if (!max_new_tokens.is_number_integer()) {
    throw std::invalid_argument("max_new_tokens must be an integer");
  } else if (max_new_tokens.get<int64_t>() <= 0) {
    throw std::invalid_argument("max_new_tokens must be greater than 0");
  } else {
  }
  const bool max_new_tokens_given =
      !FieldOr(engine, "max_new_tokens", Json()).is_null() ||
      !FieldOr(body, "max_new_tokens", Json()).is_null();
  for (const char *field : {"token_count", "duration_tokens"}) {
    if (!FieldOr(body, field, Json()).is_null() &&
        body.at(field).get<int64_t>() <= 0) {
      throw std::invalid_argument(std::string(field) +
                                  " must be greater than 0");
    } else {
    }
  }
  const Json initial_chunk =
      FieldOr(body, "initial_codec_chunk_frames", Json());
  // Note (Jiaxin Deng): as on CUDA, 0 starts at the steady chunk.
  if (initial_chunk.is_number_integer() && initial_chunk.get<int64_t>() >= 0) {
    request.first_chunk_frames = static_cast<int>(
        std::min<int64_t>(initial_chunk.get<int64_t>(), INT32_MAX));
  } else {
  }
  for (const char *field : {"initial_codec_chunk_frames", "seed"}) {
    const Json value = FieldOr(body, field, Json());
    if (value.is_number_integer() && !value.is_number_unsigned() &&
        value.get<int64_t>() < 0) {
      throw std::invalid_argument(std::string(field) +
                                  " must be greater than or equal to 0");
    } else {
    }
  }
  const qwen3_tts::SamplingConfig defaults;
  request.sampling.talker =
      SamplerSettings(FieldOr(engine, "do_sample", Json(true)),
                      SamplingField(body, engine, "temperature", "temperature",
                                    Json(defaults.talker.temperature)),
                      SamplingField(body, engine, "top_p", "top_p",
                                    Json(defaults.talker.top_p)),
                      SamplingField(body, engine, "top_k", "top_k",
                                    Json(defaults.talker.top_k)),
                      "");
  request.sampling.code_predictor = SamplerSettings(
      FieldOr(engine, "subtalker_dosample", Json(true)),
      FieldOr(engine, "subtalker_temperature",
              Json(defaults.code_predictor.temperature)),
      FieldOr(engine, "subtalker_top_p", Json(defaults.code_predictor.top_p)),
      FieldOr(engine, "subtalker_top_k", Json(defaults.code_predictor.top_k)),
      "subtalker_");
  request.sampling.repetition_penalty = static_cast<float>(NumberSetting(
      SamplingField(body, engine, "repetition_penalty", "repetition_penalty",
                    Json(defaults.repetition_penalty)),
      "repetition_penalty"));
  if (!(request.sampling.repetition_penalty > 0.0f &&
        request.sampling.repetition_penalty <= 2.0f)) {
    throw std::invalid_argument("repetition_penalty must be in (0, 2]");
  } else {
  }
  const Json seed = FieldOr(body, "seed", Json());
  if (!seed.is_null()) {
    request.sampling.seed = seed.get<uint64_t>();
  } else {
  }

  const std::string instructions = omni_common::StripUnicodeWhitespace(
      FieldOr(body, "instructions", Json("")).get<std::string>());
  if (omni_common::CodePointCount(instructions) > kMaxInstructionCharacters) {
    throw std::invalid_argument("instructions must be at most " +
                                std::to_string(kMaxInstructionCharacters) +
                                " characters");
  } else {
  }
  request.prompt = model.BuildPrompt(text, speaker, language, instructions);
  // Note (Jiaxin Deng): the CUDA engine's talker context is 8192 positions:
  // a longer prompt, or an asked-for length past it, is the client's error,
  // and the default length shrinks to fit.
  const int prompt_length = model.PromptLength(request.prompt);
  const int64_t frames = max_new_tokens.get<int64_t>();
  if (prompt_length >= kTalkerContextPositions) {
    throw std::invalid_argument("input is longer than the model's context "
                                "length of " +
                                std::to_string(kTalkerContextPositions));
  } else if (max_new_tokens_given &&
             frames > kTalkerContextPositions - prompt_length) {
    throw std::invalid_argument(
        "max_new_tokens must be at most " +
        std::to_string(kTalkerContextPositions - prompt_length) +
        " for this input, the model's context length of " +
        std::to_string(kTalkerContextPositions) + " less its prompt");
  } else {
  }
  request.max_frames = static_cast<int>(
      std::min<int64_t>(frames, kTalkerContextPositions - prompt_length));
  return request;
}

// Samples resampled linearly to 1 / speed of their length, as the CUDA
// server's apply_speed does.
std::vector<float> ApplySpeed(const std::vector<float> &samples, double speed) {
  if (speed == 1.0 || samples.empty()) {
    return samples;
  } else {
  }
  const size_t count = samples.size();
  const auto resampled_count = static_cast<size_t>(
      std::max(1.0, std::nearbyint(static_cast<double>(count) / speed)));
  std::vector<float> resampled(resampled_count);
  const double last = static_cast<double>(count - 1);
  for (size_t index = 0; index < resampled_count; ++index) {
    const double position = resampled_count == 1
                                ? 0.0
                                : last * static_cast<double>(index) /
                                      static_cast<double>(resampled_count - 1);
    const auto left = static_cast<size_t>(std::floor(position));
    const size_t right = std::min(left + 1, count - 1);
    const double fraction = position - static_cast<double>(left);
    resampled[index] = static_cast<float>(static_cast<double>(samples[left]) +
                                          (static_cast<double>(samples[right]) -
                                           static_cast<double>(samples[left])) *
                                              fraction);
  }
  return resampled;
}

class Qwen3TtsService : public speech_service::SpeechServedModel {
public:
  Qwen3TtsService(const std::filesystem::path &model_directory,
                  int max_queued_requests, double request_timeout_seconds)
      : SpeechServedModel(kMaxRequestBodyBytes, max_queued_requests,
                          request_timeout_seconds),
        model_(model_directory) {}

private:
  int SampleRate() const override {
    return model_.speech_decoder().sample_rate();
  }

  speech_service::SpeechJob ParseSpeech(const Json &body) override {
    const SpeechRequest request = ParseRequest(body, model_);
    speech_service::SpeechJob job;
    job.model = request.model;
    job.response_format = request.response_format;
    job.stream = request.stream;
    // Note (Jiaxin Deng): as CUDA, a queued stream has its headers at once.
    job.starts_stream_at_admission = true;
    job.synthesize = [this,
                      request](const speech_service::SpeechContext &context) {
      return Synthesize(request, context);
    };
    return job;
  }

  speech_service::SpeechResult
  Synthesize(const SpeechRequest &request,
             const speech_service::SpeechContext &context) const {
    qwen3_tts::SpeechStream stream(model_.speech_decoder(), request.stream,
                                   request.first_chunk_frames);
    // Note (Jiaxin Deng): a file is sped up whole, as the CUDA server does.
    std::vector<float> file_samples;
    const auto send = [&](std::vector<float> samples) {
      if (request.stream) {
        context.send(ApplySpeed(samples, request.speed));
      } else {
        file_samples.insert(file_samples.end(), samples.begin(), samples.end());
      }
    };
    int frame_count = 0;
    const omni_common::FinishReason finish_reason =
        model_.Generate(request.prompt, request.sampling, request.max_frames,
                        context.cancel, [&](const mlx::core::array &codes) {
                          context.check_deadline();
                          ++frame_count;
                          send(stream.Push(codes));
                        });
    if (frame_count == 0) {
      // Note (Minyi Tang): CUDA answers no audio output with an error.
      throw std::runtime_error("Qwen3-TTS generated no audio codes");
    } else if (context.cancel.load()) {
      throw omni_common::GenerationCancelled();
    } else {
    }
    context.check_deadline();
    send(stream.Finish());
    if (context.cancel.load()) {
      throw omni_common::GenerationCancelled();
    } else if (!request.stream && file_samples.empty()) {
      // Note (Jiaxin Deng): CUDA answers a decode with no audio as a failure.
      throw std::runtime_error("Qwen3-TTS decoded no audio");
    } else if (!request.stream) {
      context.send(ApplySpeed(file_samples, request.speed));
    } else {
    }
    context.check_deadline();
    // Note (Jiaxin Deng): CUDA counts a CustomVoice request's prompt as no
    // reference codes.
    return {finish_reason, 0, frame_count};
  }

  qwen3_tts::Qwen3TtsModel model_;
};

} // namespace

int main(int argc, char **argv) {
  asr_service::ServedKind kind;
  kind.model_kind = "qwen3_tts";
  speech_service::AddSpeechFlags(
      kind, [](const std::filesystem::path &model_directory,
               int max_queued_requests, double request_timeout_seconds) {
        return std::make_unique<Qwen3TtsService>(
            model_directory, max_queued_requests, request_timeout_seconds);
      });
  return asr_service::Serve(argc, argv, kind);
}
