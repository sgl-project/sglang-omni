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
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <filesystem>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <system_error>
#include <typeinfo>
#include <vector>

#include "civetweb.h"
#include "common/audio.h"
#include "common/http.h"
#include "common/text.h"
#include "qwen3_asr/asr_service.h"
#include "qwen3_asr/qwen3_asr_transcriber.h"
#include "qwen3_tts/qwen3_tts_model.h"

namespace {

using omni_server::Json;
using qwen3_asr::TranscriptionResult;

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
constexpr int kRequestIdHexDigits = 32;
constexpr int kMaxRequestTimeoutSeconds = 24 * 60 * 60;
// As often as the CUDA server checks whether a speech client disconnected.
constexpr auto kClientPollInterval = std::chrono::milliseconds(50);
// The status civetweb logs for a request whose client left before its answer.
constexpr int kClientLeftStatus = 499;
// The most one disconnect check reads of what a client sent after its request,
// so a client that keeps sending cannot hold the check.
constexpr size_t kMaxUnreadBytesPerCheck = 64 * 1024;
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

// Audio one synthesis hands from the worker thread to its request's thread.
struct AudioChannel {
  std::mutex mutex;
  std::condition_variable ready;
  std::deque<std::vector<float>> chunks;
  bool done = false;
  std::exception_ptr error;
  TranscriptionResult result;
};

// body's field, or fallback when it is missing or null.
Json FieldOr(const Json &body, const char *name, Json fallback) {
  const auto found = body.find(name);
  if (found == body.end() || found->is_null()) {
    return fallback;
  } else {
    return *found;
  }
}

// body's field, null included, or fallback when it is missing.
// Note (Minyi Tang): CUDA's CreateSpeechRequest takes null as the default
// only for the fields it types as optional, and rejects it for the rest.
Json PresentFieldOr(const Json &body, const char *name, Json fallback) {
  const auto found = body.find(name);
  if (found == body.end()) {
    return fallback;
  } else {
    return *found;
  }
}

std::string Lower(const std::string &text) {
  std::string lowered;
  for (const char character : text) {
    lowered.push_back(
        static_cast<char>(std::tolower(static_cast<unsigned char>(character))));
  }
  return lowered;
}

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
size_t CharacterCount(const std::string &text) {
  return static_cast<size_t>(
      std::count_if(text.begin(), text.end(), [](char byte) {
        return (static_cast<unsigned char>(byte) & 0xc0) != 0x80;
      }));
}

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
  if (CharacterCount(text) > kMaxInputCharacters) {
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
  std::string speaker = Lower(omni_common::StripUnicodeWhitespace(voice));
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
    language = Lower(
        omni_common::StripUnicodeWhitespace(language_field.get<std::string>()));
    const bool known =
        std::any_of(kLanguages.begin(), kLanguages.end(),
                    [&](const char *name) { return Lower(name) == language; });
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
  if (CharacterCount(instructions) > kMaxInstructionCharacters) {
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

std::string NewRequestId() {
  static thread_local std::mt19937_64 generator{std::random_device{}()};
  static constexpr char kHex[] = "0123456789abcdef";
  std::string id = "speech-";
  for (int index = 0; index < kRequestIdHexDigits; ++index) {
    id.push_back(kHex[generator() % 16]);
  }
  return id;
}

// The socket civetweb accepted for connection's client, found by the client's
// address and port since civetweb does not expose it, or -1.
int ClientSocket(const mg_connection *connection) {
  const mg_request_info *request_info = mg_get_request_info(connection);
  // Note (Minyi Tang): a listing that fails, as it does in a process out of
  // descriptors, ends the search, and the request is then served without
  // watching its client. Nothing here throws, since an exception would unwind
  // through civetweb's C callback and terminate the server.
  std::error_code listing_error;
  for (std::filesystem::directory_iterator listing("/dev/fd", listing_error);
       !listing_error && listing != std::filesystem::directory_iterator();
       listing.increment(listing_error)) {
    const std::string name = listing->path().filename().string();
    int file_descriptor = -1;
    const auto [parse_end, parse_error] = std::from_chars(
        name.data(), name.data() + name.size(), file_descriptor);
    if (parse_error != std::errc() || parse_end != name.data() + name.size()) {
      continue;
    } else {
    }
    sockaddr_storage peer{};
    socklen_t peer_size = sizeof(peer);
    char address[INET6_ADDRSTRLEN] = "";
    int port = -1;
    if (getpeername(file_descriptor, reinterpret_cast<sockaddr *>(&peer),
                    &peer_size) != 0) {
      continue;
    } else if (peer.ss_family == AF_INET) {
      const auto *ipv4 = reinterpret_cast<const sockaddr_in *>(&peer);
      inet_ntop(AF_INET, &ipv4->sin_addr, address, sizeof(address));
      port = ntohs(ipv4->sin_port);
    } else if (peer.ss_family == AF_INET6) {
      const auto *ipv6 = reinterpret_cast<const sockaddr_in6 *>(&peer);
      inet_ntop(AF_INET6, &ipv6->sin6_addr, address, sizeof(address));
      port = ntohs(ipv6->sin6_port);
    } else {
    }
    if (port == request_info->remote_port &&
        request_info->remote_addr == std::string(address)) {
      return file_descriptor;
    } else {
    }
  }
  return -1;
}

// Whether the client of client_socket closed it, which its socket shows by
// reading as ended or failing.
// Note (Minyi Tang): civetweb keeps no connection alive, so it never serves
// what a client sends after its request; that is read and dropped here, since
// bytes left unread would hide the end of the socket from every later check.
// Note (Minyi Tang): a client that only shuts down its sending side also reads
// as ended and is treated as gone, as uvicorn treats it for the CUDA server.
// TCP tells the two apart only once something is written to the socket.
bool ClientLeft(int client_socket) {
  if (client_socket < 0) {
    return false;
  } else {
  }
  std::array<char, 4096> unread{};
  size_t dropped_bytes = 0;
  while (dropped_bytes < kMaxUnreadBytesPerCheck) {
    const ssize_t received =
        recv(client_socket, unread.data(), unread.size(), MSG_DONTWAIT);
    if (received == 0) {
      return true;
    } else if (received < 0) {
      return errno != EAGAIN && errno != EINTR;
    } else {
      dropped_bytes += static_cast<size_t>(received);
    }
  }
  return false;
}

// Thrown by a generation that reaches its request's deadline; a cancellation,
// so it is never logged as a failure.
class DeadlineReached : public qwen3_asr::TranscriptionCancelled {};

// Whether error is a generation stopping at its request's deadline.
bool ReachedDeadline(std::exception_ptr error) {
  if (!error) {
    return false;
  } else {
  }
  try {
    std::rethrow_exception(error);
  } catch (const DeadlineReached &) {
    return true;
  } catch (...) {
    return false;
  }
}

void LogFailure(std::exception_ptr error) {
  try {
    std::rethrow_exception(error);
  } catch (const qwen3_asr::TranscriptionCancelled &) {
  } catch (const std::exception &failure) {
    // Note (Minyi Tang): only the exception type is logged, never the text.
    std::cerr << "speech generation failed: " << typeid(failure).name() << "\n";
  }
}

class Qwen3TtsService : public asr_service::ServedModel {
public:
  Qwen3TtsService(const std::filesystem::path &model_directory,
                  int max_queued_requests, double request_timeout_seconds)
      : model_(model_directory), max_queued_requests_(max_queued_requests),
        request_timeout_(
            std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                std::chrono::duration<double>(request_timeout_seconds))) {}

  asr_service::Transcription
  Prepare(std::vector<float>, const asr_service::FormFields &) const override {
    throw std::logic_error("qwen3_tts serves no transcriptions");
  }
  bool Transcribes() const override { return false; }
  // Note (Minyi Tang): /health counts the slots admission holds, which the
  // worker frees only after it marks a request done; once /health shows the
  // server idle, it admits a full queue.
  std::map<std::string, int>
  RequestStates(const qwen3_asr::TranscriptionWorker &) const override {
    std::lock_guard<std::mutex> lock(admission_mutex_);
    std::map<std::string, int> states;
    if (admitted_requests_ > 0) {
      states["running"] = 1;
    } else {
    }
    if (admitted_requests_ > 1) {
      states["queued"] = admitted_requests_ - 1;
    } else {
    }
    return states;
  }
  void AddHandlers(mg_context *context,
                   qwen3_asr::TranscriptionWorker &worker) override {
    worker_ = &worker;
    mg_set_request_handler(context, "/v1/audio/speech$", HandleSpeech, this);
  }

private:
  static int HandleSpeech(mg_connection *connection, void *data) {
    // Note (Jiaxin Deng): nothing may unwind through civetweb's C callback;
    // what the handler does not answer itself ends the connection.
    try {
      return static_cast<Qwen3TtsService *>(data)->Speak(connection);
    } catch (const std::exception &failure) {
      std::cerr << "speech request failed: " << typeid(failure).name() << "\n";
      return 500;
    }
  }

  int Speak(mg_connection *connection) {
    if (!omni_server::IsPost(connection)) {
      return omni_server::WriteJson(connection, 405,
                                    {{"detail", "Method Not Allowed"}});
    } else {
    }
    const std::optional<std::string> raw_body =
        omni_server::ReadBody(connection, kMaxRequestBodyBytes);
    if (!raw_body.has_value()) {
      return 413;
    } else {
    }
    const Json body = Json::parse(*raw_body, nullptr, false);
    if (body.is_discarded()) {
      return omni_server::BadRequest(connection,
                                     "speech request body must be valid JSON");
    } else {
    }
    SpeechRequest request;
    try {
      request = ParseRequest(body, model_);
    } catch (const std::invalid_argument &error) {
      return omni_server::BadRequest(connection, error.what());
    } catch (const Json::exception &) {
      // Note (Jiaxin Deng): a value the checks above let through but whose
      // JSON type cannot be read is still the client's error.
      return omni_server::BadRequest(connection,
                                     "speech request has an invalid field");
    }
    // Note (Minyi Tang): no client can know a generated name, and the CUDA
    // speech API ignores model, so only a chosen name is enforced.
    if (has_chosen_model_name() && request.model.has_value() &&
        *request.model != model_name()) {
      return omni_server::WriteJson(
          connection, 404, {{"detail", "Unknown model: " + *request.model}});
    } else {
    }
    const auto deadline = std::chrono::steady_clock::now() + request_timeout_;
    const std::string request_id = NewRequestId();
    const int sample_rate = model_.speech_decoder().sample_rate();
    auto channel = std::make_shared<AudioChannel>();
    const qwen3_asr::CancelFlag cancel = qwen3_asr::NewCancelFlag();
    bool admitted = false;
    {
      std::lock_guard<std::mutex> lock(admission_mutex_);
      admitted = admitted_requests_ <= max_queued_requests_;
      admitted_requests_ += admitted ? 1 : 0;
    }
    if (!admitted) {
      return omni_server::WriteJson(connection, 503,
                                    {{"detail", "The request queue is full."}});
    } else {
    }
    try {
      Submit(request, channel, cancel, deadline);
    } catch (const std::exception &) {
      // The worker never took the request, so its slot is freed here.
      {
        std::lock_guard<std::mutex> lock(admission_mutex_);
        --admitted_requests_;
      }
      return omni_server::WriteJson(connection, 500,
                                    {{"detail", "Speech generation failed."}});
    }
    if (request.stream) {
      mg_printf(connection,
                "HTTP/1.1 200 OK\r\nContent-Type: audio/pcm\r\n"
                "X-Request-Id: %s\r\nX-SGLang-Omni-Speech-Id: %s\r\n"
                "X-Sample-Rate: %d\r\nX-Channels: 1\r\nX-Bit-Depth: 16\r\n"
                "Cache-Control: no-cache\r\nConnection: close\r\n\r\n",
                request_id.c_str(), request_id.c_str(), sample_rate);
    } else {
    }

    const int client_socket = ClientSocket(connection);
    std::vector<float> samples;
    bool timed_out = false;
    bool client_left = false;
    std::unique_lock<std::mutex> lock(channel->mutex);
    while (true) {
      const bool has_audio_or_end = channel->ready.wait_until(
          lock,
          std::min(deadline,
                   std::chrono::steady_clock::now() + kClientPollInterval),
          [&] { return !channel->chunks.empty() || channel->done; });
      if (has_audio_or_end && channel->chunks.empty()) {
        break;
      } else if (std::chrono::steady_clock::now() >= deadline) {
        // Note (Minyi Tang): a write that blocked past the deadline leaves
        // audio waiting, which wait_until hands back without waiting.
        timed_out = true;
        break;
      } else if (!has_audio_or_end) {
        // Note (Minyi Tang): the client is watched while its request waits in
        // the queue too, and a request cancelled there never starts.
        client_left = ClientLeft(client_socket);
        if (client_left) {
          break;
        } else {
          continue;
        }
      } else {
      }
      std::vector<float> chunk = std::move(channel->chunks.front());
      channel->chunks.pop_front();
      lock.unlock();
      if (request.stream) {
        const std::string pcm =
            omni_common::Pcm16Bytes(ApplySpeed(chunk, request.speed));
        // Note (Minyi Tang): a write fails once the peer is gone, which
        // cancels the generation that feeds it.
        if (mg_write(connection, pcm.data(), pcm.size()) <= 0) {
          cancel->store(true);
        } else {
        }
      } else {
        samples.insert(samples.end(), chunk.begin(), chunk.end());
      }
      lock.lock();
    }
    lock.unlock();
    if (client_left) {
      // Note (Minyi Tang): nothing is read from the channel, which the
      // generation may still be writing.
      cancel->store(true);
      return kClientLeftStatus;
    } else {
    }
    // Note (Minyi Tang): the generation stops at the deadline itself, and may
    // report that before this thread's wait reaches the deadline. Any other
    // error is a failure, even one that arrives after the deadline.
    if (!timed_out && ReachedDeadline(channel->error)) {
      timed_out = true;
    } else {
    }
    if (timed_out) {
      // Note (Minyi Tang): a request still queued is dropped once the worker
      // reaches it.
      cancel->store(true);
      std::cerr << "speech generation timed out\n";
    } else if (channel->error) {
      LogFailure(channel->error);
    } else {
    }
    if (request.stream) {
      return 200;
    } else if (timed_out) {
      return omni_server::WriteJson(
          connection, 504, {{"detail", "Speech generation timed out."}});
    } else if (channel->error) {
      return omni_server::WriteJson(connection, 500,
                                    {{"detail", "Speech generation failed."}});
    } else {
    }

    const std::string pcm =
        omni_common::Pcm16Bytes(ApplySpeed(samples, request.speed));
    const std::string audio = request.response_format == "wav"
                                  ? omni_common::WavBytes(pcm, sample_rate)
                                  : pcm;
    // Note (Jiaxin Deng): CUDA counts a CustomVoice request's prompt as no
    // reference codes.
    const size_t prompt_tokens = 0;
    const int completion_tokens = channel->result.generated_token_count;
    mg_printf(
        connection,
        "HTTP/1.1 200 OK\r\nContent-Type: audio/%s\r\nContent-Length: %zu\r\n"
        "Content-Disposition: attachment; filename=\"speech.%s\"\r\n"
        "X-Request-Id: %s\r\nX-Finish-Reason: %s\r\n"
        "X-Prompt-Tokens: %zu\r\nX-Completion-Tokens: %d\r\n"
        "X-SGLang-Omni-Input-Tokens: %zu\r\n"
        "X-SGLang-Omni-Output-Tokens: %d\r\n"
        "X-SGLang-Omni-Total-Tokens: %zu\r\nConnection: close\r\n\r\n",
        request.response_format.c_str(), audio.size(),
        request.response_format.c_str(), request_id.c_str(),
        qwen3_asr::FinishReasonName(channel->result.finish_reason),
        prompt_tokens, completion_tokens, prompt_tokens, completion_tokens,
        prompt_tokens + completion_tokens);
    mg_write(connection, audio.data(), audio.size());
    return 200;
  }

  // Hands request to the worker, whose completion frees its admission slot.
  void Submit(const SpeechRequest &request,
              const std::shared_ptr<AudioChannel> &channel,
              const qwen3_asr::CancelFlag &cancel,
              std::chrono::steady_clock::time_point deadline) {
    worker_->Submit(
        [this, request, channel, deadline](const std::atomic<bool> &cancelled) {
          return Synthesize(request, *channel, cancelled, deadline);
        },
        cancel,
        [this, channel](std::optional<TranscriptionResult> result,
                        std::exception_ptr error) {
          // Note (Minyi Tang): the slot frees before the response can end, so
          // a client's next request is never refused for its previous one.
          {
            std::lock_guard<std::mutex> lock(admission_mutex_);
            --admitted_requests_;
          }
          std::lock_guard<std::mutex> lock(channel->mutex);
          channel->done = true;
          channel->error = error;
          if (result.has_value()) {
            channel->result = std::move(*result);
          } else {
          }
          channel->ready.notify_all();
        });
  }

  TranscriptionResult
  Synthesize(const SpeechRequest &request, AudioChannel &channel,
             const std::atomic<bool> &cancel,
             std::chrono::steady_clock::time_point deadline) const {
    qwen3_tts::SpeechStream stream(model_.speech_decoder(), request.stream,
                                   request.first_chunk_frames);
    const auto send = [&channel](std::vector<float> samples) {
      if (!samples.empty()) {
        std::lock_guard<std::mutex> lock(channel.mutex);
        channel.chunks.push_back(std::move(samples));
        channel.ready.notify_all();
      } else {
      }
    };
    TranscriptionResult result;
    result.finish_reason =
        model_.Generate(request.prompt, request.sampling, request.max_frames,
                        cancel, [&](const mlx::core::array &codes) {
                          // Note (Minyi Tang): the generation checks its own
                          // deadline, since the request's thread may be blocked
                          // writing to a slow client.
                          if (std::chrono::steady_clock::now() >= deadline) {
                            throw DeadlineReached();
                          } else {
                          }
                          ++result.generated_token_count;
                          send(stream.Push(codes));
                        });
    if (result.generated_token_count == 0) {
      // Note (Minyi Tang): CUDA answers no audio output with an error.
      throw std::runtime_error("Qwen3-TTS generated no audio codes");
    } else if (cancel.load()) {
      throw qwen3_asr::TranscriptionCancelled();
    } else if (std::chrono::steady_clock::now() >= deadline) {
      throw DeadlineReached();
    } else {
    }
    std::vector<float> remaining = stream.Finish();
    if (cancel.load()) {
      throw qwen3_asr::TranscriptionCancelled();
    } else if (std::chrono::steady_clock::now() >= deadline) {
      throw DeadlineReached();
    } else if (!request.stream && remaining.empty()) {
      // Note (Jiaxin Deng): CUDA answers a decode with no audio as a failure.
      throw std::runtime_error("Qwen3-TTS decoded no audio");
    } else {
    }
    send(std::move(remaining));
    return result;
  }

  qwen3_tts::Qwen3TtsModel model_;
  const int max_queued_requests_;
  const std::chrono::steady_clock::duration request_timeout_;
  mutable std::mutex admission_mutex_;
  // Requests the worker holds: running, queued, or finishing.
  int admitted_requests_ = 0;
  qwen3_asr::TranscriptionWorker *worker_ = nullptr;
};

} // namespace

int main(int argc, char **argv) {
  int max_queued_requests = 2;
  double request_timeout_seconds = 600;
  asr_service::ServedKind kind;
  kind.model_kind = "qwen3_tts";
  kind.flags = {
      {"--max-queued-requests",
       [&](const std::string &value) {
         max_queued_requests = std::stoi(value);
       }},
      {"--request-timeout-s",
       [&](const std::string &value) {
         request_timeout_seconds = std::stod(value);
       }},
  };
  kind.check_flags = [&]() {
    if (max_queued_requests < 0) {
      throw std::invalid_argument("--max-queued-requests must not be negative");
    } else if (!(request_timeout_seconds > 0 &&
                 request_timeout_seconds <= kMaxRequestTimeoutSeconds)) {
      throw std::invalid_argument(
          "--request-timeout-s must be above 0 and at most " +
          std::to_string(kMaxRequestTimeoutSeconds));
    } else {
    }
  };
  kind.load = [&](const std::filesystem::path &model_directory) {
    return std::make_unique<Qwen3TtsService>(
        model_directory, max_queued_requests, request_timeout_seconds);
  };
  return asr_service::Serve(argc, argv, kind);
}
