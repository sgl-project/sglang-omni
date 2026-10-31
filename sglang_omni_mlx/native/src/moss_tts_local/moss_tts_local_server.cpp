// SPDX-License-Identifier: Apache-2.0
// Native MOSS-TTS-Local server: POST /v1/audio/speech with the CUDA server's
// request fields, answering a stereo WAV or mono PCM file, or mono 16-bit PCM
// streamed as it is decoded, and Voxt's supervisor protocol.
//
//   moss_tts_local_server --model-path DIR --codec-path DIR
//       [--model-name NAME] [--host H] [--port P] [--quantization-bits B]
//       [--max-queued-requests N] [--request-timeout-s S]
//   moss_tts_local_server --supervised --model-kind moss_tts_local
//       --model-directory DIR --codec-path DIR [--quantization-bits B]
//       [--max-queued-requests N] [--request-timeout-s S]
//
// Request fields: input (an optional ${token:N} prefix sets the duration),
// model (the served name, when given), response_format (wav or pcm), stream
// (pcm only), speed, language, instructions, token_count (or
// duration_tokens), max_new_tokens (audio frames, 12.5 a second),
// temperature, top_p, top_k, repetition_penalty, seed and
// initial_codec_chunk_frames. Reference audio is not supported yet. One
// request runs at a time and N more (2 by default) may wait; the rest get
// 503. A request not done S seconds (600 by default) after it arrives is
// cancelled, with 504 or an early end to its stream, and so is a request whose
// client disconnects. A request body over 1 MiB gets 413.
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
#include <limits>
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
#include "common/http.h"
#include "common/text.h"
#include "moss_tts_local/moss_tts_local_model.h"
#include "qwen3_asr/asr_service.h"
#include "qwen3_asr/qwen3_asr_transcriber.h"

namespace {

using omni_server::Json;
using qwen3_asr::TranscriptionResult;

constexpr size_t kMaxInputCharacters = 4096;
// Far above the largest valid request: kMaxInputCharacters characters, each
// JSON-escaped as a 12-byte surrogate pair, take 48 KiB.
constexpr size_t kMaxRequestBodyBytes = 1024 * 1024;
constexpr int kDefaultAudioFrames = 4096;
// Note (Jiaxin Deng): the CUDA server's streaming limit, 30 minutes at 12.5
// frames a second, here for every request.
constexpr int kMaxAudioFrames = 22500;
// A WAV or PCM file holds every sample at once, so a request for one stops at
// the CUDA default, 4096 frames (about 5.5 minutes).
constexpr int kMaxFileAudioFrames = 4096;
constexpr int kMaxChunkFrames = 25;
constexpr double kMinSpeed = 0.25;
constexpr double kMaxSpeed = 4.0;
constexpr int kDefaultQuantizationBits = 8;
// A first and a second codec chunk, so every decode shape is built once.
constexpr int kWarmUpFrames = 30;
constexpr int kRequestIdHexDigits = 32;
constexpr int kMaxRequestTimeoutSeconds = 24 * 60 * 60;
// As often as the CUDA server checks whether a speech client disconnected.
constexpr auto kClientPollInterval = std::chrono::milliseconds(50);
// The status civetweb logs for a request whose client left before its answer.
constexpr int kClientLeftStatus = 499;
// The most one disconnect check reads of what a client sent after its request,
// so a client that keeps sending cannot hold the check.
constexpr size_t kMaxUnreadBytesPerCheck = 64 * 1024;
// The languages the CUDA server accepts for this model, in their canonical
// spelling; Auto means none.
constexpr std::array<const char *, 32> kLanguages = {
    "Arabic",          "Auto",      "Cantonese",  "Chinese",    "Czech",
    "Danish",          "Dutch",     "English",    "Finnish",    "French",
    "German",          "Greek",     "Hebrew",     "Hindi",      "Hungarian",
    "Italian",         "Japanese",  "Korean",     "Macedonian", "Malay",
    "Persian (Farsi)", "Polish",    "Portuguese", "Romanian",   "Russian",
    "Spanish",         "Swahili",   "Swedish",    "Tagalog",    "Thai",
    "Turkish",         "Vietnamese"};
constexpr std::array<const char *, 3> kTaskTypes = {"Base", "CustomVoice",
                                                    "VoiceDesign"};
constexpr std::array<const char *, 10> kStringFields = {
    "model",     "input",    "voice",        "speaker",   "response_format",
    "task_type", "language", "instructions", "ref_audio", "ref_text"};
constexpr std::array<const char *, 6> kIntegerFields = {
    "max_new_tokens", "initial_codec_chunk_frames",
    "token_count",    "duration_tokens",
    "seed",           "top_k"};
// Flags of CUDA's request this model passes over, still type-checked.
constexpr std::array<const char *, 3> kBooleanFields = {
    "x_vector_only_mode", "stream_codec_output", "suppress_bootstrap_silence"};
constexpr std::array<const char *, 4> kNumberFields = {
    "speed", "temperature", "top_p", "repetition_penalty"};

struct SpeechRequest {
  moss_tts_local::SpeechPrompt prompt;
  moss_tts_local::GenerationParameters generation;
  std::vector<std::vector<int>> prompt_rows;
  std::optional<std::string> model;
  std::string response_format;
  bool stream = false;
  double speed = 1.0;
  int first_chunk_frames = 5;
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

std::string AsciiLower(const std::string &text) {
  std::string lowered;
  for (const char character : text) {
    lowered.push_back(
        static_cast<char>(std::tolower(static_cast<unsigned char>(character))));
  }
  return lowered;
}

std::string Joined(const std::vector<std::string> &values) {
  std::string joined;
  for (const std::string &value : values) {
    joined += (joined.empty() ? "" : ", ") + value;
  }
  return joined;
}

// The CUDA server's raw type checks, which come before every other rule.
void CheckFieldTypes(const Json &body) {
  for (const char *field : kStringFields) {
    if (!FieldOr(body, field, Json("")).is_string()) {
      throw std::invalid_argument(std::string(field) + " must be a string");
    } else {
    }
  }
  for (const char *field : kIntegerFields) {
    if (!FieldOr(body, field, Json(0)).is_number_integer()) {
      throw std::invalid_argument(std::string(field) + " must be an integer");
    } else {
    }
  }
  for (const char *field : kNumberFields) {
    if (!FieldOr(body, field, Json(0)).is_number()) {
      throw std::invalid_argument(std::string(field) + " must be a number");
    } else {
    }
  }
  if (!PresentFieldOr(body, "stream", Json(false)).is_boolean()) {
    throw std::invalid_argument("stream must be a boolean");
  } else {
  }
  for (const char *field : kBooleanFields) {
    if (!FieldOr(body, field, Json(false)).is_boolean()) {
      throw std::invalid_argument(std::string(field) + " must be a boolean");
    } else {
    }
  }
}

// Whether an integer field holds a negative value. JSON parsing keeps every
// non-negative integer unsigned, up to 2^64 - 1.
bool IsNegative(const Json &value) {
  return !value.is_number_unsigned() && value.get<int64_t>() < 0;
}

// An integer field's value at most maximum, or maximum when it is larger.
int64_t ClampedInteger(const Json &value, int64_t maximum) {
  return value.is_number_unsigned()
             ? static_cast<int64_t>(
                   std::min<uint64_t>(value.get<uint64_t>(), maximum))
             : std::min<int64_t>(value.get<int64_t>(), maximum);
}

// An integer field that must be above 0 when given.
std::optional<int> PositiveInteger(const Json &body, const char *name,
                                   int maximum) {
  const Json value = FieldOr(body, name, Json());
  if (value.is_null()) {
    return std::nullopt;
  } else if (IsNegative(value) || value.get<uint64_t>() == 0) {
    throw std::invalid_argument(std::string(name) + " must be greater than 0");
  } else if (value.get<uint64_t>() > static_cast<uint64_t>(maximum)) {
    throw std::invalid_argument(std::string(name) + " must be at most " +
                                std::to_string(maximum));
  } else {
    return value.get<int>();
  }
}

// Note (Jiaxin Deng): as the CUDA builder, a ${token:N} prefix of the text is
// the duration in tokens when no token_count field gives one.
std::pair<std::string, std::optional<int>>
SplitTokenPrefix(const std::string &text) {
  const std::string start = "${token:";
  if (text.rfind(start, 0) != 0) {
    return {text, std::nullopt};
  } else {
  }
  const size_t digits_end = text.find_first_not_of("0123456789", start.size());
  if (digits_end == start.size() || digits_end == std::string::npos ||
      text[digits_end] != '}') {
    throw std::invalid_argument(
        "the ${token:N} prefix must hold a positive integer");
  } else {
  }
  const size_t significant_start =
      std::min(text.find_first_not_of('0', start.size()), digits_end);
  // Note (Jiaxin Deng): ten significant digits already exceed the cap.
  const std::string digits = text.substr(
      significant_start, std::min<size_t>(digits_end - significant_start, 10));
  const int64_t count = digits.empty() ? 0 : std::stoll(digits);
  if (count <= 0) {
    throw std::invalid_argument("the ${token:N} count must be greater than 0");
  } else if (count > kMaxAudioFrames) {
    throw std::invalid_argument("the ${token:N} count must be at most " +
                                std::to_string(kMaxAudioFrames));
  } else {
  }
  // As Python's str.lstrip, Unicode whitespace after the prefix goes too.
  const std::string rest = text.substr(digits_end + 1);
  const std::string stripped = omni_common::StripUnicodeWhitespace(rest);
  return {stripped.empty() ? "" : rest.substr(rest.find(stripped)),
          static_cast<int>(count)};
}

moss_tts_local::SamplingParameters
OverrideSampling(moss_tts_local::SamplingParameters defaults,
                 const Json &body) {
  const Json temperature = FieldOr(body, "temperature", Json());
  const Json top_p = FieldOr(body, "top_p", Json());
  const Json top_k = FieldOr(body, "top_k", Json());
  if (!temperature.is_null()) {
    defaults.temperature = temperature.get<double>();
  } else {
  }
  if (!top_p.is_null()) {
    defaults.top_p = top_p.get<double>();
  } else {
  }
  if (!top_k.is_null()) {
    // A top_k past the vocabulary keeps every candidate, as one at it does.
    defaults.top_k = static_cast<int>(std::max<int64_t>(
        ClampedInteger(top_k, std::numeric_limits<int>::max()),
        std::numeric_limits<int>::min()));
  } else {
  }
  return defaults;
}

// Throws when text, a request field, is over kMaxInputCharacters characters.
void CheckLength(const char *field, const std::string &text) {
  const auto character_count =
      std::count_if(text.begin(), text.end(), [](char byte) {
        return (static_cast<unsigned char>(byte) & 0xc0) != 0x80;
      });
  if (static_cast<size_t>(character_count) > kMaxInputCharacters) {
    throw std::invalid_argument(std::string(field) + " must be at most " +
                                std::to_string(kMaxInputCharacters) +
                                " characters");
  } else {
  }
}

SpeechRequest ParseRequest(const Json &body,
                           const moss_tts_local::MossTtsLocalModel &model) {
  if (!body.is_object()) {
    throw std::invalid_argument("speech request body must be a JSON object");
  } else {
  }
  CheckFieldTypes(body);
  const Json input = FieldOr(body, "input", Json(""));
  if (omni_common::StripUnicodeWhitespace(input.get<std::string>()).empty()) {
    throw std::invalid_argument("input must be a non-empty string");
  } else {
  }
  const std::string text = input.get<std::string>();
  CheckLength("input", text);

  SpeechRequest request;
  const Json model_name = FieldOr(body, "model", Json());
  if (model_name.is_string()) {
    request.model = model_name.get<std::string>();
  } else {
  }
  // Note (Minyi Tang): as CUDA's AliasChoices("voice", "speaker"), voice wins
  // whenever it is given, even as null.
  // Note (Jiaxin Deng): MOSS has no preset voices, and CUDA passes over a name
  // it has no uploaded voice for; this server keeps none.
  if (!PresentFieldOr(body, "voice",
                      PresentFieldOr(body, "speaker", Json("default")))
           .is_string()) {
    throw std::invalid_argument("voice must be a string");
  } else {
  }
  const Json response_format =
      PresentFieldOr(body, "response_format", Json("wav"));
  if (!response_format.is_string()) {
    throw std::invalid_argument("response_format must be a string");
  } else {
  }
  // Note (Minyi Tang): as CUDA's response_format.strip().lower().
  request.response_format = AsciiLower(
      omni_common::StripUnicodeWhitespace(response_format.get<std::string>()));
  if (request.response_format != "pcm" && request.response_format != "wav") {
    throw std::invalid_argument("response_format must be one of: pcm, wav");
  } else {
  }
  request.stream = PresentFieldOr(body, "stream", Json(false)).get<bool>();
  if (PresentFieldOr(body, "stream_format", Json("audio")) != "audio") {
    throw std::invalid_argument("stream_format must be audio");
  } else if (request.stream && request.response_format != "pcm") {
    throw std::invalid_argument("stream=true requires response_format='pcm'");
  } else {
  }
  const Json speed = PresentFieldOr(body, "speed", Json(1.0));
  if (!speed.is_number()) {
    throw std::invalid_argument("speed must be a number");
  } else if (!(speed.get<double>() >= kMinSpeed &&
               speed.get<double>() <= kMaxSpeed)) {
    throw std::invalid_argument("speed must be between 0.25 and 4.0");
  } else {
  }
  request.speed = speed.get<double>();

  const Json task_type = FieldOr(body, "task_type", Json());
  if (!task_type.is_null()) {
    std::string normalized;
    for (const char character :
         omni_common::StripUnicodeWhitespace(task_type.get<std::string>())) {
      if (character != '_' && character != '-') {
        normalized.push_back(character);
      } else {
      }
    }
    const bool known = std::any_of(
        kTaskTypes.begin(), kTaskTypes.end(), [&](const char *name) {
          return AsciiLower(name) == AsciiLower(normalized);
        });
    if (!known) {
      throw std::invalid_argument(
          "task_type must be one of: Base, CustomVoice, VoiceDesign");
    } else {
    }
  } else {
  }
  const Json language = FieldOr(body, "language", Json());
  if (!language.is_null()) {
    const std::string wanted = AsciiLower(
        omni_common::StripUnicodeWhitespace(language.get<std::string>()));
    const auto found = std::find_if(
        kLanguages.begin(), kLanguages.end(),
        [&](const char *name) { return AsciiLower(name) == wanted; });
    if (found == kLanguages.end()) {
      std::vector<std::string> names(kLanguages.begin(), kLanguages.end());
      throw std::invalid_argument("language must be one of: " + Joined(names));
    } else if (std::string(*found) != "Auto") {
      request.prompt.language = *found;
    } else {
    }
  } else {
  }
  if (!FieldOr(body, "ref_audio", Json("")).get<std::string>().empty() ||
      !FieldOr(body, "references", Json::array()).empty()) {
    throw std::invalid_argument(
        "reference audio (ref_audio, references) is not supported by this "
        "server yet");
  } else {
  }
  const Json instructions = FieldOr(body, "instructions", Json());
  if (!instructions.is_null()) {
    // Note (Jiaxin Deng): capped as input is, so a long instruction cannot
    // fill memory with prompt rows.
    CheckLength("instructions", instructions.get<std::string>());
    request.prompt.instruction = instructions.get<std::string>();
  } else {
  }

  const std::optional<int> max_new_tokens =
      PositiveInteger(body, "max_new_tokens",
                      request.stream ? kMaxAudioFrames : kMaxFileAudioFrames);
  const std::optional<int> token_count =
      PositiveInteger(body, "token_count", kMaxAudioFrames);
  const std::optional<int> duration_tokens =
      PositiveInteger(body, "duration_tokens", kMaxAudioFrames);
  for (const char *field : {"seed", "initial_codec_chunk_frames"}) {
    if (IsNegative(FieldOr(body, field, Json(0)))) {
      throw std::invalid_argument(std::string(field) +
                                  " must be greater than or equal to 0");
    } else {
    }
  }
  const auto [prompt_text, prefix_token_count] = SplitTokenPrefix(text);
  request.prompt.text = token_count.has_value() || duration_tokens.has_value()
                            ? text
                            : prompt_text;
  request.prompt.token_count =
      token_count.has_value()
          ? token_count
          : (duration_tokens.has_value() ? duration_tokens
                                         : prefix_token_count);

  // Note (Jiaxin Deng): the CUDA server's defaults, the checkpoint's own; the
  // public temperature, top_p and top_k set both heads.
  request.generation.text = OverrideSampling({1.0, 1.0, 50, 1.0}, body);
  request.generation.audio = OverrideSampling({1.7, 0.8, 25, 1.0}, body);
  request.generation.audio.repetition_penalty =
      FieldOr(body, "repetition_penalty", Json(1.0)).get<double>();
  if (request.generation.text.temperature < 0) {
    throw std::invalid_argument("temperature must be >= 0");
  } else if (!(request.generation.text.top_p > 0 &&
               request.generation.text.top_p <= 1)) {
    throw std::invalid_argument("top_p must be in (0, 1]");
  } else if (request.generation.text.top_k < -1) {
    throw std::invalid_argument("top_k must be >= -1");
  } else if (!(request.generation.audio.repetition_penalty > 0)) {
    throw std::invalid_argument("repetition_penalty must be > 0");
  } else {
  }
  request.generation.max_frames = max_new_tokens.value_or(kDefaultAudioFrames);
  const Json seed = FieldOr(body, "seed", Json());
  if (seed.is_null()) {
    static thread_local std::mt19937_64 generator{std::random_device{}()};
    request.generation.seed = generator();
  } else {
    request.generation.seed = seed.get<uint64_t>();
  }
  const int initial_chunk_frames = static_cast<int>(ClampedInteger(
      FieldOr(body, "initial_codec_chunk_frames", Json(5)), kMaxChunkFrames));
  request.first_chunk_frames =
      initial_chunk_frames > 0 ? initial_chunk_frames : kMaxChunkFrames;
  request.prompt_rows = model.prompt_encoder().Rows(request.prompt);
  return request;
}

// Interleaved samples to mono: each channel group's mean.
std::vector<float> Downmix(const std::vector<float> &samples,
                           int channel_count) {
  std::vector<float> mono(samples.size() / channel_count);
  for (size_t index = 0; index < mono.size(); ++index) {
    float sum = 0.0f;
    for (int channel = 0; channel < channel_count; ++channel) {
      sum += samples[index * channel_count + channel];
    }
    mono[index] = sum / static_cast<float>(channel_count);
  }
  return mono;
}

// Note (Jiaxin Deng): the CUDA server's speed change, linear interpolation of
// each channel onto round(length / speed) evenly spaced points.
std::vector<float> ApplySpeed(const std::vector<float> &samples,
                              int channel_count, double speed) {
  const size_t length = samples.size() / channel_count;
  if (speed == 1.0 || length == 0) {
    return samples;
  } else {
  }
  const size_t new_length = std::max<size_t>(
      static_cast<size_t>(std::llround(static_cast<double>(length) / speed)),
      1);
  std::vector<float> adjusted(new_length * channel_count);
  const double last = static_cast<double>(length - 1);
  for (size_t index = 0; index < new_length; ++index) {
    const double position = new_length == 1
                                ? 0.0
                                : last * static_cast<double>(index) /
                                      static_cast<double>(new_length - 1);
    const size_t left = std::min(static_cast<size_t>(position), length - 1);
    const size_t right = std::min(left + 1, length - 1);
    const double fraction = position - static_cast<double>(left);
    for (int channel = 0; channel < channel_count; ++channel) {
      const double left_value = samples[left * channel_count + channel];
      const double right_value = samples[right * channel_count + channel];
      adjusted[index * channel_count + channel] = static_cast<float>(
          left_value + (right_value - left_value) * fraction);
    }
  }
  return adjusted;
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

class MossTtsLocalService : public asr_service::ServedModel {
public:
  MossTtsLocalService(const std::filesystem::path &model_directory,
                      const std::filesystem::path &codec_directory,
                      int quantization_bits, int max_queued_requests,
                      double request_timeout_seconds)
      : model_(model_directory, codec_directory, quantization_bits),
        max_queued_requests_(max_queued_requests),
        request_timeout_(
            std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                std::chrono::duration<double>(request_timeout_seconds))) {
    WarmUp();
  }

  asr_service::Transcription
  Prepare(std::vector<float>, const asr_service::FormFields &) const override {
    throw std::logic_error("moss_tts_local serves no transcriptions");
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
    return static_cast<MossTtsLocalService *>(data)->Speak(connection);
  }

  // Samples of one chunk as the response sends them: mono for PCM, and at the
  // requested speed.
  std::vector<float> Shape(const std::vector<float> &samples,
                           const SpeechRequest &request) const {
    const int channel_count = model_.codec().channel_count();
    if (request.response_format == "pcm") {
      return ApplySpeed(Downmix(samples, channel_count), 1, request.speed);
    } else {
      return ApplySpeed(samples, channel_count, request.speed);
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
    }
    // Note (Minyi Tang): no client can know a generated name, and the CUDA
    // speech API ignores model, so only a chosen name is enforced.
    if (has_chosen_model_name() && request.model.has_value() &&
        *request.model != model_name()) {
      return omni_server::WriteJson(
          connection, 404, {{"detail", "Unknown model: " + *request.model}});
    } else {
    }
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

    const auto deadline = std::chrono::steady_clock::now() + request_timeout_;
    const std::string request_id = NewRequestId();
    const int sample_rate = model_.codec().sample_rate();
    auto channel = std::make_shared<AudioChannel>();
    const qwen3_asr::CancelFlag cancel = qwen3_asr::NewCancelFlag();
    worker_->Submit(
        [this, request, channel, deadline](const std::atomic<bool> &cancelled) {
          return Synthesize(request, *channel, cancelled, deadline);
        },
        cancel,
        [this, channel](std::optional<TranscriptionResult> result,
                        std::exception_ptr error) {
          // Note (Minyi Tang): the slot frees before the response can end,
          // so a client's next request is never refused for its previous one.
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

    // Note (Jiaxin Deng): as the CUDA server, a stream's 200 waits for its
    // first audio, so a request that fails before any still gets 500 or 504.
    bool stream_started = false;
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
        // Note (Minyi Tang): the client is watched while its request waits
        // in the queue too, and a request cancelled there never starts.
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
        if (!stream_started) {
          mg_printf(connection,
                    "HTTP/1.1 200 OK\r\nContent-Type: audio/pcm\r\n"
                    "X-Request-Id: %s\r\nX-SGLang-Omni-Speech-Id: %s\r\n"
                    "X-Sample-Rate: %d\r\nX-Channels: 1\r\nX-Bit-Depth: 16\r\n"
                    "Cache-Control: no-cache\r\nConnection: close\r\n\r\n",
                    request_id.c_str(), request_id.c_str(), sample_rate);
          stream_started = true;
        } else {
        }
        const std::string pcm =
            moss_tts_local::Pcm16Bytes(Shape(chunk, request));
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
    if (stream_started) {
      return 200;
    } else if (timed_out) {
      return omni_server::WriteJson(
          connection, 504, {{"detail", "Speech generation timed out."}});
    } else if (channel->error) {
      return omni_server::WriteJson(connection, 500,
                                    {{"detail", "Speech generation failed."}});
    } else {
    }

    const std::string pcm = moss_tts_local::Pcm16Bytes(Shape(samples, request));
    const std::string audio =
        request.response_format == "wav"
            ? moss_tts_local::WavBytes(pcm, sample_rate,
                                       model_.codec().channel_count())
            : pcm;
    const size_t prompt_tokens = request.prompt_rows.size();
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

  // Note (Jiaxin Deng): one short generation before the server reports ready,
  // as the CUDA vocoder warms up in its factory; the first request would
  // otherwise wait seconds for MLX to build its kernels.
  void WarmUp() const {
    moss_tts_local::SpeechPrompt prompt;
    prompt.text = "This sentence warms the server up before its first request.";
    moss_tts_local::GenerationParameters parameters;
    parameters.max_frames = kWarmUpFrames;
    parameters.text = {1.0, 1.0, 50, 1.0};
    parameters.audio = {1.7, 0.8, 25, 1.0};
    moss_tts_local::CodecStream stream(model_.codec());
    const std::atomic<bool> cancel(false);
    model_.Generate(model_.prompt_encoder().Rows(prompt), parameters, cancel,
                    [&](const mlx::core::array &codes) { stream.Push(codes); });
    stream.Finish();
    mlx::core::clear_cache();
  }

  TranscriptionResult
  Synthesize(const SpeechRequest &request, AudioChannel &channel,
             const std::atomic<bool> &cancel,
             std::chrono::steady_clock::time_point deadline) const {
    moss_tts_local::CodecStream stream(model_.codec(),
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
        model_.Generate(request.prompt_rows, request.generation, cancel,
                        [&](const mlx::core::array &codes) {
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
      // Note (Jiaxin Deng): the CUDA server fails a request whose first
      // decision is the end of audio rather than answer silence.
      throw std::runtime_error("MOSS-TTS-Local generated no audio frames");
    } else {
    }
    send(stream.Finish());
    return result;
  }

  moss_tts_local::MossTtsLocalModel model_;
  const int max_queued_requests_;
  const std::chrono::steady_clock::duration request_timeout_;
  mutable std::mutex admission_mutex_;
  // Requests the worker holds: running, queued, or finishing.
  int admitted_requests_ = 0;
  qwen3_asr::TranscriptionWorker *worker_ = nullptr;
};

} // namespace

int main(int argc, char **argv) {
  std::filesystem::path codec_directory;
  int quantization_bits = kDefaultQuantizationBits;
  int max_queued_requests = 2;
  double request_timeout_seconds = 600;
  asr_service::ServedKind kind;
  kind.model_kind = "moss_tts_local";
  kind.flags = {
      {"--codec-path",
       [&](const std::string &value) { codec_directory = value; }},
      {"--quantization-bits",
       [&](const std::string &value) { quantization_bits = std::stoi(value); }},
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
    if (codec_directory.empty()) {
      throw std::invalid_argument("--codec-path is required");
    } else if (quantization_bits != 4 && quantization_bits != 8 &&
               quantization_bits != 16) {
      throw std::invalid_argument("--quantization-bits must be 4, 8 or 16");
    } else if (max_queued_requests < 0) {
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
    return std::make_unique<MossTtsLocalService>(
        model_directory, codec_directory, quantization_bits,
        max_queued_requests, request_timeout_seconds);
  };
  return asr_service::Serve(argc, argv, kind);
}
