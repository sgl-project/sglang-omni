// SPDX-License-Identifier: Apache-2.0
// Native Chatterbox-Turbo server: POST /v1/audio/speech with the CUDA server's
// request fields, answering a WAV or PCM file in the checkpoint's built-in
// voice or in a voice cloned from reference audio, and Voxt's supervisor
// protocol.
//
//   chatterbox_server --model-path DIR [--model-name NAME] [--host H]
//       [--port P] [--max-queued-requests N] [--request-timeout-s S]
//       [--allowed-local-media-path DIR]
//   chatterbox_server --supervised --model-kind chatterbox
//       --model-directory DIR [--max-queued-requests N]
//       [--request-timeout-s S] [--allowed-local-media-path DIR]
//
// Request fields: input, model (the served name, when given), voice (or
// speaker; only "default", the built-in voice), references (one item whose
// audio_path, ref_audio, audio or data holds a WAV file longer than 5 s; its
// text is not needed) or ref_audio and ref_text, response_format (wav or pcm),
// max_new_tokens (speech tokens, 25 a second), temperature, top_k, top_p,
// repetition_penalty, seed and speed (1 only). Reference audio is a data URL
// or base64 WAV, or a path or file URL inside --allowed-local-media-path. One
// request runs at a time and N more (2 by default) may wait; the rest get 503.
// A request not done S seconds (600 by default) after it arrives is
// cancelled with 504, and so is a request whose client disconnects. A
// request body over 16 MiB gets 413.
#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <climits>
#include <condition_variable>
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
#include <string_view>
#include <system_error>
#include <typeinfo>
#include <vector>

#include "chatterbox/chatterbox_model.h"
#include "civetweb.h"
#include "common/http.h"
#include "common/text.h"
#include "qwen3_asr/asr_service.h"
#include "qwen3_asr/qwen3_asr_transcriber.h"
#include "voxtral_tts/voxtral_tts_codec.h"

namespace {

using omni_server::Json;
using qwen3_asr::TranscriptionResult;

constexpr size_t kMaxInputCharacters = 4096;
// Room for a base64 reference recording of a few minutes.
constexpr size_t kMaxRequestBodyBytes = 16 * 1024 * 1024;
constexpr size_t kMaxReferenceFileBytes = 12 * 1024 * 1024;
constexpr int kDefaultMaxSpeechTokens = 1000;
// Note (Jiaxin Deng): the flow's attention over prompt and speech mel frames
// grows with their square; 1024 tokens fit in a few GB, 4096 need over 25.
constexpr int kMaxSpeechTokens = 1024;
constexpr double kMaxRepetitionPenalty = 2.0;
constexpr int kRequestIdHexDigits = 32;
constexpr int kMaxRequestTimeoutSeconds = 24 * 60 * 60;
// As often as the CUDA server checks whether a speech client disconnected.
constexpr auto kClientPollInterval = std::chrono::milliseconds(50);
// The status civetweb logs for a request whose client left before its answer.
constexpr int kClientLeftStatus = 499;
// The most one disconnect check reads of what a client sent after its request,
// so a client that keeps sending cannot hold the check.
constexpr size_t kMaxUnreadBytesPerCheck = 64 * 1024;
// Fields of CUDA's CreateSpeechRequest this model has no use for, each counted
// as given when not null or "", and the collections when not empty.
constexpr std::array<const char *, 9> kUnsupportedFields = {
    "duration_tokens",     "initial_codec_chunk_frames",
    "instructions",        "language",
    "stream_codec_output", "suppress_bootstrap_silence",
    "task_type",           "token_count",
    "x_vector_only_mode"};

struct SpeechRequest {
  std::vector<int> text_ids;
  std::optional<std::string> model;
  // The reference recording's WAV bytes, its header checked; none for the
  // built-in voice.
  std::optional<std::string> reference;
  std::string response_format;
  chatterbox::SamplingOptions sampling;
  int max_speech_tokens = kDefaultMaxSpeechTokens;
  uint64_t seed = 0;
};

struct SynthesisChannel {
  std::mutex mutex;
  std::condition_variable ready;
  bool done = false;
  std::exception_ptr error;
  TranscriptionResult result;
  std::vector<float> samples;
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
Json PresentFieldOr(const Json &body, const char *name, Json fallback) {
  const auto found = body.find(name);
  if (found == body.end()) {
    return fallback;
  } else {
    return *found;
  }
}

void RejectUnsupportedFields(const Json &body) {
  std::vector<std::string> given;
  for (const char *field : kUnsupportedFields) {
    const Json value = FieldOr(body, field, Json());
    if (!value.is_null() && value != "") {
      given.push_back(field);
    } else {
    }
  }
  const Json stage_params = FieldOr(body, "stage_params", Json());
  if (!stage_params.is_null() && !stage_params.is_object()) {
    throw std::invalid_argument("stage_params must be an object");
  } else if (!stage_params.is_null() && !stage_params.empty()) {
    given.push_back("stage_params");
  } else {
  }
  if (given.empty()) {
    return;
  } else {
  }
  std::sort(given.begin(), given.end());
  std::string fields;
  for (const std::string &field : given) {
    fields += (fields.empty() ? "" : ", ") + field;
  }
  throw std::invalid_argument(
      "Chatterbox-Turbo does not support these /v1/audio/speech fields: " +
      fields);
}

// A finite number field in (low, high] (or [low, high] when low_inclusive),
// or fallback when missing or null.
float NumberField(const Json &body, const char *name, float fallback,
                  double low, bool low_inclusive, double high) {
  const Json value = FieldOr(body, name, Json(fallback));
  if (!value.is_number()) {
    throw std::invalid_argument(std::string(name) + " must be a number");
  } else {
  }
  const double number = value.get<double>();
  if (!std::isfinite(number) || number > high ||
      (low_inclusive ? number < low : number <= low)) {
    throw std::invalid_argument(
        std::string(name) + " must be in " + (low_inclusive ? "[" : "(") +
        std::to_string(low) + ", " + std::to_string(high) + "]");
  } else {
  }
  return static_cast<float>(number);
}

std::string Lowercase(std::string text) {
  for (char &character : text) {
    character =
        static_cast<char>(std::tolower(static_cast<unsigned char>(character)));
  }
  return text;
}

bool IsInside(const std::filesystem::path &path,
              const std::filesystem::path &directory) {
  const auto [directory_end, path_at] = std::mismatch(
      directory.begin(), directory.end(), path.begin(), path.end());
  return directory_end == directory.end();
}

// The bytes of a regular file inside allowed_directory.
// Note (Jiaxin Deng): the opened file's own path is checked, so a symlink
// swapped in after the name was resolved cannot lead outside the directory.
std::string ReadAllowedFile(const std::string &path,
                            const std::filesystem::path &allowed_directory) {
  const int file_descriptor = open(path.c_str(), O_RDONLY | O_NONBLOCK);
  if (file_descriptor < 0) {
    throw std::invalid_argument("reference audio file not found");
  } else {
  }
  std::string bytes;
  char opened_path[PATH_MAX] = "";
  struct stat status {};
  const bool is_allowed =
      fcntl(file_descriptor, F_GETPATH, opened_path) == 0 &&
      IsInside(std::filesystem::path(opened_path), allowed_directory);
  const bool is_regular =
      fstat(file_descriptor, &status) == 0 && S_ISREG(status.st_mode);
  const auto size = static_cast<size_t>(is_regular ? status.st_size : 0);
  bool is_read = is_allowed && is_regular && size <= kMaxReferenceFileBytes;
  if (is_read) {
    bytes.resize(size);
    size_t done = 0;
    while (is_read && done < size) {
      const ssize_t count =
          read(file_descriptor, bytes.data() + done, size - done);
      is_read = count > 0;
      done += is_read ? static_cast<size_t>(count) : 0;
    }
  } else {
  }
  close(file_descriptor);
  if (!is_allowed) {
    throw std::invalid_argument(
        "reference audio path is outside --allowed-local-media-path");
  } else if (!is_regular) {
    throw std::invalid_argument("reference audio file not found");
  } else if (size > kMaxReferenceFileBytes) {
    throw std::invalid_argument("reference audio file is larger than " +
                                std::to_string(kMaxReferenceFileBytes) +
                                " bytes");
  } else if (!is_read) {
    throw std::invalid_argument("reference audio file cannot be read");
  } else {
  }
  return bytes;
}

std::string Base64Bytes(const std::string &text) {
  const std::optional<std::vector<uint8_t>> bytes =
      omni_common::DecodeBase64(text);
  if (!bytes.has_value() || bytes->empty()) {
    throw std::invalid_argument("reference audio is not valid base64");
  } else {
  }
  return std::string(bytes->begin(), bytes->end());
}

// The WAV bytes a reference names: a base64 data URL, or a path or file URL
// inside allowed_directory.
std::string
ReferenceBytes(const std::string &source,
               const std::optional<std::filesystem::path> &allowed_directory) {
  if (source.rfind("data:", 0) == 0) {
    const size_t comma = source.find(',');
    if (comma == std::string::npos ||
        !std::string_view(source).substr(0, comma).ends_with(";base64")) {
      throw std::invalid_argument("a reference audio data URL must be base64 "
                                  "(data:audio/wav;base64,...)");
    } else {
    }
    return Base64Bytes(source.substr(comma + 1));
  } else if (source.rfind("http://", 0) == 0 ||
             source.rfind("https://", 0) == 0) {
    throw std::invalid_argument(
        "reference audio URLs are not supported; send a data URL");
  } else if (!allowed_directory.has_value()) {
    throw std::invalid_argument(
        "local reference audio paths need --allowed-local-media-path; send a "
        "data URL");
  } else {
  }
  const std::string path =
      source.rfind("file://", 0) == 0 ? source.substr(7) : source;
  std::error_code error;
  const std::filesystem::path resolved =
      std::filesystem::weakly_canonical(path, error);
  if (error || !IsInside(resolved, *allowed_directory)) {
    throw std::invalid_argument(
        "reference audio path is outside --allowed-local-media-path");
  } else {
  }
  return ReadAllowedFile(resolved.string(), *allowed_directory);
}

// The reference recording a request gives, if any.
std::optional<std::string> RequestReference(
    const Json &body,
    const std::optional<std::filesystem::path> &allowed_directory) {
  const Json references = FieldOr(body, "references", Json());
  const Json ref_audio = FieldOr(body, "ref_audio", Json());
  if (!references.is_null() && !references.is_array()) {
    throw std::invalid_argument("references must be a list");
  } else if (!ref_audio.is_null() && !ref_audio.is_string()) {
    throw std::invalid_argument("ref_audio must be a string");
  } else if (references.size() > 1) {
    throw std::invalid_argument(
        "Chatterbox-Turbo clones one voice; give at most one reference");
  } else if (!references.empty() && !ref_audio.is_null()) {
    throw std::invalid_argument("give references or ref_audio, not both");
  } else {
  }
  const Json ref_text = FieldOr(body, "ref_text", Json());
  if (!ref_text.is_null() && !ref_text.is_string()) {
    throw std::invalid_argument("ref_text must be a string");
  } else {
  }
  if (ref_audio.is_string()) {
    return ReferenceBytes(ref_audio.get<std::string>(), allowed_directory);
  } else if (references.empty()) {
    if (!ref_text.is_null() && ref_text != "") {
      throw std::invalid_argument("ref_text needs ref_audio");
    } else {
    }
    return std::nullopt;
  } else {
  }
  const Json &reference = references.front();
  if (!reference.is_object()) {
    throw std::invalid_argument("each reference must be an object");
  } else if (!FieldOr(reference, "vq_codes", Json()).is_null()) {
    throw std::invalid_argument(
        "Chatterbox-Turbo does not take vq_codes; give reference audio");
  } else {
  }
  const Json text = FieldOr(reference, "text", Json());
  if (!text.is_null() && !text.is_string()) {
    throw std::invalid_argument("references[0].text must be a string");
  } else {
  }
  for (const char *field : {"audio_path", "ref_audio", "audio"}) {
    const Json source = FieldOr(reference, field, Json());
    if (source.is_string()) {
      return ReferenceBytes(source.get<std::string>(), allowed_directory);
    } else if (!source.is_null()) {
      throw std::invalid_argument(std::string("references[0].") + field +
                                  " must be a string");
    } else {
    }
  }
  const Json data = FieldOr(reference, "data", Json());
  const Json media_type = FieldOr(reference, "media_type", Json("audio/wav"));
  if (!data.is_string()) {
    throw std::invalid_argument("references[0] has no audio");
  } else if (!media_type.is_string() ||
             (media_type != "audio/wav" && media_type != "audio/x-wav" &&
              media_type != "audio/wave")) {
    throw std::invalid_argument("reference audio must be a WAV file");
  } else {
  }
  return Base64Bytes(data.get<std::string>());
}

// Every field but the reference audio, which is read once the request is
// admitted.
SpeechRequest ParseRequest(const Json &body,
                           const chatterbox::ChatterboxModel &model) {
  if (!body.is_object()) {
    throw std::invalid_argument("speech request body must be a JSON object");
  } else {
  }
  RejectUnsupportedFields(body);
  const Json input = FieldOr(body, "input", Json());
  if (!input.is_string() ||
      omni_common::StripUnicodeWhitespace(input.get<std::string>()).empty()) {
    throw std::invalid_argument("input must be a non-empty string");
  } else {
  }
  const std::string text = input.get<std::string>();
  const auto character_count =
      std::count_if(text.begin(), text.end(), [](char byte) {
        return (static_cast<unsigned char>(byte) & 0xc0) != 0x80;
      });
  if (static_cast<size_t>(character_count) > kMaxInputCharacters) {
    throw std::invalid_argument("input must be at most " +
                                std::to_string(kMaxInputCharacters) +
                                " characters");
  } else {
  }

  SpeechRequest request;
  const Json model_name = FieldOr(body, "model", Json());
  if (model_name.is_string()) {
    request.model = model_name.get<std::string>();
  } else if (!model_name.is_null()) {
    throw std::invalid_argument("model must be a string");
  } else {
  }
  const Json voice = PresentFieldOr(
      body, "voice", PresentFieldOr(body, "speaker", Json("default")));
  if (!voice.is_string()) {
    throw std::invalid_argument("voice must be a string");
  } else if (voice != "" && voice != "default") {
    throw std::invalid_argument(
        "Chatterbox-Turbo has no voice presets; use voice 'default' or clone "
        "a voice with references");
  } else {
  }

  const Json response_format =
      PresentFieldOr(body, "response_format", Json("wav"));
  if (!response_format.is_string()) {
    throw std::invalid_argument("response_format must be a string");
  } else {
  }
  request.response_format = Lowercase(
      omni_common::StripUnicodeWhitespace(response_format.get<std::string>()));
  if (request.response_format != "pcm" && request.response_format != "wav") {
    throw std::invalid_argument("response_format must be one of: pcm, wav");
  } else {
  }
  const Json stream = PresentFieldOr(body, "stream", Json(false));
  if (!stream.is_boolean()) {
    throw std::invalid_argument("stream must be a boolean");
  } else if (stream.get<bool>()) {
    throw std::invalid_argument("stream=true is not supported on this model");
  } else if (PresentFieldOr(body, "stream_format", Json("audio")) != "audio") {
    throw std::invalid_argument("stream_format must be audio");
  } else {
  }
  const Json speed = PresentFieldOr(body, "speed", Json(1.0));
  if (!speed.is_number() || speed.get<double>() != 1.0) {
    throw std::invalid_argument("speed must be 1.0 on this model");
  } else {
  }

  const Json max_new_tokens =
      FieldOr(body, "max_new_tokens", Json(kDefaultMaxSpeechTokens));
  if (!max_new_tokens.is_number_integer()) {
    throw std::invalid_argument("max_new_tokens must be an integer");
  } else if (max_new_tokens.get<int64_t>() <= 0) {
    throw std::invalid_argument("max_new_tokens must be greater than 0");
  } else if (max_new_tokens.get<int64_t>() > kMaxSpeechTokens) {
    throw std::invalid_argument("max_new_tokens must be at most " +
                                std::to_string(kMaxSpeechTokens));
  } else {
  }
  request.max_speech_tokens = max_new_tokens.get<int>();
  request.sampling.temperature =
      NumberField(body, "temperature", request.sampling.temperature, 0.0, false,
                  std::numeric_limits<float>::max());
  request.sampling.top_p =
      NumberField(body, "top_p", request.sampling.top_p, 0.0, false, 1.0);
  request.sampling.repetition_penalty = NumberField(
      body, "repetition_penalty", request.sampling.repetition_penalty, 0.0,
      false, kMaxRepetitionPenalty);
  const Json top_k = FieldOr(body, "top_k", Json(request.sampling.top_k));
  if (!top_k.is_number_integer() ||
      (!top_k.is_number_unsigned() && top_k.get<int64_t>() != -1 &&
       top_k.get<int64_t>() < 1) ||
      (top_k.is_number_unsigned() && top_k.get<uint64_t>() == 0)) {
    throw std::invalid_argument("top_k must be -1 (off) or at least 1");
  } else if (top_k == -1) {
    request.sampling.top_k = 0;
  } else {
    // Past the speech vocabulary, top-k keeps every token.
    request.sampling.top_k = static_cast<int>(std::min<uint64_t>(
        top_k.get<uint64_t>(), chatterbox::kStopSpeechToken + 1));
  }
  const Json seed = FieldOr(body, "seed", Json());
  if (seed.is_null()) {
    request.seed = std::random_device{}();
  } else if (!seed.is_number_unsigned()) {
    throw std::invalid_argument("seed must be a non-negative integer");
  } else {
    request.seed = seed.get<uint64_t>();
  }

  request.text_ids = model.EncodeText(text);
  return request;
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

// Thrown by a generation that reaches its request's deadline.
class DeadlineReached : public qwen3_asr::TranscriptionCancelled {};

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

class ChatterboxService : public asr_service::ServedModel {
public:
  ChatterboxService(const std::filesystem::path &model_directory,
                    int max_queued_requests, double request_timeout_seconds,
                    std::optional<std::filesystem::path> allowed_directory)
      : model_(model_directory), max_queued_requests_(max_queued_requests),
        request_timeout_(
            std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                std::chrono::duration<double>(request_timeout_seconds))),
        allowed_directory_(std::move(allowed_directory)) {}

  asr_service::Transcription
  Prepare(std::vector<float>, const asr_service::FormFields &) const override {
    throw std::logic_error("chatterbox serves no transcriptions");
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
    return static_cast<ChatterboxService *>(data)->Speak(connection);
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
    auto request = std::make_shared<SpeechRequest>();
    try {
      *request = ParseRequest(body, model_);
    } catch (const std::invalid_argument &error) {
      return omni_server::BadRequest(connection, error.what());
    }
    if (has_chosen_model_name() && request->model.has_value() &&
        *request->model != model_name()) {
      return omni_server::WriteJson(
          connection, 404, {{"detail", "Unknown model: " + *request->model}});
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
    // Note (Jiaxin Deng): the reference is decoded from base64 or read from
    // disk only once admitted, so the queue bounds how many are held; only its
    // header is read here, and decoding and resampling run in the job.
    try {
      request->reference = RequestReference(body, allowed_directory_);
      if (request->reference.has_value()) {
        chatterbox::ChatterboxModel::CheckReference(*request->reference);
      } else {
      }
    } catch (const std::invalid_argument &error) {
      {
        std::lock_guard<std::mutex> lock(admission_mutex_);
        --admitted_requests_;
      }
      return omni_server::BadRequest(connection, error.what());
    }

    const auto deadline = std::chrono::steady_clock::now() + request_timeout_;
    const std::string request_id = NewRequestId();
    auto channel = std::make_shared<SynthesisChannel>();
    const qwen3_asr::CancelFlag cancel = qwen3_asr::NewCancelFlag();
    worker_->Submit(
        [this, request, channel, deadline](const std::atomic<bool> &cancelled) {
          return Synthesize(*request, *channel, cancelled, deadline);
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

    const int client_socket = ClientSocket(connection);
    bool timed_out = false;
    bool client_left = false;
    std::unique_lock<std::mutex> lock(channel->mutex);
    while (!channel->done) {
      const bool done = channel->ready.wait_until(
          lock,
          std::min(deadline,
                   std::chrono::steady_clock::now() + kClientPollInterval),
          [&] { return channel->done; });
      if (done) {
        break;
      } else if (std::chrono::steady_clock::now() >= deadline) {
        timed_out = true;
        break;
      } else {
        client_left = ClientLeft(client_socket);
        if (client_left) {
          break;
        } else {
        }
      }
    }
    lock.unlock();
    if (client_left) {
      cancel->store(true);
      return kClientLeftStatus;
    } else if (!timed_out && ReachedDeadline(channel->error)) {
      timed_out = true;
    } else {
    }
    if (timed_out) {
      // Note (Minyi Tang): a request still queued is dropped once the
      // worker reaches it.
      cancel->store(true);
      std::cerr << "speech generation timed out\n";
      return omni_server::WriteJson(
          connection, 504, {{"detail", "Speech generation timed out."}});
    } else if (channel->error) {
      LogFailure(channel->error);
      return omni_server::WriteJson(connection, 500,
                                    {{"detail", "Speech generation failed."}});
    } else {
    }

    const std::string pcm = voxtral_tts::Pcm16Bytes(channel->samples);
    const std::string audio =
        request->response_format == "wav"
            ? voxtral_tts::WavBytes(pcm, chatterbox::kOutputSampleRate)
            : pcm;
    const size_t prompt_tokens = request->text_ids.size();
    const int completion_tokens = channel->result.generated_token_count;
    mg_printf(
        connection,
        "HTTP/1.1 200 OK\r\nContent-Type: audio/%s\r\nContent-Length: %zu\r\n"
        "Content-Disposition: attachment; filename=\"speech.%s\"\r\n"
        "X-Request-Id: %s\r\nX-Finish-Reason: %s\r\nX-Sample-Rate: %d\r\n"
        "X-Prompt-Tokens: %zu\r\nX-Completion-Tokens: %d\r\n"
        "X-SGLang-Omni-Input-Tokens: %zu\r\n"
        "X-SGLang-Omni-Output-Tokens: %d\r\n"
        "X-SGLang-Omni-Total-Tokens: %zu\r\nConnection: close\r\n\r\n",
        request->response_format.c_str(), audio.size(),
        request->response_format.c_str(), request_id.c_str(),
        qwen3_asr::FinishReasonName(channel->result.finish_reason),
        chatterbox::kOutputSampleRate, prompt_tokens, completion_tokens,
        prompt_tokens, completion_tokens, prompt_tokens + completion_tokens);
    mg_write(connection, audio.data(), audio.size());
    return 200;
  }

  TranscriptionResult
  Synthesize(const SpeechRequest &request, SynthesisChannel &channel,
             const std::atomic<bool> &cancel,
             std::chrono::steady_clock::time_point deadline) const {
    const auto check_deadline = [&] {
      if (std::chrono::steady_clock::now() >= deadline) {
        throw DeadlineReached();
      } else if (cancel.load()) {
        throw qwen3_asr::TranscriptionCancelled();
      } else {
      }
    };
    check_deadline();
    const chatterbox::Voice voice = request.reference.has_value()
                                        ? model_.CloneVoice(*request.reference)
                                        : model_.builtin_voice();
    check_deadline();
    mlx::core::random::seed(request.seed);
    std::vector<int> speech_tokens;
    TranscriptionResult result;
    result.finish_reason = model_.GenerateSpeechTokens(
        voice, request.text_ids, request.sampling, request.max_speech_tokens,
        cancel, [&](int token) {
          check_deadline();
          speech_tokens.push_back(token);
        });
    result.generated_token_count = static_cast<int>(speech_tokens.size());
    check_deadline();
    std::vector<float> samples = model_.Synthesize(speech_tokens, voice);
    check_deadline();
    std::lock_guard<std::mutex> lock(channel.mutex);
    channel.samples = std::move(samples);
    return result;
  }

  chatterbox::ChatterboxModel model_;
  const int max_queued_requests_;
  const std::chrono::steady_clock::duration request_timeout_;
  const std::optional<std::filesystem::path> allowed_directory_;
  mutable std::mutex admission_mutex_;
  // Requests the worker holds: running, queued, or finishing.
  int admitted_requests_ = 0;
  qwen3_asr::TranscriptionWorker *worker_ = nullptr;
};

} // namespace

int main(int argc, char **argv) {
  int max_queued_requests = 2;
  double request_timeout_seconds = 600;
  std::optional<std::filesystem::path> allowed_directory;
  asr_service::ServedKind kind;
  kind.model_kind = "chatterbox";
  kind.flags = {
      {"--max-queued-requests",
       [&](const std::string &value) {
         max_queued_requests = std::stoi(value);
       }},
      {"--request-timeout-s",
       [&](const std::string &value) {
         request_timeout_seconds = std::stod(value);
       }},
      {"--allowed-local-media-path",
       [&](const std::string &value) {
         std::error_code error;
         allowed_directory = std::filesystem::canonical(value, error);
         if (error || !std::filesystem::is_directory(*allowed_directory)) {
           throw std::logic_error(
               "--allowed-local-media-path must be a directory");
         } else {
         }
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
    return std::make_unique<ChatterboxService>(
        model_directory, max_queued_requests, request_timeout_seconds,
        allowed_directory);
  };
  return asr_service::Serve(argc, argv, kind);
}
