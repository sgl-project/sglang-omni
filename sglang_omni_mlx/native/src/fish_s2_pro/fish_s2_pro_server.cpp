// SPDX-License-Identifier: Apache-2.0
// Native Fish S2-Pro server: POST /v1/audio/speech with the CUDA server's
// request fields, voice cloning from reference audio, answering a WAV or PCM
// file, or 16-bit PCM streamed as it is decoded, and Voxt's supervisor
// protocol.
//
//   fish_s2_pro_server --model-path DIR [--model-name NAME] [--host H]
//       [--port P] [--max-queued-requests N] [--request-timeout-s S]
//   fish_s2_pro_server --supervised --model-kind fish_s2_pro
//       --model-directory DIR [--max-queued-requests N]
//       [--request-timeout-s S]
//
// Request fields: input, model (the served name, when given), voice (or
// speaker; "default" only), response_format (wav or pcm), stream (pcm only),
// speed (1 only), max_new_tokens (audio frames, 21.5 a second; 1024 by
// default, at most 2048), the sampling fields temperature, top_p, top_k,
// repetition_penalty and seed, and the voice to clone: ref_audio (a base64
// WAV or a data: URL) with ref_text, or references, a list of {audio or data,
// text} or {vq_codes, text}. One request runs at a time and N more (2 by
// default) may wait; the rest get 503. A request not done S seconds (600 by
// default) after it arrives is cancelled, with 504 or an early end to its
// stream, and so is a request whose client disconnects. A request body over
// 16 MiB gets 413, and so does one over 1 MiB without ref_audio or references.
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
#include "fish_s2_pro/fish_s2_pro_model.h"
#include "qwen3_asr/asr_service.h"
#include "qwen3_asr/qwen3_asr_transcriber.h"

namespace {

using omni_server::Json;
using qwen3_asr::TranscriptionResult;

constexpr size_t kMaxInputCharacters = 4096;
// Bodies are read and parsed before admission, on any of civetweb's threads,
// so each stays small: 1 MiB, as Voxtral's, unless it carries reference
// audio, which 16 MiB holds for 60 s of 48 kHz stereo 16-bit WAV in base64.
constexpr size_t kMaxRequestBodyBytes = 16 * 1024 * 1024;
constexpr size_t kMaxRequestBodyBytesWithoutReferences = 1024 * 1024;
// Far above a valid request's JSON values (60 s of vq_codes are 12,920), so
// a body of millions of small values cannot build a large document.
constexpr size_t kMaxRequestJsonValues = 65536;
constexpr double kMaxReferenceSeconds = 60.0;
// Note (Jiaxin Deng): the CUDA server's defaults (stages.py) and its engine's
// cap on max_new_tokens, which it applies without an error.
constexpr int kDefaultMaxFrames = 1024;
constexpr int kMaxFrames = 2048;
constexpr int kMaxTopK = 30;
constexpr double kMaxRepetitionPenalty = 2.0;
constexpr uint32_t kSeedMask = 0x7FFFFFFF;
constexpr int kRequestIdHexDigits = 32;
constexpr int kMaxRequestTimeoutSeconds = 24 * 60 * 60;
// As often as the CUDA server checks whether a speech client disconnected.
constexpr auto kClientPollInterval = std::chrono::milliseconds(50);
// The status civetweb logs for a request whose client left before its answer.
constexpr int kClientLeftStatus = 499;
// The most one disconnect check reads of what a client sent after its request,
// so a client that keeps sending cannot hold the check.
constexpr size_t kMaxUnreadBytesPerCheck = 64 * 1024;
// A collection field of CUDA's CreateSpeechRequest and the JSON type it takes.
struct CollectionField {
  const char *name;
  Json::value_t type;
  const char *type_description;
};
// Fields of CUDA's CreateSpeechRequest this model does not use, each counted
// as given by the CUDA server's rule: text fields when not null or "", flags
// and counts when not null, and collections when not empty.
constexpr std::array<const char *, 3> kUnsupportedTextFields = {
    "instructions", "language", "task_type"};
constexpr std::array<const char *, 6> kUnsupportedValueFields = {
    "duration_tokens",     "initial_codec_chunk_frames",
    "stream_codec_output", "suppress_bootstrap_silence",
    "token_count",         "x_vector_only_mode"};
constexpr std::array<CollectionField, 1> kUnsupportedCollectionFields = {{
    {"stage_params", Json::value_t::object, "an object"},
}};

// A reference as the request gives it: its audio still base64, decoded only
// once the worker admits the request, or codes the client gave.
struct PendingReference {
  std::string text;
  // The request field that held the audio, for error messages.
  std::string audio_field;
  std::string audio_base64;
  std::vector<int32_t> codes;
  int frame_count = 0;
};

struct SpeechRequest {
  std::string text;
  std::optional<std::string> model;
  std::string response_format;
  bool stream = false;
  int max_frames = kDefaultMaxFrames;
  fish_s2_pro::SamplingOptions sampling;
  std::vector<PendingReference> references;
  // Seconds of reference audio the references' codes hold.
  double code_seconds = 0.0;
};

// A request error found once the worker decodes the request's audio; it is
// answered with 400, as a request error found before admission is.
class RejectedRequest : public std::invalid_argument {
public:
  using std::invalid_argument::invalid_argument;
};

// Audio one synthesis hands from the worker thread to its request's thread.
struct AudioChannel {
  std::mutex mutex;
  std::condition_variable ready;
  std::deque<std::vector<float>> chunks;
  bool done = false;
  std::exception_ptr error;
  TranscriptionResult result;
  size_t prompt_token_count = 0;
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
// Note (Minyi Tang): CUDA's CreateSpeechRequest takes null as the default only
// for the fields it types as optional, and rejects it for the rest.
Json PresentFieldOr(const Json &body, const char *name, Json fallback) {
  const auto found = body.find(name);
  if (found == body.end()) {
    return fallback;
  } else {
    return *found;
  }
}

// Throws naming every unsupported field body gives, in sorted order.
void RejectUnsupportedFields(const Json &body) {
  std::vector<std::string> given;
  for (const char *field : kUnsupportedTextFields) {
    const Json value = FieldOr(body, field, Json());
    if (!value.is_null() && value != "") {
      given.push_back(field);
    } else {
    }
  }
  for (const char *field : kUnsupportedValueFields) {
    if (!FieldOr(body, field, Json()).is_null()) {
      given.push_back(field);
    } else {
    }
  }
  for (const CollectionField &field : kUnsupportedCollectionFields) {
    const Json value = FieldOr(body, field.name, Json());
    if (!value.is_null() && value.type() != field.type) {
      throw std::invalid_argument(std::string(field.name) + " must be " +
                                  field.type_description);
    } else if (!value.is_null() && !value.empty()) {
      given.push_back(field.name);
    } else {
    }
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
      "Fish S2-Pro does not support these /v1/audio/speech fields: " + fields +
      ". Supported model-specific fields are ref_audio, ref_text, references, "
      "max_new_tokens, temperature, top_p, top_k, repetition_penalty and "
      "seed.");
}

// Whether value is a JSON integer too large for int64, which nlohmann would
// wrap to a negative number.
bool IsAboveInt64(const Json &value) {
  return value.is_number_unsigned() &&
         value.get<uint64_t>() >
             static_cast<uint64_t>(std::numeric_limits<int64_t>::max());
}

// Throws unless text has at most kMaxInputCharacters characters.
void CheckLength(const std::string &text, const std::string &field) {
  const auto character_count =
      std::count_if(text.begin(), text.end(), [](char byte) {
        return (static_cast<unsigned char>(byte) & 0xc0) != 0x80;
      });
  if (static_cast<size_t>(character_count) > kMaxInputCharacters) {
    throw std::invalid_argument(field + " must be at most " +
                                std::to_string(kMaxInputCharacters) +
                                " characters");
  } else {
  }
}

// An optional number field: fallback when missing or null, else a finite
// number.
double NumberOr(const Json &body, const char *name, double fallback) {
  const Json value = FieldOr(body, name, Json(fallback));
  if (!value.is_number() || !std::isfinite(value.get<double>())) {
    throw std::invalid_argument(std::string(name) + " must be a number");
  } else {
  }
  return value.get<double>();
}

// Takes seconds of reference audio from what the request has left; throws
// before a reference past kMaxReferenceSeconds in all is resampled.
void SpendReferenceSeconds(double seconds, double &remaining_seconds) {
  if (seconds > remaining_seconds) {
    throw RejectedRequest(
        "the references are longer than " +
        std::to_string(static_cast<int>(kMaxReferenceSeconds)) + " s in all");
  } else {
    remaining_seconds -= seconds;
  }
}

// The base64 payload of a reference audio string: a base64 WAV or a data:
// URL of one. Paths and URLs are the CUDA server's and not read.
std::string ReferencePayload(const std::string &audio,
                             const std::string &field) {
  if (audio.rfind("data:", 0) == 0) {
    const size_t comma = audio.find(',');
    if (comma == std::string::npos ||
        audio.substr(0, comma).find(";base64") == std::string::npos) {
      throw std::invalid_argument(field + " must be a base64 data: URL");
    } else {
      return audio.substr(comma + 1);
    }
  } else if (audio.rfind("http://", 0) == 0 ||
             audio.rfind("https://", 0) == 0 ||
             audio.rfind("file://", 0) == 0 || audio.rfind("/", 0) == 0 ||
             audio.rfind("./", 0) == 0 || audio.rfind("../", 0) == 0 ||
             audio.rfind("~", 0) == 0) {
    throw std::invalid_argument(
        field + " must be a WAV file as base64 or a data: URL; this server "
                "reads no paths or URLs");
  } else {
    return audio;
  }
}

// A reference's WAV, decoded at its own rate; throws RejectedRequest.
fish_s2_pro::DecodedWav DecodeReference(const PendingReference &reference) {
  const std::optional<std::vector<uint8_t>> bytes =
      omni_common::DecodeBase64(reference.audio_base64);
  if (!bytes.has_value() || bytes->empty()) {
    throw RejectedRequest(reference.audio_field + " is not valid base64");
  } else {
  }
  fish_s2_pro::DecodedWav wav;
  try {
    wav = fish_s2_pro::DecodeAnyWav(std::string_view(
        reinterpret_cast<const char *>(bytes->data()), bytes->size()));
  } catch (const std::invalid_argument &error) {
    throw RejectedRequest(reference.audio_field + ": " + error.what());
  }
  if (wav.samples.empty()) {
    throw RejectedRequest(reference.audio_field + " has no samples");
  } else {
  }
  return wav;
}

// A references item: {audio_path, ref_audio, audio, data, media_type, text,
// vq_codes}, as CUDA's SpeechReference.
PendingReference ParseReference(const Json &item, int index, int codebook_count,
                                int codebook_size) {
  const std::string field = "references[" + std::to_string(index) + "]";
  if (!item.is_object()) {
    throw std::invalid_argument(field + " must be an object");
  } else {
  }
  PendingReference reference;
  const Json text = FieldOr(item, "text", Json(""));
  if (!text.is_string()) {
    throw std::invalid_argument(field + ".text must be a string");
  } else {
  }
  reference.text = text.get<std::string>();
  CheckLength(reference.text, field + ".text");
  const Json codes = FieldOr(item, "vq_codes", Json());
  if (!codes.is_null()) {
    if (!codes.is_array() || static_cast<int>(codes.size()) != codebook_count) {
      throw std::invalid_argument(field + ".vq_codes must have shape (" +
                                  std::to_string(codebook_count) + ", T)");
    } else {
    }
    for (int codebook = 0; codebook < codebook_count; ++codebook) {
      const Json &row = codes.at(codebook);
      if (!row.is_array() || row.empty() ||
          (codebook > 0 && row.size() != codes.at(0).size())) {
        throw std::invalid_argument(field + ".vq_codes must have shape (" +
                                    std::to_string(codebook_count) + ", T)");
      } else {
      }
      // The semantic codebook is the model's; the residual ones are the
      // codec's.
      const int64_t size =
          codebook == 0 ? codebook_size : fish_s2_pro::kResidualCodebookSize;
      for (const Json &code : row) {
        if (!code.is_number_integer() || code.get<int64_t>() < 0 ||
            code.get<int64_t>() >= size) {
          throw std::invalid_argument(
              field + ".vq_codes row " + std::to_string(codebook) +
              " must be integers in [0, " + std::to_string(size) + ")");
        } else {
        }
        reference.codes.push_back(code.get<int32_t>());
      }
    }
    reference.frame_count = static_cast<int>(codes.at(0).size());
    return reference;
  } else {
  }
  if (!FieldOr(item, "audio_path", Json()).is_null()) {
    throw std::invalid_argument(field + ".audio_path is not supported; send "
                                        "the WAV as data or audio");
  } else {
  }
  for (const char *name : {"data", "audio", "ref_audio"}) {
    const Json audio = FieldOr(item, name, Json());
    if (audio.is_string()) {
      reference.audio_field = field + "." + name;
      reference.audio_base64 =
          ReferencePayload(audio.get<std::string>(), reference.audio_field);
      return reference;
    } else if (!audio.is_null()) {
      throw std::invalid_argument(field + "." + name + " must be a string");
    } else {
    }
  }
  throw std::invalid_argument(field + " has no audio or vq_codes");
}

SpeechRequest ParseRequest(const Json &body,
                           const fish_s2_pro::FishS2ProModel &model) {
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
  SpeechRequest request;
  request.text = input.get<std::string>();
  CheckLength(request.text, "input");

  const Json model_name = FieldOr(body, "model", Json());
  if (model_name.is_string()) {
    request.model = model_name.get<std::string>();
  } else if (!model_name.is_null()) {
    throw std::invalid_argument("model must be a string");
  } else {
  }
  // Note (Minyi Tang): as CUDA's AliasChoices("voice", "speaker"), voice wins
  // whenever it is given, even as null.
  const Json voice = PresentFieldOr(
      body, "voice", PresentFieldOr(body, "speaker", Json("default")));
  if (!voice.is_string()) {
    throw std::invalid_argument("voice must be a string");
  } else if (voice != "" && voice != "default") {
    throw std::invalid_argument("Unknown voice '" + voice.get<std::string>() +
                                "'. Supported voices: default; clone a voice "
                                "with ref_audio or references");
  } else {
  }

  const Json response_format =
      PresentFieldOr(body, "response_format", Json("wav"));
  if (!response_format.is_string()) {
    throw std::invalid_argument("response_format must be a string");
  } else {
  }
  // Note (Minyi Tang): as CUDA's response_format.strip().lower().
  for (const char character : omni_common::StripUnicodeWhitespace(
           response_format.get<std::string>())) {
    request.response_format.push_back(
        static_cast<char>(std::tolower(static_cast<unsigned char>(character))));
  }
  if (request.response_format != "pcm" && request.response_format != "wav") {
    throw std::invalid_argument("response_format must be one of: pcm, wav");
  } else {
  }

  const Json stream = PresentFieldOr(body, "stream", Json(false));
  if (!stream.is_boolean()) {
    throw std::invalid_argument("stream must be a boolean");
  } else {
  }
  request.stream = stream.get<bool>();
  if (PresentFieldOr(body, "stream_format", Json("audio")) != "audio") {
    throw std::invalid_argument("stream_format must be audio");
  } else {
  }
  if (request.stream && request.response_format != "pcm") {
    throw std::invalid_argument("stream=true requires response_format='pcm'");
  } else {
  }

  const Json speed = PresentFieldOr(body, "speed", Json(1.0));
  if (!speed.is_number() || speed.get<double>() != 1.0) {
    throw std::invalid_argument("speed must be 1.0 on this model");
  } else {
  }
  const Json max_new_tokens =
      FieldOr(body, "max_new_tokens", Json(kDefaultMaxFrames));
  if (!max_new_tokens.is_number_integer()) {
    throw std::invalid_argument("max_new_tokens must be an integer");
  } else if (IsAboveInt64(max_new_tokens)) {
    request.max_frames = kMaxFrames;
  } else if (max_new_tokens.get<int64_t>() <= 0) {
    throw std::invalid_argument("max_new_tokens must be greater than 0");
  } else {
    request.max_frames = static_cast<int>(
        std::min<int64_t>(max_new_tokens.get<int64_t>(), kMaxFrames));
  }

  // Note (Jiaxin Deng): the bounds SGLang's SamplingParams.verify and the
  // CUDA model's top_k check put on these fields.
  fish_s2_pro::SamplingOptions &sampling = request.sampling;
  const double temperature =
      NumberOr(body, "temperature", sampling.temperature);
  if (temperature < 0.0) {
    throw std::invalid_argument("temperature must be a non-negative number");
  } else {
  }
  sampling.temperature = static_cast<float>(temperature);
  const double top_p = NumberOr(body, "top_p", sampling.top_p);
  if (!(top_p > 0.0 && top_p <= 1.0)) {
    throw std::invalid_argument("top_p must be in (0, 1]");
  } else {
  }
  sampling.top_p = static_cast<float>(top_p);
  const Json top_k = FieldOr(body, "top_k", Json(sampling.top_k));
  if (!top_k.is_number_integer() || IsAboveInt64(top_k) ||
      (top_k.get<int64_t>() != -1 &&
       (top_k.get<int64_t>() < 1 || top_k.get<int64_t>() > kMaxTopK))) {
    throw std::invalid_argument("top_k must be -1 or between 1 and " +
                                std::to_string(kMaxTopK));
  } else {
  }
  sampling.top_k = top_k.get<int>();
  const double repetition_penalty =
      NumberOr(body, "repetition_penalty", sampling.repetition_penalty);
  if (!(repetition_penalty > 0.0 &&
        repetition_penalty <= kMaxRepetitionPenalty)) {
    throw std::invalid_argument("repetition_penalty must be in (0, 2]");
  } else {
  }
  sampling.repetition_penalty = static_cast<float>(repetition_penalty);
  const Json seed = FieldOr(body, "seed", Json());
  if (seed.is_null()) {
    static thread_local std::mt19937 generator{std::random_device{}()};
    sampling.seed = static_cast<uint32_t>(generator()) & kSeedMask;
  } else if (IsAboveInt64(seed)) {
    throw std::invalid_argument("seed must be in [-2**63, 2**63 - 1]");
  } else if (seed.is_number_integer() && seed.get<int64_t>() < 0) {
    // Note (Jiaxin Deng): as the CUDA server's validate_non_negative_int.
    throw std::invalid_argument("seed must be greater than or equal to 0");
  } else if (seed.is_number_integer()) {
    sampling.seed = static_cast<uint32_t>(
        static_cast<uint64_t>(seed.get<int64_t>()) & kSeedMask);
  } else {
    throw std::invalid_argument("seed must be an integer");
  }

  const Json references = FieldOr(body, "references", Json());
  double remaining_seconds = kMaxReferenceSeconds;
  if (references.is_array()) {
    for (size_t index = 0; index < references.size(); ++index) {
      request.references.push_back(
          ParseReference(references.at(index), static_cast<int>(index),
                         model.codebook_count(), model.codebook_size()));
      SpendReferenceSeconds(
          static_cast<double>(request.references.back().frame_count) *
              model.codec().samples_per_frame() / model.codec().sample_rate(),
          remaining_seconds);
    }
  } else if (!references.is_null()) {
    throw std::invalid_argument("references must be a list");
  } else {
  }
  const Json reference_audio = FieldOr(body, "ref_audio", Json());
  const Json reference_text = FieldOr(body, "ref_text", Json());
  if (!reference_text.is_null() && !reference_text.is_string()) {
    throw std::invalid_argument("ref_text must be a string");
  } else if (reference_text.is_string()) {
    CheckLength(reference_text.get<std::string>(), "ref_text");
  } else {
  }
  if (reference_audio.is_string()) {
    request.references.push_back(
        {reference_text.is_string() ? reference_text.get<std::string>() : "",
         "ref_audio",
         ReferencePayload(reference_audio.get<std::string>(), "ref_audio"),
         {},
         0});
  } else if (!reference_audio.is_null()) {
    throw std::invalid_argument("ref_audio must be a string");
  } else {
  }
  request.code_seconds = kMaxReferenceSeconds - remaining_seconds;
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

// The detail of a request error found after admission (a prompt the model
// found too long, or reference audio it could not use), or nullopt for any
// other error.
std::optional<std::string> RequestErrorDetail(std::exception_ptr error) {
  if (!error) {
    return std::nullopt;
  } else {
  }
  try {
    std::rethrow_exception(error);
  } catch (const fish_s2_pro::PromptTooLong &too_long) {
    return std::string(too_long.what());
  } catch (const RejectedRequest &rejected) {
    return std::string(rejected.what());
  } catch (...) {
    return std::nullopt;
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

class FishS2ProService : public asr_service::ServedModel {
public:
  FishS2ProService(const std::filesystem::path &model_directory,
                   int max_queued_requests, double request_timeout_seconds)
      : model_(model_directory), max_queued_requests_(max_queued_requests),
        request_timeout_(
            std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                std::chrono::duration<double>(request_timeout_seconds))) {
    // Note (Jiaxin Deng): freed MLX buffers go back to the system; MLX's
    // cache of them grows with each request's length, to twice the model's
    // memory, and an idle server then holds only the model.
    mlx::core::set_cache_limit(0);
  }

  asr_service::Transcription
  Prepare(std::vector<float>, const asr_service::FormFields &) const override {
    throw std::logic_error("fish_s2_pro serves no transcriptions");
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
    return static_cast<FishS2ProService *>(data)->Speak(connection);
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
    if (raw_body->size() > kMaxRequestBodyBytesWithoutReferences &&
        raw_body->find("\"ref_audio\"") == std::string::npos &&
        raw_body->find("\"references\"") == std::string::npos) {
      return omni_server::WriteJson(
          connection, 413,
          {{"detail",
            "request body must be at most " +
                std::to_string(kMaxRequestBodyBytesWithoutReferences) +
                " bytes without ref_audio or references"}});
    } else {
    }
    size_t value_count = 0;
    const Json::parser_callback_t count_values =
        [&value_count](int, Json::parse_event_t event, Json &) {
          if (event == Json::parse_event_t::value &&
              ++value_count > kMaxRequestJsonValues) {
            throw std::length_error("too many JSON values");
          } else {
          }
          return true;
        };
    Json body;
    try {
      body = Json::parse(*raw_body, count_values, false);
    } catch (const std::length_error &) {
      return omni_server::BadRequest(connection,
                                     "speech request body must hold at most " +
                                         std::to_string(kMaxRequestJsonValues) +
                                         " JSON values");
    }
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
    auto shared_request = std::make_shared<SpeechRequest>(std::move(request));
    const qwen3_asr::CancelFlag cancel = qwen3_asr::NewCancelFlag();
    worker_->Submit(
        [this, shared_request, channel,
         deadline](const std::atomic<bool> &cancelled) {
          return Synthesize(*shared_request, *channel, cancelled, deadline);
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
    const SpeechRequest &speech = *shared_request;
    // Note (Jiaxin Deng): a stream's 200 waits for its first audio, so a
    // request error found once the worker builds the prompt, a timeout in the
    // queue, or a generation with no audio still gets its own status.
    bool stream_started = false;
    const auto start_stream = [&]() {
      mg_printf(connection,
                "HTTP/1.1 200 OK\r\nContent-Type: audio/pcm\r\n"
                "X-Request-Id: %s\r\nX-SGLang-Omni-Speech-Id: %s\r\n"
                "X-Sample-Rate: %d\r\nX-Channels: 1\r\nX-Bit-Depth: 16\r\n"
                "Cache-Control: no-cache\r\nConnection: close\r\n\r\n",
                request_id.c_str(), request_id.c_str(), sample_rate);
      stream_started = true;
    };

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
      if (speech.stream) {
        if (!stream_started) {
          start_stream();
        } else {
        }
        const std::string pcm = fish_s2_pro::Pcm16Bytes(chunk);
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
    } else if (channel->error && !RequestErrorDetail(channel->error)) {
      LogFailure(channel->error);
    } else {
    }
    if (stream_started) {
      return 200;
    } else if (timed_out) {
      return omni_server::WriteJson(
          connection, 504, {{"detail", "Speech generation timed out."}});
    } else if (const std::optional<std::string> detail =
                   RequestErrorDetail(channel->error)) {
      return omni_server::BadRequest(connection, *detail);
    } else if (channel->error) {
      return omni_server::WriteJson(connection, 500,
                                    {{"detail", "Speech generation failed."}});
    } else if (speech.stream) {
      start_stream();
      return 200;
    } else {
    }

    const std::string pcm = fish_s2_pro::Pcm16Bytes(samples);
    const std::string audio = speech.response_format == "wav"
                                  ? fish_s2_pro::WavBytes(pcm, sample_rate)
                                  : pcm;
    const size_t prompt_tokens = channel->prompt_token_count;
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
        speech.response_format.c_str(), audio.size(),
        speech.response_format.c_str(), request_id.c_str(),
        qwen3_asr::FinishReasonName(channel->result.finish_reason),
        prompt_tokens, completion_tokens, prompt_tokens, completion_tokens,
        prompt_tokens + completion_tokens);
    mg_write(connection, audio.data(), audio.size());
    return 200;
  }

  TranscriptionResult
  Synthesize(const SpeechRequest &request, AudioChannel &channel,
             const std::atomic<bool> &cancel,
             std::chrono::steady_clock::time_point deadline) const {
    const auto check_deadline = [&deadline]() {
      // Note (Minyi Tang): the generation checks its own deadline, since the
      // request's thread may be blocked writing to a slow client.
      if (std::chrono::steady_clock::now() >= deadline) {
        throw DeadlineReached();
      } else {
      }
    };
    // Every reference's audio is decoded, and the total checked, before
    // any is resampled.
    double remaining_seconds = kMaxReferenceSeconds - request.code_seconds;
    std::vector<fish_s2_pro::DecodedWav> wavs;
    for (const PendingReference &reference : request.references) {
      if (reference.codes.empty()) {
        wavs.push_back(DecodeReference(reference));
        SpendReferenceSeconds(static_cast<double>(wavs.back().samples.size()) /
                                  wavs.back().sample_rate,
                              remaining_seconds);
      } else {
        wavs.emplace_back();
      }
    }
    std::vector<fish_s2_pro::VoiceReference> references;
    for (size_t index = 0; index < request.references.size(); ++index) {
      check_deadline();
      if (cancel.load()) {
        throw qwen3_asr::TranscriptionCancelled();
      } else {
      }
      const PendingReference &reference = request.references[index];
      if (reference.codes.empty()) {
        std::vector<float> samples;
        try {
          samples = fish_s2_pro::Resample(wavs[index].samples,
                                          wavs[index].sample_rate,
                                          model_.codec().sample_rate());
        } catch (const std::invalid_argument &error) {
          throw RejectedRequest(error.what());
        }
        references.push_back({reference.text, model_.codec().Encode(samples)});
      } else {
        references.push_back(
            {reference.text,
             mlx::core::array(reference.codes.begin(),
                              {model_.codebook_count(), reference.frame_count},
                              mlx::core::int32)});
      }
    }
    const fish_s2_pro::SpeechPrompt prompt =
        model_.BuildPrompt(request.text, references);
    {
      std::lock_guard<std::mutex> lock(channel.mutex);
      channel.prompt_token_count = prompt.ids.size();
    }
    // Note (Jiaxin Deng): as the CUDA engine, the frames stop where the
    // prompt and its audio would outrun the model's positions.
    const int max_frames = std::min(
        request.max_frames, std::max(model_.max_positions() - 1 -
                                         static_cast<int>(prompt.ids.size()),
                                     1));
    const auto send = [&channel](std::vector<float> samples) {
      if (!samples.empty()) {
        std::lock_guard<std::mutex> lock(channel.mutex);
        channel.chunks.push_back(std::move(samples));
        channel.ready.notify_all();
      } else {
      }
    };
    fish_s2_pro::CodecStream stream(model_.codec());
    std::vector<mlx::core::array> frames;
    TranscriptionResult result;
    result.finish_reason =
        model_.Generate(prompt, max_frames, request.sampling, cancel,
                        [&](const mlx::core::array &codes) {
                          check_deadline();
                          ++result.generated_token_count;
                          if (request.stream) {
                            send(stream.Push(codes));
                          } else {
                            frames.push_back(codes);
                          }
                        });
    if (result.generated_token_count == 0) {
      // Note (Minyi Tang): the CUDA vocoder raises on an end of audio at the
      // first frame rather than answer silence.
      throw std::runtime_error("Fish S2-Pro generated no audio codes");
    } else if (request.stream) {
      send(stream.Finish());
    } else {
      // Note (Jiaxin Deng): as the CUDA vocoder, a whole request decodes in
      // one window.
      check_deadline();
      const mlx::core::array audio =
          model_.codec().Decode(mlx::core::stack(frames, 1));
      mlx::core::eval(audio);
      send(std::vector<float>(audio.data<float>(),
                              audio.data<float>() + audio.size()));
    }
    return result;
  }

  fish_s2_pro::FishS2ProModel model_;
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
  kind.model_kind = "fish_s2_pro";
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
    return std::make_unique<FishS2ProService>(
        model_directory, max_queued_requests, request_timeout_seconds);
  };
  return asr_service::Serve(argc, argv, kind);
}
