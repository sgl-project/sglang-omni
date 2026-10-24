// SPDX-License-Identifier: Apache-2.0
// Native Voxtral TTS server: POST /v1/audio/speech with the CUDA server's
// request fields, answering a WAV or PCM file, or 16-bit PCM streamed as it
// is decoded, and Voxt's supervisor protocol.
//
//   voxtral_tts_server --model-path DIR [--model-name NAME] [--host H]
//       [--port P] [--max-queued-requests N] [--request-timeout-s S]
//   voxtral_tts_server --supervised --model-kind voxtral_tts
//       --model-directory DIR [--max-queued-requests N]
//       [--request-timeout-s S]
//
// Request fields: input, model (the served name, when given), voice (or
// speaker; "default" is cheerful_female), response_format (wav or pcm),
// stream (pcm only), max_new_tokens (audio frames, 12.5 a second) and speed
// (1 only). One request runs at a time and N more (2 by default) may wait; the
// rest get 503. A request not done S seconds (600 by default) after it
// arrives is cancelled, with 504 or an early end to its stream, and so is a
// request whose client disconnects. A request body over 1 MiB gets 413.
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cerrno>
#include <charconv>
#include <chrono>
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
#include "common/http.h"
#include "qwen3_asr/asr_service.h"
#include "qwen3_asr/qwen3_asr_transcriber.h"
#include "voxtral_tts/voxtral_tts_model.h"

namespace {

using omni_server::Json;
using qwen3_asr::TranscriptionResult;

constexpr const char *kDefaultVoice = "cheerful_female";
constexpr size_t kMaxInputCharacters = 4096;
// Far above the largest valid request: kMaxInputCharacters characters, each
// JSON-escaped as a 12-byte surrogate pair, take 48 KiB.
constexpr size_t kMaxRequestBodyBytes = 1024 * 1024;
constexpr int kMaxAudioFrames = 4096;
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
// Note (Minyi Tang): the fields the CUDA server's Voxtral stages reject
// (validate_voxtral_speech_params), each counted as given by the same rule:
// sampling fields when not null, text fields when not null or "", and
// collections when not empty. CUDA's request model first rejects a collection
// of another type, even an empty one.
constexpr std::array<const char *, 5> kUnsupportedSamplingFields = {
    "repetition_penalty", "seed", "temperature", "top_k", "top_p"};
constexpr std::array<const char *, 5> kUnsupportedTextFields = {
    "instructions", "language", "ref_audio", "ref_text", "task_type"};
constexpr std::array<CollectionField, 2> kUnsupportedCollectionFields = {{
    {"references", Json::value_t::array, "a list"},
    {"stage_params", Json::value_t::object, "an object"},
}};

struct SpeechRequest {
  std::vector<int> prompt_ids;
  std::optional<std::string> model;
  std::string voice;
  std::string response_format;
  bool stream = false;
  int max_frames = kMaxAudioFrames;
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
  for (const char *field : kUnsupportedSamplingFields) {
    if (!FieldOr(body, field, Json()).is_null()) {
      given.push_back(field);
    } else {
    }
  }
  for (const char *field : kUnsupportedTextFields) {
    const Json value = FieldOr(body, field, Json());
    if (!value.is_null() && value != "") {
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
      "Voxtral TTS does not support these /v1/audio/speech fields: " + fields +
      ". Supported model-specific fields are voice and max_new_tokens.");
}

SpeechRequest ParseRequest(const Json &body,
                           const voxtral_tts::SpeechTokenizer &tokenizer) {
  if (!body.is_object()) {
    throw std::invalid_argument("speech request body must be a JSON object");
  } else {
  }
  RejectUnsupportedFields(body);
  const Json input = FieldOr(body, "input", Json());
  if (!input.is_string() ||
      qwen3_asr::StripUnicodeWhitespace(input.get<std::string>()).empty()) {
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
  const Json model = FieldOr(body, "model", Json());
  if (model.is_string()) {
    request.model = model.get<std::string>();
  } else if (!model.is_null()) {
    throw std::invalid_argument("model must be a string");
  } else {
  }
  // Note (Minyi Tang): as CUDA's AliasChoices("voice", "speaker"), voice wins
  // whenever it is given, even as null.
  const Json voice = PresentFieldOr(
      body, "voice", PresentFieldOr(body, "speaker", Json("default")));
  if (!voice.is_string()) {
    throw std::invalid_argument("voice must be a string");
  } else {
  }
  request.voice = voice.get<std::string>();
  if (request.voice.empty() || request.voice == "default") {
    request.voice = kDefaultVoice;
  } else if (tokenizer.voice_frame_counts().count(request.voice) == 0) {
    std::string supported;
    for (const auto &[name, frame_count] : tokenizer.voice_frame_counts()) {
      supported += (supported.empty() ? "" : ", ") + name;
    }
    throw std::invalid_argument("Unknown voice '" + request.voice +
                                "'. Supported voices: " + supported);
  } else {
  }

  const Json response_format =
      PresentFieldOr(body, "response_format", Json("wav"));
  if (!response_format.is_string()) {
    throw std::invalid_argument("response_format must be a string");
  } else {
  }
  // Note (Minyi Tang): as CUDA's response_format.strip().lower().
  for (const char character :
       qwen3_asr::StripUnicodeWhitespace(response_format.get<std::string>())) {
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
      FieldOr(body, "max_new_tokens", Json(kMaxAudioFrames));
  if (!max_new_tokens.is_number_integer()) {
    throw std::invalid_argument("max_new_tokens must be an integer");
  } else if (max_new_tokens.get<int64_t>() <= 0) {
    throw std::invalid_argument("max_new_tokens must be greater than 0");
  } else if (max_new_tokens.get<int64_t>() > kMaxAudioFrames) {
    throw std::invalid_argument("max_new_tokens must be at most " +
                                std::to_string(kMaxAudioFrames));
  } else {
  }
  request.max_frames = max_new_tokens.get<int>();
  request.prompt_ids = tokenizer.EncodeSpeechRequest(text, request.voice);
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

void LogFailure(std::exception_ptr error) {
  try {
    std::rethrow_exception(error);
  } catch (const qwen3_asr::TranscriptionCancelled &) {
  } catch (const std::exception &failure) {
    // Note (Minyi Tang): only the exception type is logged, never the text.
    std::cerr << "speech generation failed: " << typeid(failure).name() << "\n";
  }
}

class VoxtralTtsService : public asr_service::ServedModel {
public:
  VoxtralTtsService(const std::filesystem::path &model_directory,
                    int max_queued_requests, double request_timeout_seconds)
      : model_(model_directory), max_queued_requests_(max_queued_requests),
        request_timeout_(
            std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                std::chrono::duration<double>(request_timeout_seconds))) {}

  asr_service::Transcription
  Prepare(std::vector<float>, const asr_service::FormFields &) const override {
    throw std::logic_error("voxtral_tts serves no transcriptions");
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
    return static_cast<VoxtralTtsService *>(data)->Speak(connection);
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
      request = ParseRequest(body, model_.tokenizer());
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
        const std::string pcm = voxtral_tts::Pcm16Bytes(chunk);
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

    const std::string pcm = voxtral_tts::Pcm16Bytes(samples);
    const std::string audio = request.response_format == "wav"
                                  ? voxtral_tts::WavBytes(pcm, sample_rate)
                                  : pcm;
    const size_t prompt_tokens = request.prompt_ids.size();
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

  TranscriptionResult
  Synthesize(const SpeechRequest &request, AudioChannel &channel,
             const std::atomic<bool> &cancel,
             std::chrono::steady_clock::time_point deadline) const {
    voxtral_tts::CodecStream stream(model_.codec());
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
        model_.Generate(request.prompt_ids, request.voice, request.max_frames,
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
      // Note (Minyi Tang): the CUDA vocoder raises on an end of audio at the
      // first frame rather than answer silence.
      throw std::runtime_error("Voxtral TTS generated no audio codes");
    } else {
    }
    send(stream.Finish());
    return result;
  }

  voxtral_tts::VoxtralTtsModel model_;
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
  kind.model_kind = "voxtral_tts";
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
    return std::make_unique<VoxtralTtsService>(
        model_directory, max_queued_requests, request_timeout_seconds);
  };
  return asr_service::Serve(argc, argv, kind);
}
