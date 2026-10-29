// SPDX-License-Identifier: Apache-2.0
#include "qwen3_asr/speech_service.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>

#include <array>
#include <cctype>
#include <cerrno>
#include <charconv>
#include <condition_variable>
#include <deque>
#include <filesystem>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <system_error>
#include <typeinfo>

#include "civetweb.h"
#include "common/audio.h"
#include "common/text.h"

namespace speech_service {

namespace {

constexpr int kRequestIdHexDigits = 32;
constexpr int kDefaultMaxQueuedRequests = 2;
constexpr double kDefaultRequestTimeoutSeconds = 600;
constexpr int kMaxRequestTimeoutSeconds = 24 * 60 * 60;
// As often as the CUDA server checks whether a speech client disconnected.
constexpr auto kClientPollInterval = std::chrono::milliseconds(50);
// The status civetweb logs for a request whose client left before its answer.
constexpr int kClientLeftStatus = 499;
// The most one disconnect check reads of what a client sent after its request,
// so a client that keeps sending cannot hold the check.
constexpr size_t kMaxUnreadBytesPerCheck = 64 * 1024;

// Audio one synthesis hands from the worker thread to its request's thread.
struct AudioChannel {
  std::mutex mutex;
  std::condition_variable ready;
  std::deque<std::vector<float>> chunks;
  bool done = false;
  std::exception_ptr error;
  SpeechResult result;
};

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

} // namespace

Json FieldOr(const Json &body, const char *name, Json fallback) {
  const auto found = body.find(name);
  if (found == body.end() || found->is_null()) {
    return fallback;
  } else {
    return *found;
  }
}

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

std::optional<std::string> ParseModel(const Json &body) {
  const Json model = FieldOr(body, "model", Json());
  if (model.is_string()) {
    return model.get<std::string>();
  } else if (!model.is_null()) {
    throw std::invalid_argument("model must be a string");
  } else {
    return std::nullopt;
  }
}

SpeechJob ParseCommonFields(const Json &body) {
  SpeechJob job;
  const Json response_format =
      PresentFieldOr(body, "response_format", Json("wav"));
  if (!response_format.is_string()) {
    throw std::invalid_argument("response_format must be a string");
  } else {
  }
  // Note (Minyi Tang): as CUDA's response_format.strip().lower().
  for (const char character : omni_common::StripUnicodeWhitespace(
           response_format.get<std::string>())) {
    job.response_format.push_back(
        static_cast<char>(std::tolower(static_cast<unsigned char>(character))));
  }
  if (job.response_format != "pcm" && job.response_format != "wav") {
    throw std::invalid_argument("response_format must be one of: pcm, wav");
  } else {
  }
  const Json stream = PresentFieldOr(body, "stream", Json(false));
  if (!stream.is_boolean()) {
    throw std::invalid_argument("stream must be a boolean");
  } else {
  }
  job.stream = stream.get<bool>();
  if (PresentFieldOr(body, "stream_format", Json("audio")) != "audio") {
    throw std::invalid_argument("stream_format must be audio");
  } else if (job.stream && job.response_format != "pcm") {
    throw std::invalid_argument("stream=true requires response_format='pcm'");
  } else {
  }
  const Json speed = PresentFieldOr(body, "speed", Json(1.0));
  if (!speed.is_number() || speed.get<double>() != 1.0) {
    throw std::invalid_argument("speed must be 1.0 on this model");
  } else {
  }
  return job;
}

SpeechServedModel::SpeechServedModel(size_t max_request_body_bytes,
                                     int max_queued_requests,
                                     double request_timeout_seconds)
    : max_request_body_bytes_(max_request_body_bytes),
      max_queued_requests_(max_queued_requests),
      request_timeout_(
          std::chrono::duration_cast<std::chrono::steady_clock::duration>(
              std::chrono::duration<double>(request_timeout_seconds))) {}

asr_service::Transcription
SpeechServedModel::Prepare(std::vector<float>,
                           const asr_service::FormFields &) const {
  throw std::logic_error("a speech model serves no transcriptions");
}

// Note (Minyi Tang): /health counts the slots admission holds, which the
// worker frees only after it marks a request done; once /health shows the
// server idle, it admits a full queue.
std::map<std::string, int>
SpeechServedModel::RequestStates(const qwen3_asr::TranscriptionWorker &) const {
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

void SpeechServedModel::AddHandlers(mg_context *context,
                                    qwen3_asr::TranscriptionWorker &worker) {
  worker_ = &worker;
  mg_set_request_handler(context, "/v1/audio/speech$", HandleSpeech, this);
}

int SpeechServedModel::HandleSpeech(mg_connection *connection, void *data) {
  return static_cast<SpeechServedModel *>(data)->Speak(connection);
}

int SpeechServedModel::Speak(mg_connection *connection) {
  if (!omni_server::IsPost(connection)) {
    return omni_server::WriteJson(connection, 405,
                                  {{"detail", "Method Not Allowed"}});
  } else {
  }
  const std::optional<std::string> raw_body =
      omni_server::ReadBody(connection, max_request_body_bytes_);
  if (!raw_body.has_value()) {
    return 413;
  } else {
  }
  const Json body = Json::parse(*raw_body, nullptr, false);
  if (body.is_discarded()) {
    return omni_server::BadRequest(connection,
                                   "speech request body must be valid JSON");
  } else if (!body.is_object()) {
    return omni_server::BadRequest(connection,
                                   "speech request body must be a JSON object");
  } else {
  }
  SpeechJob job;
  try {
    job = ParseSpeech(body);
  } catch (const std::invalid_argument &error) {
    return omni_server::BadRequest(connection, error.what());
  }
  // Note (Minyi Tang): no client can know a generated name, and the CUDA
  // speech API ignores model, so only a chosen name is enforced.
  if (has_chosen_model_name() && job.model.has_value() &&
      *job.model != model_name()) {
    return omni_server::WriteJson(connection, 404,
                                  {{"detail", "Unknown model: " + *job.model}});
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
  auto channel = std::make_shared<AudioChannel>();
  const qwen3_asr::CancelFlag cancel = qwen3_asr::NewCancelFlag();
  worker_->Submit(
      [synthesize = job.synthesize, channel,
       deadline](const std::atomic<bool> &cancelled) {
        const auto send = [&channel](std::vector<float> samples) {
          if (!samples.empty()) {
            std::lock_guard<std::mutex> lock(channel->mutex);
            channel->chunks.push_back(std::move(samples));
            channel->ready.notify_all();
          } else {
          }
        };
        // Note (Minyi Tang): the generation checks its own deadline, since
        // the request's thread may be blocked writing to a slow client.
        const auto check_deadline = [deadline]() {
          if (std::chrono::steady_clock::now() >= deadline) {
            throw DeadlineReached();
          } else {
          }
        };
        const SpeechResult result =
            synthesize(SpeechContext{cancelled, send, check_deadline});
        std::lock_guard<std::mutex> lock(channel->mutex);
        channel->result = result;
        return qwen3_asr::TranscriptionResult{};
      },
      cancel,
      [this, channel](std::optional<qwen3_asr::TranscriptionResult>,
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
        channel->ready.notify_all();
      });
  if (job.stream) {
    mg_printf(connection,
              "HTTP/1.1 200 OK\r\nContent-Type: audio/pcm\r\n"
              "X-Request-Id: %s\r\nX-SGLang-Omni-Speech-Id: %s\r\n"
              "X-Sample-Rate: %d\r\nX-Channels: 1\r\nX-Bit-Depth: 16\r\n"
              "Cache-Control: no-cache\r\nConnection: close\r\n\r\n",
              request_id.c_str(), request_id.c_str(), SampleRate());
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
    if (job.stream) {
      const std::string pcm = omni_common::Pcm16Bytes(chunk);
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
  if (job.stream) {
    return 200;
  } else if (timed_out) {
    return omni_server::WriteJson(connection, 504,
                                  {{"detail", "Speech generation timed out."}});
  } else if (channel->error) {
    return omni_server::WriteJson(connection, 500,
                                  {{"detail", "Speech generation failed."}});
  } else {
  }

  const std::string pcm = omni_common::Pcm16Bytes(samples);
  const std::string audio = job.response_format == "wav"
                                ? omni_common::WavBytes(pcm, SampleRate())
                                : pcm;
  const size_t prompt_tokens = channel->result.prompt_token_count;
  const int completion_tokens = channel->result.completion_token_count;
  mg_printf(
      connection,
      "HTTP/1.1 200 OK\r\nContent-Type: audio/%s\r\nContent-Length: %zu\r\n"
      "Content-Disposition: attachment; filename=\"speech.%s\"\r\n"
      "X-Request-Id: %s\r\nX-Finish-Reason: %s\r\n"
      "X-Prompt-Tokens: %zu\r\nX-Completion-Tokens: %d\r\n"
      "X-SGLang-Omni-Input-Tokens: %zu\r\n"
      "X-SGLang-Omni-Output-Tokens: %d\r\n"
      "X-SGLang-Omni-Total-Tokens: %zu\r\nConnection: close\r\n\r\n",
      job.response_format.c_str(), audio.size(), job.response_format.c_str(),
      request_id.c_str(),
      qwen3_asr::FinishReasonName(channel->result.finish_reason), prompt_tokens,
      completion_tokens, prompt_tokens, completion_tokens,
      prompt_tokens + completion_tokens);
  mg_write(connection, audio.data(), audio.size());
  return 200;
}

void AddSpeechFlags(asr_service::ServedKind &kind,
                    std::function<std::unique_ptr<asr_service::ServedModel>(
                        const std::filesystem::path &, int, double)>
                        load) {
  struct Settings {
    int max_queued_requests = kDefaultMaxQueuedRequests;
    double request_timeout_seconds = kDefaultRequestTimeoutSeconds;
  };
  auto settings = std::make_shared<Settings>();
  kind.flags["--max-queued-requests"] = [settings](const std::string &value) {
    settings->max_queued_requests = std::stoi(value);
  };
  kind.flags["--request-timeout-s"] = [settings](const std::string &value) {
    settings->request_timeout_seconds = std::stod(value);
  };
  kind.check_flags = [settings]() {
    if (settings->max_queued_requests < 0) {
      throw std::invalid_argument("--max-queued-requests must not be negative");
    } else if (!(settings->request_timeout_seconds > 0 &&
                 settings->request_timeout_seconds <=
                     kMaxRequestTimeoutSeconds)) {
      throw std::invalid_argument(
          "--request-timeout-s must be above 0 and at most " +
          std::to_string(kMaxRequestTimeoutSeconds));
    } else {
    }
  };
  kind.load = [settings, load](const std::filesystem::path &model_directory) {
    return load(model_directory, settings->max_queued_requests,
                settings->request_timeout_seconds);
  };
}

} // namespace speech_service
