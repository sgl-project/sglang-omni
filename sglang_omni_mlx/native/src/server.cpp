// SPDX-License-Identifier: Apache-2.0
// Native Qwen3-ASR server with the API of sglang_omni_mlx.qwen3_asr.server.
// --supervised speaks Voxt's supervisor protocol: "ready" or "failed" once
// serving, "stopped" after a shutdown command; end of stdin also stops it.
#include <signal.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <future>
#include <iostream>
#include <limits>
#include <mutex>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>

#include "civetweb.h"
#include "form.h"
#include "nlohmann/json.hpp"
#include "realtime.h"
#include "worker.h"

namespace {

namespace mx = mlx::core;
using qwen3_asr::AudioLayout;
using qwen3_asr::TranscriptionOptions;
using qwen3_asr::TranscriptionResult;
using qwen3_asr::TranscriptionWorker;
using Json = nlohmann::ordered_json;

constexpr auto kHeartbeatInterval = std::chrono::milliseconds(250);

struct ServerState {
  TranscriptionWorker *worker = nullptr;
  std::string model_name;
  qwen3_asr::RealtimeSettings realtime;
};

void WriteResponse(mg_connection *connection, int status,
                   const std::string &reason, const std::string &content_type,
                   const std::string &body) {
  mg_printf(connection,
            "HTTP/1.1 %d %s\r\nContent-Type: %s\r\nContent-Length: "
            "%zu\r\nConnection: close\r\n\r\n",
            status, reason.c_str(), content_type.c_str(), body.size());
  mg_write(connection, body.data(), body.size());
}

int WriteJson(mg_connection *connection, int status, const Json &body) {
  WriteResponse(connection, status,
                status == 200   ? "OK"
                : status == 400 ? "Bad Request"
                : status == 405 ? "Method Not Allowed"
                                : "Internal Server Error",
                "application/json", body.dump());
  return status;
}

int BadRequest(mg_connection *connection, const std::string &detail) {
  return WriteJson(connection, 400, {{"detail", detail}});
}

int HandleHealth(mg_connection *connection, void *data) {
  const auto *state = static_cast<ServerState *>(data);
  Json states = Json::object();
  for (const auto &[name, count] : state->worker->RequestStates())
    states[name] = count;
  return WriteJson(
      connection, 200,
      {{"status", "healthy"}, {"running", true}, {"request_states", states}});
}

int HandleModels(mg_connection *connection, void *data) {
  const auto *state = static_cast<ServerState *>(data);
  return WriteJson(connection, 200,
                   {{"object", "list"},
                    {"data", Json::array({{{"id", state->model_name},
                                           {"object", "model"}}})}});
}

std::string ReadBody(mg_connection *connection) {
  std::string body;
  char buffer[65536];
  int read = 0;
  while ((read = mg_read(connection, buffer, sizeof(buffer))) > 0) {
    body.append(buffer, static_cast<size_t>(read));
  }
  return body;
}

std::optional<std::string>
Field(const std::map<std::string, qwen3_asr::FormField> &form,
      const std::string &name) {
  const auto found = form.find(name);
  if (found == form.end()) {
    return std::nullopt;
  } else {
    return found->second.value;
  }
}

Json DoneEvent(const TranscriptionResult &result,
               bool include_generation_metadata) {
  Json event = {{"type", "transcript.text.done"}, {"text", result.text}};
  if (include_generation_metadata) {
    event["generation_metadata"] = {
        {"generated_token_count", result.generated_token_count},
        {"language",
         result.language.has_value() ? Json(*result.language) : Json()},
        {"finish_reason", qwen3_asr::FinishReasonName(result.finish_reason)}};
  } else {
  }
  return event;
}

const Json &FailureEvent() {
  static const Json event = {{"type", "error"},
                             {"error",
                              {{"type", "server_error"},
                               {"code", "transcription_failed"},
                               {"message", "Transcription failed."}}}};
  return event;
}

bool WriteSse(mg_connection *connection, const std::string &payload) {
  const std::string line = "data: " + payload + "\n\n";
  return mg_write(connection, line.data(), line.size()) > 0;
}

int HandleTranscriptions(mg_connection *connection, void *data) {
  const auto *state = static_cast<ServerState *>(data);
  const mg_request_info *request = mg_get_request_info(connection);
  if (std::strcmp(request->request_method, "POST") != 0) {
    return WriteJson(connection, 405, {{"detail", "Method Not Allowed"}});
  } else {
  }
  const char *content_type = mg_get_header(connection, "Content-Type");
  const auto form = qwen3_asr::ParseMultipartForm(
      content_type ? content_type : "", ReadBody(connection));
  if (!form.has_value() || form->count("file") == 0 ||
      !form->at("file").filename.has_value()) {
    return BadRequest(connection, "file is required");
  } else {
  }
  std::vector<float> samples;
  TranscriptionOptions options;
  try {
    samples = qwen3_asr::DecodeWav(form->at("file").value);
    const auto language = Field(*form, "language");
    options.language = language.has_value()
                           ? qwen3_asr::NormalizeLanguage(*language)
                           : std::nullopt;
    options.context = Field(*form, "prompt");
    const auto max_new_tokens = Field(*form, "max_new_tokens");
    if (max_new_tokens.has_value() && !max_new_tokens->empty()) {
      size_t parsed = 0;
      try {
        options.max_new_tokens = std::stoi(*max_new_tokens, &parsed);
      } catch (const std::invalid_argument &) {
        // Note (Jiaxin Deng): parsed stays 0, so the check below rejects it.
      }
      if (parsed != max_new_tokens->size()) {
        throw std::invalid_argument("max_new_tokens must be an integer");
      } else if (*options.max_new_tokens < 0) {
        throw std::invalid_argument("max_new_tokens must not be negative");
      } else {
      }
    } else {
    }
    options.stop_at_end_of_text =
        qwen3_asr::FormFlag(Field(*form, "stop_at_end_of_text"));
    options.stop_on_token_loop =
        qwen3_asr::FormFlag(Field(*form, "stop_on_token_loop"));
    const std::string layout = Field(*form, "audio_layout").value_or("");
    if (layout.empty() || layout == "reference") {
      options.layout = AudioLayout::kReference;
    } else if (layout == "voxt_swift") {
      options.layout = AudioLayout::kVoxtSwift;
    } else {
      throw std::invalid_argument(
          "audio_layout must be reference or voxt_swift");
    }
  } catch (const std::invalid_argument &error) {
    return BadRequest(connection, error.what());
  } catch (const std::out_of_range &) {
    return BadRequest(connection, "max_new_tokens is out of range");
  }
  const bool stream = qwen3_asr::FormFlag(Field(*form, "stream"));
  const bool include_generation_metadata =
      qwen3_asr::FormFlag(Field(*form, "include_generation_metadata"));
  if (include_generation_metadata && !stream) {
    return BadRequest(connection,
                      "include_generation_metadata requires stream=true");
  } else {
  }
  const qwen3_asr::CancelFlag cancel = qwen3_asr::NewCancelFlag();
  auto promise = std::make_shared<std::promise<TranscriptionResult>>();
  std::future<TranscriptionResult> future = promise->get_future();
  state->worker->Submit(std::move(samples), std::move(options), cancel,
                        [promise](std::optional<TranscriptionResult> result,
                                  std::exception_ptr error) {
                          if (error) {
                            promise->set_exception(error);
                          } else {
                            promise->set_value(std::move(*result));
                          }
                        });
  if (!stream) {
    try {
      return WriteJson(connection, 200, {{"text", future.get().text}});
    } catch (...) {
      return WriteJson(connection, 500, {{"detail", "Transcription failed."}});
    }
  } else {
  }
  mg_printf(connection, "HTTP/1.1 200 OK\r\nContent-Type: "
                        "text/event-stream\r\nCache-Control: no-cache\r\n"
                        "Connection: close\r\n\r\n");
  // Note (Jiaxin Deng): SSE comment heartbeats fail to write once the peer is
  // gone, which cancels the decode it was waiting for.
  while (future.wait_for(kHeartbeatInterval) != std::future_status::ready) {
    if (mg_write(connection, ":\n\n", 3) <= 0) {
      cancel->store(true);
    } else {
    }
  }
  try {
    const TranscriptionResult result = future.get();
    WriteSse(connection, DoneEvent(result, include_generation_metadata).dump());
  } catch (const qwen3_asr::TranscriptionCancelled &) {
    return 200;
  } catch (const std::exception &error) {
    std::cerr << "transcription failed: " << typeid(error).name() << "\n";
    WriteSse(connection, FailureEvent().dump());
  }
  WriteSse(connection, "[DONE]");
  return 200;
}

struct SocketState {
  std::shared_ptr<qwen3_asr::RealtimeSession> session;
  std::string fragments;
};

void SocketReady(mg_connection *connection, void *data) {
  const auto *state = static_cast<ServerState *>(data);
  auto *socket = new SocketState();
  socket->session = std::make_shared<qwen3_asr::RealtimeSession>(
      *state->worker, state->realtime, [connection](const std::string &text) {
        mg_lock_connection(connection);
        const int written = mg_websocket_write(
            connection, MG_WEBSOCKET_OPCODE_TEXT, text.data(), text.size());
        mg_unlock_connection(connection);
        return written > 0;
      });
  mg_set_user_connection_data(connection, socket);
}

int SocketData(mg_connection *connection, int bits, char *data, size_t length,
               void *) {
  auto *socket =
      static_cast<SocketState *>(mg_get_user_connection_data(connection));
  const int opcode = bits & 0x0F;
  if (socket == nullptr || opcode == MG_WEBSOCKET_OPCODE_CONNECTION_CLOSE) {
    return 0;
  } else if (opcode == MG_WEBSOCKET_OPCODE_PING) {
    mg_lock_connection(connection);
    mg_websocket_write(connection, MG_WEBSOCKET_OPCODE_PONG, data, length);
    mg_unlock_connection(connection);
    return 1;
  } else if (opcode == MG_WEBSOCKET_OPCODE_PONG) {
    return 1;
  } else {
  }
  socket->fragments.append(data, length);
  if ((bits & 0x80) == 0) {
    return 1;
  } else {
  }
  const std::string message = std::move(socket->fragments);
  socket->fragments.clear();
  nlohmann::json event;
  try {
    event = nlohmann::json::parse(message);
  } catch (const nlohmann::json::exception &) {
    socket->session->SendError("invalid_request_error", "invalid_json",
                               "Events must be JSON.");
    return 1;
  }
  if (!event.is_object()) {
    return 0;
  } else {
  }
  try {
    return socket->session->Handle(event) ? 1 : 0;
  } catch (const std::exception &error) {
    // Note (Jiaxin Deng): log the type alone, never audio or text.
    std::cerr << "realtime session failed: " << typeid(error).name() << "\n";
    return 0;
  }
}

void SocketClosed(const mg_connection *connection, void *) {
  auto *socket =
      static_cast<SocketState *>(mg_get_user_connection_data(connection));
  if (socket != nullptr) {
    socket->session->Close();
    delete socket;
  } else {
  }
}

std::string RandomHex(int length) {
  std::mt19937_64 generator{std::random_device{}()};
  static constexpr char kHex[] = "0123456789abcdef";
  std::string text;
  for (int i = 0; i < length; ++i)
    text.push_back(kHex[generator() % 16]);
  return text;
}

void Emit(const Json &event) {
  static std::mutex emit_mutex;
  std::lock_guard<std::mutex> lock(emit_mutex);
  std::cout << event.dump() << std::endl;
}

struct Arguments {
  std::string model_path;
  std::string model_name;
  std::string host = "127.0.0.1";
  int port = 0;
  int decode_interval_ms = 1000;
  int first_decode_ms = 100;
  double max_segment_seconds = 30.0;
  bool supervised = false;
};

Arguments ParseArguments(int argc, char **argv) {
  Arguments arguments;
  std::string model_kind = "qwen3_asr";
  for (int i = 1; i < argc; ++i) {
    const std::string flag = argv[i];
    const auto value = [&]() -> std::string {
      if (i + 1 >= argc)
        throw std::invalid_argument(flag + " needs a value");
      return argv[++i];
    };
    if (flag == "--model-path" || flag == "--model-directory") {
      arguments.model_path = value();
    } else if (flag == "--model-name") {
      arguments.model_name = value();
    } else if (flag == "--host") {
      arguments.host = value();
    } else if (flag == "--port") {
      arguments.port = std::stoi(value());
    } else if (flag == "--decode-interval-ms") {
      arguments.decode_interval_ms = std::stoi(value());
    } else if (flag == "--first-decode-ms") {
      arguments.first_decode_ms = std::stoi(value());
    } else if (flag == "--max-segment-seconds") {
      arguments.max_segment_seconds = std::stod(value());
    } else if (flag == "--supervised") {
      arguments.supervised = true;
    } else if (flag == "--model-kind") {
      model_kind = value();
    } else if (flag == "--startup-timeout-s") {
      value(); // Note (Jiaxin Deng): Voxt passes it; unused here.
    } else {
      throw std::invalid_argument("unknown argument " + flag);
    }
  }
  const double max_segment_samples =
      arguments.max_segment_seconds * qwen3_asr::kSampleRate;
  if (model_kind != "qwen3_asr") {
    throw std::invalid_argument("only --model-kind qwen3_asr is served");
  } else if (arguments.model_path.empty()) {
    throw std::invalid_argument("--model-path is required");
  } else if (arguments.decode_interval_ms <= 0) {
    throw std::invalid_argument("--decode-interval-ms must be positive");
  } else if (arguments.first_decode_ms < 0) {
    throw std::invalid_argument("--first-decode-ms must not be negative");
  } else if (!(max_segment_samples >= 1 &&
               max_segment_samples <= std::numeric_limits<int>::max())) {
    throw std::invalid_argument(
        "--max-segment-seconds must span one sample to 134217 s");
  } else {
  }
  if (arguments.model_name.empty()) {
    arguments.model_name = "voxt-qwen3_asr-" + RandomHex(12);
  } else {
  }
  return arguments;
}

class StopSignal {
public:
  void Set(const std::string &reason) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (reason_.empty())
      reason_ = reason;
    stopped_.notify_all();
  }
  std::string Wait() {
    std::unique_lock<std::mutex> lock(mutex_);
    stopped_.wait(lock, [&] { return !reason_.empty(); });
    return reason_;
  }
  std::string Reason() {
    std::lock_guard<std::mutex> lock(mutex_);
    return reason_;
  }

private:
  std::mutex mutex_;
  std::condition_variable stopped_;
  std::string reason_;
};

} // namespace

int main(int argc, char **argv) {
  const auto started = std::chrono::steady_clock::now();
  // Note (Jiaxin Deng): stop signals go to one waiting thread, not to
  // whichever thread happens to run.
  sigset_t stop_signals;
  sigemptyset(&stop_signals);
  for (const int signal_number : {SIGTERM, SIGINT, SIGHUP, SIGQUIT})
    sigaddset(&stop_signals, signal_number);
  pthread_sigmask(SIG_BLOCK, &stop_signals, nullptr);
  signal(SIGPIPE, SIG_IGN);

  Arguments arguments;
  try {
    arguments = ParseArguments(argc, argv);
  } catch (const std::exception &error) {
    std::cerr << "qwen3_asr_server: " << error.what() << "\n";
    return 2;
  }
  StopSignal stop;
  std::thread([&stop, stop_signals]() {
    int signal_number = 0;
    sigwait(&stop_signals, &signal_number);
    stop.Set("signal");
  }).detach();
  if (arguments.supervised) {
    std::thread([&stop]() {
      std::string line;
      while (std::getline(std::cin, line)) {
        try {
          if (nlohmann::json::parse(line).value("command", "") == "shutdown") {
            stop.Set("shutdown");
            return;
          } else {
          }
        } catch (const nlohmann::json::exception &) {
          // Note (Jiaxin Deng): malformed control lines are ignored.
        }
      }
      stop.Set("closed");
    }).detach();
  } else {
  }

  // Note (Jiaxin Deng): freed MLX buffers go back to the system, so an idle
  // server holds only the model.
  mx::set_cache_limit(0);
  std::unique_ptr<TranscriptionWorker> worker;
  std::atomic<bool> loaded(false);
  std::thread loader([&]() {
    try {
      worker = std::make_unique<TranscriptionWorker>(arguments.model_path);
      loaded.store(true);
    } catch (const std::exception &error) {
      if (arguments.supervised) {
        Emit({{"event", "failed"},
              {"reason", std::string("model load failed: ") + error.what()}});
      } else {
        std::cerr << "model load failed: " << error.what() << "\n";
      }
      std::_Exit(1);
    }
  });
  // Note (Jiaxin Deng): a stop before the model is ready ends the process at
  // once.
  while (true) {
    if (loaded.load()) {
      break;
    } else {
    }
    const std::string reason = stop.Reason();
    if (!reason.empty()) {
      if (reason == "shutdown")
        Emit({{"event", "stopped"}});
      std::_Exit(0);
    } else {
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  loader.join();

  ServerState state{worker.get(), arguments.model_name,
                    qwen3_asr::MakeRealtimeSettings(
                        arguments.decode_interval_ms, arguments.first_decode_ms,
                        arguments.max_segment_seconds)};
  mg_init_library(0);
  const std::string listening =
      arguments.host + ":" + std::to_string(arguments.port);
  const char *options[] = {"listening_ports",
                           listening.c_str(),
                           "num_threads",
                           "16",
                           "request_timeout_ms",
                           "3600000",
                           "websocket_timeout_ms",
                           "3600000",
                           nullptr};
  mg_callbacks callbacks{};
  mg_context *context = mg_start(&callbacks, &state, options);
  if (context == nullptr) {
    if (arguments.supervised) {
      Emit({{"event", "failed"}, {"reason", "cannot listen on " + listening}});
    } else {
      std::cerr << "cannot listen on " << listening << "\n";
    }
    return 1;
  } else {
  }
  // Note (Jiaxin Deng): port 0 lets the system pick a free port at bind time.
  mg_server_port server_port{};
  mg_get_server_ports(context, 1, &server_port);
  const std::string endpoint =
      arguments.host + ":" + std::to_string(server_port.port);
  mg_set_request_handler(context, "/health$", HandleHealth, &state);
  mg_set_request_handler(context, "/v1/models$", HandleModels, &state);
  mg_set_request_handler(context, "/v1/audio/transcriptions$",
                         HandleTranscriptions, &state);
  mg_set_websocket_handler(context, "/v1/realtime", nullptr, SocketReady,
                           SocketData, SocketClosed, &state);
  const double startup_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - started)
          .count();
  if (arguments.supervised) {
    Emit({{"event", "ready"},
          {"host", arguments.host},
          {"port", server_port.port},
          {"model_name", arguments.model_name},
          {"server_pid", static_cast<int>(getpid())},
          {"startup_s", std::round(startup_seconds * 1000) / 1000}});
  } else {
    std::cerr << "serving " << arguments.model_name << " on " << endpoint
              << " after " << startup_seconds << " s\n";
  }

  const std::string reason = stop.Wait();
  // Note (Jiaxin Deng): cancel and exit at once; civetweb's own stop waits out
  // its 2 s poll quantum, and the owner has no use for the open responses.
  worker->CancelAll();
  if (reason == "shutdown") {
    Emit({{"event", "stopped"}});
  } else {
  }
  std::_Exit(0);
}
