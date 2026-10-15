// SPDX-License-Identifier: Apache-2.0
#include "asr_service.h"

#include <signal.h>
#include <unistd.h>

#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <future>
#include <iostream>
#include <mutex>
#include <random>
#include <stdexcept>
#include <thread>
#include <typeinfo>

#include "audio.h"
#include "civetweb.h"
#include "http.h"
#include "nlohmann/json.hpp"
#include "worker.h"

namespace asr_service {

namespace {

namespace mx = mlx::core;
using omni_server::BadRequest;
using omni_server::Json;
using omni_server::ReadBody;
using omni_server::WriteJson;
using qwen3_asr::CancelFlag;
using qwen3_asr::TranscriptionResult;
using qwen3_asr::TranscriptionWorker;

constexpr auto kHeartbeatInterval = std::chrono::milliseconds(250);

struct ServerState {
  TranscriptionWorker *worker = nullptr;
  const ServedModel *model = nullptr;
  std::string model_name;
};

int HandleHealth(mg_connection *connection, void *data) {
  const auto *state = static_cast<ServerState *>(data);
  Json states = Json::object();
  for (const auto &[name, count] : state->model->RequestStates(*state->worker))
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

// Speaker segments, for models that produce them.
void AddSegments(Json &body, const TranscriptionResult &result) {
  if (!result.segments.empty()) {
    body["segments"] = qwen3_asr::SpeakerSegmentsJson(result.segments);
  } else {
  }
}

Json DoneEvent(const TranscriptionResult &result,
               bool include_generation_metadata) {
  Json event = {{"type", "transcript.text.done"}, {"text", result.text}};
  AddSegments(event, result);
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
  if (!omni_server::IsPost(connection)) {
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
  Transcription transcription;
  try {
    transcription = state->model->Prepare(
        qwen3_asr::DecodeWav(form->at("file").value), *form);
  } catch (const std::invalid_argument &error) {
    return BadRequest(connection, error.what());
  } catch (const std::out_of_range &) {
    return BadRequest(connection, "a numeric field is out of range");
  }
  const bool stream = qwen3_asr::FormFlag(TextField(*form, "stream"));
  const bool include_generation_metadata =
      qwen3_asr::FormFlag(TextField(*form, "include_generation_metadata"));
  if (include_generation_metadata && !stream) {
    return BadRequest(connection,
                      "include_generation_metadata requires stream=true");
  } else {
  }
  const CancelFlag cancel = qwen3_asr::NewCancelFlag();
  auto promise = std::make_shared<std::promise<TranscriptionResult>>();
  std::future<TranscriptionResult> future = promise->get_future();
  state->worker->Submit(std::move(transcription), cancel,
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
      const TranscriptionResult result = future.get();
      Json body = {{"text", result.text}};
      AddSegments(body, result);
      return WriteJson(connection, 200, body);
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
  std::shared_ptr<qwen3_asr::RealtimeConnection> session;
  std::string fragments;
};

void SocketReady(mg_connection *connection, void *data) {
  const auto *factory = static_cast<const RealtimeFactory *>(data);
  auto *socket = new SocketState();
  socket->session = (*factory)([connection](const std::string &text) {
    mg_lock_connection(connection);
    const int written = mg_websocket_write(connection, MG_WEBSOCKET_OPCODE_TEXT,
                                           text.data(), text.size());
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
  const ServedKind *kind = nullptr;
  std::string model_path;
  std::string model_name;
  std::string host = "127.0.0.1";
  int port = 0;
  bool supervised = false;
};

// The kind the last --model-kind names, the first by default; nullptr for
// another. Every flag but --supervised takes a value, which is skipped.
const ServedKind *NamedKind(int argc, char **argv,
                            const std::vector<ServedKind> &kinds) {
  std::optional<std::string> name;
  for (int i = 1; i + 1 < argc; ++i) {
    const std::string flag = argv[i];
    if (flag == "--supervised") {
      continue;
    } else if (flag == "--model-kind") {
      name = argv[i + 1];
    } else {
    }
    ++i;
  }
  if (!name.has_value()) {
    return &kinds.front();
  } else {
  }
  for (const ServedKind &kind : kinds) {
    if (kind.model_kind == *name) {
      return &kind;
    } else {
    }
  }
  return nullptr;
}

std::string KindNames(const std::vector<ServedKind> &kinds) {
  std::string names;
  for (size_t i = 0; i < kinds.size(); ++i) {
    names += (i == 0                  ? ""
              : i + 1 == kinds.size() ? " or "
                                      : ", ") +
             kinds[i].model_kind;
  }
  return names;
}

Arguments ParseArguments(int argc, char **argv,
                         const std::vector<ServedKind> &kinds) {
  const ServedKind *named = NamedKind(argc, argv, kinds);
  if (named == nullptr) {
    throw std::invalid_argument(
        kinds.size() == 1
            ? "only --model-kind " + kinds.front().model_kind + " is served"
            : "--model-kind must be " + KindNames(kinds));
  } else {
  }
  const ServedKind &kind = *named;
  Arguments arguments;
  arguments.kind = named;
  for (int i = 1; i < argc; ++i) {
    const std::string flag = argv[i];
    const auto value = [&]() -> std::string {
      if (i + 1 >= argc) {
        throw std::invalid_argument(flag + " needs a value");
      } else {
      }
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
    } else if (flag == "--supervised") {
      arguments.supervised = true;
    } else if (flag == "--model-kind") {
      value(); // Note (khazic): NamedKind already read it.
    } else if (flag == "--startup-timeout-s") {
      value(); // Note (Jiaxin Deng): Voxt passes it; unused here.
    } else if (kind.flags.count(flag) != 0) {
      kind.flags.at(flag)(value());
    } else {
      throw std::invalid_argument("unknown argument " + flag);
    }
  }
  if (arguments.model_path.empty()) {
    throw std::invalid_argument("--model-path is required");
  } else if (kind.check_flags) {
    kind.check_flags();
  } else {
  }
  if (arguments.model_name.empty()) {
    arguments.model_name = "voxt-" + kind.model_kind + "-" + RandomHex(12);
  } else {
  }
  return arguments;
}

class StopSignal {
public:
  void Set(const std::string &reason) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (reason_.empty()) {
      reason_ = reason;
    } else {
    }
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

std::optional<std::string> TextField(const FormFields &form,
                                     const std::string &name) {
  return omni_server::Field(form, name);
}

std::optional<int> IntegerField(const FormFields &form,
                                const std::string &name) {
  const std::optional<std::string> text = TextField(form, name);
  if (!text.has_value() || text->empty()) {
    return std::nullopt;
  } else {
  }
  size_t parsed = 0;
  int value = 0;
  try {
    value = std::stoi(*text, &parsed);
  } catch (const std::logic_error &) {
    // Note (khazic): parsed stays 0, so the check below names the field.
  }
  if (parsed != text->size()) {
    throw std::invalid_argument(name + " must be an integer");
  } else {
  }
  return value;
}

std::optional<float> NumberField(const FormFields &form,
                                 const std::string &name) {
  const std::optional<std::string> text = TextField(form, name);
  if (!text.has_value() || text->empty()) {
    return std::nullopt;
  } else {
  }
  size_t parsed = 0;
  float value = 0;
  try {
    value = std::stof(*text, &parsed);
  } catch (const std::logic_error &) {
    // Note (khazic): parsed stays 0, so the check below names the field.
  }
  if (parsed != text->size() || !std::isfinite(value)) {
    throw std::invalid_argument(name + " must be a finite number");
  } else {
  }
  return value;
}

void AddRealtimeHandler(mg_context *context, const RealtimeFactory &factory) {
  mg_set_websocket_handler(context, "/v1/realtime", nullptr, SocketReady,
                           SocketData, SocketClosed,
                           const_cast<RealtimeFactory *>(&factory));
}

int Serve(int argc, char **argv, const std::vector<ServedKind> &kinds) {
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
    arguments = ParseArguments(argc, argv, kinds);
  } catch (const std::exception &error) {
    std::cerr << argv[0] << ": " << error.what() << "\n";
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

  std::unique_ptr<ServedModel> model;
  std::unique_ptr<TranscriptionWorker> worker;
  std::atomic<bool> loaded(false);
  std::thread loader([&]() {
    try {
      worker = std::make_unique<TranscriptionWorker>(
          [&]() { model = arguments.kind->load(arguments.model_path); });
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
  while (!loaded.load()) {
    const std::string reason = stop.Reason();
    if (!reason.empty()) {
      if (reason == "shutdown") {
        Emit({{"event", "stopped"}});
      } else {
      }
      std::_Exit(0);
    } else {
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  loader.join();

  ServerState state{worker.get(), model.get(), arguments.model_name};
  mg_init_library(0);
  const std::string listening =
      arguments.host + ":" + std::to_string(arguments.port);
  // Note (Jiaxin Deng): each open WebSocket keeps a civetweb worker thread
  // and Voxt opens a VAD stream per stream ID: allow far more than it uses.
  const char *options[] = {"listening_ports",
                           listening.c_str(),
                           "num_threads",
                           "64",
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
  if (model->Transcribes()) {
    mg_set_request_handler(context, "/v1/audio/transcriptions$",
                           HandleTranscriptions, &state);
  } else {
  }
  model->AddHandlers(context, *worker);
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

} // namespace asr_service
