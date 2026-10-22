// SPDX-License-Identifier: Apache-2.0
#include "common/tts_service.h"

#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <optional>
#include <random>
#include <stdexcept>
#include <thread>
#include <utility>

#include "common/http.h"
#include "common/supervisor.h"

namespace omni_server {

namespace {

constexpr int kMaximumNewTokens = 4096;
constexpr int kQueueRetryAfterSeconds = 1;
constexpr int kMaximumQueuedRequests = 2;
constexpr std::chrono::milliseconds kDisconnectCheckInterval{250};
constexpr std::chrono::milliseconds kStartupPollInterval{20};
constexpr char kProcessingResponse[] = "HTTP/1.1 102 Processing\r\n\r\n";

struct ServerArguments {
  std::filesystem::path model_directory;
  std::string model_name;
  std::string host = "127.0.0.1";
  int port = 8000;
  bool supervised = false;
};

ServerArguments ParseArguments(int argument_count, char **argument_values,
                               const std::string &model_kind) {
  ServerArguments arguments;
  for (int argument_index = 1; argument_index < argument_count;
       ++argument_index) {
    const std::string flag = argument_values[argument_index];
    const auto read_value = [&]() {
      if (argument_index + 1 >= argument_count) {
        throw std::invalid_argument(flag + " needs a value");
      } else {
      }
      return std::string(argument_values[++argument_index]);
    };
    if (flag == "--model-path" || flag == "--model-directory") {
      arguments.model_directory = read_value();
    } else if (flag == "--model-name") {
      arguments.model_name = read_value();
    } else if (flag == "--host") {
      arguments.host = read_value();
    } else if (flag == "--port") {
      arguments.port = std::stoi(read_value());
    } else if (flag == "--supervised") {
      arguments.supervised = true;
    } else if (flag == "--model-kind") {
      if (read_value() != model_kind) {
        throw std::invalid_argument("only --model-kind " + model_kind +
                                    " is served");
      } else {
      }
    } else if (flag == "--startup-timeout-s") {
      read_value();
    } else {
      throw std::invalid_argument("unknown argument " + flag);
    }
  }
  if (arguments.model_directory.empty()) {
    throw std::invalid_argument("--model-path is required");
  } else if (arguments.port < 0 || arguments.port > 65535) {
    throw std::invalid_argument("--port must be between 0 and 65535");
  } else {
  }
  if (arguments.model_name.empty()) {
    arguments.model_name =
        std::string("voxt-") + model_kind + "-" + RandomHex(12);
  } else {
  }
  return arguments;
}

void ReportFailure(bool supervised, const std::string &reason) {
  if (supervised) {
    EmitControlEvent({{"event", "failed"}, {"reason", reason}});
  } else {
    std::cerr << reason << "\n";
  }
}

} // namespace

TTSService::TTSService(std::string default_voice, std::set<std::string> voices,
                       SynthesisWorker &worker)
    : default_voice_(std::move(default_voice)), voices_(std::move(voices)),
      worker_(worker) {}

void TTSService::Register(mg_context *context) {
  mg_set_request_handler(context, "/v1/audio/speech$", HandleSpeech, this);
}

int TTSService::HandleSpeech(mg_connection *connection, void *data) {
  auto *service = static_cast<TTSService *>(data);
  return service->Synthesize(connection);
}

int TTSService::Synthesize(mg_connection *connection) {
  if (!IsPost(connection)) {
    return WriteJson(connection, 405, {{"detail", "Method Not Allowed"}});
  } else {
  }
  Json body;
  try {
    body = Json::parse(ReadBody(connection));
  } catch (const Json::exception &) {
    return BadRequest(connection, "request body must be valid JSON");
  }
  if (!body.is_object()) {
    return BadRequest(connection, "request body must be a JSON object");
  } else {
  }
  const auto text_value = body.find("input");
  if (text_value == body.end() || !text_value->is_string()) {
    return BadRequest(connection, "input must be a string");
  } else {
  }
  const std::string text = text_value->get<std::string>();
  if (text.empty() ||
      std::all_of(text.begin(), text.end(), [](unsigned char character) {
        return std::isspace(character) != 0;
      })) {
    return BadRequest(connection,
                      "input must contain non-whitespace characters");
  } else {
  }
  const auto model_value = body.find("model");
  if (model_value != body.end() && !model_value->is_string()) {
    return BadRequest(connection, "model must be a string");
  } else {
  }
  const auto voice_value = body.find("voice");
  if (voice_value != body.end() && !voice_value->is_string()) {
    return BadRequest(connection, "voice must be a string");
  } else {
  }
  const std::string voice = voice_value == body.end()
                                ? default_voice_
                                : voice_value->get<std::string>();
  if (!voices_.contains(voice)) {
    return BadRequest(connection, "unknown voice: " + voice);
  } else {
  }
  const auto format_value = body.find("response_format");
  if (format_value != body.end() &&
      (!format_value->is_string() ||
       format_value->get<std::string>() != "wav")) {
    return BadRequest(connection, "response_format must be wav");
  } else {
  }
  int max_new_tokens = kMaximumNewTokens;
  const auto max_tokens_value = body.find("max_new_tokens");
  if (max_tokens_value != body.end()) {
    if (!max_tokens_value->is_number_integer()) {
      return BadRequest(connection, "max_new_tokens must be an integer");
    } else {
    }
    if (max_tokens_value->is_number_unsigned()) {
      const uint64_t requested_max_new_tokens =
          max_tokens_value->get<uint64_t>();
      if (requested_max_new_tokens < 1 ||
          requested_max_new_tokens > kMaximumNewTokens) {
        return BadRequest(connection,
                          "max_new_tokens is outside the supported range");
      } else {
      }
      max_new_tokens = static_cast<int>(requested_max_new_tokens);
    } else {
      const int64_t requested_max_new_tokens = max_tokens_value->get<int64_t>();
      if (requested_max_new_tokens < 1 ||
          requested_max_new_tokens > kMaximumNewTokens) {
        return BadRequest(connection,
                          "max_new_tokens is outside the supported range");
      } else {
      }
      max_new_tokens = static_cast<int>(requested_max_new_tokens);
    }
  } else {
  }
  uint64_t seed = std::mt19937_64(std::random_device{}())();
  const auto seed_value = body.find("seed");
  if (seed_value != body.end()) {
    if (!seed_value->is_number_unsigned() &&
        (!seed_value->is_number_integer() || seed_value->get<int64_t>() < 0)) {
      return BadRequest(connection, "seed must be a nonnegative integer");
    } else {
    }
    seed = seed_value->get<uint64_t>();
  } else {
  }
  bool heartbeat = false;
  const auto heartbeat_value = body.find("heartbeat");
  if (heartbeat_value != body.end()) {
    if (!heartbeat_value->is_boolean()) {
      return BadRequest(connection, "heartbeat must be a boolean");
    } else {
    }
    heartbeat = heartbeat_value->get<bool>();
  } else {
  }

  SynthesisTask task;
  try {
    task = worker_.Submit({text, voice, max_new_tokens, seed});
  } catch (const std::length_error &) {
    const std::string body =
        Json{{"detail", "TTS request queue is full"}}.dump();
    mg_printf(connection,
              "HTTP/1.1 429 Too Many Requests\r\n"
              "Content-Type: application/json\r\n"
              "Retry-After: %d\r\n"
              "Content-Length: %zu\r\n"
              "Connection: close\r\n\r\n",
              kQueueRetryAfterSeconds, body.size());
    mg_write(connection, body.data(), body.size());
    return 429;
  } catch (const std::exception &error) {
    std::cerr << "cannot enqueue TTS request: " << error.what() << "\n";
    return WriteJson(connection, 503,
                     {{"detail", "TTS service is unavailable"}});
  }
  bool disconnected = false;
  if (heartbeat) {
    while (task.result.wait_for(kDisconnectCheckInterval) !=
           std::future_status::ready) {
      if (mg_write(connection, kProcessingResponse,
                   sizeof(kProcessingResponse) - 1) <= 0) {
        task.cancelled->store(true);
        disconnected = true;
        break;
      } else {
      }
    }
  }
  if (disconnected) {
    task.result.wait();
    return 200;
  } else {
  }
  try {
    const SynthesisResult result = task.result.get();
    const std::string finish_reason =
        result.finish_reason == "length" ? "length" : "stop";
    mg_printf(connection,
              "HTTP/1.1 200 OK\r\nContent-Type: audio/wav\r\n"
              "Content-Disposition: attachment; filename=\"speech.wav\"\r\n"
              "X-Speech-Finish-Reason: %s\r\nContent-Length: %zu\r\n"
              "Connection: close\r\n\r\n",
              finish_reason.c_str(), result.wav_bytes.size());
    mg_write(connection, result.wav_bytes.data(), result.wav_bytes.size());
    return 200;
  } catch (const std::exception &error) {
    std::cerr << "TTS request failed: " << error.what() << "\n";
    return WriteJson(connection, 500,
                     {{"detail", "TTS speech generation failed"}});
  }
}

int ServeTTS(int argument_count, char **argument_values, const TTSKind &kind) {
  const auto started = std::chrono::steady_clock::now();
  BlockTerminationSignals();
  IgnoreBrokenPipe();
  ServerArguments arguments;
  try {
    arguments =
        ParseArguments(argument_count, argument_values, kind.model_kind);
  } catch (const std::exception &error) {
    std::cerr << argument_values[0] << ": " << error.what() << "\n";
    return 2;
  }

  StopSignal stop;
  std::thread(WaitForTerminationSignal, std::ref(stop)).detach();
  if (arguments.supervised) {
    std::thread(WatchShutdownCommands, std::ref(stop)).detach();
  } else {
  }

  std::optional<TTSModel> model;
  SynthesisWorker worker(
      [&model](const SynthesisRequest &request,
               const std::atomic<bool> &cancelled) {
        return model->synthesize(request, cancelled);
      },
      [&model, &kind, &arguments]() {
        model = kind.load(arguments.model_directory);
        if (model->voices.empty() ||
            !model->voices.contains(model->default_voice) ||
            !model->synthesize) {
          throw std::runtime_error(
              "model loader returned incomplete TTS model configuration");
        } else {
        }
      },
      kMaximumQueuedRequests);
  try {
    while (!worker.WaitUntilReady(kStartupPollInterval)) {
      const std::string reason = stop.Reason();
      if (!reason.empty()) {
        if (reason == "shutdown") {
          EmitControlEvent({{"event", "stopped"}});
        } else {
        }
        std::_Exit(0);
      } else {
      }
    }
  } catch (const std::exception &error) {
    ReportFailure(arguments.supervised,
                  std::string("model load failed: ") + error.what());
    return 1;
  }

  InfoRoutes info_routes(arguments.model_name,
                         [&worker]() { return worker.RequestStates(); });
  TTSService speech_service(model->default_voice, model->voices, worker);
  HttpServer http_server = StartHttpServer(arguments.host, arguments.port);
  if (http_server.context == nullptr) {
    ReportFailure(arguments.supervised, "cannot listen on " + arguments.host +
                                            ":" +
                                            std::to_string(arguments.port));
    return 1;
  } else {
  }
  info_routes.Register(http_server.context);
  speech_service.Register(http_server.context);
  const double startup_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - started)
          .count();
  if (arguments.supervised) {
    EmitControlEvent(
        {{"event", "ready"},
         {"host", arguments.host},
         {"port", http_server.port},
         {"model_name", arguments.model_name},
         {"server_pid", static_cast<int>(getpid())},
         {"startup_s", std::round(startup_seconds * 1000) / 1000}});
  } else {
    std::cerr << "serving " << arguments.model_name << " on " << arguments.host
              << ":" << http_server.port << " after " << startup_seconds
              << " s\n";
  }

  const std::string reason = stop.Wait();
  worker.CancelAll();
  if (reason == "shutdown") {
    EmitControlEvent({{"event", "stopped"}});
  } else {
  }
  std::_Exit(0);
}

} // namespace omni_server
