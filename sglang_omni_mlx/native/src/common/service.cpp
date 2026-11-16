// SPDX-License-Identifier: Apache-2.0
#include "common/service.h"

#include <signal.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <optional>
#include <random>
#include <stdexcept>
#include <thread>

#include "common/http.h"

namespace omni_server {

namespace {

struct ServerState {
  Service *service;
  std::string model_name;
};

int HandleHealth(mg_connection *connection, void *data) {
  const auto *state = static_cast<ServerState *>(data);
  Json states = Json::object();
  for (const auto &[name, count] : state->service->RequestStates())
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
  bool has_chosen_model_name = false;
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
  arguments.has_chosen_model_name = !arguments.model_name.empty();
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

  std::unique_ptr<Service> service;
  std::atomic<bool> loaded(false);
  std::thread loader([&]() {
    try {
      service = arguments.kind->load(arguments.model_path);
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

  ServerState state{service.get(), arguments.model_name};
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
  service->SetModelName(arguments.model_name, arguments.has_chosen_model_name);
  service->Register(context, arguments.model_name);
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
  service->CancelAll();
  if (reason == "shutdown") {
    Emit({{"event", "stopped"}});
  } else {
  }
  std::_Exit(0);
}

} // namespace omni_server
