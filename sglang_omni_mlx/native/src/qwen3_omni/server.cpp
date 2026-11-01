// SPDX-License-Identifier: Apache-2.0
#include "common/http.h"
#include "qwen3_omni/serving.h"

#include <arpa/inet.h>
#include <cerrno>
#include <csignal>
#include <cstring>
#include <iostream>
#include <libproc.h>
#include <memory>
#include <mutex>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

namespace qwen3_omni {
namespace {
volatile std::sig_atomic_t interrupted = 0;

struct Server {
  Arguments arguments;
  OmniModel model;
  std::timed_mutex inference;
  std::atomic<bool> stopping{false};
  std::atomic<bool> busy{false};
  std::atomic<int> active_socket{-1};
  explicit Server(Arguments arguments)
      : arguments(std::move(arguments)), model(this->arguments.model_path) {}
};
int error_response(mg_connection *connection, int status,
                   const std::string &message) {
  return omni_server::WriteJson(
      connection, status,
      {{"error",
        {{"message", message},
         {"type", status == 400 ? "invalid_request_error" : "server_error"}}}});
}
bool socket_disconnected(int descriptor) {
  char byte;
  int available = recv(descriptor, &byte, 1, MSG_PEEK | MSG_DONTWAIT);
  return available == 0 || (available < 0 && errno != EAGAIN &&
                            errno != EWOULDBLOCK && errno != EINTR);
}
int connection_socket(const mg_request_info &request) {
  int bytes = proc_pidinfo(getpid(), PROC_PIDLISTFDS, 0, nullptr, 0);
  if (bytes <= 0) {
    throw std::runtime_error("cannot inspect own connection socket");
  } else {
  }
  std::vector<proc_fdinfo> descriptors(bytes / sizeof(proc_fdinfo) + 32);
  bytes = proc_pidinfo(getpid(), PROC_PIDLISTFDS, 0, descriptors.data(),
                       descriptors.size() * sizeof(proc_fdinfo));
  for (int i = 0; i < bytes / int(sizeof(proc_fdinfo)); ++i) {
    if (descriptors[i].proc_fdtype != PROX_FDTYPE_SOCKET) {
      continue;
    } else {
    }
    int descriptor = descriptors[i].proc_fd;
    sockaddr_storage peer{}, local{};
    socklen_t peer_size = sizeof(peer), local_size = sizeof(local);
    if (getpeername(descriptor, reinterpret_cast<sockaddr *>(&peer),
                    &peer_size) != 0 ||
        getsockname(descriptor, reinterpret_cast<sockaddr *>(&local),
                    &local_size) != 0) {
      continue;
    } else {
    }
    int peer_port =
        peer.ss_family == AF_INET
            ? ntohs(reinterpret_cast<sockaddr_in *>(&peer)->sin_port)
        : peer.ss_family == AF_INET6
            ? ntohs(reinterpret_cast<sockaddr_in6 *>(&peer)->sin6_port)
            : -1;
    int local_port =
        local.ss_family == AF_INET
            ? ntohs(reinterpret_cast<sockaddr_in *>(&local)->sin_port)
        : local.ss_family == AF_INET6
            ? ntohs(reinterpret_cast<sockaddr_in6 *>(&local)->sin6_port)
            : -1;
    char peer_address[INET6_ADDRSTRLEN]{};
    if (peer.ss_family == AF_INET) {
      inet_ntop(AF_INET, &reinterpret_cast<sockaddr_in *>(&peer)->sin_addr,
                peer_address, sizeof(peer_address));
    } else if (peer.ss_family == AF_INET6) {
      inet_ntop(AF_INET6, &reinterpret_cast<sockaddr_in6 *>(&peer)->sin6_addr,
                peer_address, sizeof(peer_address));
    } else {
    }
    if (peer_port == request.remote_port && local_port == request.server_port &&
        std::strcmp(peer_address, request.remote_addr) == 0) {
      return descriptor;
    } else {
    }
  }
  throw std::runtime_error("cannot find own HTTP connection socket");
}
int route(mg_connection *connection, void *opaque) {
  auto &server = *static_cast<Server *>(opaque);
  auto info = mg_get_request_info(connection);
  std::string uri = info->local_uri ? info->local_uri : "";
  if (uri == "/health" && std::strcmp(info->request_method, "GET") == 0) {
    return omni_server::WriteJson(
        connection, server.stopping ? 503 : 200,
        {{"status", server.stopping ? "stopping" : "healthy"},
         {"running", !server.stopping},
         {"busy", bool(server.busy)},
         {"request_states", omni_server::Json::object()}});
  } else if (uri == "/v1/models" &&
             std::strcmp(info->request_method, "GET") == 0) {
    return omni_server::WriteJson(connection, 200,
                                  {{"object", "list"},
                                   {"data",
                                    {{{"id", server.arguments.model_name},
                                      {"object", "model"},
                                      {"created", 0},
                                      {"owned_by", "sglang_omni_mlx"}}}}});
  } else if (uri != "/v1/chat/completions") {
    return error_response(connection, 404, "unknown route");
  } else if (!omni_server::IsPost(connection)) {
    return error_response(connection, 405, "POST required");
  } else {
  }
  bool streaming_started = false;
  try {
    if (server.stopping) {
      return error_response(connection, 503, "server stopping");
    } else {
    }
    if (info->content_length > 64 * 1024 * 1024) {
      throw std::invalid_argument("request exceeds 64 MiB");
    } else {
    }
    std::string body;
    char buffer[65536];
    int received;
    while ((received = mg_read(connection, buffer, sizeof(buffer))) > 0) {
      if (body.size() + received > 64 * 1024 * 1024) {
        throw std::invalid_argument("request exceeds 64 MiB");
      } else {
        body.append(buffer, received);
      }
    }
    if (received < 0) {
      throw std::invalid_argument("incomplete request body");
    } else {
    }
    auto request = parse_request(Json::parse(body), server.arguments.model_name,
                                 server.arguments.allowed_paths);
    std::unique_lock lock(server.inference, std::try_to_lock);
    if (!lock.owns_lock() && (server.active_socket < 0 ||
                              socket_disconnected(server.active_socket))) {
      lock.try_lock_for(std::chrono::seconds(2));
    } else {
    }
    if (!lock.owns_lock()) {
      return error_response(connection, 429, "model busy");
    } else if (server.stopping) {
      return error_response(connection, 503, "server stopping");
    } else {
    }
    struct BusyGuard {
      std::atomic<bool> &busy;
      std::atomic<int> &active_socket;
      BusyGuard(std::atomic<bool> &busy, std::atomic<int> &active_socket)
          : busy(busy), active_socket(active_socket) {
        busy = true;
      }
      ~BusyGuard() {
        active_socket = -1;
        busy = false;
      }
    } busy(server.busy, server.active_socket);
    int descriptor = connection_socket(*info);
    server.active_socket = descriptor;
    auto cancelled = [&]() {
      if (server.stopping || interrupted) {
        return true;
      } else {
        return socket_disconnected(descriptor);
      }
    };
    auto completion = make_completion(request.model);
    server.model.generate(
        request,
        [&](const GenerationChunk &chunk) {
          check_cancelled(cancelled);
          if (request.stream) {
            if (!streaming_started) {
              if (mg_printf(connection,
                            "HTTP/1.1 200 OK\r\nContent-Type: "
                            "text/event-stream\r\nCache-Control: "
                            "no-cache\r\nConnection: close\r\n\r\n") <= 0) {
                throw Cancelled();
              } else {
                streaming_started = true;
              }
            } else {
            }
            auto payload =
                "data: " +
                streaming_payload(chunk, completion, request.include_usage)
                    .dump() +
                "\n\n";
            if (mg_write(connection, payload.data(), payload.size()) !=
                int(payload.size())) {
              throw Cancelled();
            } else {
            }
          } else {
            completion.append(chunk);
          }
        },
        cancelled);
    if (request.stream) {
      mg_write(connection, "data: [DONE]\n\n", 14);
      return 200;
    } else {
      return omni_server::WriteJson(connection, 200, completion.json());
    }
  } catch (const Cancelled &) {
    return streaming_started
               ? 200
               : error_response(connection, 503, "request cancelled");
  } catch (const Json::exception &error) {
    return streaming_started ? 200
                             : error_response(connection, 400, error.what());
  } catch (const std::invalid_argument &error) {
    return streaming_started ? 200
                             : error_response(connection, 400, error.what());
  } catch (const std::exception &error) {
    std::cerr << "request failed: " << error.what() << '\n';
    return streaming_started ? 200
                             : error_response(connection, 500, error.what());
  }
}
} // namespace
} // namespace qwen3_omni
int main(int count, char **values) {
  using namespace qwen3_omni;
  try {
    auto arguments = parse_arguments(count, values, true);
    if (arguments.version) {
      std::cout << version_json().dump() << '\n';
      return 0;
    } else {
    }
    std::signal(SIGPIPE, SIG_IGN);
    std::signal(SIGINT, [](int) { interrupted = 1; });
    std::signal(SIGTERM, [](int) { interrupted = 1; });
    mx::set_default_device(mx::Device::gpu);
    Server server(arguments);
    std::string listening =
        arguments.host + ":" + std::to_string(arguments.port);
    const char *options[] = {"listening_ports",
                             listening.c_str(),
                             "num_threads",
                             "4",
                             "enable_keep_alive",
                             "no",
                             "request_timeout_ms",
                             "30000",
                             nullptr};
    mg_callbacks callbacks{};
    auto stop_listener = [&server](mg_context *context) {
      server.stopping = true;
      mg_stop(context);
    };
    std::unique_ptr<mg_context, decltype(stop_listener)> context(
        mg_start(&callbacks, &server, options), stop_listener);
    if (!context) {
      throw std::runtime_error("cannot start HTTP listener");
    } else {
    }
    for (const auto *path : {"/health", "/v1/models", "/v1/chat/completions"}) {
      mg_set_request_handler(context.get(), path, route, &server);
    }
    mg_server_port ports[1];
    mg_get_server_ports(context.get(), 1, ports);
    if (arguments.supervised) {
      std::cout << Json({{"event", "ready"},
                         {"host", arguments.host},
                         {"port", ports[0].port},
                         {"server_pid", getpid()},
                         {"model_name", arguments.model_name}})
                       .dump()
                << std::endl;
    } else {
      std::cerr << "listening on " << arguments.host << ':' << ports[0].port
                << '\n';
    }
    while (!interrupted && !server.stopping) {
      pollfd input{STDIN_FILENO, POLLIN, 0};
      int ready = poll(arguments.supervised ? &input : nullptr,
                       arguments.supervised ? 1 : 0, 100);
      if (arguments.supervised && ready > 0) {
        std::string line;
        if (!std::getline(std::cin, line)) {
          server.stopping = true;
        } else {
          auto command = Json::parse(line, nullptr, false);
          if (command.is_object() && command.contains("command") &&
              command.at("command").is_string() &&
              command.at("command") == "shutdown") {
            server.stopping = true;
          } else {
            std::cerr << "ignored invalid supervisor command\n";
          }
        }
      } else {
      }
    }
    context.reset();
    mx::synchronize();
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
