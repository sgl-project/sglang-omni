// SPDX-License-Identifier: Apache-2.0
#include "common/supervisor.h"

#include <pthread.h>
#include <signal.h>

#include <iostream>
#include <random>

namespace omni_server {

void StopSignal::Set(const std::string &reason) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (reason_.empty()) {
    reason_ = reason;
  } else {
  }
  stopped_.notify_all();
}

std::string StopSignal::Wait() {
  std::unique_lock<std::mutex> lock(mutex_);
  stopped_.wait(lock, [&] { return !reason_.empty(); });
  return reason_;
}

std::string StopSignal::Reason() {
  std::lock_guard<std::mutex> lock(mutex_);
  return reason_;
}

void BlockTerminationSignals() {
  sigset_t stop_signals;
  sigemptyset(&stop_signals);
  for (const int signal_number : {SIGTERM, SIGINT, SIGHUP, SIGQUIT}) {
    sigaddset(&stop_signals, signal_number);
  }
  pthread_sigmask(SIG_BLOCK, &stop_signals, nullptr);
}

void IgnoreBrokenPipe() { signal(SIGPIPE, SIG_IGN); }

void WaitForTerminationSignal(StopSignal &stop) {
  sigset_t stop_signals;
  sigemptyset(&stop_signals);
  for (const int signal_number : {SIGTERM, SIGINT, SIGHUP, SIGQUIT}) {
    sigaddset(&stop_signals, signal_number);
  }
  int signal_number = 0;
  sigwait(&stop_signals, &signal_number);
  stop.Set("signal");
}

void WatchShutdownCommands(StopSignal &stop) {
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
}

std::string RandomHex(int character_count) {
  std::mt19937_64 generator{std::random_device{}()};
  static constexpr char kHex[] = "0123456789abcdef";
  std::string text;
  for (int index = 0; index < character_count; ++index) {
    text.push_back(kHex[generator() % 16]);
  }
  return text;
}

void EmitControlEvent(const Json &event) {
  static std::mutex emit_mutex;
  std::lock_guard<std::mutex> lock(emit_mutex);
  std::cout << event.dump() << std::endl;
}

} // namespace omni_server
