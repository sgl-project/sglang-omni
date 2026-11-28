// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <atomic>
#include <chrono>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>

#include "qwen3_asr/realtime.h"

namespace qwen3_asr::tests {

using PauseRefreshTestDecoder = Qwen3ASRTranscriber::Decoder;

inline std::shared_ptr<RealtimeSession> MakePauseRefreshSession(
    omni_server::TranscriptionWorker &worker, PauseRefreshTestDecoder decoder,
    RealtimeSettings settings, omni_server::RealtimeConnection::Sender sender) {
  static Qwen3ASRTranscriber transcriber;
  transcriber.decoder = std::move(decoder);
  return std::make_shared<RealtimeSession>(worker, transcriber, settings,
                                           std::move(sender));
}

inline constexpr auto kPauseRefreshTestWait = std::chrono::seconds(10);

inline void Require(bool condition, const std::string &message) {
  if (!condition) {
    throw std::runtime_error(message);
  } else {
  }
}

inline void WaitFor(const std::shared_future<void> &event,
                    const std::string &message) {
  Require(event.wait_for(kPauseRefreshTestWait) == std::future_status::ready,
          message);
}

inline void WaitFor(std::future<void> &event, const std::string &message) {
  Require(event.wait_for(kPauseRefreshTestWait) == std::future_status::ready,
          message);
}

class TestSignal {
public:
  TestSignal() : event_(promise_.get_future().share()) {}

  void Notify(std::exception_ptr error = nullptr) {
    std::call_once(notify_once_, [this, error] {
      if (error != nullptr) {
        promise_.set_exception(error);
      } else {
        promise_.set_value();
      }
    });
  }

  void Wait(const std::string &message) const {
    WaitFor(event_, message);
    event_.get();
  }

private:
  std::promise<void> promise_;
  std::shared_future<void> event_;
  std::once_flag notify_once_;
};

class TestGate {
public:
  TestGate() : release_(promise_.get_future().share()) {}
  ~TestGate() { Release(); }

  TestGate(const TestGate &) = delete;
  TestGate &operator=(const TestGate &) = delete;

  std::shared_future<void> Future() const { return release_; }

  void Release() {
    if (!released_) {
      promise_.set_value();
      released_ = true;
    } else {
    }
  }

private:
  std::promise<void> promise_;
  std::shared_future<void> release_;
  bool released_ = false;
};

} // namespace qwen3_asr::tests
