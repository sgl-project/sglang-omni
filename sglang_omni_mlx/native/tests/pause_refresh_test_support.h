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
  TestSignal() = default;

  void Notify(std::exception_ptr error = nullptr) const {
    const std::shared_ptr<State> state = state_;
    std::call_once(state->notify_once, [state, error] {
      if (error != nullptr) {
        state->promise.set_exception(error);
      } else {
        state->promise.set_value();
      }
    });
  }

  void Wait(const std::string &message) const {
    const std::shared_ptr<State> state = state_;
    WaitFor(state->event, message);
    state->event.get();
  }

private:
  struct State {
    State() : event(promise.get_future().share()) {}

    std::promise<void> promise;
    std::shared_future<void> event;
    std::once_flag notify_once;
  };

  std::shared_ptr<State> state_ = std::make_shared<State>();
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
