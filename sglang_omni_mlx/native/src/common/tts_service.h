// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <exception>
#include <functional>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <variant>

#include "common/civetweb_connection.h"

namespace tts_service {

class SynthesisCancelled : public std::runtime_error {
public:
  SynthesisCancelled() : std::runtime_error("Synthesis cancelled") {}
};

class SynthesisWorker {
  struct Task {
    virtual ~Task() = default;
    virtual void Synthesize(const std::function<void()> &check_cancelled) = 0;
    virtual void Complete(std::exception_ptr error) = 0;
    std::atomic<bool> cancelled{false};
  };

public:
  template <typename Result> struct Job final : Task {
    std::function<Result(const std::function<void()> &)> synthesize;
    std::promise<Result> completion;

  private:
    void Synthesize(const std::function<void()> &check_cancelled) override {
      if constexpr (std::is_void_v<Result>) {
        synthesize(check_cancelled);
      } else {
        result_.emplace(synthesize(check_cancelled));
      }
    }

    void Complete(std::exception_ptr error) override {
      if (error) {
        completion.set_exception(error);
      } else if constexpr (std::is_void_v<Result>) {
        completion.set_value();
      } else {
        completion.set_value(std::move(*result_));
      }
    }

    // Note (altale): Publish the result only after worker cleanup completes.
    std::optional<
        std::conditional_t<std::is_void_v<Result>, std::monostate, Result>>
        result_;
  };
  enum class Admission { Accepted, Busy, Stopping };

  SynthesisWorker(std::function<void()> load, std::function<void()> unload);
  ~SynthesisWorker();
  template <typename Result>
  Admission Submit(const std::shared_ptr<Job<Result>> &job) {
    return SubmitTask(job);
  }

  template <typename Result>
  static Result Wait(mg_connection *connection,
                     const std::shared_ptr<Job<Result>> &job) {
    constexpr auto kConnectionPollInterval = std::chrono::milliseconds(100);
    auto future = job->completion.get_future();
    while (future.wait_for(kConnectionPollInterval) ==
           std::future_status::timeout) {
      if (omni_connection_closed(connection)) {
        job->cancelled.store(true);
        break;
      } else {
      }
    }
    return future.get();
  }
  std::map<std::string, int> RequestStates() const;
  void Stop();

private:
  Admission SubmitTask(const std::shared_ptr<Task> &job);
  void Run(std::function<void()> load, std::function<void()> unload,
           std::promise<void> loaded);
  mutable std::mutex mutex_;
  std::condition_variable wake_;
  std::shared_ptr<Task> job_;
  bool stopping_ = false;
  std::thread thread_;
};

} // namespace tts_service
