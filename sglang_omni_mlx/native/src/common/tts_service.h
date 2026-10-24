// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <exception>
#include <functional>
#include <future>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <typeinfo>
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

enum class AudioFormat { Wav, Pcm, JsonWav, StreamingPcm };

class AudioResponse {
public:
  AudioResponse(mg_connection *connection, AudioFormat format)
      : connection_(connection), format_(format) {}

  void WritePcm(std::string_view pcm, int sample_rate, int channels);

  template <typename Result, typename Respond>
  int SubmitAndRespond(SynthesisWorker &worker,
                       const std::shared_ptr<SynthesisWorker::Job<Result>> &job,
                       Respond &&respond,
                       const std::string &busy_detail = "TTS service is busy") {
    const auto admission = worker.Submit(job);
    if (admission == SynthesisWorker::Admission::Stopping) {
      return Error(503, "TTS service is stopping");
    } else if (admission == SynthesisWorker::Admission::Busy) {
      return Error(429, busy_detail);
    } else {
    }
    try {
      if constexpr (std::is_void_v<Result>) {
        SynthesisWorker::Wait(connection_, job);
        respond();
      } else {
        respond(SynthesisWorker::Wait(connection_, job));
      }
      return Finish();
    } catch (const SynthesisCancelled &) {
      return Error(503, "Synthesis cancelled");
    } catch (const std::invalid_argument &error) {
      std::cerr << "synthesis failed: " << typeid(error).name() << "\n";
      return Error(400, error.what());
    } catch (const std::exception &error) {
      std::cerr << "synthesis failed: " << typeid(error).name() << "\n";
      return Error(500, "Synthesis failed");
    }
  }

private:
  int Error(int status, const std::string &detail);
  int Finish();
  mg_connection *connection_;
  AudioFormat format_;
  bool headers_sent_ = false;
  int sample_rate_ = 0;
  int channels_ = 0;
};

} // namespace tts_service
