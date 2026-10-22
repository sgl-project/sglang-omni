// SPDX-License-Identifier: Apache-2.0
// Bounded serial worker for native TTS synthesis.
#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <exception>
#include <functional>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

namespace omni_server {

struct SynthesisRequest {
  std::string text;
  std::string voice;
  int max_new_tokens;
  uint64_t seed;
};

struct SynthesisResult {
  std::string wav_bytes;
  std::string finish_reason;
};

struct SynthesisTask {
  std::future<SynthesisResult> result;
  std::shared_ptr<std::atomic<bool>> cancelled;
};

using SynthesisFunction = std::function<SynthesisResult(
    const SynthesisRequest &, const std::atomic<bool> &)>;
using SynthesisInitializer = std::function<void()>;

class SynthesisWorker {
public:
  SynthesisWorker(SynthesisFunction synthesize, SynthesisInitializer initialize,
                  int max_queued_requests);
  ~SynthesisWorker();
  SynthesisWorker(const SynthesisWorker &) = delete;
  SynthesisWorker &operator=(const SynthesisWorker &) = delete;

  SynthesisTask Submit(SynthesisRequest request);
  void WaitUntilReady();
  bool WaitUntilReady(std::chrono::milliseconds timeout);
  std::map<std::string, int> RequestStates() const;
  void CancelAll();

private:
  struct Job {
    SynthesisRequest request;
    std::shared_ptr<std::atomic<bool>> cancelled;
    std::promise<SynthesisResult> result;
  };

  void Run();

  SynthesisFunction synthesize_;
  SynthesisInitializer initialize_;
  const int max_queued_requests_;
  std::promise<void> startup_result_;
  std::future<void> startup_future_;
  mutable std::mutex mutex_;
  std::condition_variable wake_;
  std::deque<std::unique_ptr<Job>> queue_;
  std::shared_ptr<std::atomic<bool>> running_cancelled_;
  bool running_ = false;
  bool stopping_ = false;
  std::thread thread_;
};

} // namespace omni_server
