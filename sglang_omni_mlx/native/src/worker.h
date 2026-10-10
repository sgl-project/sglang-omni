// SPDX-License-Identifier: Apache-2.0
// Runs every transcription on one thread, the thread that loaded the model.
#pragma once

#include <atomic>
#include <condition_variable>
#include <deque>
#include <exception>
#include <functional>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>

#include "transcriber.h"

namespace qwen3_asr {

using CancelFlag = std::shared_ptr<std::atomic<bool>>;

inline CancelFlag NewCancelFlag() {
  return std::make_shared<std::atomic<bool>>(false);
}

// One transcription bound to its audio and options; runs on the worker
// thread and stops with TranscriptionCancelled once its flag is set.
using Transcription =
    std::function<TranscriptionResult(const std::atomic<bool> &)>;

// Serializes requests: one model, one MLX stream, one request at a time.
class TranscriptionWorker {
public:
  // Called on the worker thread with the result, or with the exception the
  // transcription raised (TranscriptionCancelled included).
  using Completion = std::function<void(std::optional<TranscriptionResult>,
                                        std::exception_ptr)>;

  // Runs load on the worker thread; throws what load throws.
  explicit TranscriptionWorker(std::function<void()> load);
  ~TranscriptionWorker();
  TranscriptionWorker(const TranscriptionWorker &) = delete;
  TranscriptionWorker &operator=(const TranscriptionWorker &) = delete;

  void Submit(Transcription transcription, CancelFlag cancel,
              Completion completion);
  TranscriptionResult Transcribe(Transcription transcription,
                                 CancelFlag cancel);
  // Requests waiting for the worker and running on it; empty when idle.
  std::map<std::string, int> RequestStates() const;
  // Cancels everything queued or running, as on shutdown.
  void CancelAll();

private:
  struct Job {
    Transcription transcription;
    CancelFlag cancel;
    Completion completion;
  };

  void Run(std::promise<void> loaded, std::function<void()> load);

  mutable std::mutex mutex_;
  std::condition_variable wake_;
  std::deque<Job> queue_;
  CancelFlag running_cancel_;
  bool running_ = false;
  bool stopping_ = false;
  std::thread thread_;
};

} // namespace qwen3_asr
