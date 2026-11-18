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
#include <vector>

#include "common/transcription.h"

namespace omni_server {

using CancelFlag = std::shared_ptr<std::atomic<bool>>;

inline CancelFlag NewCancelFlag() {
  return std::make_shared<std::atomic<bool>>(false);
}

// One transcription bound to its audio and options; runs on the worker
// thread and stops with GenerationCancelled once its flag is set.
using Transcription =
    std::function<omni_common::TranscriptionResult(const std::atomic<bool> &)>;

// Called on the worker thread with the result, or with the exception the
// transcription raised (GenerationCancelled included).
using Completion = std::function<void(
    std::optional<omni_common::TranscriptionResult>, std::exception_ptr)>;

// One queued request of a batch: its transcription and its cancel flag.
struct BatchItem {
  const Transcription *transcription;
  const std::atomic<bool> *cancel;
};

// Reports the batch item at an index the moment it finishes.
using BatchCompletion =
    std::function<void(size_t, std::optional<omni_common::TranscriptionResult>,
                       std::exception_ptr)>;

// How a model decodes queued requests together: which transcriptions can
// join a batch, and the decode of a batch of them.
struct BatchRunner {
  std::function<bool(const Transcription &)> accepts;
  std::function<void(const std::vector<BatchItem> &, const BatchCompletion &)>
      run;
};

// Runs requests in queue order on one model and one MLX stream. Each runs
// alone unless max_batch is above 1 and a batch runner is given, in which case
// consecutive queued requests it accepts are decoded together.
class TranscriptionWorker {
public:
  // Runs load on the worker thread; throws what load throws. Up to max_batch
  // queued requests batch_runner accepts decode together.
  TranscriptionWorker(std::function<void()> load,
                      std::function<void()> release_idle_memory,
                      int max_batch = 1,
                      std::optional<BatchRunner> batch_runner = std::nullopt);
  ~TranscriptionWorker();
  TranscriptionWorker(const TranscriptionWorker &) = delete;
  TranscriptionWorker &operator=(const TranscriptionWorker &) = delete;

  void Submit(Transcription transcription, CancelFlag cancel,
              Completion completion);
  omni_common::TranscriptionResult Transcribe(Transcription transcription,
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

  void Run(std::promise<void> loaded, std::function<void()> load,
           std::function<void()> release_idle_memory);

  mutable std::mutex mutex_;
  std::condition_variable wake_;
  std::deque<Job> queue_;
  std::vector<CancelFlag> running_cancels_;
  // Most queued batchable requests decoded together. Batched output can differ
  // from serial on borderline tokens, so the default of 1 keeps every request
  // serial and deterministic.
  int max_batch_;
  std::optional<BatchRunner> batch_runner_;
  int running_count_ = 0;
  bool running_ = false;
  bool stopping_ = false;
  std::thread thread_;
};

} // namespace omni_server
