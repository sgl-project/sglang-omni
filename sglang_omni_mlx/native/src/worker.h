// SPDX-License-Identifier: Apache-2.0
// Runs every transcription on one thread, the thread that loaded the model.
#pragma once

#include <atomic>
#include <condition_variable>
#include <deque>
#include <exception>
#include <filesystem>
#include <functional>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "transcriber.h"

namespace qwen3_asr {

using CancelFlag = std::shared_ptr<std::atomic<bool>>;

inline CancelFlag NewCancelFlag() {
  return std::make_shared<std::atomic<bool>>(false);
}

// Serializes requests: one model, one MLX stream, one request at a time.
class TranscriptionWorker {
public:
  // Called on the worker thread with the result, or with the exception the
  // transcription raised (TranscriptionCancelled included).
  using Completion = std::function<void(std::optional<TranscriptionResult>,
                                        std::exception_ptr)>;

  // Loads the model on the worker thread; throws what loading throws.
  explicit TranscriptionWorker(const std::filesystem::path &model_directory);
  ~TranscriptionWorker();
  TranscriptionWorker(const TranscriptionWorker &) = delete;
  TranscriptionWorker &operator=(const TranscriptionWorker &) = delete;

  void Submit(std::vector<float> samples, TranscriptionOptions options,
              CancelFlag cancel, Completion completion);
  TranscriptionResult Transcribe(std::vector<float> samples,
                                 TranscriptionOptions options,
                                 CancelFlag cancel);
  // Requests waiting for the worker and running on it; empty when idle.
  std::map<std::string, int> RequestStates() const;
  // Cancels everything queued or running, as on shutdown.
  void CancelAll();

  const Qwen3ASRTranscriber &transcriber() const { return *transcriber_; }

private:
  struct Job {
    std::vector<float> samples;
    TranscriptionOptions options;
    CancelFlag cancel;
    Completion completion;
  };

  void Run(std::promise<void> loaded, std::filesystem::path model_directory);

  std::unique_ptr<Qwen3ASRTranscriber> transcriber_;
  mutable std::mutex mutex_;
  std::condition_variable wake_;
  std::deque<Job> queue_;
  CancelFlag running_cancel_;
  bool running_ = false;
  bool stopping_ = false;
  std::thread thread_;
};

} // namespace qwen3_asr
