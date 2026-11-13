// SPDX-License-Identifier: Apache-2.0
#include "qwen3_asr/worker.h"

#include <algorithm>

namespace qwen3_asr {

TranscriptionWorker::TranscriptionWorker(
    std::function<void()> load, std::function<void()> release_idle_memory,
    int max_batch)
    : max_batch_(max_batch) {
  std::promise<void> loaded;
  std::future<void> loaded_future = loaded.get_future();
  thread_ = std::thread(&TranscriptionWorker::Run, this, std::move(loaded),
                        std::move(load), std::move(release_idle_memory));
  try {
    loaded_future.get();
  } catch (...) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stopping_ = true;
    }
    wake_.notify_all();
    thread_.join();
    throw;
  }
}

TranscriptionWorker::~TranscriptionWorker() {
  CancelAll();
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stopping_ = true;
  }
  wake_.notify_all();
  if (thread_.joinable()) {
    thread_.join();
  } else {
  }
}

void TranscriptionWorker::Run(std::promise<void> loaded,
                              std::function<void()> load,
                              std::function<void()> release_idle_memory) {
  try {
    load();
    loaded.set_value();
  } catch (...) {
    loaded.set_exception(std::current_exception());
    return;
  }
  while (true) {
    std::vector<Job> jobs;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      if (queue_.empty()) {
        lock.unlock();
        release_idle_memory();
        lock.lock();
      } else {
      }
      wake_.wait(lock, [&] { return stopping_ || !queue_.empty(); });
      if (queue_.empty()) {
        return;
      } else {
      }
      jobs.push_back(std::move(queue_.front()));
      queue_.pop_front();
      // Whatever batchable requests are already waiting decode together;
      // nothing waits for more to arrive.
      if (jobs.front().transcription.target<BatchableTranscription>()) {
        while (static_cast<int>(jobs.size()) < max_batch_ && !queue_.empty() &&
               queue_.front().transcription.target<BatchableTranscription>()) {
          jobs.push_back(std::move(queue_.front()));
          queue_.pop_front();
        }
      } else {
      }
      running_ = true;
      running_count_ = static_cast<int>(jobs.size());
      running_cancels_.clear();
      for (const Job &job : jobs) {
        running_cancels_.push_back(job.cancel);
      }
    }
    std::vector<bool> finished(jobs.size(), false);
    // A request completes the moment it finishes, not when the whole batch
    // does, so a short clip batched with a long one does not wait for it.
    const auto finish = [&](size_t i, std::optional<TranscriptionResult> result,
                            std::exception_ptr error) {
      finished[i] = true;
      {
        std::lock_guard<std::mutex> lock(mutex_);
        --running_count_;
      }
      jobs[i].completion(std::move(result), error);
    };
    if (jobs.size() > 1) {
      // Note (khazic): a request cancelled while queued never starts; the
      // batch checks each flag before it prefills.
      std::vector<BatchRequest> requests;
      const Qwen3ASRTranscriber *transcriber = nullptr;
      for (const Job &job : jobs) {
        const auto *batchable =
            job.transcription.target<BatchableTranscription>();
        transcriber = batchable->transcriber;
        requests.push_back(
            {&batchable->samples, &batchable->options, job.cancel.get()});
      }
      try {
        transcriber->TranscribeBatch(
            requests, [&](size_t i, BatchOutcome outcome) {
              finish(i, std::move(outcome.result), outcome.error);
            });
      } catch (...) {
        for (size_t i = 0; i < jobs.size(); ++i) {
          if (!finished[i]) {
            finish(i, std::nullopt, std::current_exception());
          } else {
          }
        }
      }
    } else {
      std::optional<TranscriptionResult> result;
      std::exception_ptr error;
      try {
        // Note (khazic): a request cancelled while queued never starts.
        if (jobs[0].cancel->load()) {
          throw TranscriptionCancelled();
        } else {
        }
        result = jobs[0].transcription(*jobs[0].cancel);
      } catch (...) {
        error = std::current_exception();
      }
      finish(0, std::move(result), error);
    }
    {
      std::lock_guard<std::mutex> lock(mutex_);
      running_ = false;
      running_count_ = 0;
      running_cancels_.clear();
    }
  }
}

void TranscriptionWorker::Submit(Transcription transcription, CancelFlag cancel,
                                 Completion completion) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    queue_.push_back(
        {std::move(transcription), std::move(cancel), std::move(completion)});
  }
  wake_.notify_one();
}

TranscriptionResult TranscriptionWorker::Transcribe(Transcription transcription,
                                                    CancelFlag cancel) {
  auto promise = std::make_shared<std::promise<TranscriptionResult>>();
  std::future<TranscriptionResult> future = promise->get_future();
  Submit(std::move(transcription), std::move(cancel),
         [promise](std::optional<TranscriptionResult> result,
                   std::exception_ptr error) {
           if (error) {
             promise->set_exception(error);
           } else {
             promise->set_value(std::move(*result));
           }
         });
  return future.get();
}

std::map<std::string, int> TranscriptionWorker::RequestStates() const {
  std::lock_guard<std::mutex> lock(mutex_);
  std::map<std::string, int> states;
  if (running_ && running_count_ > 0) {
    states["running"] = running_count_;
  } else {
  }
  if (!queue_.empty()) {
    states["queued"] = static_cast<int>(queue_.size());
  } else {
  }
  return states;
}

void TranscriptionWorker::CancelAll() {
  std::lock_guard<std::mutex> lock(mutex_);
  for (const Job &job : queue_) {
    job.cancel->store(true);
  }
  for (const CancelFlag &cancel : running_cancels_) {
    cancel->store(true);
  }
}

} // namespace qwen3_asr
