// SPDX-License-Identifier: Apache-2.0
#include "qwen3_asr/worker.h"

#include <algorithm>
#include <cstdlib>

namespace qwen3_asr {

TranscriptionWorker::TranscriptionWorker(std::function<void()> load) {
  if (const char *value = std::getenv("QWEN3_ASR_MAX_BATCH")) {
    max_batch_ = std::max(1, std::atoi(value));
  } else {
  }
  std::promise<void> loaded;
  std::future<void> loaded_future = loaded.get_future();
  thread_ = std::thread(&TranscriptionWorker::Run, this, std::move(loaded),
                        std::move(load));
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
                              std::function<void()> load) {
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
      // Note (khazic): a decode keeps MLX's buffer cache, since freeing every
      // step's buffers slows each token; an idle server returns it, so it
      // holds only the model. The synchronize lets the step a decode queued
      // before it stopped return its buffers first.
      if (queue_.empty()) {
        lock.unlock();
        mlx::core::synchronize();
        mlx::core::clear_cache();
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
    std::vector<std::optional<TranscriptionResult>> results(jobs.size());
    std::vector<std::exception_ptr> errors(jobs.size());
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
        std::vector<BatchOutcome> outcomes =
            transcriber->TranscribeBatch(requests);
        for (size_t i = 0; i < jobs.size(); ++i) {
          results[i] = std::move(outcomes[i].result);
          errors[i] = outcomes[i].error;
        }
      } catch (...) {
        std::fill(errors.begin(), errors.end(), std::current_exception());
      }
    } else {
      try {
        // Note (khazic): a request cancelled while queued never starts.
        if (jobs[0].cancel->load()) {
          throw TranscriptionCancelled();
        } else {
        }
        results[0] = jobs[0].transcription(*jobs[0].cancel);
      } catch (...) {
        errors[0] = std::current_exception();
      }
    }
    {
      std::lock_guard<std::mutex> lock(mutex_);
      running_ = false;
      running_count_ = 0;
      running_cancels_.clear();
    }
    for (size_t i = 0; i < jobs.size(); ++i) {
      jobs[i].completion(std::move(results[i]), errors[i]);
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
  if (running_) {
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
