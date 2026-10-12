// SPDX-License-Identifier: Apache-2.0
#include "worker.h"

namespace qwen3_asr {

TranscriptionWorker::TranscriptionWorker(std::function<void()> load) {
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
    Job job;
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
      job = std::move(queue_.front());
      queue_.pop_front();
      running_ = true;
      running_cancel_ = job.cancel;
    }
    std::optional<TranscriptionResult> result;
    std::exception_ptr error;
    try {
      // Note (khazic): a request cancelled while queued never starts.
      if (job.cancel->load()) {
        throw TranscriptionCancelled();
      } else {
      }
      result = job.transcription(*job.cancel);
    } catch (...) {
      error = std::current_exception();
    }
    {
      std::lock_guard<std::mutex> lock(mutex_);
      running_ = false;
      running_cancel_.reset();
    }
    job.completion(std::move(result), error);
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
    states["running"] = 1;
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
  if (running_cancel_) {
    running_cancel_->store(true);
  } else {
  }
}

} // namespace qwen3_asr
