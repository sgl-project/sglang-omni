// SPDX-License-Identifier: Apache-2.0
#include "worker.h"

namespace qwen3_asr {

TranscriptionWorker::TranscriptionWorker(
    const std::filesystem::path &model_directory) {
  std::promise<void> loaded;
  std::future<void> loaded_future = loaded.get_future();
  thread_ = std::thread(&TranscriptionWorker::Run, this, std::move(loaded),
                        model_directory);
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
                              std::filesystem::path model_directory) {
  try {
    transcriber_ = std::make_unique<Qwen3ASRTranscriber>(model_directory);
    loaded.set_value();
  } catch (...) {
    loaded.set_exception(std::current_exception());
    return;
  }
  while (true) {
    Job job;
    {
      std::unique_lock<std::mutex> lock(mutex_);
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
      result = transcriber_->Transcribe(job.samples, job.options, *job.cancel);
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

void TranscriptionWorker::Submit(std::vector<float> samples,
                                 TranscriptionOptions options,
                                 CancelFlag cancel, Completion completion) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    queue_.push_back({std::move(samples), std::move(options), std::move(cancel),
                      std::move(completion)});
  }
  wake_.notify_one();
}

TranscriptionResult
TranscriptionWorker::Transcribe(std::vector<float> samples,
                                TranscriptionOptions options,
                                CancelFlag cancel) {
  auto promise = std::make_shared<std::promise<TranscriptionResult>>();
  std::future<TranscriptionResult> future = promise->get_future();
  Submit(std::move(samples), std::move(options), std::move(cancel),
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
