// SPDX-License-Identifier: Apache-2.0
#include "common/synthesis.h"

#include <stdexcept>
#include <utility>

namespace omni_server {

SynthesisWorker::SynthesisWorker(SynthesisFunction synthesize,
                                 SynthesisInitializer initialize,
                                 int max_queued_requests)
    : synthesize_(std::move(synthesize)), initialize_(std::move(initialize)),
      max_queued_requests_(max_queued_requests), thread_() {
  if (!synthesize_ || !initialize_ || max_queued_requests_ < 0) {
    throw std::invalid_argument(
        "TTS synthesis worker needs an initializer, synthesizer, and "
        "nonnegative queue limit");
  } else {
  }
  startup_future_ = startup_result_.get_future();
  thread_ = std::thread(&SynthesisWorker::Run, this);
}

SynthesisWorker::~SynthesisWorker() {
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

SynthesisTask SynthesisWorker::Submit(SynthesisRequest request) {
  auto job = std::make_unique<Job>();
  job->request = std::move(request);
  job->cancelled = std::make_shared<std::atomic<bool>>(false);
  std::shared_ptr<std::atomic<bool>> cancelled = job->cancelled;
  std::future<SynthesisResult> result = job->result.get_future();
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (stopping_) {
      throw std::runtime_error("TTS synthesis worker is shutting down");
    } else if (queue_.size() >= static_cast<size_t>(max_queued_requests_)) {
      throw std::length_error("TTS synthesis queue is full");
    } else {
    }
    queue_.push_back(std::move(job));
  }
  wake_.notify_one();
  return {std::move(result), std::move(cancelled)};
}

void SynthesisWorker::WaitUntilReady() { startup_future_.get(); }

bool SynthesisWorker::WaitUntilReady(std::chrono::milliseconds timeout) {
  if (startup_future_.wait_for(timeout) != std::future_status::ready) {
    return false;
  } else {
  }
  startup_future_.get();
  return true;
}

std::map<std::string, int> SynthesisWorker::RequestStates() const {
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

void SynthesisWorker::CancelAll() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (running_cancelled_ != nullptr) {
    running_cancelled_->store(true);
  } else {
  }
  for (const std::unique_ptr<Job> &job : queue_) {
    job->cancelled->store(true);
  }
}

void SynthesisWorker::Run() {
  try {
    initialize_();
    startup_result_.set_value();
  } catch (...) {
    startup_result_.set_exception(std::current_exception());
    return;
  }
  while (true) {
    std::unique_ptr<Job> job;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      wake_.wait(lock, [this]() { return stopping_ || !queue_.empty(); });
      if (stopping_ && queue_.empty()) {
        return;
      } else {
      }
      job = std::move(queue_.front());
      queue_.pop_front();
      running_ = true;
      running_cancelled_ = job->cancelled;
    }
    try {
      if (job->cancelled->load()) {
        throw std::runtime_error("TTS request cancelled");
      } else {
      }
      job->result.set_value(synthesize_(job->request, *job->cancelled));
    } catch (...) {
      job->result.set_exception(std::current_exception());
    }
    {
      std::lock_guard<std::mutex> lock(mutex_);
      running_ = false;
      running_cancelled_.reset();
    }
  }
}

} // namespace omni_server
