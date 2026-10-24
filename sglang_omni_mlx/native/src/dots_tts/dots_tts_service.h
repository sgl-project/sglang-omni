// SPDX-License-Identifier: Apache-2.0
// The dots.tts TTS service for the native runtime: a JSON POST /v1/audio/speech
// (OpenAI-style) returning WAV bytes. Request counts are read by asr_service's
// /health handler.
//
// note (guozhihao-224): MLX models must be loaded and run on one thread, so
// every synthesis runs on a dedicated worker thread that owns the model; each
// civetweb handler enqueues a job and waits for it while polling its client for
// disconnects, the same way the Voxtral TTS server does.
#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <exception>
#include <filesystem>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>

#include "civetweb.h"

namespace dots_tts {

inline constexpr int kMaxRequestTimeoutSeconds = 24 * 60 * 60;

class DotsTtsService {
public:
  DotsTtsService(const std::filesystem::path &model_directory,
                 int max_queued_requests, double request_timeout_seconds);
  ~DotsTtsService();

  void Register(mg_context *context);
  std::map<std::string, int> RequestStates() const;
  std::pair<int, std::string> Synthesize(mg_connection *connection,
                                         const std::string &body);

  void RequestStarted();
  void RequestFinished();

private:
  struct Job {
    std::string body;
    std::string wav;
    std::exception_ptr error;
    std::shared_ptr<std::atomic<bool>> cancelled;
    bool finished = false;
  };
  void RunWorker();

  std::thread worker_;
  std::promise<void> loaded_;
  std::mutex jobs_mutex_;
  std::condition_variable jobs_cv_;
  std::deque<std::shared_ptr<Job>> jobs_;
  bool shutdown_ = false;
  std::filesystem::path model_directory_;
  const int max_queued_requests_;
  const std::chrono::steady_clock::duration request_timeout_;
  mutable std::mutex admission_mutex_;
  int admitted_requests_ = 0;
  mutable std::mutex states_mutex_;
  int running_requests_ = 0;
  int completed_requests_ = 0;
};

} // namespace dots_tts
