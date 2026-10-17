// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <atomic>
#include <condition_variable>
#include <filesystem>
#include <future>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>

#include "auk/auk_model.h"
#include "common/service.h"

struct mg_connection;

namespace auk {

class SynthesisCancelled : public std::runtime_error {
public:
  SynthesisCancelled() : std::runtime_error("Synthesis cancelled") {}
};

class SpeechService : public omni_server::Service {
public:
  SpeechService(const std::filesystem::path &model_directory,
                mlx::core::Dtype dtype, float max_seconds);
  ~SpeechService() override;
  void Register(mg_context *context, const std::string &model_name) override;
  std::map<std::string, int> RequestStates() const override;
  void CancelAll() override;

private:
  struct Job {
    SynthesisRequest request;
    std::promise<SynthesisResult> completion;
    std::atomic<bool> cancelled{false};
  };
  int Handle(mg_connection *connection, bool raw_instruction);
  void Run(const std::filesystem::path &model_directory, mlx::core::Dtype dtype,
           std::promise<void> loaded);
  float max_seconds_;
  std::string model_name_;
  mutable std::mutex mutex_;
  std::condition_variable wake_;
  std::shared_ptr<Job> job_;
  std::atomic<bool> stopping_{false};
  std::thread thread_;
};

} // namespace auk
