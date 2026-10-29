// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <condition_variable>
#include <future>
#include <memory>
#include <mutex>
#include <thread>

#include "common/service.h"
#include "music3/music3_model.h"

struct mg_connection;

namespace music3 {

class SpeechService : public omni_server::Service {
public:
  explicit SpeechService(const std::filesystem::path &directory,
                         bool is_serial_offload_enabled = false);
  ~SpeechService() override;
  void SetModelName(const std::string &model_name, bool chosen) override;
  void Register(mg_context *context, const std::string &model_name) override;
  std::map<std::string, int> RequestStates() const override;
  void CancelAll() override;

private:
  struct Job {
    std::string body;
    Request request;
    std::atomic<bool> cancelled{false};
    std::promise<Result> completion;
  };
  void Run(const std::filesystem::path &directory,
           bool is_serial_offload_enabled, std::promise<void> loaded);
  int Handle(mg_connection *connection);

  std::string model_name_;
  bool has_chosen_model_name_ = false;
  std::unique_ptr<Model> model_;
  mutable std::mutex mutex_;
  std::condition_variable wake_;
  std::shared_ptr<Job> job_;
  bool stopping_ = false;
  std::thread thread_;
};

} // namespace music3
