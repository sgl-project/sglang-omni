// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <stdexcept>

#include "common/service.h"
#include "nlohmann/json.hpp"

struct mg_connection;

namespace chat_service {

struct Chunk {
  std::string text;
  std::string audio_base64;
};

struct Result {
  std::string text;
  std::string audio_base64;
  std::string finish_reason;
  int prompt_tokens = 0;
  int completion_tokens = 0;
};

using Cancellation = std::function<bool()>;
using ChunkEmitter = std::function<void(const Chunk &)>;
using Generation =
    std::function<Result(const ChunkEmitter &, const Cancellation &)>;

class Cancelled : public std::runtime_error {
public:
  Cancelled() : std::runtime_error("chat generation cancelled") {}
};

class ServedModel {
public:
  virtual ~ServedModel() = default;
  // Prepare validates without inference; the returned generation runs
  // exclusively.
  virtual Generation Prepare(const nlohmann::json &request,
                             const std::string &model_name) const = 0;
};

class ChatService : public omni_server::Service {
public:
  explicit ChatService(std::unique_ptr<ServedModel> model,
                       size_t max_body_bytes);
  void Register(mg_context *context, const std::string &model_name) override;
  std::map<std::string, int> RequestStates() const override;
  void CancelAll() override;

private:
  int Handle(mg_connection *connection);

  std::unique_ptr<ServedModel> model_;
  size_t max_body_bytes_;
  std::string model_name_;
  std::mutex inference_;
  std::atomic<bool> running_{false};
  std::atomic<bool> stopping_{false};
  std::atomic<uint64_t> request_count_{0};
};

} // namespace chat_service
