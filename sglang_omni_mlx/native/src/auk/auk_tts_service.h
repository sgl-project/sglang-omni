// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <filesystem>
#include <memory>

#include "auk/auk_model.h"
#include "common/service.h"
#include "common/tts_service.h"

struct mg_connection;

namespace auk {

class SpeechService : public omni_server::Service {
public:
  SpeechService(const std::filesystem::path &model_directory,
                mlx::core::Dtype dtype, float max_seconds);
  void Register(mg_context *context, const std::string &model_name) override;
  std::map<std::string, int> RequestStates() const override;
  void CancelAll() override;

private:
  int Handle(mg_connection *connection, bool raw_instruction);
  float max_seconds_;
  std::string model_name_;
  std::unique_ptr<Model> model_;
  tts_service::SynthesisWorker worker_;
};

} // namespace auk
