// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <atomic>
#include <condition_variable>
#include <filesystem>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include "nlohmann/json.hpp"

#include "ming_tts_audio.h"
#include "ming_tts_generator.h"
#include "service.h"

struct mg_connection;

namespace ming_tts {

struct SpeechRequest {
  std::string text;
  std::string prompt;
  std::string instructions;
  bool stream;
  bool wav;
  int max_new_tokens;
  float guidance_scale;
  float sigma;
  float temperature;
  static SpeechRequest Parse(const nlohmann::json &request);
};

class MingSpeechService : public omni_server::Service {
public:
  explicit MingSpeechService(const std::filesystem::path &directory);
  ~MingSpeechService() override;
  void Register(mg_context *context, const std::string &model_name) override;
  std::map<std::string, int> RequestStates() const override;
  void CancelAll() override;

private:
  struct Job {
    mg_connection *connection;
    SpeechRequest request;
    std::promise<int> completion;
    bool headers_sent = false;
    std::string audio;
    int sample_rate = 0;
  };
  int HandleSpeech(mg_connection *connection);
  void Run(const std::filesystem::path &directory, std::promise<void> loaded);
  int Generate(Job &job, const MingFrontend &frontend,
               const MingGenerator &generator, const MingAudioDecoder &decoder);
  mutable std::mutex mutex_;
  std::condition_variable wake_;
  std::shared_ptr<Job> job_;
  std::atomic<bool> stopping_{false};
  std::thread thread_;
};

} // namespace ming_tts
