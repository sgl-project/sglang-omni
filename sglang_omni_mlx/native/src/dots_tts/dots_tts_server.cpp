// SPDX-License-Identifier: Apache-2.0
#include "dots_tts/dots_tts_service.h"

#include <filesystem>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "qwen3_asr/asr_service.h"

namespace {

class DotsTtsModelService : public asr_service::ServedModel {
public:
  DotsTtsModelService(const std::filesystem::path &model_directory,
                      int max_queued_requests, double request_timeout_seconds)
      : service_(model_directory, max_queued_requests,
                 request_timeout_seconds) {}

  asr_service::Transcription
  Prepare(std::vector<float>, const asr_service::FormFields &) const override {
    throw std::logic_error("dots_tts serves no transcriptions");
  }
  bool Transcribes() const override { return false; }
  void AddHandlers(mg_context *context,
                   qwen3_asr::TranscriptionWorker &) override {
    service_.Register(context);
  }
  std::map<std::string, int>
  RequestStates(const qwen3_asr::TranscriptionWorker &) const override {
    return service_.RequestStates();
  }

private:
  dots_tts::DotsTtsService service_;
};

} // namespace

int main(int argc, char **argv) {
  int max_queued_requests = 2;
  double request_timeout_seconds = 600;
  asr_service::ServedKind kind;
  kind.model_kind = "dots_tts";
  kind.flags = {
      {"--max-queued-requests",
       [&](const std::string &value) {
         max_queued_requests = std::stoi(value);
       }},
      {"--request-timeout-s",
       [&](const std::string &value) {
         request_timeout_seconds = std::stod(value);
       }},
  };
  kind.check_flags = [&]() {
    if (max_queued_requests < 0) {
      throw std::invalid_argument("--max-queued-requests must not be negative");
    }
    if (!(request_timeout_seconds > 0 &&
          request_timeout_seconds <= dots_tts::kMaxRequestTimeoutSeconds)) {
      throw std::invalid_argument(
          "--request-timeout-s must be above 0 and at most " +
          std::to_string(dots_tts::kMaxRequestTimeoutSeconds));
    }
  };
  kind.load = [&](const std::filesystem::path &model_directory) {
    return std::make_unique<DotsTtsModelService>(
        model_directory, max_queued_requests, request_timeout_seconds);
  };
  return asr_service::Serve(argc, argv, kind);
}
