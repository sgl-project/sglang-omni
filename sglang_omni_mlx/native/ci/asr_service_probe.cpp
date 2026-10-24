// SPDX-License-Identifier: Apache-2.0
#include "qwen3_asr/asr_service.h"

#include <stdexcept>

#include "common/http.h"

namespace {

class ServiceProbe : public asr_service::ServedModel {
public:
  asr_service::Transcription
  Prepare(std::vector<float>, const asr_service::FormFields &) const override {
    throw std::logic_error("service probe serves no transcriptions");
  }

  bool Transcribes() const override { return false; }

  void AddHandlers(mg_context *context,
                   qwen3_asr::TranscriptionWorker &) override {
    registration_ = {{"model_name", model_name()},
                     {"has_chosen_model_name", has_chosen_model_name()}};
    mg_set_request_handler(
        context, "/service-name$",
        [](mg_connection *connection, void *model) {
          return omni_server::WriteJson(
              connection, 200,
              static_cast<const ServiceProbe *>(model)->registration_);
        },
        this);
  }

private:
  omni_server::Json registration_;
};

} // namespace

int main(int argc, char **argv) {
  return asr_service::Serve(argc, argv,
                            {"service_probe",
                             [](const std::filesystem::path &) {
                               return std::make_unique<ServiceProbe>();
                             },
                             {},
                             {}});
}
