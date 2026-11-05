// SPDX-License-Identifier: Apache-2.0
// Native ARK-ASR server: transcriptions over HTTP (JSON or SSE) with the API
// of qwen3_asr_server, without its realtime API, and Voxt's supervisor
// protocol.
//
//   arkasr_server --model-path DIR [--model-name NAME] [--host H] [--port P]
//   arkasr_server --supervised --model-kind arkasr --model-directory DIR
//
// Request fields beside the audio: language (reported back; the model
// detects the language itself), max_new_tokens (per 30 s window), stream and
// include_generation_metadata.
#include <memory>
#include <stdexcept>

#include "arkasr/arkasr_transcriber.h"
#include "qwen3_asr/asr_service.h"

namespace {

class ArkModelService : public asr_service::ServedModel {
public:
  explicit ArkModelService(const std::filesystem::path &model_directory)
      : transcriber_(model_directory) {}

  asr_service::Transcription
  Prepare(std::vector<float> samples,
          const asr_service::FormFields &form) const override {
    arkasr::ArkOptions options;
    options.language = asr_service::TextField(form, "language");
    options.max_new_tokens = asr_service::IntegerField(form, "max_new_tokens");
    if (options.max_new_tokens.has_value() && *options.max_new_tokens <= 0) {
      throw std::invalid_argument("max_new_tokens must be positive");
    } else {
    }
    return [this, samples = std::move(samples),
            options](const std::atomic<bool> &cancel) {
      return transcriber_.Transcribe(samples, options, cancel);
    };
  }

private:
  arkasr::ArkTranscriber transcriber_;
};

} // namespace

int main(int argc, char **argv) {
  asr_service::ServedKind kind;
  kind.model_kind = "arkasr";
  kind.load = [](const std::filesystem::path &model_directory) {
    return std::make_unique<ArkModelService>(model_directory);
  };
  return asr_service::Serve(argc, argv, kind);
}
