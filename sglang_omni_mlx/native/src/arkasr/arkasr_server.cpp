// SPDX-License-Identifier: Apache-2.0
// Native ARK-ASR server: transcriptions over HTTP (JSON or SSE) with the API
// of qwen3_asr_server, without its realtime API, and Voxt's supervisor
// protocol.
//
//   arkasr_server --model-path DIR [--model-name NAME] [--host H] [--port P]
//     [--draft-model-path DIR [--draft-bits 4|6|8]]
//   arkasr_server --supervised --model-kind arkasr --model-directory DIR
//
// Request fields beside the audio: language (reported back; the model
// detects the language itself), max_new_tokens (one budget for the whole
// request; unset or 0 is the default), stream and
// include_generation_metadata.
#include <memory>
#include <stdexcept>

#include "arkasr/arkasr_transcriber.h"
#include "common/asr_service.h"

namespace {

class ArkModelService : public asr_service::ServedModel {
public:
  ArkModelService(const std::filesystem::path &model_directory,
                  const std::filesystem::path &draft_directory,
                  int draft_quantization_bits)
      : transcriber_(model_directory, draft_directory,
                     draft_quantization_bits) {}

  asr_service::Transcription
  Prepare(std::vector<float> samples,
          const asr_service::FormFields &form) const override {
    arkasr::ArkOptions options;
    options.language = asr_service::TextField(form, "language");
    options.max_new_tokens = asr_service::IntegerField(form, "max_new_tokens");
    if (options.max_new_tokens.value_or(0) < 0) {
      throw std::invalid_argument("max_new_tokens must not be negative");
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
  std::filesystem::path draft_directory;
  int draft_quantization_bits = 0;
  asr_service::ServedKind kind;
  kind.model_kind = "arkasr";
  kind.flags = {
      {"--draft-model-path",
       [&](const std::string &value) { draft_directory = value; }},
      {"--draft-bits",
       [&](const std::string &value) {
         draft_quantization_bits = std::stoi(value);
       }},
  };
  kind.check_flags = [&]() {
    if (draft_quantization_bits != 0 && draft_quantization_bits != 4 &&
        draft_quantization_bits != 6 && draft_quantization_bits != 8) {
      throw std::invalid_argument("--draft-bits must be 4, 6 or 8");
    } else if (draft_quantization_bits != 0 && draft_directory.empty()) {
      throw std::invalid_argument("--draft-bits needs --draft-model-path");
    } else {
    }
  };
  kind.load = [&](const std::filesystem::path &model_directory) {
    return std::make_unique<ArkModelService>(model_directory, draft_directory,
                                             draft_quantization_bits);
  };
  return asr_service::Serve(argc, argv, kind);
}
