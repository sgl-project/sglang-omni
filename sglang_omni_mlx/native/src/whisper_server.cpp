// SPDX-License-Identifier: Apache-2.0
// Native Whisper server: asr_service's transcription API (JSON or SSE) and
// supervisor protocol, without qwen3_asr_server's realtime API.
//
//   whisper_server --model-path DIR [--model-name NAME] [--host H] [--port P]
//   whisper_server --supervised --model-kind whisper --model-directory DIR
//
// Request fields beside the audio: language (a code in the checkpoint's
// tokenizer, or one of the English names Swift's WhisperTokenizer maps; any
// other value sends no language token), max_new_tokens, temperature, stream
// and include_generation_metadata.
#include <stdexcept>

#include "asr_service.h"
#include "whisper_transcriber.h"

namespace {

class WhisperModelService : public asr_service::ServedModel {
public:
  explicit WhisperModelService(const std::filesystem::path &model_directory)
      : transcriber_(model_directory) {}

  asr_service::Transcription
  Prepare(std::vector<float> samples,
          const asr_service::FormFields &form) const override {
    whisper::WhisperOptions options;
    options.language =
        asr_service::TextField(form, "language").value_or(options.language);
    options.max_new_tokens = asr_service::IntegerField(form, "max_new_tokens")
                                 .value_or(options.max_new_tokens);
    options.temperature = asr_service::NumberField(form, "temperature")
                              .value_or(options.temperature);
    if (options.max_new_tokens < 0) {
      throw std::invalid_argument("max_new_tokens must be nonnegative");
    } else if (options.temperature < 0) {
      throw std::invalid_argument("temperature must be nonnegative");
    } else {
    }
    return [this, samples = std::move(samples),
            options](const std::atomic<bool> &cancel) {
      return transcriber_.Transcribe(samples, options, cancel);
    };
  }

private:
  whisper::WhisperTranscriber transcriber_;
};

} // namespace

int main(int argc, char **argv) {
  asr_service::ServedKind kind;
  kind.model_kind = "whisper";
  kind.load = [](const std::filesystem::path &model_directory) {
    return std::make_unique<WhisperModelService>(model_directory);
  };
  return asr_service::Serve(argc, argv, kind);
}
