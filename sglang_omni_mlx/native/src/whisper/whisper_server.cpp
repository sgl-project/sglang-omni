// SPDX-License-Identifier: Apache-2.0
// Native Whisper server: asr_service's transcription API (JSON or SSE) and
// supervisor protocol, without qwen3_asr_server's realtime API.
//
//   whisper_server --model-path DIR [--model-name NAME] [--host H] [--port P]
//     [--draft-model-path DIR [--draft-steps N]]
//   whisper_server --supervised --model-kind whisper --model-directory DIR
//
// Request fields beside the audio: language (a code in the checkpoint's
// tokenizer, or one of the English names Swift's WhisperTokenizer maps; a
// missing or unknown value detects the language), max_new_tokens, temperature,
// stream and include_generation_metadata.
#include <optional>
#include <stdexcept>

#include "qwen3_asr/asr_service.h"
#include "whisper/whisper_transcriber.h"

namespace {

class WhisperModelService : public asr_service::ServedModel {
public:
  explicit WhisperModelService(const std::filesystem::path &model_directory,
                               const std::filesystem::path &draft_directory,
                               int draft_steps)
      : transcriber_(model_directory, draft_directory, draft_steps) {}

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
  std::filesystem::path draft_directory;
  std::optional<int> draft_steps;
  asr_service::ServedKind kind;
  kind.flags = {{"--draft-model-path",
                 [&](const std::string &value) { draft_directory = value; }},
                {"--draft-steps", [&](const std::string &value) {
                   draft_steps = std::stoi(value);
                 }}};
  kind.check_flags = [&]() {
    if (draft_steps.has_value() && *draft_steps < 1) {
      throw std::invalid_argument("--draft-steps must be at least 1");
    } else if (draft_steps.has_value() && draft_directory.empty()) {
      throw std::invalid_argument("--draft-steps needs --draft-model-path");
    } else {
    }
  };
  kind.model_kind = "whisper";
  kind.load = [&](const std::filesystem::path &model_directory) {
    return std::make_unique<WhisperModelService>(
        model_directory, draft_directory,
        draft_steps.value_or(whisper::kDefaultDraftSteps));
  };
  return asr_service::Serve(argc, argv, kind);
}
