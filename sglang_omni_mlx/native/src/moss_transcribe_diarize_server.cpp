// SPDX-License-Identifier: Apache-2.0
// Native MOSS-Transcribe-Diarize server: asr_service's transcription API (JSON
// or SSE, with timestamped speaker segments) and supervisor protocol, plus
// the realtime API on /v1/realtime.
//
//   moss_transcribe_diarize_server --model-path DIR [--model-name NAME]
//     [--host H] [--port P]
//   moss_transcribe_diarize_server --supervised
//     --model-kind moss_transcribe_diarize --model-directory DIR
//
// Request fields beside the audio: prompt (the task instruction; blank is the
// model's diarization prompt), max_new_tokens, stop_at_end_of_text,
// stop_on_token_loop, stream and include_generation_metadata.
#include <memory>
#include <stdexcept>

#include "asr_service.h"
#include "moss_realtime.h"
#include "moss_transcriber.h"

namespace {

class MossModelService : public asr_service::ServedModel {
public:
  explicit MossModelService(const std::filesystem::path &model_directory)
      : transcriber_(model_directory) {}

  asr_service::Transcription
  Prepare(std::vector<float> samples,
          const asr_service::FormFields &form) const override {
    moss::MossOptions options;
    options.prompt = asr_service::TextField(form, "prompt");
    moss::ValidatePrompt(options.prompt);
    options.max_new_tokens = asr_service::IntegerField(form, "max_new_tokens");
    if (options.max_new_tokens.value_or(0) < 0) {
      throw std::invalid_argument("max_new_tokens must not be negative");
    } else {
    }
    options.stop_at_end_of_text = qwen3_asr::FormFlag(
        asr_service::TextField(form, "stop_at_end_of_text"));
    options.stop_on_token_loop =
        qwen3_asr::FormFlag(asr_service::TextField(form, "stop_on_token_loop"));
    return [this, samples = std::move(samples),
            options = std::move(options)](const std::atomic<bool> &cancel) {
      return transcriber_.Transcribe(samples, options, cancel);
    };
  }

  void AddHandlers(mg_context *context,
                   qwen3_asr::TranscriptionWorker &worker) override {
    realtime_sessions_ = [this,
                          &worker](moss::MossRealtimeSession::Sender sender) {
      return std::make_shared<moss::MossRealtimeSession>(worker, transcriber_,
                                                         std::move(sender));
    };
    asr_service::AddRealtimeHandler(context, realtime_sessions_);
  }

private:
  moss::MossTranscriber transcriber_;
  asr_service::RealtimeFactory realtime_sessions_;
};

} // namespace

int main(int argc, char **argv) {
  asr_service::ServedKind kind;
  kind.model_kind = "moss_transcribe_diarize";
  kind.load = [](const std::filesystem::path &model_directory) {
    return std::make_unique<MossModelService>(model_directory);
  };
  return asr_service::Serve(argc, argv, kind);
}
