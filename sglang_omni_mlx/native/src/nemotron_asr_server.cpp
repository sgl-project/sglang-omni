// SPDX-License-Identifier: Apache-2.0
// Native Nemotron 3.5 ASR streaming server: asr_service's transcription API
// (JSON or SSE, with sentence segments) and supervisor protocol, plus the
// realtime API on /v1/realtime.
//
//   nemotron_asr_server --model-path DIR [--model-name NAME] [--host H]
//     [--port P]
//   nemotron_asr_server --supervised --model-kind nemotron_asr
//     --model-directory DIR
//
// Request fields beside the audio: language (a prompt dictionary key such as
// en-US; unknown or blank is the model's default), stream and
// include_generation_metadata. A realtime session.update may carry language
// and chunk_ms, the live latency in milliseconds.
#include <memory>

#include "asr_service.h"
#include "nemotron_realtime.h"
#include "nemotron_transcriber.h"

namespace {

class NemotronModelService : public asr_service::ServedModel {
public:
  explicit NemotronModelService(const std::filesystem::path &model_directory)
      : transcriber_(model_directory) {}

  asr_service::Transcription
  Prepare(std::vector<float> samples,
          const asr_service::FormFields &form) const override {
    nemotron::NemotronOptions options;
    options.language = asr_service::TextField(form, "language");
    return [this, samples = std::move(samples),
            options = std::move(options)](const std::atomic<bool> &cancel) {
      return transcriber_.Transcribe(samples, options, cancel);
    };
  }

  void AddHandlers(mg_context *context,
                   qwen3_asr::TranscriptionWorker &worker) override {
    realtime_sessions_ =
        [this, &worker](nemotron::NemotronRealtimeSession::Sender sender) {
          return std::make_shared<nemotron::NemotronRealtimeSession>(
              worker, transcriber_, std::move(sender));
        };
    asr_service::AddRealtimeHandler(context, realtime_sessions_);
  }

private:
  nemotron::NemotronTranscriber transcriber_;
  asr_service::RealtimeFactory realtime_sessions_;
};

} // namespace

int main(int argc, char **argv) {
  asr_service::ServedKind kind;
  kind.model_kind = "nemotron_asr";
  kind.load = [](const std::filesystem::path &model_directory) {
    return std::make_unique<NemotronModelService>(model_directory);
  };
  return asr_service::Serve(argc, argv, kind);
}
