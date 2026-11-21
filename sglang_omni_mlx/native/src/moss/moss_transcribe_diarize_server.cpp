// SPDX-License-Identifier: Apache-2.0
// Native MOSS-Transcribe-Diarize server: asr_service's transcription API (JSON
// or SSE, with timestamped speaker segments) and supervisor protocol, plus
// the realtime API on /v1/realtime.
//
//   moss_transcribe_diarize_server --model-path DIR [--model-name NAME]
//     [--host H] [--port P] [--max-batch N]
//   moss_transcribe_diarize_server --supervised
//     --model-kind moss_transcribe_diarize --model-directory DIR
//
// Request fields beside the audio: prompt (the task instruction; blank is the
// model's diarization prompt), max_new_tokens, stop_at_end_of_text,
// stop_on_token_loop, stream and include_generation_metadata.
#include <memory>
#include <stdexcept>

#include "common/asr_service.h"
#include "moss/moss_realtime.h"
#include "moss/moss_transcriber.h"

namespace {

class MossModelService : public asr_service::ServedModel {
public:
  MossModelService(const std::filesystem::path &model_directory,
                   const std::filesystem::path &draft_directory,
                   int draft_quantization_bits)
      : transcriber_(model_directory, draft_directory,
                     draft_quantization_bits) {
    mlx::core::set_cache_limit(asr_service::kDecodeBufferCacheLimitBytes);
  }

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
    options.stop_at_end_of_text = omni_common::FormFlag(
        asr_service::TextField(form, "stop_at_end_of_text"));
    options.stop_on_token_loop = omni_common::FormFlag(
        asr_service::TextField(form, "stop_on_token_loop"));
    if (transcriber_.CanBatch(samples, options)) {
      return moss::BatchableTranscription{&transcriber_, std::move(samples),
                                          std::move(options)};
    } else {
    }
    return [this, samples = std::move(samples),
            options = std::move(options)](const std::atomic<bool> &cancel) {
      return transcriber_.Transcribe(samples, options, cancel);
    };
  }

  void AddHandlers(mg_context *context,
                   omni_server::TranscriptionWorker &worker) override {
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
  std::filesystem::path draft_directory;
  int draft_quantization_bits = 0;
  int max_batch = 1;
  asr_service::ServedKind kind;
  kind.model_kind = "moss_transcribe_diarize";
  kind.flags = {
      {"--draft-model-path",
       [&](const std::string &value) { draft_directory = value; }},
      {"--draft-bits",
       [&](const std::string &value) {
         draft_quantization_bits = std::stoi(value);
       }},
      {"--max-batch",
       [&](const std::string &value) { max_batch = std::stoi(value); }},
  };
  kind.check_flags = [&]() {
    if (draft_quantization_bits != 0 && draft_quantization_bits != 4 &&
        draft_quantization_bits != 6 && draft_quantization_bits != 8) {
      throw std::invalid_argument("--draft-bits must be 4, 6 or 8");
    } else if (draft_quantization_bits != 0 && draft_directory.empty()) {
      throw std::invalid_argument("--draft-bits needs --draft-model-path");
    } else if (max_batch < 1) {
      throw std::invalid_argument("--max-batch must be at least 1");
    } else {
    }
  };
  kind.max_batch = [&]() { return max_batch; };
  kind.batch_runner = moss::BatchableTranscriptionRunner();
  kind.load = [&](const std::filesystem::path &model_directory) {
    return std::make_unique<MossModelService>(model_directory, draft_directory,
                                              draft_quantization_bits);
  };
  return asr_service::Serve(argc, argv, kind);
}
