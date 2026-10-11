// SPDX-License-Identifier: Apache-2.0
// Native Parakeet TDT server: transcriptions over HTTP (JSON or SSE) with the
// API of qwen3_asr_server, without its realtime API, and Voxt's supervisor
// protocol.
//
//   parakeet_server --model-path DIR [--model-name NAME] [--host H] [--port P]
//   parakeet_server --supervised --model-kind parakeet --model-directory DIR
//
// Request fields beside the audio: chunk_duration (the chunk length Voxt
// asks for, in seconds), stream and include_generation_metadata. Segments
// are the transcript's timed sentences.
#include <memory>
#include <stdexcept>

#include "asr_service.h"
#include "parakeet_transcriber.h"

namespace {

// Note (Jiaxin Deng): chunks overlap by 1 s, so a shorter request advances a
// sample at a time, and the encoder's attention grows with the square of a
// longer one; 1199 s or more asks for 5 s chunks.
constexpr double kMinChunkDurationSeconds = 2.0;
constexpr double kMaxChunkDurationSeconds = 300.0;
constexpr double kWholeRecordingChunkDurationSeconds = 1199.0;

class ParakeetModelService : public asr_service::ServedModel {
public:
  explicit ParakeetModelService(const std::filesystem::path &model_directory)
      : transcriber_(model_directory) {}

  asr_service::Transcription
  Prepare(std::vector<float> samples,
          const asr_service::FormFields &form) const override {
    parakeet::ParakeetOptions options;
    options.chunk_duration_seconds =
        asr_service::NumberField(form, "chunk_duration")
            .value_or(options.chunk_duration_seconds);
    if (options.chunk_duration_seconds < kMinChunkDurationSeconds ||
        (options.chunk_duration_seconds > kMaxChunkDurationSeconds &&
         options.chunk_duration_seconds <
             kWholeRecordingChunkDurationSeconds)) {
      throw std::invalid_argument(
          "chunk_duration must be 2 to 300 seconds, or 1199 or more for a "
          "whole recording");
    } else {
    }
    return [this, samples = std::move(samples),
            options](const std::atomic<bool> &cancel) {
      return transcriber_.Transcribe(samples, options, cancel);
    };
  }

private:
  parakeet::ParakeetTranscriber transcriber_;
};

} // namespace

int main(int argc, char **argv) {
  asr_service::ServedKind kind;
  kind.model_kind = "parakeet";
  kind.load = [](const std::filesystem::path &model_directory) {
    return std::make_unique<ParakeetModelService>(model_directory);
  };
  return asr_service::Serve(argc, argv, kind);
}
