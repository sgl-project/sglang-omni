// SPDX-License-Identifier: Apache-2.0
// Native Parakeet TDT server: transcriptions over HTTP (JSON or SSE) with the
// API of qwen3_asr_server, without its realtime API, and Voxt's supervisor
// protocol.
//
//   parakeet_server --model-path DIR [--model-name NAME] [--host H] [--port P]
//   parakeet_server --supervised --model-kind parakeet --model-directory DIR
//
// Request fields beside the audio: chunk_duration (fixed windows in seconds,
// or 1199 or more for automatic pause-aligned chunks), stream and
// include_generation_metadata. Segments are the transcript's timed sentences.
#include <memory>
#include <optional>

#include "parakeet/parakeet_transcriber.h"
#include "qwen3_asr/asr_service.h"

namespace {

class ParakeetModelService : public asr_service::ServedModel {
public:
  explicit ParakeetModelService(const std::filesystem::path &model_directory)
      : transcriber_(model_directory) {
    mlx::core::set_cache_limit(parakeet::kBufferCacheLimitBytes);
  }

  asr_service::Transcription
  Prepare(std::vector<float> samples,
          const asr_service::FormFields &form) const override {
    const std::optional<float> chunk_duration_seconds =
        asr_service::NumberField(form, "chunk_duration");
    const parakeet::ParakeetOptions options =
        chunk_duration_seconds.has_value()
            ? parakeet::ChunkDurationOptions(*chunk_duration_seconds)
            : parakeet::ParakeetOptions();
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
