// SPDX-License-Identifier: Apache-2.0
// Native Cohere Transcribe server: transcriptions over HTTP (JSON or SSE) with
// the API of qwen3_asr_server, without its realtime API, and Voxt's supervisor
// protocol.
//
//   cohere_transcribe_server --model-path DIR [--model-name NAME] [--host H]
//     [--port P]
//   cohere_transcribe_server --supervised --model-kind cohere_transcribe
//     --model-directory DIR
//
// Request fields beside the audio: language (an ISO code or an English name),
// use_punctuation, max_new_tokens, temperature, chunk_duration,
// min_chunk_duration, stream and include_generation_metadata. With
// vad_model_directory (a Silero VAD checkpoint), long audio is cut at speech
// by vad_threshold, vad_min_speech_ms, vad_min_silence_ms, vad_speech_pad_ms,
// vad_merge_gap_seconds and vad_max_chunk_seconds, all required then.
#include <cmath>
#include <memory>
#include <stdexcept>

#include "asr_service.h"
#include "cohere_transcriber.h"
#include "speech_segments.h"

namespace {

template <typename Value>
Value Required(const std::optional<Value> &value, const std::string &name) {
  if (!value.has_value()) {
    throw std::invalid_argument(name + " is required with vad_model_directory");
  } else {
  }
  return *value;
}

class CohereModelService : public asr_service::ServedModel {
public:
  explicit CohereModelService(const std::filesystem::path &model_directory)
      : transcriber_(model_directory) {}

  asr_service::Transcription
  Prepare(std::vector<float> samples,
          const asr_service::FormFields &form) const override {
    cohere_transcribe::CohereOptions options;
    options.language =
        asr_service::TextField(form, "language").value_or(options.language);
    const auto use_punctuation =
        asr_service::TextField(form, "use_punctuation");
    if (use_punctuation.has_value()) {
      options.use_punctuation = qwen3_asr::FormFlag(use_punctuation);
    } else {
    }
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
    options.chunk_duration_seconds =
        asr_service::NumberField(form, "chunk_duration")
            .value_or(options.chunk_duration_seconds);
    options.min_chunk_duration_seconds =
        asr_service::NumberField(form, "min_chunk_duration")
            .value_or(options.min_chunk_duration_seconds);
    if (!std::isfinite(options.chunk_duration_seconds) ||
        options.chunk_duration_seconds <= 0 ||
        !std::isfinite(options.min_chunk_duration_seconds) ||
        options.min_chunk_duration_seconds < 0) {
      throw std::invalid_argument(
          "chunk durations must be finite and nonnegative");
    }
    const std::optional<std::string> vad_directory =
        asr_service::TextField(form, "vad_model_directory");
    if (vad_directory.has_value()) {
      options.speech_segments = {
          Required(asr_service::NumberField(form, "vad_threshold"),
                   "vad_threshold"),
          Required(asr_service::IntegerField(form, "vad_min_speech_ms"),
                   "vad_min_speech_ms"),
          Required(asr_service::IntegerField(form, "vad_min_silence_ms"),
                   "vad_min_silence_ms"),
          Required(asr_service::IntegerField(form, "vad_speech_pad_ms"),
                   "vad_speech_pad_ms"),
          Required(asr_service::NumberField(form, "vad_merge_gap_seconds"),
                   "vad_merge_gap_seconds"),
          Required(asr_service::NumberField(form, "vad_max_chunk_seconds"),
                   "vad_max_chunk_seconds")};
      const auto &segments = options.speech_segments;
      if (!std::isfinite(segments.threshold) || segments.threshold < 0 ||
          segments.threshold > 1 || segments.min_speech_milliseconds < 0 ||
          segments.min_silence_milliseconds < 0 ||
          segments.speech_pad_milliseconds < 0 ||
          !std::isfinite(segments.merge_gap_seconds) ||
          segments.merge_gap_seconds < 0 ||
          !std::isfinite(segments.max_chunk_seconds) ||
          segments.max_chunk_seconds <= 0) {
        throw std::invalid_argument(
            "VAD settings must be finite and nonnegative");
      }
    } else {
    }
    return [this, samples = std::move(samples), options,
            vad_directory](const std::atomic<bool> &cancel) mutable {
      if (vad_directory.has_value()) {
        if (detector_ == nullptr || detector_directory_ != *vad_directory) {
          // Note (khazic): the old model goes before the new one loads, so two
          // never sit in memory together.
          detector_.reset();
          detector_ = std::make_unique<silero_vad::SileroVAD>(*vad_directory);
          detector_directory_ = *vad_directory;
        } else {
        }
        options.voice_activity_detector = detector_.get();
      } else {
      }
      return transcriber_.Transcribe(samples, options, cancel);
    };
  }

private:
  cohere_transcribe::CohereTranscriber transcriber_;
  // Note (khazic): the Silero VAD of the last directory a request named, loaded
  // on first use and touched on the worker thread only: one model at most,
  // since Voxt sends the same directory every time.
  mutable std::unique_ptr<silero_vad::SileroVAD> detector_;
  mutable std::string detector_directory_;
};

} // namespace

int main(int argc, char **argv) {
  asr_service::ServedKind kind;
  kind.model_kind = "cohere_transcribe";
  kind.load = [](const std::filesystem::path &model_directory) {
    return std::make_unique<CohereModelService>(model_directory);
  };
  return asr_service::Serve(argc, argv, kind);
}
