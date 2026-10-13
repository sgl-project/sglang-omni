// SPDX-License-Identifier: Apache-2.0
// Native SenseVoice Small server: transcriptions over HTTP (JSON or SSE) with
// the API of qwen3_asr_server, without its realtime API, and Voxt's supervisor
// protocol.
//
//   sensevoice_server --model-path DIR [--model-name NAME] [--host H]
//     [--port P]
//   sensevoice_server --supervised --model-kind sensevoice
//     --model-directory DIR
//
// Request fields beside the audio: language (Voxt's hint), use_itn, stream
// and include_generation_metadata. With vad_model_directory (a Silero VAD
// checkpoint), the speech is cut by vad_threshold, vad_min_speech_ms,
// vad_min_silence_ms and vad_speech_pad_ms into chunks of at most
// vad_max_chunk_seconds overlapping by vad_chunk_overlap_seconds, all
// required then. Segments carry each pass's language, emotion and event.
#include <cmath>
#include <memory>
#include <stdexcept>

#include "asr_service.h"
#include "sensevoice_transcriber.h"

namespace {

template <typename Value>
Value Required(const std::optional<Value> &value, const std::string &name) {
  if (!value.has_value()) {
    throw std::invalid_argument(name + " is required with vad_model_directory");
  } else {
  }
  return *value;
}

// Note (Jiaxin Deng): a finite number of seconds read as a double, as Voxt
// holds the chunk lengths it sends.
double Seconds(const asr_service::FormFields &form, const std::string &name) {
  Required(asr_service::NumberField(form, name), name);
  return std::stod(*asr_service::TextField(form, name));
}

class SenseVoiceModelService : public asr_service::ServedModel {
public:
  explicit SenseVoiceModelService(const std::filesystem::path &model_directory)
      : transcriber_(model_directory) {}

  asr_service::Transcription
  Prepare(std::vector<float> samples,
          const asr_service::FormFields &form) const override {
    sensevoice::SenseVoiceOptions options;
    options.language = asr_service::TextField(form, "language");
    const auto use_itn = asr_service::TextField(form, "use_itn");
    if (use_itn.has_value()) {
      options.use_itn = qwen3_asr::FormFlag(use_itn);
    } else {
    }
    const std::optional<std::string> vad_directory =
        asr_service::TextField(form, "vad_model_directory");
    if (vad_directory.has_value()) {
      options.speech = {
          Required(asr_service::NumberField(form, "vad_threshold"),
                   "vad_threshold"),
          Required(asr_service::IntegerField(form, "vad_min_speech_ms"),
                   "vad_min_speech_ms"),
          Required(asr_service::IntegerField(form, "vad_min_silence_ms"),
                   "vad_min_silence_ms"),
          Required(asr_service::IntegerField(form, "vad_speech_pad_ms"),
                   "vad_speech_pad_ms")};
      options.max_chunk_seconds = Seconds(form, "vad_max_chunk_seconds");
      options.chunk_overlap_seconds =
          Seconds(form, "vad_chunk_overlap_seconds");
      if (options.speech.threshold < 0 || options.speech.threshold > 1 ||
          options.speech.min_speech_ms < 0 ||
          options.speech.min_silence_ms < 0 ||
          options.speech.speech_pad_ms < 0) {
        throw std::invalid_argument("VAD settings must be nonnegative");
      } else {
      }
      // Note (Dayuxiaoshui): an overlap as long as a chunk would never move
      // past the chunk's start; compared in samples, as chunking counts it.
      sensevoice::ChunkSampleCounts(options.max_chunk_seconds,
                                    options.chunk_overlap_seconds);
    } else {
    }
    return [this, samples = std::move(samples), options,
            vad_directory](const std::atomic<bool> &cancel) mutable {
      if (vad_directory.has_value()) {
        if (detector_ == nullptr || detector_directory_ != *vad_directory) {
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
  sensevoice::SenseVoiceTranscriber transcriber_;
  // Note (Dayuxiaoshui): as the Cohere server keeps it, the Silero VAD of the
  // last directory a request named, used on the worker thread only.
  mutable std::unique_ptr<silero_vad::SileroVAD> detector_;
  mutable std::string detector_directory_;
};

} // namespace

int main(int argc, char **argv) {
  asr_service::ServedKind kind;
  kind.model_kind = "sensevoice";
  kind.load = [](const std::filesystem::path &model_directory) {
    return std::make_unique<SenseVoiceModelService>(model_directory);
  };
  return asr_service::Serve(argc, argv, kind);
}
