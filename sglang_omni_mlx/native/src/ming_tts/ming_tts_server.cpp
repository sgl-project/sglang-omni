// SPDX-License-Identifier: Apache-2.0
// Native Ming-Omni-TTS server: POST /v1/audio/speech with the CUDA server's
// request fields, answering 44.1 kHz WAV or PCM, or 16-bit PCM streamed as it
// is decoded, and Voxt's supervisor protocol.
//
//   ming_tts_server --model-path DIR [--model-name NAME] [--host H] [--port P]
//       [--max-queued-requests N] [--request-timeout-s S]
//   ming_tts_server --supervised --model-kind ming_tts --model-directory DIR
//       [--max-queued-requests N] [--request-timeout-s S]
//
// Requests are read as ming_tts_request.h describes, and as on CUDA:
// max_new_tokens (or stage_params.tts_engine.max_decode_steps) counts
// acoustic patches, 200 by default and at most 256, and ref_audio with
// ref_text clones a voice, ref_audio a base64 mono WAV data URL of 0.1 to 30 s
// at any common rate, which needs campplus.safetensors or campplus.onnx in the
// model directory. One request runs at a time and N more (2 by default) may
// wait; the rest get 503. A request not done S seconds (600 by default) after
// it arrives is cancelled, with 504 or an early end to its stream, and so is a
// request whose client disconnects.
#include <filesystem>
#include <memory>
#include <optional>
#include <stdexcept>
#include <vector>

#include "common/transcription.h"
#include "ming_tts/ming_tts_audio.h"
#include "ming_tts/ming_tts_generator.h"
#include "ming_tts/ming_tts_reference.h"
#include "ming_tts/ming_tts_request.h"
#include "qwen3_asr/asr_service.h"
#include "qwen3_asr/speech_service.h"

namespace {

namespace mx = mlx::core;
using speech_service::Json;

constexpr int kInitialChunkPatches = 2;
constexpr int kSteadyChunkPatches = 4;

std::vector<float> Samples(const mx::array &waveform) {
  const mx::array samples = mx::astype(mx::flatten(waveform), mx::float32);
  mx::eval(samples);
  return std::vector<float>(samples.data<float>(),
                            samples.data<float>() + samples.size());
}

// One parsed request and its prompt.
struct Synthesis {
  ming_tts::SpeechRequest request;
  ming_tts::Prompt prompt;
  bool stream = false;
};

class MingSpeechService : public speech_service::SpeechServedModel {
public:
  MingSpeechService(const std::filesystem::path &model_directory,
                    int max_queued_requests, double request_timeout_seconds)
      : SpeechServedModel(ming_tts::kMaxRequestBodyBytes, max_queued_requests,
                          request_timeout_seconds),
        frontend_(model_directory), generator_(model_directory),
        decoder_(model_directory) {
    if (ming_tts::HasCamPlusWeights(model_directory)) {
      reference_encoder_.emplace(model_directory);
    } else {
    }
  }

private:
  int SampleRate() const override { return decoder_.sample_rate(); }

  speech_service::SpeechJob ParseSpeech(const Json &body) override {
    const std::optional<std::string> model = speech_service::ParseModel(body);
    speech_service::SpeechJob job = speech_service::ParseCommonFields(body);
    job.model = model;
    Synthesis synthesis{ming_tts::ParseSpeechRequest(body), {}, job.stream};
    const ming_tts::SpeechRequest &request = synthesis.request;
    int reference_patches = 0;
    if (request.reference_audio.has_value()) {
      if (!reference_encoder_.has_value()) {
        throw std::invalid_argument(
            "voice cloning requires campplus.safetensors or campplus.onnx "
            "in the model directory");
      } else {
      }
      reference_patches = reference_encoder_->PatchCount(
          request.reference_audio->samples_44k.size());
    } else {
    }
    synthesis.prompt = frontend_.BuildPrompt(
        request.text, request.reference_text.value_or(""), reference_patches);
    generator_.CheckLength(synthesis.prompt, request.max_patches);
    job.synthesize = [this,
                      synthesis](const speech_service::SpeechContext &context) {
      return Synthesize(synthesis, context);
    };
    return job;
  }

  speech_service::SpeechResult
  Synthesize(const Synthesis &synthesis,
             const speech_service::SpeechContext &context) const {
    const auto stop_if_cancelled = [&context]() {
      if (context.cancel.load()) {
        throw omni_common::TranscriptionCancelled();
      } else {
      }
      context.check_deadline();
    };
    const ming_tts::SpeechRequest &request = synthesis.request;
    std::optional<ming_tts::ReferenceConditioning> reference;
    if (request.reference_audio.has_value()) {
      reference = reference_encoder_->Encode(*request.reference_audio);
    } else {
    }
    stop_if_cancelled();
    ming_tts::GenerationState state =
        generator_.Start(synthesis.prompt, request.max_patches, reference);
    ming_tts::AudioDecoderState audio_state;
    std::vector<mx::array> pending;
    int chunk_patches = kInitialChunkPatches;
    while (!state.finished) {
      stop_if_cancelled();
      pending.push_back(generator_.Step(state, request.cfg, request.sigma,
                                        request.temperature));
      if (synthesis.stream &&
          (static_cast<int>(pending.size()) >= chunk_patches ||
           state.finished)) {
        context.send(Samples(decoder_.DecodeChunk(
            mx::expand_dims(mx::concatenate(pending, 0), 0), audio_state,
            state.finished)));
        pending.clear();
        chunk_patches = kSteadyChunkPatches;
      } else {
      }
    }
    if (!synthesis.stream) {
      stop_if_cancelled();
      context.send(Samples(decoder_.DecodeFull(
          mx::expand_dims(mx::concatenate(pending, 0), 0))));
    } else {
    }
    return speech_service::SpeechResult{
        state.stopped ? omni_common::FinishReason::kStop
                      : omni_common::FinishReason::kLength,
        synthesis.prompt.input_ids.size(), state.steps};
  }

  ming_tts::MingFrontend frontend_;
  ming_tts::MingGenerator generator_;
  ming_tts::MingAudioDecoder decoder_;
  std::optional<ming_tts::MingReferenceEncoder> reference_encoder_;
};

} // namespace

int main(int argc, char **argv) {
  asr_service::ServedKind kind;
  kind.model_kind = "ming_tts";
  speech_service::AddSpeechFlags(
      kind, [](const std::filesystem::path &model_directory,
               int max_queued_requests, double request_timeout_seconds) {
        return std::make_unique<MingSpeechService>(
            model_directory, max_queued_requests, request_timeout_seconds);
      });
  return asr_service::Serve(argc, argv, kind);
}
