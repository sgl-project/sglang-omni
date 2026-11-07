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
// Request fields: input, model, voice (default only), speed (1 only),
// response_format (wav or pcm), stream (pcm only), prompt (replaces the
// default description), instructions, max_new_tokens (acoustic patches, 200
// by default and at most 256), temperature, stage_params.tts_engine (cfg,
// sigma, temperature), and ref_audio with ref_text to clone a voice: ref_audio
// a base64 mono WAV data URL of 0.1 to 30 s at any common rate, which needs
// campplus.safetensors in the model directory. As on CUDA, the reference is
// resampled from its own rate to 44.1 kHz for AudioVAE and 16 kHz for
// CampPlus. One request runs at a time and N more (2 by default) may wait; the
// rest get 503. A request not done S seconds (600 by default) after it arrives
// is cancelled, with 504 or an early end to its stream, and so is a request
// whose client disconnects.
#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "common/reference_audio.h"
#include "common/text.h"
#include "ming_tts/ming_tts_audio.h"
#include "ming_tts/ming_tts_generator.h"
#include "ming_tts/ming_tts_reference.h"
#include "qwen3_asr/asr_service.h"
#include "qwen3_asr/speech_service.h"

namespace {

namespace mx = mlx::core;
using speech_service::FieldOr;
using speech_service::Json;
using speech_service::PresentFieldOr;

constexpr int kCfmSteps = 10;
constexpr int kInitialChunkPatches = 2;
constexpr int kSteadyChunkPatches = 4;
// CUDA's MING_TTS_DEFAULT_MAX_DECODE_STEPS and its serving cap.
constexpr int kDefaultMaxPatches = 200;
constexpr int kMaxPatches = 256;
constexpr float kDefaultCfg = 2.0f;
constexpr float kMinCfg = 1e-5f;
constexpr float kDefaultSigma = 0.25f;
constexpr double kMinReferenceSeconds = 0.1;
constexpr double kMaxReferenceSeconds = 30;
// Room for kMaxReferenceSeconds of mono float32 WAV at up to 192 kHz as
// base64 (29 MiB).
constexpr size_t kMaxRequestBodyBytes = 32 * 1024 * 1024;
constexpr std::string_view kReferenceAudioPrefix = "data:audio/wav;base64,";
constexpr std::array<std::string_view, 14> kAcceptedFields = {
    "model",         "input",   "response_format", "stream",
    "stream_format", "voice",   "speed",           "max_new_tokens",
    "temperature",   "prompt",  "instructions",    "stage_params",
    "ref_audio",     "ref_text"};
constexpr std::array<std::string_view, 3> kSamplingFields = {"cfg", "sigma",
                                                             "temperature"};

// A string field, empty when it is missing or null.
std::string TextField(const Json &body, const char *name) {
  const Json value = FieldOr(body, name, Json(""));
  if (!value.is_string()) {
    throw std::invalid_argument(std::string(name) + " must be a string");
  } else {
  }
  return value.get<std::string>();
}

// A finite float32 field, or fallback when it is missing or null.
float NumberField(const Json &source, const char *name, float fallback) {
  const Json value = FieldOr(source, name, Json(fallback));
  if (!value.is_number() || !std::isfinite(value.get<float>())) {
    throw std::invalid_argument(std::string(name) + " must be a finite number");
  } else {
  }
  return value.get<float>();
}

// A base64 mono WAV data URL at the rates the reference encoder takes;
// vae_sample_rate is AudioVAE's.
ming_tts::ReferenceAudio DecodeReferenceAudio(const std::string &url,
                                              int vae_sample_rate) {
  if (!url.starts_with(kReferenceAudioPrefix)) {
    throw std::invalid_argument(
        "ref_audio must be a data:audio/wav;base64 URL on this server");
  } else {
  }
  const std::optional<std::vector<uint8_t>> bytes =
      omni_common::DecodeBase64(url.substr(kReferenceAudioPrefix.size()));
  if (!bytes.has_value()) {
    throw std::invalid_argument("ref_audio must be valid base64");
  } else {
  }
  try {
    const omni_common::DecodedAudio decoded =
        omni_common::DecodeAnyWav(std::string_view(
            reinterpret_cast<const char *>(bytes->data()), bytes->size()));
    const double seconds =
        static_cast<double>(decoded.samples.size()) / decoded.sample_rate;
    if (seconds < kMinReferenceSeconds || seconds > kMaxReferenceSeconds) {
      throw std::invalid_argument(
          "reference audio must be between 0.1 and 30 seconds long");
    } else if (decoded.channel_count != 1) {
      throw std::invalid_argument("reference audio must be mono");
    } else {
    }
    return {omni_common::Resample(decoded.samples, decoded.sample_rate,
                                  ming_tts::kSpeakerSampleRate),
            omni_common::Resample(decoded.samples, decoded.sample_rate,
                                  vae_sample_rate)};
  } catch (const std::invalid_argument &error) {
    throw std::invalid_argument(std::string("ref_audio: ") + error.what());
  }
}

std::vector<float> Samples(const mx::array &waveform) {
  const mx::array samples = mx::astype(mx::flatten(waveform), mx::float32);
  mx::eval(samples);
  return std::vector<float>(samples.data<float>(),
                            samples.data<float>() + samples.size());
}

// One parsed request: its prompt, reference audio and sampling.
struct Synthesis {
  ming_tts::Prompt prompt;
  std::optional<ming_tts::ReferenceAudio> reference;
  int max_patches = 0;
  float cfg = 0, sigma = 0, temperature = 0;
  bool stream = false;
};

class MingSpeechService : public speech_service::SpeechServedModel {
public:
  MingSpeechService(const std::filesystem::path &model_directory,
                    int max_queued_requests, double request_timeout_seconds)
      : SpeechServedModel(kMaxRequestBodyBytes, max_queued_requests,
                          request_timeout_seconds),
        frontend_(model_directory), generator_(model_directory),
        decoder_(model_directory) {
    if (std::filesystem::is_regular_file(model_directory /
                                         "campplus.safetensors")) {
      reference_encoder_.emplace(model_directory);
    } else {
    }
  }

private:
  int SampleRate() const override { return decoder_.sample_rate(); }

  speech_service::SpeechJob ParseSpeech(const Json &body) override {
    for (const auto &[name, value] : body.items()) {
      if (std::find(kAcceptedFields.begin(), kAcceptedFields.end(), name) ==
          kAcceptedFields.end()) {
        throw std::invalid_argument("unsupported speech field: " + name);
      } else {
      }
    }
    const std::optional<std::string> model = speech_service::ParseModel(body);
    speech_service::SpeechJob job = speech_service::ParseCommonFields(body);
    job.model = model;
    const Json input = FieldOr(body, "input", Json());
    if (!input.is_string()) {
      throw std::invalid_argument("input must be a string");
    } else {
    }
    if (PresentFieldOr(body, "voice", Json("default")) != "default") {
      throw std::invalid_argument("only voice=default is supported");
    } else {
    }
    const Json speed = PresentFieldOr(body, "speed", Json(1.0));
    if (!speed.is_number() || speed.get<double>() != 1.0) {
      throw std::invalid_argument("speed must be 1.0 on this model");
    } else {
    }
    const Json max_new_tokens =
        FieldOr(body, "max_new_tokens", Json(kDefaultMaxPatches));
    if (!max_new_tokens.is_number_integer() ||
        max_new_tokens.get<int64_t>() < 1 ||
        max_new_tokens.get<int64_t>() > kMaxPatches) {
      throw std::invalid_argument("max_new_tokens must be an integer in [1, " +
                                  std::to_string(kMaxPatches) + "]");
    } else {
    }

    const Json stages = FieldOr(body, "stage_params", Json::object());
    if (!stages.is_object() || stages.size() > stages.count("tts_engine")) {
      throw std::invalid_argument(
          "only tts_engine sampling parameters are supported");
    } else {
    }
    const Json sampling = PresentFieldOr(stages, "tts_engine", Json::object());
    if (!sampling.is_object()) {
      throw std::invalid_argument("tts_engine must be an object");
    } else {
    }
    for (const auto &[name, value] : sampling.items()) {
      if (std::find(kSamplingFields.begin(), kSamplingFields.end(), name) ==
          kSamplingFields.end()) {
        throw std::invalid_argument("unsupported sampling field: " + name);
      } else {
      }
    }
    Synthesis synthesis;
    synthesis.max_patches = max_new_tokens.get<int>();
    synthesis.stream = job.stream;
    synthesis.cfg = NumberField(sampling, "cfg", kDefaultCfg);
    synthesis.sigma = NumberField(sampling, "sigma", kDefaultSigma);
    synthesis.temperature = FieldOr(sampling, "temperature", Json()).is_null()
                                ? NumberField(body, "temperature", 0.0f)
                                : NumberField(sampling, "temperature", 0.0f);
    if (synthesis.cfg < kMinCfg || synthesis.cfg == 1.0f) {
      throw std::invalid_argument(
          "Ming-Omni-TTS currently supports only guided-branch cfg values; "
          "cfg must be >= 1e-5 and not equal to 1.0");
    } else if (synthesis.sigma < 0 || synthesis.temperature < 0) {
      throw std::invalid_argument("sigma and temperature must be nonnegative");
    } else {
    }

    const Json ref_audio = FieldOr(body, "ref_audio", Json());
    const Json ref_text = FieldOr(body, "ref_text", Json());
    std::string reference_text;
    int reference_patches = 0;
    if (ref_audio.is_null() != ref_text.is_null()) {
      throw std::invalid_argument(
          "ref_audio and ref_text must be provided together");
    } else if (!ref_audio.is_null()) {
      if (!ref_audio.is_string() || !ref_text.is_string()) {
        throw std::invalid_argument("ref_audio and ref_text must be strings");
      } else if (!reference_encoder_.has_value()) {
        throw std::invalid_argument(
            "voice cloning requires campplus.safetensors in the model "
            "directory");
      } else {
      }
      synthesis.reference = DecodeReferenceAudio(ref_audio.get<std::string>(),
                                                 decoder_.sample_rate());
      reference_text = ref_text.get<std::string>();
      reference_patches = reference_encoder_->PatchCount(
          synthesis.reference->samples_44k.size());
    } else {
    }
    synthesis.prompt = frontend_.BuildPrompt(
        input.get<std::string>(), TextField(body, "prompt"),
        TextField(body, "instructions"), reference_text, reference_patches);
    generator_.CheckLength(synthesis.prompt, synthesis.max_patches);
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
        throw qwen3_asr::TranscriptionCancelled();
      } else {
      }
      context.check_deadline();
    };
    std::optional<ming_tts::ReferenceConditioning> reference;
    if (synthesis.reference.has_value()) {
      reference = reference_encoder_->Encode(*synthesis.reference);
    } else {
    }
    stop_if_cancelled();
    const ming_tts::MingAcoustic &acoustic = generator_.acoustic();
    ming_tts::GenerationState state =
        generator_.Start(synthesis.prompt, synthesis.max_patches, reference);
    const mx::array timesteps = ming_tts::CfmTimesteps(kCfmSteps);
    ming_tts::AudioDecoderState audio_state;
    std::vector<mx::array> pending;
    int chunk_patches = kInitialChunkPatches;
    while (!state.finished) {
      stop_if_cancelled();
      const mx::array noise =
          mx::random::normal({1, acoustic.latent_dim(), acoustic.patch_size()});
      const mx::array sde_random =
          mx::random::normal({timesteps.shape(0) - 2, 1, acoustic.patch_size(),
                              acoustic.latent_dim()});
      pending.push_back(generator_.Step(state, noise, timesteps, sde_random,
                                        synthesis.cfg, synthesis.sigma,
                                        synthesis.temperature));
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
        state.stopped ? qwen3_asr::FinishReason::kStop
                      : qwen3_asr::FinishReason::kLength,
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
