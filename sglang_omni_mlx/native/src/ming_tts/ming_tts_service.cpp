// SPDX-License-Identifier: Apache-2.0
#include "ming_tts/ming_tts_service.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <set>
#include <stdexcept>
#include <typeinfo>

#include "common/audio.h"
#include "common/http.h"

namespace ming_tts {

namespace mx = mlx::core;

namespace {

constexpr int kCfmSteps = 10;
constexpr int kInitialChunkPatches = 2;
constexpr int kSteadyChunkPatches = 4;
constexpr size_t kMaxBodyBytes = 4 * 1024 * 1024;

std::string Pcm(const mx::array &waveform) {
  if (waveform.size() == 0) {
    return {};
  } else {
  }
  const auto samples =
      mx::astype(mx::clip(mx::astype(waveform, mx::float32) * 32768.0f,
                          mx::array(-32768.0f), mx::array(32767.0f)),
                 mx::int16);
  mx::eval(samples);
  return {reinterpret_cast<const char *>(samples.data<int16_t>()),
          samples.size() * sizeof(int16_t)};
}

} // namespace

SpeechRequest SpeechRequest::Parse(const nlohmann::json &request) {
  static const std::set<std::string> fields{
      "model",   "input",        "response_format", "stream",
      "voice",   "speed",        "max_new_tokens",  "temperature",
      "prompt",  "instructions", "stage_params",    "ref_audio",
      "ref_text"};
  if (!request.is_object()) {
    throw std::invalid_argument("request must be a JSON object");
  } else {
  }
  for (const auto &[name, value] : request.items()) {
    if (!fields.contains(name)) {
      throw std::invalid_argument("unsupported speech field: " + name);
    } else {
    }
  }
  request.at("model").get<std::string>();
  const auto text = request.at("input").get<std::string>();
  const auto value_or_default = [](const nlohmann::json &source,
                                   const std::string &name,
                                   const nlohmann::json &default_value) {
    const auto value = source.value(name, nlohmann::json());
    return value.is_null() ? default_value : value;
  };
  const auto prompt =
      value_or_default(request, "prompt", "").get<std::string>();
  const auto instructions =
      value_or_default(request, "instructions", "").get<std::string>();
  std::vector<float> reference_samples;
  std::string reference_text;
  if (request.contains("ref_audio") || request.contains("ref_text")) {
    if (!request.contains("ref_audio") || !request.contains("ref_text")) {
      throw std::invalid_argument(
          "ref_audio and ref_text must be provided together");
    } else {
    }
    reference_text = request.at("ref_text").get<std::string>();
    const auto audio = request.at("ref_audio").get<std::string>();
    constexpr std::string_view prefix = "data:audio/wav;base64,";
    if (reference_text.empty() || !audio.starts_with(prefix)) {
      throw std::invalid_argument("voice cloning requires nonempty ref_text "
                                  "and inline data:audio/wav;base64 audio");
    } else {
    }
    const auto encoded = std::string_view(audio).substr(prefix.size());
    const size_t padding = encoded.ends_with("==")  ? 2
                           : encoded.ends_with("=") ? 1
                                                    : 0;
    if (encoded.empty() || encoded.size() % 4 != 0 ||
        encoded.substr(0, encoded.size() - padding)
                .find_first_not_of("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopq"
                                   "rstuvwxyz0123456789+/") !=
            std::string_view::npos) {
      throw std::invalid_argument("ref_audio contains invalid base64");
    } else {
    }
    std::string decoded(encoded.size(), '\0');
    size_t bytes = decoded.size();
    if (mg_base64_decode(encoded.data(), encoded.size(),
                         reinterpret_cast<unsigned char *>(decoded.data()),
                         &bytes) != -1) {
      throw std::invalid_argument("ref_audio contains invalid base64");
    } else {
      decoded.resize(bytes - 1);
    }
    reference_samples = qwen3_asr::DecodeWav(decoded);
    if (reference_samples.size() < qwen3_asr::kSampleRate / 10 ||
        reference_samples.size() > 30 * qwen3_asr::kSampleRate) {
      throw std::invalid_argument(
          "ref_audio must contain 0.1 to 30 seconds of mono 16 kHz audio");
    } else if (!std::all_of(
                   reference_samples.begin(), reference_samples.end(),
                   [](float sample) { return std::isfinite(sample); })) {
      throw std::invalid_argument("ref_audio samples must be finite");
    } else {
    }
  } else {
  }
  const auto format = request.value("response_format", std::string("wav"));
  const bool stream = request.value("stream", false);
  const auto length = value_or_default(request, "max_new_tokens", 200);
  if ((format != "wav" && format != "pcm") || (stream && format != "pcm")) {
    throw std::invalid_argument("use wav or pcm; streaming requires pcm");
  } else if (request.value("voice", std::string("default")) != "default" ||
             request.value("speed", 1.0) != 1.0) {
    throw std::invalid_argument("only voice=default and speed=1 are supported");
  } else if (!length.is_number_integer() || length < 1 || length > 256) {
    throw std::invalid_argument("max_new_tokens must be an integer in [1,256]");
  } else {
  }
  const auto stages =
      value_or_default(request, "stage_params", nlohmann::json::object());
  if (!stages.is_object() || stages.size() > stages.count("tts_engine")) {
    throw std::invalid_argument(
        "only tts_engine sampling parameters are supported");
  } else {
  }
  const auto sampling = stages.value("tts_engine", nlohmann::json::object());
  if (!sampling.is_object()) {
    throw std::invalid_argument("tts_engine must be an object");
  } else {
  }
  for (const auto &[name, value] : sampling.items()) {
    if (name != "cfg" && name != "sigma" && name != "temperature") {
      throw std::invalid_argument("unsupported sampling field: " + name);
    } else {
    }
  }
  const auto temperature_value =
      sampling.value("temperature", nlohmann::json());
  const float temperature =
      temperature_value.is_null()
          ? value_or_default(request, "temperature", 0.0f).get<float>()
          : temperature_value.get<float>();
  const float guidance_scale =
      value_or_default(sampling, "cfg", 2.0f).get<float>();
  const float sigma = value_or_default(sampling, "sigma", 0.25f).get<float>();
  if (!std::isfinite(guidance_scale) || guidance_scale < 1e-5f ||
      guidance_scale == 1.0f) {
    throw std::invalid_argument(
        "Ming-Omni-TTS currently supports only guided-branch cfg values; "
        "cfg must be >= 1e-5 and not equal to 1.0");
  } else if (!std::isfinite(sigma) || sigma < 0 ||
             !std::isfinite(temperature) || temperature < 0) {
    throw std::invalid_argument(
        "sigma and temperature must be finite and nonnegative");
  } else {
    return {text,
            prompt,
            instructions,
            stream,
            format == "wav",
            length.get<int>(),
            guidance_scale,
            sigma,
            temperature,
            std::move(reference_samples),
            std::move(reference_text)};
  }
}

MingSpeechService::MingSpeechService(const std::filesystem::path &directory)
    : directory_(directory),
      worker_(
          [this] {
            frontend_ = std::make_unique<MingFrontend>(directory_);
            generator_ = std::make_unique<MingGenerator>(directory_);
            decoder_ = std::make_unique<MingAudioDecoder>(directory_);
            mx::synchronize();
            mx::clear_cache();
          },
          [this] {
            reference_encoder_.reset();
            decoder_.reset();
            generator_.reset();
            frontend_.reset();
          }) {}

void MingSpeechService::Register(mg_context *context, const std::string &) {
  mg_set_request_handler(
      context, "/v1/audio/speech$",
      [](mg_connection *connection, void *service) {
        return static_cast<MingSpeechService *>(service)->HandleSpeech(
            connection);
      },
      this);
}

int MingSpeechService::HandleSpeech(mg_connection *connection) {
  if (!omni_server::IsPost(connection)) {
    return omni_server::WriteJson(connection, 405, {{"detail", "use POST"}});
  } else {
  }
  Job job;
  try {
    const auto body = omni_server::ReadBody(connection, kMaxBodyBytes);
    job.request = SpeechRequest::Parse(nlohmann::json::parse(body));
    const auto format = job.request.stream
                            ? tts_service::AudioFormat::StreamingPcm
                        : job.request.wav ? tts_service::AudioFormat::Wav
                                          : tts_service::AudioFormat::Pcm;
    tts_service::AudioResponse response(connection, format);
    auto synthesis =
        std::make_shared<tts_service::SynthesisWorker::Job<void>>();
    synthesis->synthesize = [&](const std::function<void()> &check_cancelled) {
      Generate(job, response, check_cancelled);
    };
    return response.SubmitAndRespond(worker_, synthesis, [&] {
      if (!job.request.stream) {
        response.WritePcm(job.audio, job.sample_rate, 1);
      } else {
      }
    });
  } catch (const nlohmann::json::exception &error) {
    return omni_server::BadRequest(connection, error.what());
  } catch (const std::length_error &error) {
    return omni_server::BadRequest(connection, error.what());
  } catch (const std::invalid_argument &error) {
    std::cerr << "Speech request failed: " << typeid(error).name() << '\n';
    return omni_server::BadRequest(connection, error.what());
  } catch (const std::exception &error) {
    std::cerr << "Speech generation failed: " << typeid(error).name() << '\n';
    return omni_server::WriteJson(connection, 500, {{"detail", error.what()}});
  }
}

std::map<std::string, int> MingSpeechService::RequestStates() const {
  return worker_.RequestStates();
}

void MingSpeechService::CancelAll() { worker_.Stop(); }

void MingSpeechService::Generate(Job &job, tts_service::AudioResponse &response,
                                 const std::function<void()> &check_cancelled) {
  const auto &request = job.request;
  const auto &acoustic = generator_->acoustic();
  std::optional<ReferenceConditioning> reference;
  if (!request.reference_samples.empty()) {
    if (!reference_encoder_) {
      if (!std::filesystem::is_regular_file(directory_ /
                                            "campplus.safetensors")) {
        throw std::runtime_error("voice cloning requires campplus.safetensors "
                                 "in the model directory");
      } else {
        reference_encoder_ = std::make_unique<MingReferenceEncoder>(directory_);
      }
    } else {
    }
    check_cancelled();
    reference = reference_encoder_->Encode(request.reference_samples);
  } else {
  }
  check_cancelled();
  const int reference_tokens =
      reference ? reference->latents.shape(1) / acoustic.patch_size() : 0;
  const auto prompt =
      frontend_->BuildPrompt(request.text, request.prompt, request.instructions,
                             request.reference_text, reference_tokens);
  auto state = generator_->Start(prompt, request.max_new_tokens, reference);
  const auto timesteps = CfmTimesteps(kCfmSteps);
  AudioDecoderState audio_state;
  std::vector<mx::array> pending;
  int chunk_patches = kInitialChunkPatches;
  while (!state.finished) {
    check_cancelled();
    const auto noise =
        mx::random::normal({1, acoustic.latent_dim(), acoustic.patch_size()});
    const auto sde_random =
        mx::random::normal({timesteps.shape(0) - 2, 1, acoustic.patch_size(),
                            acoustic.latent_dim()});
    pending.push_back(generator_->Step(state, noise, timesteps, sde_random,
                                       request.guidance_scale, request.sigma,
                                       request.temperature));
    if (request.stream &&
        (static_cast<int>(pending.size()) >= chunk_patches || state.finished)) {
      const auto waveform =
          decoder_->DecodeChunk(mx::expand_dims(mx::concatenate(pending, 0), 0),
                                audio_state, state.finished);
      pending.clear();
      chunk_patches = kSteadyChunkPatches;
      response.WritePcm(Pcm(waveform), decoder_->sample_rate(), 1);
    } else {
    }
  }
  check_cancelled();
  if (!request.stream) {
    job.audio = Pcm(
        decoder_->DecodeFull(mx::expand_dims(mx::concatenate(pending, 0), 0)));
    job.sample_rate = decoder_->sample_rate();
  } else {
  }
}

} // namespace ming_tts

int main(int argc, char **argv) {
  omni_server::ServedKind kind;
  kind.model_kind = "ming_tts";
  kind.load = [](const std::filesystem::path &model_directory) {
    return std::make_unique<ming_tts::MingSpeechService>(model_directory);
  };
  return omni_server::Serve(argc, argv, {kind});
}
