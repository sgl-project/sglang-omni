// SPDX-License-Identifier: Apache-2.0
#include "auk/auk_tts_service.h"

#include <iostream>
#include <typeinfo>

#include "auk/auk_request.h"
#include "common/audio.h"
#include "common/http.h"

namespace auk {

namespace {

using omni_server::Json;
constexpr size_t kMaximumBodyBytes = 16 * 1024 * 1024;

} // namespace

SpeechService::SpeechService(const std::filesystem::path &model_directory,
                             mlx::core::Dtype dtype, float max_seconds)
    : max_seconds_(max_seconds),
      worker_(
          [this, model_directory, dtype] {
            model_ =
                std::make_unique<Model>(model_directory, dtype, max_seconds_);
          },
          [this] { model_.reset(); }) {}

void SpeechService::Register(mg_context *context,
                             const std::string &model_name) {
  model_name_ = model_name;
  mg_set_request_handler(
      context, "/v1/audio/speech$",
      [](mg_connection *connection, void *service) {
        return static_cast<SpeechService *>(service)->Handle(connection, false);
      },
      this);
  mg_set_request_handler(
      context, "/generate$",
      [](mg_connection *connection, void *service) {
        return static_cast<SpeechService *>(service)->Handle(connection, true);
      },
      this);
}

int SpeechService::Handle(mg_connection *connection, bool raw_instruction) {
  if (!omni_server::IsPost(connection)) {
    return omni_server::WriteJson(connection, 405,
                                  {{"detail", "Method Not Allowed"}});
  } else {
  }
  ParsedRequest parsed;
  try {
    const std::string body =
        omni_server::ReadBody(connection, kMaximumBodyBytes);
    parsed = ParseRequest(Json::parse(body), raw_instruction, max_seconds_,
                          model_name_);
  } catch (const std::length_error &) {
    return omni_server::BadRequest(connection, "request body exceeds 16 MiB");
  } catch (const Json::exception &) {
    return omni_server::BadRequest(connection,
                                   "invalid request JSON or field type");
  } catch (const std::invalid_argument &error) {
    return omni_server::BadRequest(connection, error.what());
  }
  auto job = std::make_shared<
      tts_service::SynthesisWorker::Job<qwen3_asr::DecodedAudio>>();
  job->synthesize = [this, request = std::move(parsed.synthesis)](
                        const std::function<void()> &check_cancelled) {
    auto result = model_->Synthesize(request, check_cancelled);
    return qwen3_asr::DecodedAudio{std::move(result.samples),
                                   result.sample_rate};
  };
  const auto admission = worker_.Submit(job);
  if (admission == tts_service::SynthesisWorker::Admission::Stopping) {
    return omni_server::WriteJson(connection, 503,
                                  {{"detail", "TTS service is stopping"}});
  } else if (admission == tts_service::SynthesisWorker::Admission::Busy) {
    return omni_server::WriteJson(connection, 429, {{"detail", "AuK is busy"}});
  } else {
  }
  try {
    const auto result = tts_service::SynthesisWorker::Wait(connection, job);
    std::string audio = qwen3_asr::EncodePcm16(result.samples);
    if (!parsed.pcm) {
      audio = qwen3_asr::EncodeWavPcm16(audio, result.sample_rate, 1);
    } else {
    }
    if (raw_instruction) {
      std::string encoded(4 * ((audio.size() + 2) / 3) + 1, '\0');
      size_t length = encoded.size();
      mg_base64_encode(reinterpret_cast<const unsigned char *>(audio.data()),
                       audio.size(), encoded.data(), &length);
      encoded.resize(length - 1);
      return omni_server::WriteJson(
          connection, 200,
          {{"audio",
            {{"data", encoded},
             {"format", "wav"},
             {"sample_rate", result.sample_rate}}},
           {"meta_info", {{"finish_reason", {{"type", "stop"}}}}}});
    } else {
      omni_server::WriteResponse(connection, 200, "OK",
                                 parsed.pcm ? "audio/pcm" : "audio/wav", audio);
      return 200;
    }
  } catch (const tts_service::SynthesisCancelled &) {
    return omni_server::WriteJson(connection, 503,
                                  {{"detail", "Synthesis cancelled"}});
  } catch (const std::exception &error) {
    std::cerr << "synthesis failed: " << typeid(error).name() << "\n";
    return omni_server::WriteJson(connection, 500,
                                  {{"detail", "Synthesis failed"}});
  }
}

std::map<std::string, int> SpeechService::RequestStates() const {
  return worker_.RequestStates();
}

void SpeechService::CancelAll() { worker_.Stop(); }

} // namespace auk
