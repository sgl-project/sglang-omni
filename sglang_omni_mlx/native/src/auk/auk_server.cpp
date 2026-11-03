// SPDX-License-Identifier: Apache-2.0
// Native AuK and AuK-Flash server: POST /v1/audio/speech answered with a WAV
// or PCM file, POST /generate with a raw AuK instruction answered with JSON
// around a base64 WAV, and Voxt's supervisor protocol.
//
//   auk_server --model-path DIR --text-encoder-path QWEN_DIR
//       [--model-name NAME] [--host H] [--port P] [--dtype bfloat16|float32]
//       [--max-seconds S] [--cache-limit-gb G] [--max-queued-requests N]
//       [--request-timeout-s T]
//   auk_server --supervised --model-kind auk --model-directory DIR
//       --text-encoder-path QWEN_DIR [...]
//
// DIR is a released AuK or AuK-Flash checkpoint and QWEN_DIR is
// Qwen2.5-Omni-3B, both as published.
//
// One request runs at a time and N more (2 by default) may wait; the rest get
// 503. A request not done T seconds (600 by default) after it arrives gets
// 504, and is cancelled at its next stage or sampling step, as is one whose
// client disconnects. A request body over 16 MiB gets 413.
#include <cmath>
#include <filesystem>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

#include "auk/auk_model.h"
#include "auk/auk_request.h"
#include "qwen3_asr/asr_service.h"
#include "qwen3_asr/speech_service.h"

namespace {

using speech_service::Json;

constexpr size_t kMaxRequestBodyBytes = 16 * 1024 * 1024;

class AukSpeechService : public speech_service::SpeechServedModel {
public:
  AukSpeechService(const std::filesystem::path &model_directory,
                   const std::filesystem::path &text_encoder_directory,
                   mlx::core::Dtype dtype, float max_seconds,
                   int max_queued_requests, double request_timeout_seconds)
      : SpeechServedModel(kMaxRequestBodyBytes, max_queued_requests,
                          request_timeout_seconds),
        max_seconds_(max_seconds),
        model_(model_directory, text_encoder_directory, dtype, max_seconds) {}

  void AddHandlers(mg_context *context,
                   qwen3_asr::TranscriptionWorker &worker) override {
    SpeechServedModel::AddHandlers(context, worker);
    mg_set_request_handler(context, "/generate$", HandleGenerate, this);
  }

private:
  static int HandleGenerate(mg_connection *connection, void *data) {
    auto *service = static_cast<AukSpeechService *>(data);
    return service->Speak(connection, [service](const Json &body) {
      speech_service::SpeechJob job;
      job.synthesize = service->SynthesisFor(
          auk::ParseRequest(body, true, service->max_seconds_));
      job.model = speech_service::ParseModel(body);
      job.response_format = "wav";
      job.is_generate_response = true;
      return job;
    });
  }

  int SampleRate() const override { return auk::kSampleRate; }

  speech_service::SpeechJob ParseSpeech(const Json &body) override {
    auto synthesize =
        SynthesisFor(auk::ParseRequest(body, false, max_seconds_));
    speech_service::SpeechJob job = speech_service::ParseCommonFields(body);
    job.model = speech_service::ParseModel(body);
    job.synthesize = std::move(synthesize);
    return job;
  }

  std::function<
      speech_service::SpeechResult(const speech_service::SpeechContext &)>
  SynthesisFor(auk::SynthesisRequest request) {
    return [this, request = std::move(request)](
               const speech_service::SpeechContext &context) {
      const auto check_cancelled = [&context] {
        if (context.cancel.load()) {
          std::cerr << "synthesis cancelled\n";
          throw qwen3_asr::TranscriptionCancelled();
        } else {
        }
        context.check_deadline();
      };
      context.send(model_.Synthesize(request, check_cancelled).samples);
      return speech_service::SpeechResult{};
    };
  }

  const float max_seconds_;
  auk::Model model_;
};

} // namespace

int main(int argc, char **argv) {
  constexpr size_t GIBIBYTE = size_t{1} << 30;
  mlx::core::Dtype dtype = mlx::core::bfloat16;
  float max_seconds = 30.0f;
  size_t cache_limit_bytes = GIBIBYTE;
  std::filesystem::path text_encoder_directory;
  asr_service::ServedKind kind;
  kind.model_kind = "auk";
  speech_service::AddSpeechFlags(
      kind, [&](const std::filesystem::path &model_directory,
                int max_queued_requests, double request_timeout_seconds) {
        if (text_encoder_directory.empty()) {
          throw std::invalid_argument(
              "--text-encoder-path must name the Qwen2.5-Omni-3B directory");
        } else {
        }
        mlx::core::set_cache_limit(cache_limit_bytes);
        return std::make_unique<AukSpeechService>(
            model_directory, text_encoder_directory, dtype, max_seconds,
            max_queued_requests, request_timeout_seconds);
      });
  kind.flags["--text-encoder-path"] = [&](const std::string &value) {
    text_encoder_directory = value;
  };
  kind.flags["--dtype"] = [&](const std::string &value) {
    if (value == "bfloat16") {
      dtype = mlx::core::bfloat16;
    } else if (value == "float32") {
      dtype = mlx::core::float32;
    } else {
      throw std::invalid_argument("--dtype must be bfloat16 or float32");
    }
  };
  kind.flags["--max-seconds"] = [&](const std::string &value) {
    size_t parsed = 0;
    max_seconds = std::stof(value, &parsed);
    if (parsed != value.size() || !std::isfinite(max_seconds) ||
        max_seconds < 0.02f || max_seconds > 30.0f) {
      throw std::invalid_argument("--max-seconds must be in [0.02, 30]");
    } else {
    }
  };
  kind.flags["--cache-limit-gb"] = [&](const std::string &value) {
    size_t parsed = 0;
    const double cache_limit_gibibytes = std::stod(value, &parsed);
    const double requested_cache_bytes = cache_limit_gibibytes * GIBIBYTE;
    if (parsed != value.size() || !std::isfinite(cache_limit_gibibytes) ||
        cache_limit_gibibytes < 0 ||
        requested_cache_bytes >=
            static_cast<double>(std::numeric_limits<size_t>::max())) {
      throw std::invalid_argument("--cache-limit-gb must be finite, "
                                  "non-negative, and fit in size_t bytes");
    } else {
      cache_limit_bytes = static_cast<size_t>(requested_cache_bytes);
    }
  };
  return asr_service::Serve(argc, argv, kind);
}
