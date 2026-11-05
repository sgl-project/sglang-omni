// SPDX-License-Identifier: Apache-2.0
// The servers' shared protocol around stand-in models, for the protocol tests:
// no checkpoint is read and no model runs.
//
//   stub_server --supervised --model-kind stub_asr|stub_speech
//       --model-directory DIR [--load-delay-ms N] [--max-queued-requests N]
//       [--request-timeout-s S]
//
// stub_asr transcribes audio to its decoded sample count, with the language
// and max_new_tokens fields as its generation metadata, and serves
// /v1/realtime, where transcription.done answers the sample count appended.
// stub_speech answers chunks chunks of chunk_samples silent samples. Both
// read an outcome field: "fail" throws, "block" runs, past any chunks, until it
// is cancelled or its deadline passes, and stub_asr's "segments" returns one
// speaker segment.
#include <chrono>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "qwen3_asr/asr_service.h"
#include "qwen3_asr/realtime.h"
#include "qwen3_asr/speech_service.h"

namespace {

using speech_service::FieldOr;
using speech_service::Json;

constexpr auto kBlockPollInterval = std::chrono::milliseconds(10);
// Unlike Voxt's 16 kHz input, so a test sees which rate a header reports.
constexpr int kSpeechSampleRate = 24000;
constexpr size_t kMaxRequestBodyBytes = 64 * 1024;

class StubRealtimeConnection : public qwen3_asr::RealtimeConnection {
public:
  using RealtimeConnection::RealtimeConnection;

  bool Handle(const nlohmann::json &message) override {
    const nlohmann::json type = message.value("type", nlohmann::json());
    if (type == "session.update") {
      if (ManualTurnSession(message).has_value()) {
        Send({{"type", "transcription_session.updated"}});
      } else {
      }
      return true;
    } else if (type == "input_audio_buffer.append") {
      const std::optional<std::vector<float>> samples =
          AppendedSamples(message.value("audio", nlohmann::json()));
      appended_sample_count_ += samples.has_value() ? samples->size() : 0;
      return true;
    } else if (type == "transcription.done") {
      Send({{"type", "transcription.completed"},
            {"text", std::to_string(appended_sample_count_)}});
      return false;
    } else {
      SendError("invalid_request_error", "invalid_event",
                "Unknown event type.");
      return true;
    }
  }

private:
  size_t appended_sample_count_ = 0;
};

class StubAsrModel : public asr_service::ServedModel {
public:
  void ReleaseIdleMemory() override {}
  asr_service::Transcription
  Prepare(std::vector<float> samples,
          const asr_service::FormFields &form) const override {
    const std::string outcome =
        asr_service::TextField(form, "outcome").value_or("text");
    const std::optional<std::string> language =
        asr_service::TextField(form, "language");
    const int max_new_tokens =
        asr_service::IntegerField(form, "max_new_tokens").value_or(0);
    return [outcome, language, max_new_tokens,
            sample_count = samples.size()](const std::atomic<bool> &cancel) {
      if (outcome == "fail") {
        throw std::runtime_error("stub transcription failed");
      } else if (outcome == "block") {
        while (!cancel.load()) {
          std::this_thread::sleep_for(kBlockPollInterval);
        }
        throw qwen3_asr::TranscriptionCancelled();
      } else {
      }
      qwen3_asr::TranscriptionResult result;
      result.text = std::to_string(sample_count);
      result.language = language;
      result.generated_token_count = max_new_tokens;
      result.finish_reason = qwen3_asr::FinishReason::kStop;
      if (outcome == "segments") {
        result.segments.push_back({.start_seconds = 0.5,
                                   .end_seconds = 1.5,
                                   .speaker = "S01",
                                   .text = result.text});
      } else {
      }
      return result;
    };
  }

  void AddHandlers(mg_context *context,
                   qwen3_asr::TranscriptionWorker &) override {
    asr_service::AddRealtimeHandler(context, realtime_factory_);
  }

private:
  const asr_service::RealtimeFactory realtime_factory_ =
      [](qwen3_asr::RealtimeConnection::Sender sender) {
        return std::make_shared<StubRealtimeConnection>(std::move(sender));
      };
};

class StubSpeechModel : public speech_service::SpeechServedModel {
public:
  using SpeechServedModel::SpeechServedModel;
  void ReleaseIdleMemory() override {}

protected:
  speech_service::SpeechJob ParseSpeech(const Json &body) override {
    speech_service::SpeechJob job = speech_service::ParseCommonFields(body);
    job.model = speech_service::ParseModel(body);
    const std::string outcome =
        FieldOr(body, "outcome", Json("audio")).get<std::string>();
    const int chunk_count = FieldOr(body, "chunks", Json(1)).get<int>();
    const int chunk_sample_count =
        FieldOr(body, "chunk_samples", Json(kSpeechSampleRate)).get<int>();
    job.synthesize = [outcome, chunk_count, chunk_sample_count](
                         const speech_service::SpeechContext &context) {
      if (outcome == "fail") {
        throw std::runtime_error("stub speech failed");
      } else if (outcome == "invalid") {
        throw std::invalid_argument("stub invalid speech request");
      } else {
      }
      for (int chunk = 0; chunk < chunk_count; ++chunk) {
        context.send(std::vector<float>(chunk_sample_count, 0.0f));
      }
      if (outcome == "fail_after_audio") {
        throw std::runtime_error("stub speech failed after audio");
      } else if (outcome == "block") {
        while (!context.cancel.load()) {
          context.check_deadline();
          std::this_thread::sleep_for(kBlockPollInterval);
        }
        throw qwen3_asr::TranscriptionCancelled();
      } else {
      }
      speech_service::SpeechResult result;
      result.finish_reason = qwen3_asr::FinishReason::kStop;
      result.completion_token_count = chunk_count;
      return result;
    };
    return job;
  }

  int SampleRate() const override { return kSpeechSampleRate; }
};

} // namespace

int main(int argc, char **argv) {
  int load_delay_ms = 0;
  const auto set_load_delay = [&](const std::string &value) {
    load_delay_ms = std::stoi(value);
  };
  asr_service::ServedKind asr;
  asr.model_kind = "stub_asr";
  asr.flags = {{"--load-delay-ms", set_load_delay}};
  asr.load = [&](const std::filesystem::path &) {
    std::this_thread::sleep_for(std::chrono::milliseconds(load_delay_ms));
    return std::make_unique<StubAsrModel>();
  };
  asr_service::ServedKind speech;
  speech.model_kind = "stub_speech";
  speech_service::AddSpeechFlags(speech, [&](const std::filesystem::path &,
                                             int max_queued_requests,
                                             double request_timeout_seconds) {
    std::this_thread::sleep_for(std::chrono::milliseconds(load_delay_ms));
    return std::make_unique<StubSpeechModel>(
        kMaxRequestBodyBytes, max_queued_requests, request_timeout_seconds);
  });
  speech.flags["--load-delay-ms"] = set_load_delay;
  return asr_service::Serve(argc, argv, {asr, speech});
}
