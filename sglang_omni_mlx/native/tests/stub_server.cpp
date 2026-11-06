// SPDX-License-Identifier: Apache-2.0
// The servers' shared protocol around stand-in models, for the protocol tests:
// no checkpoint is read and no model runs.
//
//   stub_server --supervised --model-kind stub_asr|stub_speech|stub_chat
//       --model-directory DIR [--load-delay-ms N] [--max-queued-requests N]
//       [--request-timeout-s S]
//
// stub_asr transcribes audio to its decoded sample count, with the language
// and max_new_tokens fields as its generation metadata, and serves
// /v1/realtime, where transcription.done answers the sample count appended.
// stub_speech answers chunks chunks of chunk_samples silent samples. Both
// read an outcome field: "fail" throws, "block" runs until it is cancelled or
// its deadline passes, and stub_asr's "segments" returns one speaker segment.
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "common/http.h"
#include "common/service.h"
#include "ming3_audio/ming3_audio_service.h"
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
constexpr int kChatPromptTokenCount = 17;
constexpr int kChatCompletionTokenCount = 3;

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
      } else if (outcome == "draft") {
        result.verify_round_count = 2;
        result.proposed_draft_token_count = 7;
        result.accepted_draft_token_count = 3;
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
      } else if (outcome == "block") {
        while (!context.cancel.load()) {
          context.check_deadline();
          std::this_thread::sleep_for(kBlockPollInterval);
        }
        throw qwen3_asr::TranscriptionCancelled();
      } else {
      }
      for (int chunk = 0; chunk < chunk_count; ++chunk) {
        context.send(std::vector<float>(chunk_sample_count, 0.0f));
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

class StubChatService : public omni_server::Service {
public:
  StubChatService()
      : service_(ming3_audio::CreateChatService(
            [](ming3_audio::ChatRequest request) {
              if (request.system_prompt == "invalid") {
                throw std::invalid_argument("stub chat preparation failed");
              } else {
                return ming3_audio::PreparedChatRequest{
                    std::move(request), {}, 0, 0};
              }
            },
            [this](const ming3_audio::PreparedChatRequest &prepared,
                   const ming3_audio::Cancellation &cancelled) {
              const auto &request = prepared.request;
              if (request.system_prompt == "fail") {
                throw std::runtime_error("stub chat failed");
              } else if (request.system_prompt == "cancel") {
                throw ming3_audio::Cancelled();
              } else if (request.system_prompt == "block") {
                std::unique_lock lock(generation_mutex_);
                is_blocked_ = true;
                is_released_ = false;
                generation_ready_.notify_all();
                while (!is_released_) {
                  if (cancelled()) {
                    is_blocked_ = false;
                    throw ming3_audio::Cancelled();
                  } else {
                    generation_ready_.wait_for(lock, kBlockPollInterval);
                  }
                }
                is_blocked_ = false;
              } else {
              }
              const int completion_tokens =
                  std::min(request.max_new_tokens, kChatCompletionTokenCount);
              return ming3_audio::ChatResult{
                  .text = Json({{"system_prompt", request.system_prompt},
                                {"before_audio", request.before_audio},
                                {"after_audio", request.after_audio},
                                {"samples", request.samples},
                                {"max_new_tokens", request.max_new_tokens}})
                              .dump(),
                  .finish_reason = completion_tokens < kChatCompletionTokenCount
                                       ? "length"
                                       : "stop",
                  .prompt_tokens = kChatPromptTokenCount,
                  .completion_tokens = completion_tokens};
            })) {}

  void Register(mg_context *context, const std::string &model_name) override {
    service_->Register(context, model_name);
    mg_set_request_handler(
        context, "/stub/chat/started$",
        [](mg_connection *connection, void *instance) {
          auto &service = *static_cast<StubChatService *>(instance);
          std::unique_lock lock(service.generation_mutex_);
          service.generation_ready_.wait(lock,
                                         [&] { return service.is_blocked_; });
          return omni_server::WriteJson(connection, 200, Json::object());
        },
        this);
    mg_set_request_handler(
        context, "/stub/chat/release$",
        [](mg_connection *connection, void *instance) {
          auto &service = *static_cast<StubChatService *>(instance);
          {
            std::lock_guard lock(service.generation_mutex_);
            service.is_released_ = true;
          }
          service.generation_ready_.notify_all();
          return omni_server::WriteJson(connection, 200, Json::object());
        },
        this);
  }

  std::map<std::string, int> RequestStates() const override {
    return service_->RequestStates();
  }

  void CancelAll() override { service_->CancelAll(); }

private:
  std::mutex generation_mutex_;
  std::condition_variable generation_ready_;
  bool is_blocked_ = false;
  bool is_released_ = false;
  std::unique_ptr<omni_server::Service> service_;
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
  const omni_server::ServedKind chat{
      "stub_chat",
      [](const std::filesystem::path &) {
        return std::make_unique<StubChatService>();
      },
      {},
      {}};
  return omni_server::Serve(argc, argv,
                            {asr_service::CreateServedKind(asr),
                             asr_service::CreateServedKind(speech), chat});
}
