// SPDX-License-Identifier: Apache-2.0
// The speech API the native TTS servers share: POST /v1/audio/speech answered
// with a WAV or PCM file, or 16-bit PCM streamed as it is decoded. One request
// runs at a time and a few more may wait; the rest get 503. A request not done
// by its deadline is cancelled, with 504 or an early end to its stream, and so
// is a request whose client disconnects. A body over the model's limit gets
// 413.
#pragma once

#include <atomic>
#include <chrono>
#include <functional>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "common/http.h"
#include "qwen3_asr/asr_service.h"
#include "qwen3_asr/qwen3_asr_transcriber.h"

namespace speech_service {

using omni_server::Json;

struct SpeechResult {
  qwen3_asr::FinishReason finish_reason = qwen3_asr::FinishReason::kStop;
  size_t prompt_token_count = 0;
  int completion_token_count = 0;
};

// What a synthesis on the worker thread is handed: its cancel flag, where
// its decoded audio goes, and a check that throws once its request's deadline
// has passed.
struct SpeechContext {
  const std::atomic<bool> &cancel;
  std::function<void(std::vector<float>)> send;
  std::function<void()> check_deadline;
};

// A speech request a model parsed: the fields every server reads, and the
// synthesis that answers it.
struct SpeechJob {
  std::optional<std::string> model;
  std::string response_format;
  bool stream = false;
  std::function<SpeechResult(const SpeechContext &)> synthesize;
};

// body's field, or fallback when it is missing or null.
Json FieldOr(const Json &body, const char *name, Json fallback);
// body's field, null included, or fallback when it is missing.
Json PresentFieldOr(const Json &body, const char *name, Json fallback);
// The model field; throws std::invalid_argument unless it is a string or
// null.
std::optional<std::string> ParseModel(const Json &body);
// The response_format, stream and stream_format fields, as the CUDA
// server's CreateSpeechRequest reads them; throws std::invalid_argument.
SpeechJob ParseCommonFields(const Json &body);

class SpeechServedModel : public asr_service::ServedModel {
public:
  SpeechServedModel(size_t max_request_body_bytes, int max_queued_requests,
                    double request_timeout_seconds);

  asr_service::Transcription
  Prepare(std::vector<float>, const asr_service::FormFields &) const override;
  bool Transcribes() const override { return false; }
  std::map<std::string, int>
  RequestStates(const qwen3_asr::TranscriptionWorker &) const override;
  void AddHandlers(mg_context *context,
                   qwen3_asr::TranscriptionWorker &worker) override;

protected:
  // The job of a request body; throws std::invalid_argument for a request
  // to answer with 400.
  virtual SpeechJob ParseSpeech(const Json &body) = 0;
  virtual int SampleRate() const = 0;

private:
  static int HandleSpeech(mg_connection *connection, void *data);
  int Speak(mg_connection *connection);

  const size_t max_request_body_bytes_;
  const int max_queued_requests_;
  const std::chrono::steady_clock::duration request_timeout_;
  mutable std::mutex admission_mutex_;
  // Requests the worker holds: running, queued, or finishing.
  int admitted_requests_ = 0;
  qwen3_asr::TranscriptionWorker *worker_ = nullptr;
};

// Adds --max-queued-requests and --request-timeout-s to kind; load then
// receives their values.
void AddSpeechFlags(asr_service::ServedKind &kind,
                    std::function<std::unique_ptr<asr_service::ServedModel>(
                        const std::filesystem::path &, int, double)>
                        load);

} // namespace speech_service
