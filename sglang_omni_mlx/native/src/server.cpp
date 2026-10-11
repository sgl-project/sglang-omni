// SPDX-License-Identifier: Apache-2.0
// Native MLX server for Voxt, one model per process on asr_service:
// qwen3_asr (the API of sglang_omni_mlx.qwen3_asr.server, with the realtime
// API on /v1/realtime), silero_vad (vad_service.h's API, no transcriptions) or
// sortformer (sortformer_service.h's API, no transcriptions).
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

#include "asr_service.h"
#include "realtime.h"
#include "sortformer_service.h"
#include "vad_service.h"

namespace {

using qwen3_asr::AudioLayout;
using qwen3_asr::TranscriptionOptions;

class Qwen3ASRModel : public asr_service::ServedModel {
public:
  Qwen3ASRModel(const std::filesystem::path &model_directory,
                qwen3_asr::RealtimeSettings realtime)
      : transcriber_(model_directory), realtime_(realtime) {}

  asr_service::Transcription
  Prepare(std::vector<float> samples,
          const asr_service::FormFields &form) const override {
    TranscriptionOptions options;
    const auto language = asr_service::TextField(form, "language");
    options.language = language.has_value()
                           ? qwen3_asr::NormalizeLanguage(*language)
                           : std::nullopt;
    options.context = asr_service::TextField(form, "prompt");
    options.max_new_tokens = asr_service::IntegerField(form, "max_new_tokens");
    if (options.max_new_tokens.value_or(0) < 0) {
      throw std::invalid_argument("max_new_tokens must not be negative");
    } else {
    }
    options.stop_at_end_of_text = qwen3_asr::FormFlag(
        asr_service::TextField(form, "stop_at_end_of_text"));
    options.stop_on_token_loop =
        qwen3_asr::FormFlag(asr_service::TextField(form, "stop_on_token_loop"));
    const std::string layout =
        asr_service::TextField(form, "audio_layout").value_or("");
    if (layout.empty() || layout == "reference") {
      options.layout = AudioLayout::kReference;
    } else if (layout == "voxt_swift") {
      options.layout = AudioLayout::kVoxtSwift;
    } else {
      throw std::invalid_argument(
          "audio_layout must be reference or voxt_swift");
    }
    return [this, samples = std::move(samples),
            options = std::move(options)](const std::atomic<bool> &cancel) {
      return transcriber_.Transcribe(samples, options, cancel);
    };
  }

  void AddHandlers(mg_context *context,
                   qwen3_asr::TranscriptionWorker &worker) override {
    realtime_sessions_ = [this,
                          &worker](qwen3_asr::RealtimeSession::Sender sender) {
      return std::make_shared<qwen3_asr::RealtimeSession>(
          worker, transcriber_, realtime_, std::move(sender));
    };
    asr_service::AddRealtimeHandler(context, realtime_sessions_);
  }

private:
  qwen3_asr::Qwen3ASRTranscriber transcriber_;
  const qwen3_asr::RealtimeSettings realtime_;
  asr_service::RealtimeFactory realtime_sessions_;
};

class SileroVADModel : public asr_service::ServedModel {
public:
  explicit SileroVADModel(const std::filesystem::path &model_directory)
      : service_(model_directory) {
    // Note (Jiaxin Deng): freed MLX buffers go back to the system, so an idle
    // server holds only the model.
    mlx::core::set_cache_limit(0);
  }

  asr_service::Transcription
  Prepare(std::vector<float>, const asr_service::FormFields &) const override {
    throw std::logic_error("silero_vad serves no transcriptions");
  }
  bool Transcribes() const override { return false; }
  std::map<std::string, int>
  RequestStates(const qwen3_asr::TranscriptionWorker &) const override {
    return service_.RequestStates();
  }
  void AddHandlers(mg_context *context,
                   qwen3_asr::TranscriptionWorker &) override {
    service_.Register(context);
  }

private:
  silero_vad::VADService service_;
};

class SortformerModel : public asr_service::ServedModel {
public:
  explicit SortformerModel(const std::filesystem::path &model_directory)
      : service_(model_directory) {
    mlx::core::set_cache_limit(0);
  }

  asr_service::Transcription
  Prepare(std::vector<float>, const asr_service::FormFields &) const override {
    throw std::logic_error("sortformer serves no transcriptions");
  }
  bool Transcribes() const override { return false; }
  std::map<std::string, int>
  RequestStates(const qwen3_asr::TranscriptionWorker &) const override {
    return service_.RequestStates();
  }
  void AddHandlers(mg_context *context,
                   qwen3_asr::TranscriptionWorker &) override {
    service_.Register(context);
  }

private:
  sortformer::SortformerService service_;
};

} // namespace

int main(int argc, char **argv) {
  int decode_interval_ms = 1000;
  int first_decode_ms = 100;
  double max_segment_seconds = 30.0;
  asr_service::ServedKind qwen3_asr;
  qwen3_asr.model_kind = "qwen3_asr";
  qwen3_asr.flags = {
      {"--decode-interval-ms",
       [&](const std::string &value) {
         decode_interval_ms = std::stoi(value);
       }},
      {"--first-decode-ms",
       [&](const std::string &value) { first_decode_ms = std::stoi(value); }},
      {"--max-segment-seconds",
       [&](const std::string &value) {
         max_segment_seconds = std::stod(value);
       }},
  };
  qwen3_asr.check_flags = [&]() {
    const double max_segment_samples =
        max_segment_seconds * qwen3_asr::kSampleRate;
    if (decode_interval_ms <= 0) {
      throw std::invalid_argument("--decode-interval-ms must be positive");
    } else if (first_decode_ms < 0) {
      throw std::invalid_argument("--first-decode-ms must not be negative");
    } else if (!(max_segment_samples >= 1 &&
                 max_segment_samples <= std::numeric_limits<int>::max())) {
      throw std::invalid_argument(
          "--max-segment-seconds must span one sample to 134217 s");
    } else {
    }
  };
  qwen3_asr.load = [&](const std::filesystem::path &model_directory) {
    return std::make_unique<Qwen3ASRModel>(
        model_directory,
        qwen3_asr::MakeRealtimeSettings(decode_interval_ms, first_decode_ms,
                                        max_segment_seconds));
  };
  // Note (khazic): Voxt passes the realtime flags to every kind.
  asr_service::ServedKind silero_vad = qwen3_asr;
  silero_vad.model_kind = "silero_vad";
  silero_vad.load = [](const std::filesystem::path &model_directory) {
    return std::make_unique<SileroVADModel>(model_directory);
  };
  asr_service::ServedKind sortformer = qwen3_asr;
  sortformer.model_kind = "sortformer";
  sortformer.load = [](const std::filesystem::path &model_directory) {
    return std::make_unique<SortformerModel>(model_directory);
  };
  return asr_service::Serve(argc, argv, {qwen3_asr, silero_vad, sortformer});
}
