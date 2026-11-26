// SPDX-License-Identifier: Apache-2.0
// Native MLX server for Voxt, one model per process on asr_service:
// qwen3_asr (OpenAI-style transcriptions, with the realtime API on
// /v1/realtime), silero_vad (vad_service.h's API, no transcriptions) or
// sortformer (sortformer_service.h's API, no transcriptions).
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>

#include "common/asr_service.h"
#include "qwen3_asr/realtime.h"
#include "silero_vad/vad_service.h"
#include "sortformer/sortformer_service.h"

namespace {

using qwen3_asr::AudioLayout;
using qwen3_asr::TranscriptionOptions;

class Qwen3ASRModel : public asr_service::ServedModel {
public:
  Qwen3ASRModel(const std::filesystem::path &model_directory,
                qwen3_asr::RealtimeSettings realtime,
                const omni_common::DraftOptions &draft,
                omni_common::DraftPolicy draft_policy)
      : transcriber_(model_directory, draft, draft_policy),
        realtime_(realtime) {
    mlx::core::set_cache_limit(asr_service::kDecodeBufferCacheLimitBytes);
  }

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
    options.repetition_penalty =
        asr_service::NumberField(form, "repetition_penalty")
            .value_or(options.repetition_penalty);
    if (options.max_new_tokens.value_or(0) < 0) {
      throw std::invalid_argument("max_new_tokens must not be negative");
    } else if (options.repetition_penalty <= 0) {
      throw std::invalid_argument("repetition_penalty must be positive");
    } else {
    }
    const auto guessed_text = asr_service::TextField(form, "guessed_text");
    if (guessed_text.has_value()) {
      // note (XXXXRT666): a live preview already has most of the text; only
      // the part greedy decoding also writes is kept.
      options.guessed_token_ids =
          transcriber_.RetainedPrefix(*guessed_text, 0).prefix_ids;
    } else {
    }
    options.stop_at_end_of_text = omni_common::FormFlag(
        asr_service::TextField(form, "stop_at_end_of_text"));
    const auto stop_on_token_loop =
        asr_service::TextField(form, "stop_on_token_loop");
    // Note (Minyi Tang): greedy decoding loops on audio with little speech, so
    // the guard is on unless a request turns it off.
    if (stop_on_token_loop.has_value()) {
      options.stop_on_token_loop = omni_common::FormFlag(stop_on_token_loop);
    } else {
      options.stop_on_token_loop = true;
    }
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
    // note (chenye): the caller opts in when the request covers the whole
    // recording. Other requests retain their existing behavior, including
    // explicit guessed_text.
    const bool guess_auto_language =
        omni_common::FormFlag(
            asr_service::TextField(form, "guess_auto_language")) &&
        !options.language.has_value() && !guessed_text.has_value() &&
        options.repetition_penalty == 1.0f &&
        !omni_common::FormFlag(asr_service::TextField(form, "preview"));
    // note (XXXXRT666): a session holds one prompt's windows, and longer audio
    // is decoded in chunks.
    const qwen3_asr::FinalReuse final_reuse =
        samples.size() <= qwen3_asr::kMaxChunkSamples
            ? TakeFinalReuse(
                  asr_service::TextField(form, "realtime_session").value_or(""),
                  options.context, guess_auto_language)
            : qwen3_asr::FinalReuse();
    if (final_reuse.reuse != nullptr) {
      // note (chenye): auto-language decoding generates a language header
      // before the transcript. Propose both from the session's preview and
      // verify them through the existing guessed-token path. Leave
      // options.language unset so the decoder still detects the language.
      if (guess_auto_language && final_reuse.auto_language_guess.has_value()) {
        const auto &guess = *final_reuse.auto_language_guess;
        options.guessed_token_ids =
            transcriber_.AutoLanguageGuesses(guess.language, guess.text);
      }
      return [this, final_reuse, samples = std::move(samples),
              options = std::move(options)](const std::atomic<bool> &cancel) {
        // note (XXXXRT666): a copy the refreshes keep using shares the
        // buffers, so the decode copies them; one taken for the Final does
        // not.
        qwen3_asr::RefreshReuse reuse = final_reuse.is_taken
                                            ? std::move(*final_reuse.reuse)
                                            : *final_reuse.reuse;
        return transcriber_.Transcribe(samples, options, cancel, &reuse);
      };
    } else if (transcriber_.CanBatch(samples.size(), options)) {
      return qwen3_asr::BatchableTranscription{
          &transcriber_, std::move(samples), std::move(options)};
    } else {
      return [this, samples = std::move(samples),
              options = std::move(options)](const std::atomic<bool> &cancel) {
        return transcriber_.Transcribe(samples, options, cancel);
      };
    }
  }

  void AddHandlers(mg_context *context,
                   omni_server::TranscriptionWorker &worker) override {
    realtime_sessions_ = [this,
                          &worker](qwen3_asr::RealtimeSession::Sender sender) {
      const auto session = std::make_shared<qwen3_asr::RealtimeSession>(
          worker, transcriber_, realtime_, std::move(sender));
      std::lock_guard<std::mutex> lock(live_sessions_mutex_);
      std::erase_if(live_sessions_,
                    [](const auto &entry) { return entry.second.expired(); });
      live_sessions_[session->session_id()] = session;
      return session;
    };
    asr_service::AddRealtimeHandler(context, realtime_sessions_);
  }

private:
  // What the live session of this id holds for a Final with this prompt, null
  // when it is not open.
  qwen3_asr::FinalReuse
  TakeFinalReuse(const std::string &session_id,
                 const std::optional<std::string> &context,
                 bool guess_auto_language) const {
    std::lock_guard<std::mutex> lock(live_sessions_mutex_);
    const auto entry = live_sessions_.find(session_id);
    const std::shared_ptr<qwen3_asr::RealtimeSession> session =
        entry != live_sessions_.end() ? entry->second.lock() : nullptr;
    return session != nullptr
               ? session->TakeFinalReuse(context, guess_auto_language)
               : qwen3_asr::FinalReuse();
  }

  qwen3_asr::Qwen3ASRTranscriber transcriber_;
  const qwen3_asr::RealtimeSettings realtime_;
  asr_service::RealtimeFactory realtime_sessions_;
  mutable std::mutex live_sessions_mutex_;
  std::map<std::string, std::weak_ptr<qwen3_asr::RealtimeSession>>
      live_sessions_;
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
  RequestStates(const omni_server::TranscriptionWorker &) const override {
    return service_.RequestStates();
  }
  void AddHandlers(mg_context *context,
                   omni_server::TranscriptionWorker &) override {
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
  RequestStates(const omni_server::TranscriptionWorker &) const override {
    return service_.RequestStates();
  }
  void AddHandlers(mg_context *context,
                   omni_server::TranscriptionWorker &) override {
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
  long long cache_limit_mb = -1;
  int max_batch = 1;
  omni_common::DraftOptions draft;
  omni_common::DraftPolicy draft_policy = omni_common::DraftPolicy::kAdaptive;
  asr_service::ServedKind qwen3_asr;
  qwen3_asr.model_kind = "qwen3_asr";
  qwen3_asr.flags = draft.Flags();
  qwen3_asr.flags.insert({
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
      {"--cache-limit-mb",
       [&](const std::string &value) { cache_limit_mb = std::stoll(value); }},
      {"--max-batch",
       [&](const std::string &value) { max_batch = std::stoi(value); }},
      {"--draft-policy",
       [&](const std::string &value) {
         draft_policy = omni_common::ParseDraftPolicy(value);
       }},
  });
  qwen3_asr.check_flags = [&]() {
    const double max_segment_samples =
        max_segment_seconds * omni_common::kSampleRate;
    if (decode_interval_ms <= 0) {
      throw std::invalid_argument("--decode-interval-ms must be positive");
    } else if (first_decode_ms < 0) {
      throw std::invalid_argument("--first-decode-ms must not be negative");
    } else if (!(max_segment_samples >= 1 &&
                 max_segment_samples <= std::numeric_limits<int>::max())) {
      throw std::invalid_argument(
          "--max-segment-seconds must span one sample to 134217 s");
    } else if (cache_limit_mb < -1 || cache_limit_mb > (1LL << 30)) {
      throw std::invalid_argument(
          "--cache-limit-mb must be 0 to 1073741824, or -1 for MLX's limit");
    } else if (max_batch < 1) {
      throw std::invalid_argument("--max-batch must be at least 1");
    } else {
    }
    draft.Check();
  };
  qwen3_asr.max_batch = [&]() { return max_batch; };
  qwen3_asr.batch_runner = qwen3_asr::BatchableTranscriptionRunner();
  qwen3_asr.load = [&](const std::filesystem::path &model_directory) {
    auto model = std::make_unique<Qwen3ASRModel>(
        model_directory,
        qwen3_asr::MakeRealtimeSettings(decode_interval_ms, first_decode_ms,
                                        max_segment_seconds),
        draft, draft_policy);
    // Note (hehehai): within one long Final MLX keeps every freed buffer
    // for reuse (the growing KV cache, the encoder's activations), several
    // GB beside the weights; a client can bound it. Decoding reuses buffers
    // of a few sizes, so a few hundred MB keep its speed.
    // Note (cosekii): set after construction so the flag overrides the
    // model's default cap.
    if (cache_limit_mb >= 0) {
      mlx::core::set_cache_limit(static_cast<size_t>(cache_limit_mb) << 20);
    } else {
    }
    return model;
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
