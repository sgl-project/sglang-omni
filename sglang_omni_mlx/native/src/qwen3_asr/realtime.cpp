// SPDX-License-Identifier: Apache-2.0
#include "qwen3_asr/realtime.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "common/audio.h"
#include "common/generation.h"
#include "common/text.h"
#include "common/transcription.h"

namespace qwen3_asr {

using omni_common::FinishReason;
using omni_common::GenerationCancelled;
using omni_common::JoinTranscriptParts;
using omni_common::TranscriptionResult;
using omni_server::Completion;
using omni_server::Transcription;
using omni_server::TranscriptionWorker;

namespace {

// Note (Jiaxin Deng): after two full re-decodes a refresh continues from the
// shown text minus its last tokens, which may still change.
constexpr int kPrefixAfterRefreshCount = 2;
constexpr int kPrefixRollbackTokenCount = 5;
// A decode with a prefix has tokens for the rolled-back ones and for the
// audio after the transcript's decode, at the offline default's rate, plus
// slack; speech uses half of that at most.
constexpr double kPrefixedTokensPerNewSecond = 10.0;
constexpr int kPrefixedTokenSlack = 8;
// The audio from where its Final starts that a session prefills, and that it
// keeps for that: Voxt recognises a longer recording in pieces while it
// records, and the Final that is left starts where the last piece ended.
constexpr size_t kMaxFinalPrefillSamples = 64 * omni_common::kSampleRate;
constexpr size_t kMaxFinalRecordingSamples = 128 * omni_common::kSampleRate;
// A shadow Final runs once per window of the recording, a little after the
// window completes.
constexpr long kShadowIntervalSamples = 8 * omni_common::kSampleRate;
constexpr long kShadowDelaySamples = omni_common::kSampleRate / 10;
// The shadow's last tokens, which the audio after it may change, and the
// tokens before them that find where the preview goes on.
constexpr int kShadowUnsettledTokens = 4;
constexpr int kShadowAnchorTokens = 3;
// How long after a request shadow Finals wait: a request prepared but not yet
// queued would otherwise wait behind one queued meanwhile.
constexpr std::chrono::seconds kShadowQuietTime{2};

std::mutex attribution_output_mutex;

std::string_view PauseHintDispositionName(PauseHintDisposition disposition) {
  switch (disposition) {
  case PauseHintDisposition::kInvalid:
    return "invalid";
  case PauseHintDisposition::kDuplicate:
    return "duplicate";
  case PauseHintDisposition::kDisabled:
    return "disabled";
  case PauseHintDisposition::kInvalidDeferredMetadata:
    return "invalid_deferred_metadata";
  case PauseHintDisposition::kRefreshBusy:
    return "refresh_busy";
  case PauseHintDisposition::kCovered:
    return "covered_by_submitted_refresh";
  case PauseHintDisposition::kInsufficientNewSpeech:
    return "insufficient_new_speech";
  case PauseHintDisposition::kSupersededByCadence:
    return "superseded_by_cadence";
  case PauseHintDisposition::kSilent:
    return "silent";
  case PauseHintDisposition::kFinalHandoff:
    return "final_handoff";
  case PauseHintDisposition::kSegmentFinalization:
    return "segment_finalization";
  case PauseHintDisposition::kCancelled:
    return "cancelled";
  case PauseHintDisposition::kSpeechResumption:
    return "speech_resumption";
  case PauseHintDisposition::kSegmentRollover:
    return "segment_rollover";
  case PauseHintDisposition::kDeferred:
    return "deferred";
  case PauseHintDisposition::kSuperseded:
    return "superseded";
  case PauseHintDisposition::kAdmittedImmediate:
    return "admitted_immediate";
  case PauseHintDisposition::kAdmittedDeferred:
    return "admitted_deferred";
  case PauseHintDisposition::kSessionClose:
    return "session_close";
  case PauseHintDisposition::kNone:
    return "none";
  }
  return "none";
}

int64_t SteadyMicros(std::chrono::steady_clock::time_point point) {
  return std::chrono::duration_cast<std::chrono::microseconds>(
             point.time_since_epoch())
      .count();
}

nlohmann::json SteadyMicrosJson(
    const std::optional<std::chrono::steady_clock::time_point> &point) {
  return point.has_value() ? nlohmann::json(SteadyMicros(*point))
                           : nlohmann::json();
}

nlohmann::json ElapsedMicrosJson(
    const std::optional<std::chrono::steady_clock::time_point> &start,
    const std::optional<std::chrono::steady_clock::time_point> &finish) {
  if (!start.has_value() || !finish.has_value()) {
    return nlohmann::json();
  }
  return std::chrono::duration_cast<std::chrono::microseconds>(*finish - *start)
      .count();
}

std::atomic<std::chrono::steady_clock::rep> last_foreground_request{0};

bool IsShadowQuiet() {
  return std::chrono::steady_clock::now().time_since_epoch().count() -
             last_foreground_request.load() <
         std::chrono::duration_cast<std::chrono::steady_clock::duration>(
             kShadowQuietTime)
             .count();
}

std::optional<long> NonnegativeLong(const nlohmann::json &value) {
  const uint64_t maximum =
      static_cast<uint64_t>(std::numeric_limits<long>::max());
  if (value.is_number_unsigned()) {
    const uint64_t number = value.get<uint64_t>();
    return number <= maximum ? std::optional<long>(static_cast<long>(number))
                             : std::nullopt;
  }
  if (value.is_number_integer()) {
    const int64_t number = value.get<int64_t>();
    return number >= 0 && static_cast<uint64_t>(number) <= maximum
               ? std::optional<long>(static_cast<long>(number))
               : std::nullopt;
  }
  return std::nullopt;
}

} // namespace

void NoteForegroundRequest() {
  last_foreground_request.store(
      std::chrono::steady_clock::now().time_since_epoch().count());
}

std::vector<int> MergedGuesses(const std::vector<int> &shadow_ids,
                               const std::vector<int> &preview_ids) {
  const int settled_count =
      static_cast<int>(shadow_ids.size()) - kShadowUnsettledTokens;
  if (settled_count < kShadowAnchorTokens) {
    return preview_ids.empty() ? shadow_ids : preview_ids;
  } else {
  }
  const auto settled_end = shadow_ids.begin() + settled_count;
  // The place in the preview after the anchor nearest the settled tokens'
  // count.
  std::optional<int> best_end;
  for (int end = kShadowAnchorTokens;
       end <= static_cast<int>(preview_ids.size()); ++end) {
    if (std::equal(settled_end - kShadowAnchorTokens, settled_end,
                   preview_ids.begin() + end - kShadowAnchorTokens) &&
        (!best_end.has_value() ||
         std::abs(end - settled_count) < std::abs(*best_end - settled_count))) {
      best_end = end;
    } else {
    }
  }
  if (!best_end.has_value()) {
    return shadow_ids;
  } else {
  }
  std::vector<int> merged(shadow_ids.begin(), settled_end);
  merged.insert(merged.end(), preview_ids.begin() + *best_end,
                preview_ids.end());
  return merged;
}

RealtimeSettings MakeRealtimeSettings(int decode_interval_ms,
                                      int first_decode_ms,
                                      double max_segment_seconds) {
  return {decode_interval_ms * omni_common::kSampleRate / 1000,
          first_decode_ms * omni_common::kSampleRate / 1000,
          static_cast<int>(max_segment_seconds * omni_common::kSampleRate)};
}

RealtimeSession::RealtimeSession(TranscriptionWorker &worker,
                                 const Qwen3ASRTranscriber &transcriber,
                                 RealtimeSettings settings, Sender sender,
                                 bool hold_gpu_clock)
    : RealtimeConnection(std::move(sender)), worker_(worker),
      transcriber_(transcriber), settings_(settings) {
  if (hold_gpu_clock) {
    gpu_clock_.emplace();
  } else {
  }
}

bool RealtimeSession::Handle(const nlohmann::json &message) {
  const nlohmann::json type = message.value("type", nlohmann::json());
  if (type == "session.update") {
    const std::optional<nlohmann::json> session = ManualTurnSession(message);
    if (session.has_value()) {
      const nlohmann::json language =
          session->value("language", nlohmann::json());
      const nlohmann::json final_context =
          session->value("final_prompt", nlohmann::json());
      const auto preview_refresh_enabled =
          session->value("preview_refresh_enabled", nlohmann::json());
      const auto defer_busy_enabled = session->value(
          "preview_refresh_defer_busy_enabled", nlohmann::json());
      const auto attribution_enabled = session->value(
          "preview_refresh_attribution_enabled", nlohmann::json());
      {
        std::lock_guard<std::mutex> lock(mutex_);
        language_ = language.is_string()
                        ? NormalizeLanguage(language.get<std::string>())
                        : std::nullopt;
        if (session->contains("preview_refresh_enabled")) {
          preview_refresh_enabled_ = preview_refresh_enabled.is_boolean() &&
                                     preview_refresh_enabled.get<bool>();
        } else {
        }
        const bool was_deferred = preview_refresh_defer_busy_enabled_;
        preview_refresh_defer_busy_enabled_ = defer_busy_enabled.is_boolean() &&
                                              defer_busy_enabled.get<bool>() &&
                                              preview_refresh_enabled_;
        preview_refresh_attribution_enabled_ =
            attribution_enabled.is_boolean() && attribution_enabled.get<bool>();
        if (was_deferred && !preview_refresh_defer_busy_enabled_) {
          DiscardPendingPauseHint(PauseHintDisposition::kDisabled);
          if (segment_.has_value()) {
            segment_->speech_epoch.reset();
            segment_->pause_metadata_valid = true;
          }
        } else {
        }
        // note (XXXXRT666): the prefill takes the recording from its first
        // sample, so a prompt sent after audio has none.
        if (final_context.is_string() && !should_prefill_final_ &&
            final_reuse_ == nullptr && EndSample() == 0) {
          final_context_ = final_context.get<std::string>();
          should_prefill_final_ = true;
        } else {
        }
      }
      LogAttributionLazy([&] {
        return nlohmann::ordered_json{
            {"event", "session"},
            {"pause_refresh_enabled", preview_refresh_enabled.is_boolean() &&
                                          preview_refresh_enabled.get<bool>()},
            {"deferred_busy_enabled", defer_busy_enabled.is_boolean() &&
                                          defer_busy_enabled.get<bool>()}};
      });
      Send({{"type", "transcription_session.updated"},
            {"session_id", session_id()}});
    } else {
    }
    return true;
  } else if (type == "input_audio_buffer.append") {
    if (gpu_clock_.has_value()) {
      gpu_clock_->Touch();
    } else {
    }
    Append(message.value("audio", nlohmann::json()),
           message.value("preview_refresh", nlohmann::json()),
           message.value("speech_epoch", nlohmann::json()),
           message.contains("preview_refresh"));
    return true;
  } else if (type == "input_audio_buffer.final_start") {
    StartFinalAt(message.value("start_sample", nlohmann::json()));
    return true;
  } else if (type == "input_audio_buffer.commit") {
    FinalizeThrough(LockedEndSample());
    return true;
  } else if (type == "transcription.done") {
    FinalizeThrough(LockedEndSample());
    std::vector<std::pair<int, std::string>> committed;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      committed = committed_;
    }
    std::sort(committed.begin(), committed.end());
    std::vector<std::string> texts;
    for (const auto &[segment_id, text] : committed)
      texts.push_back(text);
    Send({{"type", "transcription.completed"},
          {"text", JoinTranscriptParts(texts)}});
    return false;
  } else {
    SendError("invalid_request_error", "invalid_event", "Unknown event type.");
    return true;
  }
}

void RealtimeSession::Close() {
  omni_server::RealtimeConnection::Close();
  std::lock_guard<std::mutex> lock(mutex_);
  DiscardPendingPauseHint(PauseHintDisposition::kSessionClose);
  LogAttributionSummary();
}

void RealtimeSession::DiscardPendingPauseHint(
    PauseHintDisposition disposition) {
  if (!segment_.has_value() || !segment_->pending_pause.has_value()) {
    return;
  } else {
  }
  RecordPauseHintDisposition(disposition, segment_->pending_pause,
                             segment_->last_submitted_refresh_end_sample,
                             segment_->last_completed_refresh_end_sample);
  segment_->pending_pause.reset();
}

void RealtimeSession::RecordPauseHintDisposition(
    PauseHintDisposition disposition, std::optional<PauseRefreshHint> hint,
    long submitted_refresh_end_sample, long completed_refresh_end_sample) {
  if (disposition == PauseHintDisposition::kNone) {
    return;
  }
  LogAttributionLazy([&] {
    nlohmann::ordered_json event = {
        {"event", "pause_hint"},
        {"disposition", PauseHintDispositionName(disposition)}};
    if (segment_.has_value()) {
      event["segment_id"] = segment_->segment_id;
    }
    if (hint.has_value()) {
      event["speech_start_sample"] = hint->speech_start_sample;
      event["speech_end_sample"] = hint->speech_end_sample;
    }
    if (submitted_refresh_end_sample >= 0) {
      event["last_submitted_refresh_end_sample"] = submitted_refresh_end_sample;
    }
    if (completed_refresh_end_sample >= 0) {
      event["last_completed_refresh_end_sample"] = completed_refresh_end_sample;
    }
    return event;
  });
}

void RealtimeSession::LogAttribution(nlohmann::ordered_json event) const {
  if (!preview_refresh_attribution_enabled_.load()) {
    return;
  }
  event["session_id"] = session_id();
  std::lock_guard<std::mutex> lock(attribution_output_mutex);
  std::clog << "qwen3_asr_pause_refresh " << event.dump() << '\n';
}

void RealtimeSession::LogAttributionSummary() {
  if (!preview_refresh_attribution_enabled_.load() ||
      attribution_summary_logged_) {
    return;
  } else {
  }
  attribution_summary_logged_ = true;
  LogAttributionLazy([&] {
    return nlohmann::ordered_json{{"event", "summary"},
                                  {"hints_received", pause_hints_received_}};
  });
}

long RealtimeSession::LockedEndSample() {
  std::lock_guard<std::mutex> lock(mutex_);
  return EndSample();
}

void RealtimeSession::Append(const nlohmann::json &audio,
                             const nlohmann::json &preview_refresh,
                             const nlohmann::json &speech_epoch,
                             bool pause_hint_received) {
  const std::optional<std::vector<float>> appended = AppendedSamples(audio);
  if (!appended.has_value()) {
    return;
  } else {
  }
  std::optional<PauseRefreshHint> pause_hint;
  bool pause_hint_malformed = false;
  if (pause_hint_received && !preview_refresh.is_object()) {
    pause_hint_malformed = true;
  } else {
  }
  if (preview_refresh.is_object()) {
    const std::optional<long> speech_start_sample = NonnegativeLong(
        preview_refresh.value("speech_start_sample", nlohmann::json()));
    const std::optional<long> speech_end_sample = NonnegativeLong(
        preview_refresh.value("speech_end_sample", nlohmann::json()));
    if (speech_start_sample.has_value() && speech_end_sample.has_value()) {
      pause_hint = PauseRefreshHint{*speech_start_sample, *speech_end_sample};
    } else {
      pause_hint_malformed = true;
    }
  } else {
  }
  const std::optional<long> parsed_speech_epoch = NonnegativeLong(speech_epoch);
  long cut_sample = -1;
  bool moved_final_start = false;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    const long start_sample = EndSample();
    samples_.insert(samples_.end(), appended->begin(), appended->end());
    if (!segment_.has_value()) {
      StartSegment(start_sample);
    }
    if (preview_refresh_defer_busy_enabled_) {
      if (parsed_speech_epoch.has_value() &&
          (!segment_->speech_epoch.has_value() ||
           *parsed_speech_epoch >= *segment_->speech_epoch)) {
        if (segment_->speech_epoch.has_value() &&
            *parsed_speech_epoch > *segment_->speech_epoch) {
          DiscardPendingPauseHint(PauseHintDisposition::kSpeechResumption);
        } else {
        }
        segment_->speech_epoch = parsed_speech_epoch;
      } else {
        segment_->pause_metadata_valid = false;
        DiscardPendingPauseHint(PauseHintDisposition::kInvalidDeferredMetadata);
        LogAttributionLazy([&] {
          return nlohmann::ordered_json{{"event", "pause_metadata"},
                                        {"status", "invalid"},
                                        {"audio_end_sample", EndSample()}};
        });
      }
    } else {
    }
    if (should_prefill_final_ && !is_recording_cut_off_) {
      is_recording_cut_off_ = recording_samples_.size() + appended->size() >
                              kMaxFinalRecordingSamples;
      if (!is_recording_cut_off_) {
        recording_samples_.insert(recording_samples_.end(), appended->begin(),
                                  appended->end());
      } else {
      }
    } else {
    }
    if (pending_final_start_sample_.has_value() &&
        *pending_final_start_sample_ <= EndSample()) {
      MoveFinalStart(*std::exchange(pending_final_start_sample_, std::nullopt));
      moved_final_start = true;
    } else {
    }
    const long overflow =
        (EndSample() - segment_->start_sample) / settings_.max_segment_samples;
    if (overflow > 0) {
      cut_sample =
          segment_->start_sample + overflow * settings_.max_segment_samples;
      if (preview_refresh_enabled_) {
        finalizing_ = true;
      }
      DiscardPendingPauseHint(PauseHintDisposition::kSegmentRollover);
      if (pause_hint_received) {
        if (preview_refresh_attribution_enabled_.load()) {
          ++pause_hints_received_;
        } else {
        }
        RecordPauseHintDisposition(
            pause_hint_malformed ? PauseHintDisposition::kInvalid
                                 : PauseHintDisposition::kSegmentRollover,
            pause_hint, segment_->last_submitted_refresh_end_sample,
            segment_->last_completed_refresh_end_sample);
      }
    } else {
    }
  }
  if (cut_sample >= 0) {
    if (before_rollover_finalize_for_test_) {
      before_rollover_finalize_for_test_();
    } else {
    }
    FinalizeThrough(cut_sample);
    std::lock_guard<std::mutex> lock(mutex_);
    StartSegment(cut_sample);
  } else {
  }
  if (moved_final_start) {
    MaybeStartFinalPrefill();
  } else {
  }
  MaybeStartRefresh(cut_sample < 0 ? pause_hint : std::nullopt,
                    cut_sample < 0 && pause_hint_received,
                    cut_sample < 0 && pause_hint_malformed);
}

void RealtimeSession::StartSegment(long start_sample) {
  DiscardPendingPauseHint(PauseHintDisposition::kSegmentRollover);
  Segment segment;
  segment.segment_id = next_segment_id_++;
  segment.start_sample = start_sample;
  segment.next_refresh_sample = start_sample + settings_.first_decode_samples;
  segment.last_submitted_refresh_end_sample = start_sample;
  segment.last_completed_refresh_end_sample = start_sample;
  segment.last_pause_speech_end_sample = start_sample;
  segment.language = language_;
  segment.reuse = std::make_shared<RefreshReuse>();
  if (recording_reuse_ == nullptr) {
    recording_reuse_ = segment.reuse;
  } else {
  }
  segment_ = std::move(segment);
}

void RealtimeSession::StartFinalAt(const nlohmann::json &start_sample) {
  bool is_valid = false;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (start_sample.is_number_integer()) {
      const long sample = start_sample.get<long>();
      is_valid = sample >= final_start_sample_ && sample <= EndSample();
      if (is_valid) {
        MoveFinalStart(sample);
      } else {
      }
    } else {
    }
  }
  if (is_valid) {
    MaybeStartFinalPrefill();
  } else {
    SendError("invalid_request_error", "invalid_start_sample",
              "start_sample must be a sample the session has, from the last "
              "one on.");
  }
}

void RealtimeSession::MoveFinalStart(long sample) {
  if (is_recording_cut_off_) {
    recording_samples_.clear();
  } else {
    recording_samples_.erase(recording_samples_.begin(),
                             recording_samples_.begin() +
                                 std::min<size_t>(sample - final_start_sample_,
                                                  recording_samples_.size()));
  }
  final_start_sample_ = sample;
  // The shadow decoded the audio from the old start.
  shadow_guess_.reset();
  shadow_cancel_->store(true);
}

void RealtimeSession::CancelShadowFinal() {
  std::lock_guard<std::mutex> lock(mutex_);
  shadow_cancel_->store(true);
}

FinalReuse RealtimeSession::TakeFinalReuse(
    const std::optional<std::string> &context, bool guess_auto_language,
    bool take_reuse, bool final_handoff, std::optional<long> piece_end_sample) {
  std::lock_guard<std::mutex> lock(mutex_);
  shadow_cancel_->store(true);
  if (final_handoff) {
    pause_refresh_final_handoff_ = true;
    DiscardPendingPauseHint(PauseHintDisposition::kFinalHandoff);
    if (preview_refresh_attribution_enabled_.load()) {
      const std::map<std::string, int> request_states = worker_.RequestStates();
      LogAttributionLazy([&] {
        return nlohmann::ordered_json{
            {"event", "final_handoff"},
            {"handoff_at_us", SteadyMicros(std::chrono::steady_clock::now())},
            {"segment_id", segment_.has_value() ? segment_->segment_id : -1},
            {"audio_end_sample", EndSample()},
            {"worker_queued", request_states.contains("queued")
                                  ? request_states.at("queued")
                                  : 0},
            {"worker_running", request_states.contains("running")
                                   ? request_states.at("running")
                                   : 0}};
      });
    } else {
    }
  } else {
  }
  if (piece_end_sample.has_value() &&
      *piece_end_sample <= final_start_sample_) {
    return FinalReuse();
  } else {
  }
  if (!take_reuse) {
    return FinalReuse();
  } else {
  }
  // A shadow that finished after the start moved decoded other audio.
  const std::optional<FinalPreviewGuess> shadow_guess =
      final_start_sample_ == 0 ? shadow_guess_ : std::nullopt;
  const bool has_final_prompt =
      should_prefill_final_ && context.value_or("") == final_context_;
  FinalReuse reuse;
  if (piece_end_sample.has_value()) {
    reuse.piece = RecordingPiece{
        std::static_pointer_cast<RealtimeSession>(shared_from_this()), context,
        *piece_end_sample};
  } else if (!has_final_prompt) {
    reuse = {recording_reuse_, false};
  } else {
    // note (XXXXRT666): the prefill starts after the first preview, and without
    // a dictionary after the first segment, so a Final before it finds none
    // and reuses what the refreshes keep.
    should_prefill_final_ = false;
    std::shared_ptr<RefreshReuse> final_reuse =
        std::exchange(final_reuse_, nullptr);
    reuse = final_reuse != nullptr ? FinalReuse{std::move(final_reuse), true}
                                   : FinalReuse{recording_reuse_, false};
  }
  reuse.shadow_guess = shadow_guess;
  // note (chenye): only use an auto-language preview of the same recording
  // start. A committed segment or a moved start can leave a different tail.
  if (has_final_prompt && guess_auto_language && !language_.has_value() &&
      final_start_sample_ == 0 && committed_.empty() && !finalizing_ &&
      segment_.has_value() && segment_->start_sample == 0 &&
      !segment_->ran_out_of_tokens && segment_->language.has_value() &&
      !segment_->transcript.empty()) {
    reuse.auto_language_guess =
        FinalPreviewGuess{*segment_->language, segment_->transcript};
  } else {
  }
  return reuse;
}

RefreshReuse
RealtimeSession::TakePieceReuse(const std::optional<std::string> &context,
                                long end_sample) {
  RefreshReuse reuse;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (end_sample <= final_start_sample_) {
      return reuse;
    } else {
    }
    // note (XXXXRT666): a piece leaves the prefill to the session, which moves
    // on from the piece's end, so it decodes a snapshot; the worker runs the
    // prefill's jobs too, so none changes it meanwhile.
    const std::shared_ptr<RefreshReuse> &source =
        should_prefill_final_ && context.value_or("") == final_context_ &&
                final_reuse_ != nullptr
            ? final_reuse_
            : recording_reuse_;
    if (source != nullptr) {
      reuse = *source;
    } else {
    }
    if (end_sample <= EndSample()) {
      MoveFinalStart(end_sample);
    } else {
      pending_final_start_sample_ = end_sample;
    }
  }
  MaybeStartFinalPrefill();
  return reuse;
}

void RealtimeSession::MaybeStartFinalPrefill() {
  std::shared_ptr<RefreshReuse> final_reuse;
  TranscriptionOptions options;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    const bool is_waiting_for_segment = final_context_.empty() &&
                                        next_segment_id_ < 2 &&
                                        final_start_sample_ == 0;
    if (!should_prefill_final_ || !has_preview_ || is_prefilling_final_ ||
        is_waiting_for_segment) {
      return;
    }
    if (final_reuse_ == nullptr) {
      final_reuse_ = std::make_shared<RefreshReuse>();
    }
    is_prefilling_final_ = true;
    final_reuse = final_reuse_;
    options.context = final_context_;
  }
  const auto has_more = std::make_shared<bool>(false);
  std::weak_ptr<RealtimeSession> weak_self =
      std::static_pointer_cast<RealtimeSession>(shared_from_this());
  const Qwen3ASRTranscriber *transcriber = &transcriber_;
  Transcription transcribe = [transcriber, weak_self, final_reuse,
                              options = std::move(options),
                              has_more](const std::atomic<bool> &cancel) {
    RefreshReuse recording_reuse;
    std::vector<float> recording_samples;
    long recording_start_sample = 0;
    if (const std::shared_ptr<RealtimeSession> self = weak_self.lock()) {
      std::lock_guard<std::mutex> lock(self->mutex_);
      recording_start_sample = self->final_start_sample_;
      if (self->recording_reuse_ != nullptr && recording_start_sample == 0) {
        recording_reuse = *self->recording_reuse_;
      }
      recording_samples.assign(self->recording_samples_.begin(),
                               self->recording_samples_.begin() +
                                   std::min(self->recording_samples_.size(),
                                            kMaxFinalPrefillSamples));
    }
    *has_more = transcriber->ExtendFinalPrefill(
        recording_reuse, recording_samples, recording_start_sample, options,
        cancel, *final_reuse);
    return TranscriptionResult();
  };
  Completion completion = [weak_self,
                           has_more](std::optional<TranscriptionResult>,
                                     std::exception_ptr error) {
    const std::shared_ptr<RealtimeSession> self = weak_self.lock();
    if (!self) {
      return;
    }
    {
      std::lock_guard<std::mutex> lock(self->mutex_);
      self->is_prefilling_final_ = false;
    }
    if (error == nullptr && *has_more) {
      self->MaybeStartFinalPrefill();
    }
  };
  worker_.Submit(std::move(transcribe), cancel_, std::move(completion));
}

void RealtimeSession::MaybeStartShadowFinal() {
  const Qwen3ASRTranscriber *transcriber = &transcriber_;
  std::vector<float> samples;
  TranscriptionOptions options;
  std::string preview_text;
  std::optional<std::string> preview_language;
  std::optional<FinalPreviewGuess> shadow_guess;
  omni_server::CancelFlag cancel;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    const long recording_end = static_cast<long>(recording_samples_.size());
    if (!should_prefill_final_ || !has_preview_ || is_shadowing_ ||
        finalizing_ || is_recording_cut_off_ || final_start_sample_ != 0 ||
        IsShadowQuiet() || !segment_.has_value() ||
        recording_end <
            std::max(next_shadow_sample_,
                     kShadowIntervalSamples + kShadowDelaySamples) ||
        recording_samples_.size() > kMaxChunkSamples) {
      return;
    }
    next_shadow_sample_ =
        ((recording_end - kShadowDelaySamples) / kShadowIntervalSamples + 1) *
            kShadowIntervalSamples +
        kShadowDelaySamples;
    samples = recording_samples_;
    options.language = language_;
    options.context = final_context_;
    options.stop_at_end_of_text = true;
    options.stop_on_token_loop = true;
    std::vector<std::pair<int, std::string>> committed = committed_;
    std::sort(committed.begin(), committed.end());
    std::vector<std::string> texts;
    for (const auto &[segment_id, text] : committed) {
      texts.push_back(text);
    }
    texts.push_back(segment_->transcript);
    preview_text = JoinTranscriptParts(texts);
    preview_language = segment_->language;
    shadow_guess = shadow_guess_;
    is_shadowing_ = true;
    shadow_cancel_ = omni_server::NewCancelFlag();
    cancel = shadow_cancel_;
  }
  std::weak_ptr<RealtimeSession> weak_self =
      std::static_pointer_cast<RealtimeSession>(shared_from_this());
  Transcription transcribe = [transcriber, weak_self,
                              samples = std::move(samples),
                              options = std::move(options),
                              preview_text = std::move(preview_text),
                              preview_language, shadow_guess](
                                 const std::atomic<bool> &cancel) mutable {
    if (IsShadowQuiet()) {
      return TranscriptionResult();
    }
    const auto guesses = [&](const std::optional<std::string> &language,
                             const std::string &text) {
      return options.language.has_value()
                 ? transcriber->RetainedPrefix(text, 0).prefix_ids
             : language.has_value()
                 ? transcriber->AutoLanguageGuesses(*language, text)
                 : std::vector<int>();
    };
    const std::vector<int> preview_ids =
        guesses(preview_language, preview_text);
    options.guessed_token_ids =
        shadow_guess.has_value()
            ? MergedGuesses(guesses(shadow_guess->language, shadow_guess->text),
                            preview_ids)
            : preview_ids;
    RefreshReuse reuse;
    if (const std::shared_ptr<RealtimeSession> self = weak_self.lock()) {
      std::lock_guard<std::mutex> lock(self->mutex_);
      const std::shared_ptr<RefreshReuse> &source =
          self->final_reuse_ != nullptr ? self->final_reuse_
                                        : self->recording_reuse_;
      if (source == nullptr) {
        return TranscriptionResult();
      }
      reuse = *source;
    } else {
      return TranscriptionResult();
    }
    return transcriber->Transcribe(samples, options, cancel, &reuse);
  };
  Completion completion = [weak_self](std::optional<TranscriptionResult> result,
                                      std::exception_ptr error) {
    const std::shared_ptr<RealtimeSession> self = weak_self.lock();
    if (!self) {
      return;
    }
    std::lock_guard<std::mutex> lock(self->mutex_);
    self->is_shadowing_ = false;
    if (error == nullptr && result.has_value() &&
        result->finish_reason == FinishReason::kStop && !result->text.empty()) {
      self->shadow_guess_ =
          FinalPreviewGuess{result->language.value_or(""), result->text};
    }
  };
  worker_.Submit(std::move(transcribe), cancel, std::move(completion));
}

std::vector<float> RealtimeSession::SegmentSamples(const Segment &segment,
                                                   long end_sample) const {
  const long begin = segment.start_sample - buffer_start_sample_;
  const long end = end_sample - buffer_start_sample_;
  return std::vector<float>(samples_.begin() + begin, samples_.begin() + end);
}

TranscriptionOptions RealtimeSession::DecodeOptions(Segment &segment,
                                                    long end_sample) {
  TranscriptionOptions options;
  options.language = segment.language;
  // Note (XXXXRT666): on as for HTTP requests; the prefix would carry a loop
  // into every later decode of the segment.
  options.stop_on_token_loop = true;
  const bool use_prefix = segment.decode_count >= kPrefixAfterRefreshCount &&
                          !segment.transcript.empty() &&
                          segment.language.has_value() &&
                          !segment.ran_out_of_tokens;
  if (use_prefix) {
    RetainedText retained = transcriber_.RetainedPrefix(
        segment.transcript, kPrefixRollbackTokenCount);
    options.prefix_token_ids = std::move(retained.prefix_ids);
    options.prefix_text = std::move(retained.prefix_text);
    // Note (XXXXRT666): greedy decoding can repeat the prefix's last sentence
    // and still stop; the budget makes that run out of tokens instead.
    options.max_new_tokens =
        static_cast<int>(retained.rest_ids.size()) +
        static_cast<int>(std::ceil(
            static_cast<double>(end_sample - segment.transcript_end_sample) /
            omni_common::kSampleRate * kPrefixedTokensPerNewSecond)) +
        kPrefixedTokenSlack;
    // note (chenye): XXXXRT666's refresh path verifies rolled-back tokens on
    // refreshes and finals.
    options.guessed_token_ids = std::move(retained.rest_ids);
  } else {
  }
  segment.decode_count += 1;
  return options;
}

Transcription RealtimeSession::Decode(std::vector<float> samples,
                                      TranscriptionOptions options,
                                      std::shared_ptr<RefreshReuse> reuse,
                                      bool snapshot_reuse) const {
  const Qwen3ASRTranscriber *transcriber = &transcriber_;
  return [transcriber, samples = std::move(samples),
          options = std::move(options), reuse = std::move(reuse),
          snapshot_reuse](const std::atomic<bool> &cancel) {
    if (reuse == nullptr || !snapshot_reuse) {
      return transcriber->Transcribe(samples, options, cancel, reuse.get());
    }
    RefreshReuse snapshot = *reuse;
    return transcriber->Transcribe(samples, options, cancel, &snapshot);
  };
}

void RealtimeSession::ApplyResult(Segment &segment,
                                  const TranscriptionResult &result,
                                  long end_sample) {
  if (result.language.has_value() && !result.language->empty()) {
    segment.language = result.language;
  } else {
  }
  // Note (XXXXRT666): speech never fills the token budget, so a decode that
  // does repeats itself, often a whole sentence the loop guard does not see.
  segment.ran_out_of_tokens = result.finish_reason == FinishReason::kLength;
  if (!segment.ran_out_of_tokens) {
    segment.transcript = result.text;
    segment.transcript_end_sample = end_sample;
  } else {
  }
}

void RealtimeSession::MaybeStartRefresh(
    std::optional<PauseRefreshHint> pause_hint, bool pause_hint_received,
    bool pause_hint_malformed) {
  std::vector<float> samples;
  TranscriptionOptions options;
  std::shared_ptr<RefreshReuse> reuse;
  int segment_id = 0;
  long end_sample = 0;
  long speech_end_sample = 0;
  std::unique_lock<std::mutex> lock(mutex_);
  if (pause_hint_received && preview_refresh_attribution_enabled_.load()) {
    ++pause_hints_received_;
  }
  if (!segment_.has_value()) {
    if (pause_hint_received) {
      RecordPauseHintDisposition(PauseHintDisposition::kSegmentRollover,
                                 pause_hint);
    }
    return;
  }

  Segment &segment = *segment_;
  const long audio_end_sample = EndSample();
  PauseHintConsumption consumption;
  if (!preview_refresh_enabled_) {
    if (pause_hint_received) {
      RecordPauseHintDisposition(PauseHintDisposition::kDisabled, pause_hint,
                                 segment.last_submitted_refresh_end_sample,
                                 segment.last_completed_refresh_end_sample);
    }
    DiscardPendingPauseHint(PauseHintDisposition::kDisabled);
  } else if (pause_hint_received && pause_hint_malformed) {
    RecordPauseHintDisposition(PauseHintDisposition::kInvalid, std::nullopt,
                               segment.last_submitted_refresh_end_sample,
                               segment.last_completed_refresh_end_sample);
    DiscardPendingPauseHint(PauseHintDisposition::kInvalidDeferredMetadata);
  } else {
    consumption = ConsumePauseRefreshHint(pause_hint, segment.start_sample,
                                          audio_end_sample,
                                          segment.last_pause_speech_end_sample);
    if (consumption.disposition == PauseHintDisposition::kInvalid) {
      RecordPauseHintDisposition(PauseHintDisposition::kInvalid, pause_hint);
      DiscardPendingPauseHint(PauseHintDisposition::kInvalidDeferredMetadata);
    } else if (consumption.disposition == PauseHintDisposition::kDuplicate) {
      RecordPauseHintDisposition(PauseHintDisposition::kDuplicate, pause_hint);
    } else if (consumption.hint.has_value()) {
      segment.last_pause_speech_end_sample = consumption.last_speech_end_sample;
    }
  }

  const bool has_current_hint = consumption.hint.has_value();
  const bool cancelled = cancel_->load();
  const bool cadence_due = audio_end_sample >= segment.next_refresh_sample;
  const long minimum_new_speech_samples = omni_common::kSampleRate / 5;
  const auto admission_input = [&](std::optional<PauseRefreshHint> hint,
                                   bool deferred, bool silent) {
    return RefreshAdmissionInput{
        .segment_start_sample = segment.start_sample,
        .audio_end_sample = audio_end_sample,
        .next_refresh_sample = segment.next_refresh_sample,
        .last_submitted_refresh_end_sample =
            segment.last_submitted_refresh_end_sample,
        .decode_interval_samples = settings_.decode_interval_samples,
        .minimum_new_speech_samples = minimum_new_speech_samples,
        .pause_hint = hint,
        .pause_hint_deferred = deferred,
        .refreshing = refreshing_,
        .finalizing = finalizing_,
        .cancelled = cancelled,
        .final_handoff = pause_refresh_final_handoff_,
        .silent = silent,
        .has_decoded = segment.decode_count > 0};
  };

  std::optional<PauseRefreshHint> candidate = consumption.hint;
  bool candidate_is_deferred = false;
  if (cancelled || finalizing_) {
    if (!candidate.has_value() && segment.pending_pause.has_value()) {
      candidate = segment.pending_pause;
      candidate_is_deferred = true;
    }
    if (candidate.has_value()) {
      const RefreshAdmission admission = SelectRefreshAdmission(
          admission_input(candidate, candidate_is_deferred, false));
      if (candidate_is_deferred) {
        DiscardPendingPauseHint(admission.disposition);
      } else {
        RecordPauseHintDisposition(admission.disposition, candidate,
                                   segment.last_submitted_refresh_end_sample,
                                   segment.last_completed_refresh_end_sample);
      }
    }
    if (segment.pending_pause.has_value()) {
      DiscardPendingPauseHint(cancelled
                                  ? PauseHintDisposition::kCancelled
                                  : PauseHintDisposition::kSegmentFinalization);
    }
    return;
  }

  if (refreshing_) {
    if (cadence_due && !candidate.has_value() &&
        segment.pending_pause.has_value()) {
      candidate = segment.pending_pause;
      candidate_is_deferred = true;
    }
  } else if (has_current_hint && segment.pending_pause.has_value()) {
    DiscardPendingPauseHint(PauseHintDisposition::kSuperseded);
  } else if (!has_current_hint && segment.pending_pause.has_value()) {
    candidate = segment.pending_pause;
    candidate_is_deferred = true;
  }

  if (!candidate.has_value() && !cadence_due) {
    return;
  }

  bool silent = false;
  if (!refreshing_ && (cadence_due || (candidate.has_value() &&
                                       !pause_refresh_final_handoff_))) {
    samples = SegmentSamples(segment, audio_end_sample);
    silent =
        omni_common::PeakIsSilent(samples.data(), samples.size(), kSilentPeak);
  }
  const RefreshAdmission admission = SelectRefreshAdmission(
      admission_input(candidate, candidate_is_deferred, silent));
  segment.next_refresh_sample = admission.next_refresh_sample;

  const auto record_candidate_disposition =
      [&](PauseHintDisposition disposition) {
        if (candidate_is_deferred) {
          DiscardPendingPauseHint(disposition);
        } else if (candidate.has_value()) {
          RecordPauseHintDisposition(disposition, candidate,
                                     segment.last_submitted_refresh_end_sample,
                                     segment.last_completed_refresh_end_sample);
        }
      };

  if (refreshing_ && cadence_due && segment.pending_pause.has_value() &&
      !candidate_is_deferred) {
    DiscardPendingPauseHint(PauseHintDisposition::kSupersededByCadence);
  }

  if (admission.disposition == PauseHintDisposition::kRefreshBusy &&
      has_current_hint) {
    if (preview_refresh_defer_busy_enabled_) {
      if (!segment.pause_metadata_valid || !segment.speech_epoch.has_value()) {
        RecordPauseHintDisposition(
            PauseHintDisposition::kInvalidDeferredMetadata, consumption.hint,
            segment.last_submitted_refresh_end_sample,
            segment.last_completed_refresh_end_sample);
      } else {
        DiscardPendingPauseHint(PauseHintDisposition::kSuperseded);
        segment.pending_pause = consumption.hint;
        RecordPauseHintDisposition(PauseHintDisposition::kDeferred,
                                   consumption.hint,
                                   segment.last_submitted_refresh_end_sample,
                                   segment.last_completed_refresh_end_sample);
      }
    } else {
      RecordPauseHintDisposition(PauseHintDisposition::kRefreshBusy,
                                 consumption.hint,
                                 segment.last_submitted_refresh_end_sample,
                                 segment.last_completed_refresh_end_sample);
    }
    return;
  }

  if (admission.disposition != PauseHintDisposition::kNone &&
      admission.disposition != PauseHintDisposition::kAdmittedImmediate &&
      admission.disposition != PauseHintDisposition::kAdmittedDeferred) {
    record_candidate_disposition(admission.disposition);
  }

  if (admission.source == RefreshSource::kNone) {
    return;
  }
  end_sample = admission.end_sample;
  speech_end_sample = candidate.has_value()
                          ? candidate->speech_end_sample
                          : segment.last_pause_speech_end_sample;
  options = DecodeOptions(segment, end_sample);
  reuse = segment.reuse;
  segment_id = segment.segment_id;
  segment.last_submitted_refresh_end_sample = end_sample;
  refreshing_ = true;

  if (admission.source == RefreshSource::kPause) {
    RecordPauseHintDisposition(PauseHintDisposition::kAdmittedImmediate,
                               candidate,
                               segment.last_submitted_refresh_end_sample,
                               segment.last_completed_refresh_end_sample);
  } else if (admission.source == RefreshSource::kDeferredPause) {
    RecordPauseHintDisposition(PauseHintDisposition::kAdmittedDeferred,
                               candidate,
                               segment.last_submitted_refresh_end_sample,
                               segment.last_completed_refresh_end_sample);
    segment.pending_pause.reset();
  }

  SubmitRefresh(lock, std::move(samples), std::move(options), std::move(reuse),
                segment_id, end_sample, speech_end_sample, admission.source);
}

void RealtimeSession::SubmitRefresh(std::unique_lock<std::mutex> &lock,
                                    std::vector<float> samples,
                                    TranscriptionOptions options,
                                    std::shared_ptr<RefreshReuse> reuse,
                                    int segment_id, long end_sample,
                                    long speech_end_sample,
                                    RefreshSource source) {
  std::shared_ptr<RefreshTiming> timing;
  if (preview_refresh_attribution_enabled_.load()) {
    timing = std::make_shared<RefreshTiming>();
    timing->decode_index = next_refresh_decode_index_++;
    timing->source = source;
    timing->segment_id = segment_id;
    timing->end_sample = end_sample;
    timing->speech_end_sample = speech_end_sample;
  }

  std::weak_ptr<RealtimeSession> weak_self =
      std::static_pointer_cast<RealtimeSession>(shared_from_this());
  Transcription transcribe =
      Decode(std::move(samples), std::move(options), std::move(reuse), false);
  if (timing != nullptr) {
    Transcription decode = std::move(transcribe);
    transcribe = [decode = std::move(decode),
                  timing](const std::atomic<bool> &cancel) mutable {
      timing->worker_started_at = std::chrono::steady_clock::now();
      try {
        TranscriptionResult result = decode(cancel);
        timing->decode_finished_at = std::chrono::steady_clock::now();
        return result;
      } catch (...) {
        timing->decode_finished_at = std::chrono::steady_clock::now();
        throw;
      }
    };
  }
  Completion completion =
      [weak_self, segment_id, end_sample, speech_end_sample, source, timing](
          std::optional<TranscriptionResult> result, std::exception_ptr error) {
        if (const std::shared_ptr<RealtimeSession> self = weak_self.lock()) {
          self->CompleteRefresh(segment_id, end_sample, speech_end_sample,
                                source, timing, std::move(result), error);
        }
      };

  const bool pause_admission = source == RefreshSource::kPause ||
                               source == RefreshSource::kDeferredPause;
  if (pause_admission) {
    // note (XXXXRT666): Queue the pause while Final cannot take this mutex.
    if (timing != nullptr) {
      timing->submitted_at = std::chrono::steady_clock::now();
    }
    worker_.Submit(std::move(transcribe), cancel_, std::move(completion));
    return;
  }
  lock.unlock();
  if (timing != nullptr) {
    timing->submitted_at = std::chrono::steady_clock::now();
  }
  worker_.Submit(std::move(transcribe), cancel_, std::move(completion));
}

void RealtimeSession::CompleteRefresh(int segment_id, long end_sample,
                                      long speech_end_sample,
                                      RefreshSource source,
                                      std::shared_ptr<RefreshTiming> timing,
                                      std::optional<TranscriptionResult> result,
                                      std::exception_ptr error) {
  std::optional<std::string> preview;
  bool transcript_changed = false;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    refreshing_ = false;
    if (result.has_value() && segment_.has_value() &&
        segment_->segment_id == segment_id) {
      Segment &segment = *segment_;
      segment.last_completed_refresh_end_sample =
          std::max(segment.last_completed_refresh_end_sample, end_sample);
      ApplyResult(segment, *result, end_sample);
      if (segment.transcript != segment.last_text) {
        segment.last_text = segment.transcript;
        preview = segment.transcript;
        transcript_changed = true;
        has_preview_ = true;
      }
    }
  }
  refresh_done_.notify_all();
  // note (XXXXRT666): queued before the event, so a Final sent in answer to it
  // finds the windows this refresh finished.
  MaybeStartFinalPrefill();
  MaybeStartShadowFinal();
  if (error) {
    ReportDecodeFailure(error);
  } else if (preview.has_value()) {
    if (timing != nullptr) {
      timing->preview_send_attempt_at = std::chrono::steady_clock::now();
    }
    Send({{"type", "transcription.segment"},
          {"segment_id", segment_id},
          {"text", *preview},
          {"is_final", false}});
  }
  MaybeStartRefresh();

  if (timing == nullptr) {
    return;
  }
  timing->callback_finished_at = std::chrono::steady_clock::now();
  const std::string_view source_name =
      source == RefreshSource::kCadence ? "cadence"
      : source == RefreshSource::kPause ? "immediate_pause"
                                        : "deferred_pause";
  std::string status = result.has_value() ? "success" : "error";
  bool cancelled = false;
  if (error != nullptr) {
    try {
      std::rethrow_exception(error);
    } catch (const GenerationCancelled &) {
      status = "cancelled";
      cancelled = true;
    } catch (...) {
    }
  }
  LogAttributionLazy([&] {
    nlohmann::ordered_json event = {
        {"event", "refresh"},
        {"decode_index", timing->decode_index},
        {"source", source_name},
        {"segment_id", timing->segment_id},
        {"snapshot_end_sample", timing->end_sample},
        {"latest_speech_end_sample", speech_end_sample},
        {"completed_refresh_end_sample",
         result.has_value() ? nlohmann::json(end_sample) : nlohmann::json()},
        {"submitted_at_us", SteadyMicrosJson(timing->submitted_at)},
        {"worker_started_at_us", SteadyMicrosJson(timing->worker_started_at)},
        {"decode_finished_at_us", SteadyMicrosJson(timing->decode_finished_at)},
        {"callback_finished_at_us",
         SteadyMicrosJson(timing->callback_finished_at)},
        {"preview_send_attempt_at_us",
         SteadyMicrosJson(timing->preview_send_attempt_at)},
        {"queue_wait_us",
         ElapsedMicrosJson(timing->submitted_at, timing->worker_started_at)},
        {"decode_execution_wall_us",
         ElapsedMicrosJson(timing->worker_started_at,
                           timing->decode_finished_at)},
        {"completion_overhead_us",
         ElapsedMicrosJson(timing->decode_finished_at,
                           timing->callback_finished_at)},
        {"status", status},
        {"cancelled", cancelled},
        {"error", error != nullptr},
        {"transcript_changed", transcript_changed},
        {"preview_send_attempted", preview.has_value()}};
    return event;
  });
}
void RealtimeSession::FinalizeThrough(long end_sample) {
  while (true) {
    Segment segment;
    std::vector<float> samples;
    long cut_sample = 0;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      if (!segment_.has_value() || end_sample <= segment_->start_sample) {
        finalizing_ = false;
        return;
      } else {
      }
      // Note (Jiaxin Deng): wait out an in-flight preview and start no new one,
      // as the Python server's FIFO decode lock does.
      finalizing_ = true;
      const PauseHintDisposition finalization_disposition =
          end_sample < EndSample() ? PauseHintDisposition::kSegmentRollover
          : pause_refresh_final_handoff_
              ? PauseHintDisposition::kFinalHandoff
              : PauseHintDisposition::kSegmentFinalization;
      DiscardPendingPauseHint(finalization_disposition);
      refresh_done_.wait(lock, [&] { return !refreshing_; });
      if (!segment_.has_value() || end_sample <= segment_->start_sample) {
        finalizing_ = false;
        return;
      } else {
      }
      segment = *segment_;
      segment_.reset();
      cut_sample = std::min(end_sample, segment.start_sample +
                                            settings_.max_segment_samples);
      samples = SegmentSamples(segment, cut_sample);
    }
    std::optional<std::string> text;
    std::optional<TranscriptionResult> result;
    if (omni_common::PeakIsSilent(samples.data(), samples.size(),
                                  kSilentPeak)) {
      text = std::string();
    } else {
      TranscriptionOptions options;
      {
        std::lock_guard<std::mutex> lock(mutex_);
        options = DecodeOptions(segment, cut_sample);
      }
      const bool segment_cache_reuse_eligible =
          samples.size() <= kMaxChunkSamples;
      const auto reuse = segment_cache_reuse_eligible ? segment.reuse : nullptr;
      if (!segment_cache_reuse_eligible && !options.prefix_token_ids.empty()) {
        // note (chenye): Long prefixed Finals keep the original path without
        // rollback guesses.
        options.guessed_token_ids.clear();
      } else {
      }
      try {
        result =
            worker_.Transcribe(Decode(samples, options, reuse, true), cancel_);
        // A final that ran out of tokens on the prefix decodes the segment
        // again without it, as the next refresh would have.
        if (result->finish_reason == FinishReason::kLength &&
            !options.prefix_token_ids.empty()) {
          options.prefix_token_ids.clear();
          options.prefix_text.clear();
          options.guessed_token_ids.clear();
          options.max_new_tokens.reset();
          result = worker_.Transcribe(
              Decode(std::move(samples), std::move(options), reuse, true),
              cancel_);
        } else {
        }
        ApplyResult(segment, *result, cut_sample);
        text = result->text;
      } catch (...) {
        ReportDecodeFailure(std::current_exception());
      }
    }
    // Note (Jiaxin Deng): a failed or cancelled decode was already reported, so
    // its segment is dropped rather than committed empty.
    if (text.has_value()) {
      {
        std::lock_guard<std::mutex> lock(mutex_);
        committed_.emplace_back(segment.segment_id, *text);
      }
      nlohmann::ordered_json event = {{"type", "transcription.segment"},
                                      {"segment_id", segment.segment_id},
                                      {"text", *text},
                                      {"is_final", true}};
      if (result.has_value()) {
        event["generation_metadata"] = {
            {"reused_audio_token_count", result->reused_audio_token_count},
            {"reused_prompt_token_count", result->reused_prompt_token_count},
            {"verified_guess_token_count", result->verified_guess_token_count},
            {"accepted_guess_token_count", result->accepted_guess_token_count}};
      } else {
      }
      Send(std::move(event));
    } else {
    }
    {
      std::lock_guard<std::mutex> lock(mutex_);
      samples_.erase(samples_.begin(),
                     samples_.begin() + (cut_sample - buffer_start_sample_));
      buffer_start_sample_ = cut_sample;
      if (cut_sample < end_sample) {
        StartSegment(cut_sample);
      } else {
      }
    }
  }
}

} // namespace qwen3_asr
