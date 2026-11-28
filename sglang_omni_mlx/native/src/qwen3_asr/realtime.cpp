// SPDX-License-Identifier: Apache-2.0
#include "qwen3_asr/realtime.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <string>
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

std::atomic<std::chrono::steady_clock::rep> last_foreground_request{0};

bool IsShadowQuiet() {
  return std::chrono::steady_clock::now().time_since_epoch().count() -
             last_foreground_request.load() <
         std::chrono::duration_cast<std::chrono::steady_clock::duration>(
             kShadowQuietTime)
             .count();
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
                                 RealtimeSettings settings, Sender sender)
    : RealtimeConnection(std::move(sender)), worker_(worker),
      transcriber_(transcriber), settings_(settings),
      profile_session_id_(settings.profile ? NextRealtimeProfileSessionId()
                                           : 0) {}

bool RealtimeSession::Handle(const nlohmann::json &message) {
  const nlohmann::json type = message.value("type", nlohmann::json());
  if (type == "session.update") {
    const std::optional<nlohmann::json> session = ManualTurnSession(message);
    if (session.has_value()) {
      const nlohmann::json language =
          session->value("language", nlohmann::json());
      const nlohmann::json final_context =
          session->value("final_prompt", nlohmann::json());
      {
        std::lock_guard<std::mutex> lock(mutex_);
        language_ = language.is_string()
                        ? NormalizeLanguage(language.get<std::string>())
                        : std::nullopt;
        // note (XXXXRT666): the prefill takes the recording from its first
        // sample, so a prompt sent after audio has none.
        if (final_context.is_string() && !should_prefill_final_ &&
            final_reuse_ == nullptr && EndSample() == 0) {
          final_context_ = final_context.get<std::string>();
          should_prefill_final_ = true;
        } else {
        }
      }
      Send({{"type", "transcription_session.updated"},
            {"session_id", session_id()}});
    } else {
    }
    return true;
  } else if (type == "input_audio_buffer.append") {
    gpu_clock_.Touch();
    Append(message.value("audio", nlohmann::json()));
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

long RealtimeSession::LockedEndSample() {
  std::lock_guard<std::mutex> lock(mutex_);
  return EndSample();
}

void RealtimeSession::Append(const nlohmann::json &audio) {
  const std::optional<std::vector<float>> appended = AppendedSamples(audio);
  if (!appended.has_value()) {
    return;
  } else {
  }
  long cut_sample = -1;
  bool moved_final_start = false;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    const long start_sample = EndSample();
    samples_.insert(samples_.end(), appended->begin(), appended->end());
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
    if (!segment_.has_value()) {
      StartSegment(start_sample);
    } else {
    }
    const long overflow =
        (EndSample() - segment_->start_sample) / settings_.max_segment_samples;
    if (overflow > 0) {
      cut_sample =
          segment_->start_sample + overflow * settings_.max_segment_samples;
    } else {
    }
  }
  if (cut_sample >= 0) {
    FinalizeThrough(cut_sample);
    std::lock_guard<std::mutex> lock(mutex_);
    StartSegment(cut_sample);
  } else {
  }
  if (moved_final_start) {
    MaybeStartFinalPrefill();
  } else {
  }
  MaybeStartRefresh();
}

void RealtimeSession::StartSegment(long start_sample) {
  Segment segment;
  segment.segment_id = next_segment_id_++;
  segment.start_sample = start_sample;
  segment.next_refresh_sample = start_sample + settings_.first_decode_samples;
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

FinalReuse
RealtimeSession::TakeFinalReuse(const std::optional<std::string> &context,
                                bool guess_auto_language,
                                std::optional<long> piece_end_sample) {
  std::lock_guard<std::mutex> lock(mutex_);
  shadow_cancel_->store(true);
  if (piece_end_sample.has_value() &&
      *piece_end_sample <= final_start_sample_) {
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
    // note (XXXXRT666): without a dictionary a Final of the first segment
    // reuses the previews' keys and values, so the prefill waits for the
    // second, or for a Final that starts later.
    const bool is_waiting_for_segment = final_context_.empty() &&
                                        next_segment_id_ < 2 &&
                                        final_start_sample_ == 0;
    if (!should_prefill_final_ || !has_preview_ || is_prefilling_final_ ||
        is_waiting_for_segment) {
      return;
    } else {
    }
    if (final_reuse_ == nullptr) {
      final_reuse_ = std::make_shared<RefreshReuse>();
    } else {
    }
    is_prefilling_final_ = true;
    final_reuse = final_reuse_;
    options.context = final_context_;
  }
  const auto has_more = std::make_shared<bool>(false);
  std::weak_ptr<RealtimeSession> weak_self =
      std::static_pointer_cast<RealtimeSession>(shared_from_this());
  worker_.Submit(
      [&transcriber = transcriber_, weak_self, final_reuse,
       options = std::move(options),
       has_more](const std::atomic<bool> &cancel) {
        // Windows a refresh finished since the job was queued count too.
        RefreshReuse recording_reuse;
        std::vector<float> recording_samples;
        long recording_start_sample = 0;
        if (const std::shared_ptr<RealtimeSession> self = weak_self.lock()) {
          std::lock_guard<std::mutex> lock(self->mutex_);
          recording_start_sample = self->final_start_sample_;
          // The previews' windows are those of the recording's first segment.
          if (self->recording_reuse_ != nullptr &&
              recording_start_sample == 0) {
            recording_reuse = *self->recording_reuse_;
          } else {
          }
          recording_samples.assign(self->recording_samples_.begin(),
                                   self->recording_samples_.begin() +
                                       std::min(self->recording_samples_.size(),
                                                kMaxFinalPrefillSamples));
        } else {
        }
        *has_more = transcriber.ExtendFinalPrefill(
            recording_reuse, recording_samples, recording_start_sample, options,
            cancel, *final_reuse);
        return TranscriptionResult();
      },
      cancel_,
      [weak_self, has_more](std::optional<TranscriptionResult>,
                            std::exception_ptr error) {
        const std::shared_ptr<RealtimeSession> self = weak_self.lock();
        if (!self) {
          return;
        } else {
        }
        {
          std::lock_guard<std::mutex> lock(self->mutex_);
          self->is_prefilling_final_ = false;
        }
        // A failed prefill leaves the rest to the Final.
        if (error == nullptr && *has_more) {
          self->MaybeStartFinalPrefill();
        } else {
        }
      });
}

void RealtimeSession::MaybeStartShadowFinal() {
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
    } else {
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
  worker_.Submit(
      [&transcriber = transcriber_, weak_self, samples = std::move(samples),
       options = std::move(options), preview_text = std::move(preview_text),
       preview_language,
       shadow_guess](const std::atomic<bool> &cancel) mutable {
        if (IsShadowQuiet()) {
          return TranscriptionResult();
        } else {
        }
        const auto guesses = [&](const std::optional<std::string> &language,
                                 const std::string &text) {
          return options.language.has_value()
                     ? transcriber.RetainedPrefix(text, 0).prefix_ids
                 : language.has_value()
                     ? transcriber.AutoLanguageGuesses(*language, text)
                     : std::vector<int>();
        };
        const std::vector<int> preview_ids =
            guesses(preview_language, preview_text);
        options.guessed_token_ids =
            shadow_guess.has_value()
                ? MergedGuesses(
                      guesses(shadow_guess->language, shadow_guess->text),
                      preview_ids)
                : preview_ids;
        // The reuse the Final will take, copied: the shadow's decode must not
        // change it.
        RefreshReuse reuse;
        if (const std::shared_ptr<RealtimeSession> self = weak_self.lock()) {
          std::lock_guard<std::mutex> lock(self->mutex_);
          const std::shared_ptr<RefreshReuse> &source =
              self->final_reuse_ != nullptr ? self->final_reuse_
                                            : self->recording_reuse_;
          if (source == nullptr) {
            return TranscriptionResult();
          } else {
          }
          reuse = *source;
        } else {
          return TranscriptionResult();
        }
        return transcriber.Transcribe(samples, options, cancel, &reuse);
      },
      cancel,
      [weak_self](std::optional<TranscriptionResult> result,
                  std::exception_ptr error) {
        const std::shared_ptr<RealtimeSession> self = weak_self.lock();
        if (!self) {
          return;
        } else {
        }
        std::lock_guard<std::mutex> lock(self->mutex_);
        self->is_shadowing_ = false;
        // A failed or cut-short shadow leaves the last one.
        if (error == nullptr && result.has_value() &&
            result->finish_reason == FinishReason::kStop &&
            !result->text.empty()) {
          self->shadow_guess_ =
              FinalPreviewGuess{result->language.value_or(""), result->text};
        } else {
        }
      });
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

std::shared_ptr<RealtimeDecodeTiming>
RealtimeSession::NewDecodeTiming(const Segment &segment, std::string kind,
                                 long end_sample,
                                 int audio_sample_count) const {
  if (!settings_.profile) {
    return nullptr;
  }
  auto timing = std::make_shared<RealtimeDecodeTiming>();
  timing->session_id = profile_session_id_;
  timing->segment_id = segment.segment_id;
  timing->decode_index = segment.decode_count;
  timing->kind = std::move(kind);
  timing->audio_sample_count = audio_sample_count;
  timing->end_sample = end_sample;
  timing->submitted_at_ms = NowRealtimeMilliseconds();
  return timing;
}

Transcription RealtimeSession::Decode(
    std::vector<float> samples, TranscriptionOptions options,
    std::shared_ptr<RefreshReuse> reuse, bool snapshot_reuse,
    std::shared_ptr<RealtimeDecodeTiming> timing) const {
  return [&transcriber = transcriber_, samples = std::move(samples),
          options = std::move(options), reuse = std::move(reuse),
          snapshot_reuse, timing](const std::atomic<bool> &cancel) {
    return RunRealtimeDecodeTiming(timing, [&] {
      if (reuse == nullptr || !snapshot_reuse) {
        return transcriber.Transcribe(samples, options, cancel, reuse.get());
      } else {
        RefreshReuse snapshot = *reuse;
        return transcriber.Transcribe(samples, options, cancel, &snapshot);
      }
    });
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

void RealtimeSession::MaybeStartRefresh() {
  std::vector<float> samples;
  TranscriptionOptions options;
  std::shared_ptr<RefreshReuse> reuse;
  std::shared_ptr<RealtimeDecodeTiming> timing;
  int segment_id = 0;
  long end_sample = 0;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (refreshing_ || finalizing_ || !segment_.has_value() ||
        EndSample() < segment_->next_refresh_sample) {
      return;
    } else {
    }
    Segment &segment = *segment_;
    end_sample = EndSample();
    samples = SegmentSamples(segment, end_sample);
    const bool silent =
        omni_common::PeakIsSilent(samples.data(), samples.size(), kSilentPeak);
    // Note (Jiaxin Deng): leading silence keeps the early first decode for the
    // first audible audio.
    if (!(silent && segment.decode_count == 0)) {
      segment.next_refresh_sample =
          end_sample + settings_.decode_interval_samples;
    } else {
    }
    if (silent) {
      return;
    } else {
    }
    options = DecodeOptions(segment, end_sample);
    reuse = segment.reuse;
    segment_id = segment.segment_id;
    refreshing_ = true;
    timing = NewDecodeTiming(segment, "preview", end_sample,
                             static_cast<int>(samples.size()));
  }
  std::weak_ptr<RealtimeSession> weak_self =
      std::static_pointer_cast<RealtimeSession>(shared_from_this());
  worker_.Submit(
      Decode(std::move(samples), std::move(options), std::move(reuse), false,
             std::move(timing)),
      cancel_,
      [weak_self, segment_id, end_sample](
          std::optional<TranscriptionResult> result, std::exception_ptr error) {
        const std::shared_ptr<RealtimeSession> self = weak_self.lock();
        if (!self) {
          return;
        } else {
        }
        std::optional<std::string> preview;
        {
          std::lock_guard<std::mutex> lock(self->mutex_);
          self->refreshing_ = false;
          if (result.has_value() && self->segment_.has_value() &&
              self->segment_->segment_id == segment_id) {
            Segment &segment = *self->segment_;
            self->ApplyResult(segment, *result, end_sample);
            if (segment.transcript != segment.last_text) {
              segment.last_text = segment.transcript;
              preview = segment.transcript;
              self->has_preview_ = true;
            } else {
            }
          } else {
          }
        }
        self->refresh_done_.notify_all();
        // note (XXXXRT666): queued before the event, so a Final sent in answer
        // to it finds the windows this refresh finished.
        self->MaybeStartFinalPrefill();
        self->MaybeStartShadowFinal();
        if (error) {
          self->ReportDecodeFailure(error);
        } else if (preview.has_value()) {
          self->Send({{"type", "transcription.segment"},
                      {"segment_id", segment_id},
                      {"text", *preview},
                      {"is_final", false}});
        } else {
        }
        self->MaybeStartRefresh();
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
        auto timing = NewDecodeTiming(segment, "final", cut_sample,
                                      static_cast<int>(samples.size()));
        result = worker_.Transcribe(
            Decode(samples, options, reuse, true, std::move(timing)), cancel_);
        // A final that ran out of tokens on the prefix decodes the segment
        // again without it, as the next refresh would have.
        if (result->finish_reason == FinishReason::kLength &&
            !options.prefix_token_ids.empty()) {
          options.prefix_token_ids.clear();
          options.prefix_text.clear();
          options.guessed_token_ids.clear();
          options.max_new_tokens.reset();
          segment.decode_count += 1;
          timing = NewDecodeTiming(segment, "final_retry", cut_sample,
                                   static_cast<int>(samples.size()));
          result =
              worker_.Transcribe(Decode(std::move(samples), std::move(options),
                                        reuse, true, std::move(timing)),
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
