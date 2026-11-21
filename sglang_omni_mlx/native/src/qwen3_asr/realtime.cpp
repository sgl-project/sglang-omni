// SPDX-License-Identifier: Apache-2.0
#include "qwen3_asr/realtime.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <iostream>

#include "common/audio.h"
#include "common/generation.h"
#include "common/text.h"
#include "common/transcription.h"

namespace qwen3_asr {

using omni_common::CodePoints;
using omni_common::FinishReason;
using omni_common::GenerationCancelled;
using omni_common::IsUnicodeSpace;
using omni_common::StripUnicodeWhitespace;
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
constexpr float kSilentPeak = 1e-3f;

constexpr std::pair<uint32_t, uint32_t> kUnspacedScriptRanges[] = {
    {0x0E00, 0x0EFF}, {0x1000, 0x109F}, {0x1780, 0x17FF},
    {0x2E80, 0x303F}, {0x3040, 0x30FF}, {0x3400, 0x9FFF},
    {0xF900, 0xFAFF}, {0xFF00, 0xFFEF}, {0x20000, 0x2FA1F},
};

bool IsSpacedScript(uint32_t code_point) {
  if (IsUnicodeSpace(code_point)) {
    return false;
  } else {
  }
  for (const auto &[low, high] : kUnspacedScriptRanges) {
    if (code_point >= low && code_point <= high) {
      return false;
    } else {
    }
  }
  return true;
}

} // namespace

RealtimeSettings MakeRealtimeSettings(int decode_interval_ms,
                                      int first_decode_ms,
                                      double max_segment_seconds) {
  return {decode_interval_ms * omni_common::kSampleRate / 1000,
          first_decode_ms * omni_common::kSampleRate / 1000,
          static_cast<int>(max_segment_seconds * omni_common::kSampleRate)};
}

std::string JoinTranscriptParts(const std::vector<std::string> &parts) {
  std::string joined;
  for (const std::string &raw_part : parts) {
    const std::string part = StripUnicodeWhitespace(raw_part);
    if (part.empty()) {
      continue;
    } else if (!joined.empty() && IsSpacedScript(CodePoints(joined).back()) &&
               IsSpacedScript(CodePoints(part).front())) {
      joined += " " + part;
    } else {
      joined += part;
    }
  }
  return joined;
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
      {
        std::lock_guard<std::mutex> lock(mutex_);
        language_ = language.is_string()
                        ? NormalizeLanguage(language.get<std::string>())
                        : std::nullopt;
      }
      Send({{"type", "transcription_session.updated"}});
    } else {
    }
    return true;
  } else if (type == "input_audio_buffer.append") {
    Append(message.value("audio", nlohmann::json()));
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
  {
    std::lock_guard<std::mutex> lock(mutex_);
    const long start_sample = EndSample();
    samples_.insert(samples_.end(), appended->begin(), appended->end());
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
  MaybeStartRefresh();
}

void RealtimeSession::StartSegment(long start_sample) {
  Segment segment;
  segment.segment_id = next_segment_id_++;
  segment.start_sample = start_sample;
  segment.next_refresh_sample = start_sample + settings_.first_decode_samples;
  segment.language = language_;
  segment.reuse = std::make_shared<RefreshReuse>();
  segment_ = std::move(segment);
}

std::vector<float> RealtimeSession::SegmentSamples(const Segment &segment,
                                                   long end_sample) const {
  const long begin = segment.start_sample - buffer_start_sample_;
  const long end = end_sample - buffer_start_sample_;
  return std::vector<float>(samples_.begin() + begin, samples_.begin() + end);
}

TranscriptionOptions RealtimeSession::DecodeOptions(Segment &segment,
                                                    bool is_refresh,
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
    // Note (XXXXRT666): a refresh checks the rolled-back tokens as guesses;
    // the final decode decodes them as before.
    if (is_refresh) {
      options.guessed_token_ids = std::move(retained.rest_ids);
    } else {
    }
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

Transcription
RealtimeSession::Decode(std::vector<float> samples,
                        TranscriptionOptions options,
                        std::shared_ptr<RefreshReuse> reuse,
                        std::shared_ptr<RealtimeDecodeTiming> timing) const {
  return [&transcriber = transcriber_, samples = std::move(samples),
          options = std::move(options), reuse = std::move(reuse),
          timing](const std::atomic<bool> &cancel) {
    return RunRealtimeDecodeTiming(timing, [&] {
      return transcriber.Transcribe(samples, options, cancel, reuse.get());
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
    options = DecodeOptions(segment, true, end_sample);
    reuse = segment.reuse;
    segment_id = segment.segment_id;
    refreshing_ = true;
    timing = NewDecodeTiming(segment, "preview", end_sample,
                             static_cast<int>(samples.size()));
  }
  std::weak_ptr<RealtimeSession> weak_self =
      std::static_pointer_cast<RealtimeSession>(shared_from_this());
  worker_.Submit(
      Decode(std::move(samples), std::move(options), std::move(reuse),
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
            } else {
            }
          } else {
          }
        }
        self->refresh_done_.notify_all();
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
    if (omni_common::PeakIsSilent(samples.data(), samples.size(),
                                  kSilentPeak)) {
      text = std::string();
    } else {
      TranscriptionOptions options;
      {
        std::lock_guard<std::mutex> lock(mutex_);
        options = DecodeOptions(segment, false, cut_sample);
      }
      try {
        auto timing = NewDecodeTiming(segment, "final", cut_sample,
                                      static_cast<int>(samples.size()));
        TranscriptionResult result = worker_.Transcribe(
            Decode(samples, options, nullptr, std::move(timing)), cancel_);
        // A final that ran out of tokens on the prefix decodes the segment
        // again without it, as the next refresh would have.
        if (result.finish_reason == FinishReason::kLength &&
            !options.prefix_token_ids.empty()) {
          options.prefix_token_ids.clear();
          options.prefix_text.clear();
          options.max_new_tokens.reset();
          timing = NewDecodeTiming(segment, "final_retry", cut_sample,
                                   static_cast<int>(samples.size()));
          result =
              worker_.Transcribe(Decode(std::move(samples), std::move(options),
                                        nullptr, std::move(timing)),
                                 cancel_);
        } else {
        }
        ApplyResult(segment, result, cut_sample);
        text = result.text;
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
      Send({{"type", "transcription.segment"},
            {"segment_id", segment.segment_id},
            {"text", *text},
            {"is_final", true}});
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
