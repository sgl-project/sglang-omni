// SPDX-License-Identifier: Apache-2.0
#include "moss/moss_realtime.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include "common/audio.h"
#include "common/text.h"
#include "common/transcription.h"

namespace moss {

using omni_common::kSampleRate;
using omni_common::SpeakerSegment;
using omni_common::SpeakerSegmentsJson;
using omni_common::StripUnicodeWhitespace;
using omni_common::TranscriptionResult;

// No prompt, transcript or speaker content. A stage's cost is the difference
// of two consecutive stamps.
struct DecodeTiming {
  int window_id = 0;
  std::string kind;
  int audio_sample_count = 0;
  // The append, stop or finished decode that made this decode due.
  int64_t append_at_ms = 0;
  int64_t decision_at_ms = 0;
  int64_t submit_at_ms = 0;
  int64_t transcribe_started_at_ms = 0;
  int64_t transcribe_at_ms = 0;
  int64_t publish_at_ms = 0;
  int verify_round_count = 0;
  int proposed_draft_token_count = 0;
  int accepted_draft_token_count = 0;
};

namespace {

// Note (Dayuxiaoshui): Voxt's Swift session: 4 s windows, and at most one
// preview a second of the last 2.5 s pending, once 1.25 s has arrived.
constexpr int kWindowSamples = 4 * kSampleRate;
constexpr int kPreviewSamples = kSampleRate * 5 / 2;
// 0.75 s, not Swift's 1.25 s: first text 0.55 s sooner, same finals (MacBook
// Air A/B, 3 runs per arm).
constexpr int kMinPreviewSamples = kSampleRate * 3 / 4;
constexpr auto kPreviewInterval = std::chrono::seconds(1);
// Note (Dayuxiaoshui): Voxt's intermediate budget, scaled by window length; a
// preview gets half a final window's, since it is decoded again.
constexpr int kMaxTokensPerPass = 1024;
constexpr double kPreviewTokensPerSecond = 16.0;
constexpr int kMinPreviewTokens = 48;
constexpr double kFinalTokensPerSecond = 32.0;
constexpr int kMinFinalTokens = 96;

int64_t SteadyClockMilliseconds(std::chrono::steady_clock::time_point time) {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             time.time_since_epoch())
      .count();
}

int64_t NowSteadyClockMilliseconds() {
  return SteadyClockMilliseconds(std::chrono::steady_clock::now());
}

nlohmann::ordered_json DecodeTimingJson(const DecodeTiming &timing) {
  return {{"event", "moss_preview_timing"},
          {"window_id", timing.window_id},
          {"kind", timing.kind},
          {"audio_sample_count", timing.audio_sample_count},
          {"append_at_ms", timing.append_at_ms},
          {"decision_at_ms", timing.decision_at_ms},
          {"submit_at_ms", timing.submit_at_ms},
          {"transcribe_started_at_ms", timing.transcribe_started_at_ms},
          {"transcribe_at_ms", timing.transcribe_at_ms},
          {"publish_at_ms", timing.publish_at_ms},
          {"min_preview_samples", kMinPreviewSamples},
          {"verify_round_count", timing.verify_round_count},
          {"proposed_draft_token_count", timing.proposed_draft_token_count},
          {"accepted_draft_token_count", timing.accepted_draft_token_count}};
}

} // namespace

MossRealtimeSession::MossRealtimeSession(
    omni_server::TranscriptionWorker &worker, TranscribeFn transcribe,
    Sender sender, ClockFn clock)
    : RealtimeConnection(std::move(sender)), worker_(worker),
      transcribe_(std::move(transcribe)), clock_(std::move(clock)) {}

void MossRealtimeSession::Close() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    CancelPreviewLocked();
  }
  RealtimeConnection::Close();
}

void MossRealtimeSession::CancelPreviewLocked() {
  if (preview_in_flight_) {
    preview_cancel_->store(true);
  } else {
  }
}

bool MossRealtimeSession::Handle(const nlohmann::json &message) {
  const nlohmann::json type = message.value("type", nlohmann::json());
  if (type == "session.update") {
    const std::optional<nlohmann::json> session = ManualTurnSession(message);
    if (session.has_value()) {
      const nlohmann::json value = session->value("prompt", nlohmann::json());
      const std::optional<std::string> prompt =
          value.is_string()
              ? std::optional<std::string>(value.get<std::string>())
              : std::nullopt;
      try {
        ValidatePrompt(prompt);
      } catch (const std::invalid_argument &error) {
        SendError("invalid_request_error", "invalid_prompt", error.what());
        return true;
      }
      {
        std::lock_guard<std::mutex> lock(mutex_);
        prompt_ = prompt;
      }
      Send({{"type", "transcription_session.updated"}});
    } else {
    }
    return true;
  } else if (type == "input_audio_buffer.append") {
    Append(message.value("audio", nlohmann::json()));
    return true;
  } else if (type == "input_audio_buffer.commit") {
    Flush();
    return true;
  } else if (type == "transcription.done") {
    Flush();
    std::string text;
    std::vector<SpeakerSegment> segments;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      for (const std::string &part : committed_texts_) {
        text += (text.empty() ? "" : "\n") + part;
      }
      segments = committed_segments_;
    }
    Send({{"type", "transcription.completed"},
          {"text", text},
          {"segments", SpeakerSegmentsJson(segments)}});
    return false;
  } else {
    SendError("invalid_request_error", "invalid_event", "Unknown event type.");
    return true;
  }
}

void MossRealtimeSession::Append(const nlohmann::json &audio) {
  const auto appended_at = std::chrono::steady_clock::now();
  const std::optional<std::vector<float>> appended = AppendedSamples(audio);
  if (!appended.has_value()) {
    return;
  } else {
  }
  std::optional<Decode> decode;
  MossOptions options;
  omni_server::CancelFlag flag;
  std::chrono::steady_clock::time_point decided_at;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    pending_samples_.insert(pending_samples_.end(), appended->begin(),
                            appended->end());
    decode = NextDecode();
    if (decode.has_value()) {
      options = DecodeOptions(*decode);
      flag = decode->kind == DecodeKind::kPreview ? preview_cancel_ : cancel_;
      decided_at = std::chrono::steady_clock::now();
    } else {
      return;
    }
  }
  SubmitDecode(std::move(*decode), std::move(options), std::move(flag),
               appended_at, decided_at);
}

void MossRealtimeSession::SubmitDecode(
    Decode decode, MossOptions options, omni_server::CancelFlag flag,
    std::chrono::steady_clock::time_point due_at,
    std::chrono::steady_clock::time_point decided_at) {
  const std::shared_ptr<DecodeTiming> timing =
      StartDecodeTiming(decode, due_at, decided_at);
  omni_server::Transcription job =
      DecodeJob(decode, std::move(options), timing);
  std::weak_ptr<MossRealtimeSession> weak_self =
      std::static_pointer_cast<MossRealtimeSession>(shared_from_this());
  const omni_server::CancelFlag callback_flag = flag;
  timing->submit_at_ms = NowSteadyClockMilliseconds();
  worker_.Submit(
      std::move(job), std::move(flag),
      [weak_self, decode = std::move(decode), flag = callback_flag, timing](
          std::optional<TranscriptionResult> result, std::exception_ptr error) {
        const std::shared_ptr<MossRealtimeSession> self = weak_self.lock();
        if (!self) {
          return;
        } else if (error) {
          self->ReportDecodeFailure(error);
        } else if (decode.kind == DecodeKind::kPreview && flag->load()) {
          // The transcriber can return after its last cancellation check; the
          // cancelled preview's result is already replaced, so it is dropped.
        } else {
          self->PublishWithTiming(decode, *result, *timing);
        }
        self->OnDecodeDone(decode.kind);
      });
}

void MossRealtimeSession::OnDecodeDone(DecodeKind kind) {
  const auto finished_at = std::chrono::steady_clock::now();
  std::optional<Decode> next;
  MossOptions options;
  std::chrono::steady_clock::time_point decided_at;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    decoding_ = false;
    if (kind == DecodeKind::kPreview) {
      preview_in_flight_ = false;
    } else {
    }
    // A window that filled while this decode ran is due now, not at the next
    // append or stop; a cancelled preview's completion starts it.
    if (static_cast<int>(pending_samples_.size()) >= kWindowSamples) {
      next = NextDecode();
      options = DecodeOptions(*next);
      decided_at = std::chrono::steady_clock::now();
    } else {
    }
  }
  decode_done_.notify_all();
  if (next.has_value()) {
    SubmitDecode(std::move(*next), std::move(options), cancel_, finished_at,
                 decided_at);
  } else {
  }
}

MossRealtimeSession::Decode
MossRealtimeSession::TakeFinalWindow(int sample_count) {
  Decode decode;
  decode.kind = DecodeKind::kFinal;
  decode.window_id = next_window_id_++;
  decode.samples.assign(pending_samples_.begin(),
                        pending_samples_.begin() + sample_count);
  decode.offset_seconds =
      static_cast<double>(pending_start_sample_) / kSampleRate;
  pending_samples_.erase(pending_samples_.begin(),
                         pending_samples_.begin() + sample_count);
  pending_start_sample_ += sample_count;
  return decode;
}

std::optional<MossRealtimeSession::Decode> MossRealtimeSession::NextDecode() {
  const auto now = clock_();
  const int pending_count = static_cast<int>(pending_samples_.size());
  Decode decode;
  if (decoding_) {
    // A full window outranks the preview in flight: cancelling it starts the
    // window's final at the preview's completion, not after work it replaces.
    if (pending_count >= kWindowSamples) {
      CancelPreviewLocked();
    } else {
    }
    return std::nullopt;
  } else if (pending_count >= kWindowSamples) {
    decode = TakeFinalWindow(kWindowSamples);
  } else if (pending_count < kMinPreviewSamples ||
             (last_decode_time_.has_value() &&
              now - *last_decode_time_ < kPreviewInterval)) {
    return std::nullopt;
  } else {
    const int preview_count = std::min(pending_count, kPreviewSamples);
    const int preview_start = pending_count - preview_count;
    decode.kind = DecodeKind::kPreview;
    decode.window_id = next_window_id_;
    decode.samples.assign(pending_samples_.begin() + preview_start,
                          pending_samples_.end());
    decode.offset_seconds =
        static_cast<double>(pending_start_sample_ + preview_start) /
        kSampleRate;
  }
  preview_in_flight_ = decode.kind == DecodeKind::kPreview;
  if (preview_in_flight_) {
    preview_cancel_ = omni_server::NewCancelFlag();
  } else {
  }
  decoding_ = true;
  last_decode_time_ = now;
  return decode;
}

MossOptions MossRealtimeSession::DecodeOptions(const Decode &decode) const {
  const double seconds =
      static_cast<double>(decode.samples.size()) / kSampleRate;
  MossOptions options;
  options.prompt = prompt_;
  options.window_offset_seconds = decode.offset_seconds;
  // Note (Dayuxiaoshui): Voxt's Swift session always applied both stops.
  options.stop_at_end_of_text = true;
  options.stop_on_token_loop = true;
  options.max_new_tokens =
      decode.kind == DecodeKind::kPreview
          ? std::min(kMaxTokensPerPass,
                     std::max(kMinPreviewTokens,
                              static_cast<int>(std::ceil(
                                  seconds * kPreviewTokensPerSecond))))
          : std::min(kMaxTokensPerPass,
                     std::max(kMinFinalTokens,
                              static_cast<int>(
                                  std::ceil(seconds * kFinalTokensPerSecond))));
  return options;
}

std::shared_ptr<DecodeTiming> MossRealtimeSession::StartDecodeTiming(
    const Decode &decode, std::chrono::steady_clock::time_point due_at,
    std::chrono::steady_clock::time_point decided_at) {
  auto timing = std::make_shared<DecodeTiming>();
  timing->window_id = decode.window_id;
  timing->kind = decode.kind == DecodeKind::kPreview ? "preview" : "final";
  timing->audio_sample_count = static_cast<int>(decode.samples.size());
  timing->append_at_ms = SteadyClockMilliseconds(due_at);
  timing->decision_at_ms = SteadyClockMilliseconds(decided_at);
  return timing;
}

omni_server::Transcription
MossRealtimeSession::DecodeJob(const Decode &decode, MossOptions options,
                               std::shared_ptr<DecodeTiming> timing) const {
  return [transcribe = transcribe_, samples = decode.samples,
          options = std::move(options),
          timing = std::move(timing)](const std::atomic<bool> &cancel) {
    timing->transcribe_started_at_ms = NowSteadyClockMilliseconds();
    return transcribe(samples, options, cancel);
  };
}

void MossRealtimeSession::Publish(const Decode &decode,
                                  const TranscriptionResult &result) {
  const std::string text = StripUnicodeWhitespace(result.text);
  if (decode.kind == DecodeKind::kPreview) {
    // Note (Dayuxiaoshui): an empty preview is still sent, to clear the one on
    // screen.
    Send({{"type", "transcription.segment"},
          {"segment_id", decode.window_id},
          {"text", text},
          {"is_final", false}});
    return;
  } else {
  }
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!text.empty()) {
      committed_texts_.push_back(text);
    } else {
    }
    committed_segments_.insert(committed_segments_.end(),
                               result.segments.begin(), result.segments.end());
  }
  Send({{"type", "transcription.segment"},
        {"segment_id", decode.window_id},
        {"text", text},
        {"is_final", true},
        {"segments", SpeakerSegmentsJson(result.segments)}});
}

void MossRealtimeSession::PublishWithTiming(const Decode &decode,
                                            const TranscriptionResult &result,
                                            DecodeTiming &timing) {
  timing.transcribe_at_ms = NowSteadyClockMilliseconds();
  timing.verify_round_count = result.verify_round_count;
  timing.proposed_draft_token_count = result.proposed_draft_token_count;
  timing.accepted_draft_token_count = result.accepted_draft_token_count;
  Publish(decode, result);
  timing.publish_at_ms = NowSteadyClockMilliseconds();
  std::cerr << DecodeTimingJson(timing).dump() << "\n";
}

void MossRealtimeSession::Flush() {
  const auto stopped_at = std::chrono::steady_clock::now();
  while (true) {
    Decode decode;
    MossOptions options;
    std::chrono::steady_clock::time_point decided_at;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      // The stop's final replaces the preview in flight, so the preview is
      // cancelled rather than waited out.
      CancelPreviewLocked();
      decode_done_.wait(lock, [&] { return !decoding_; });
      const int pending_count = static_cast<int>(pending_samples_.size());
      if (pending_count == 0) {
        return;
      } else if (pending_count >= 2 * kWindowSamples) {
        // Note (Dayuxiaoshui): a backlog goes out window by window; as one
        // window, as Swift sent it, it would hit the budget and lose its tail.
        decode = TakeFinalWindow(kWindowSamples);
      } else {
        // Note (Dayuxiaoshui): the rest is one window, as Swift ended; under
        // two windows it stays within the token budget.
        decode = TakeFinalWindow(pending_count);
      }
      options = DecodeOptions(decode);
      decoding_ = true;
      decided_at = std::chrono::steady_clock::now();
    }
    const std::shared_ptr<DecodeTiming> timing =
        StartDecodeTiming(decode, stopped_at, decided_at);
    try {
      omni_server::Transcription job =
          DecodeJob(decode, std::move(options), timing);
      timing->submit_at_ms = NowSteadyClockMilliseconds();
      PublishWithTiming(decode, worker_.Transcribe(std::move(job), cancel_),
                        *timing);
    } catch (...) {
      ReportDecodeFailure(std::current_exception());
    }
    {
      std::lock_guard<std::mutex> lock(mutex_);
      decoding_ = false;
    }
    decode_done_.notify_all();
  }
}

} // namespace moss
