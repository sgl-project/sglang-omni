// SPDX-License-Identifier: Apache-2.0
#include "moss_realtime.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "audio.h"

namespace moss {

using qwen3_asr::kSampleRate;
using qwen3_asr::SpeakerSegment;
using qwen3_asr::SpeakerSegmentsJson;
using qwen3_asr::StripUnicodeWhitespace;
using qwen3_asr::TranscriptionResult;

namespace {

// Note (Dayuxiaoshui): Voxt's Swift session: 4 s windows, and at most one
// preview a second of the last 2.5 s pending, once 1.25 s has arrived.
constexpr int kWindowSamples = 4 * kSampleRate;
constexpr int kPreviewSamples = kSampleRate * 5 / 2;
constexpr int kMinPreviewSamples = kSampleRate * 5 / 4;
constexpr auto kPreviewInterval = std::chrono::seconds(1);
// Note (Dayuxiaoshui): Voxt's intermediate budget, scaled by window length; a
// preview gets half a final window's, since it is decoded again.
constexpr int kMaxTokensPerPass = 1024;
constexpr double kPreviewTokensPerSecond = 16.0;
constexpr int kMinPreviewTokens = 48;
constexpr double kFinalTokensPerSecond = 32.0;
constexpr int kMinFinalTokens = 96;

} // namespace

MossRealtimeSession::MossRealtimeSession(qwen3_asr::TranscriptionWorker &worker,
                                         const MossTranscriber &transcriber,
                                         Sender sender)
    : RealtimeConnection(std::move(sender)), worker_(worker),
      transcriber_(transcriber) {}

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
  const std::optional<std::vector<float>> appended = AppendedSamples(audio);
  if (!appended.has_value()) {
    return;
  } else {
  }
  std::optional<Decode> decode;
  MossOptions options;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    pending_samples_.insert(pending_samples_.end(), appended->begin(),
                            appended->end());
    decode = NextDecode();
    if (decode.has_value()) {
      options = DecodeOptions(*decode);
    } else {
      return;
    }
  }
  qwen3_asr::Transcription job = DecodeJob(*decode, std::move(options));
  std::weak_ptr<MossRealtimeSession> weak_self =
      std::static_pointer_cast<MossRealtimeSession>(shared_from_this());
  worker_.Submit(
      std::move(job), cancel_,
      [weak_self, decode = std::move(*decode)](
          std::optional<TranscriptionResult> result, std::exception_ptr error) {
        const std::shared_ptr<MossRealtimeSession> self = weak_self.lock();
        if (!self) {
          return;
        } else if (error) {
          self->ReportDecodeFailure(error);
        } else {
          self->Publish(decode, *result);
        }
        {
          std::lock_guard<std::mutex> lock(self->mutex_);
          self->decoding_ = false;
        }
        self->decode_done_.notify_all();
      });
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
  const auto now = std::chrono::steady_clock::now();
  const int pending_count = static_cast<int>(pending_samples_.size());
  Decode decode;
  if (decoding_) {
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

qwen3_asr::Transcription
MossRealtimeSession::DecodeJob(const Decode &decode,
                               MossOptions options) const {
  return [&transcriber = transcriber_, samples = decode.samples,
          options = std::move(options)](const std::atomic<bool> &cancel) {
    return transcriber.Transcribe(samples, options, cancel);
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

void MossRealtimeSession::Flush() {
  while (true) {
    Decode decode;
    MossOptions options;
    {
      std::unique_lock<std::mutex> lock(mutex_);
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
    }
    try {
      Publish(decode, worker_.Transcribe(DecodeJob(decode, std::move(options)),
                                         cancel_));
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
