// SPDX-License-Identifier: Apache-2.0
#include "nemotron_realtime.h"

#include <array>
#include <cmath>
#include <cstdint>

namespace nemotron {

namespace {

// Note (Dayuxiaoshui): Voxt's live latency settings; a requested chunk length
// snaps to the nearest, as the Swift session's custom delay preset does.
constexpr std::array<int, 5> kChunkMilliseconds = {80, 160, 320, 560, 1120};
// Note (Dayuxiaoshui): the stream counts samples in int, which a 16 kHz
// session overflows after about 37 hours; one day stays well below that.
constexpr int64_t kMaxSessionSeconds = 24 * 60 * 60;

} // namespace

NemotronRealtimeSession::NemotronRealtimeSession(
    qwen3_asr::TranscriptionWorker &worker,
    const NemotronTranscriber &transcriber, Sender sender)
    : RealtimeConnection(std::move(sender)), worker_(worker),
      transcriber_(transcriber) {}

bool NemotronRealtimeSession::Handle(const nlohmann::json &message) {
  const nlohmann::json type = message.value("type", nlohmann::json());
  if (type == "session.update") {
    const std::optional<nlohmann::json> session = ManualTurnSession(message);
    if (!session.has_value()) {
      return true;
    } else {
    }
    const nlohmann::json language =
        session->value("language", nlohmann::json());
    const nlohmann::json chunk = session->value("chunk_ms", nlohmann::json());
    std::unique_lock<std::mutex> lock(mutex_);
    if (stream_ != nullptr) {
      lock.unlock();
      SendError("invalid_request_error", "session_started",
                "Settings apply before the first audio.");
    } else if (!chunk.is_null() &&
               !(chunk.is_number() && chunk.get<double>() > 0)) {
      lock.unlock();
      SendError("invalid_request_error", "invalid_chunk",
                "chunk_ms must be a positive number.");
    } else {
      options_.language =
          language.is_string()
              ? std::optional<std::string>(language.get<std::string>())
              : std::nullopt;
      if (chunk.is_null()) {
        options_.chunk_frames.reset();
      } else {
        const double requested = chunk.get<double>();
        int nearest = kChunkMilliseconds.front();
        for (const int candidate : kChunkMilliseconds) {
          if (std::abs(candidate - requested) < std::abs(nearest - requested)) {
            nearest = candidate;
          } else {
          }
        }
        options_.chunk_frames = static_cast<int>(
            std::lround(nearest / (transcriber_.frame_seconds() * 1000.0)));
      }
      lock.unlock();
      Send({{"type", "transcription_session.updated"}});
    }
    return true;
  } else if (type == "input_audio_buffer.append") {
    Append(message.value("audio", nlohmann::json()));
    return true;
  } else if (type == "input_audio_buffer.commit") {
    Finish();
    return true;
  } else if (type == "transcription.done") {
    Finish();
    std::unique_lock<std::mutex> lock(mutex_);
    const qwen3_asr::TranscriptionResult result = final_result_;
    lock.unlock();
    Send({{"type", "transcription.completed"},
          {"text", result.text},
          {"segments", qwen3_asr::SpeakerSegmentsJson(result.segments)}});
    return false;
  } else {
    SendError("invalid_request_error", "invalid_event", "Unknown event type.");
    return true;
  }
}

void NemotronRealtimeSession::Append(const nlohmann::json &audio) {
  const std::optional<std::vector<float>> appended = AppendedSamples(audio);
  if (!appended.has_value()) {
    return;
  } else {
  }
  std::unique_lock<std::mutex> lock(mutex_);
  if (finished_) {
    lock.unlock();
    SendError("invalid_request_error", "session_finished",
              "The stream was committed; audio after it is not decoded.");
    return;
  } else if (failed_) {
    return;
  } else if (received_samples_ + static_cast<int64_t>(appended->size()) >
             kMaxSessionSeconds * transcriber_.front_end().sampling_rate) {
    lock.unlock();
    SendError("invalid_request_error", "session_too_long",
              "A session takes at most 24 hours of audio.");
    return;
  } else {
  }
  received_samples_ += static_cast<int64_t>(appended->size());
  pending_samples_.insert(pending_samples_.end(), appended->begin(),
                          appended->end());
  if (!decoding_) {
    SubmitDecode();
  } else {
  }
}

void NemotronRealtimeSession::SubmitDecode() {
  if (stream_ == nullptr) {
    stream_ = std::make_shared<NemotronStream>(transcriber_, options_);
  } else {
  }
  decoding_ = true;
  std::vector<float> samples = std::move(pending_samples_);
  pending_samples_.clear();
  std::weak_ptr<NemotronRealtimeSession> weak_self =
      std::static_pointer_cast<NemotronRealtimeSession>(shared_from_this());
  worker_.Submit(
      [stream = stream_,
       samples = std::move(samples)](const std::atomic<bool> &cancel) {
        stream->Append(samples, cancel);
        qwen3_asr::TranscriptionResult preview;
        preview.text = stream->Text();
        return preview;
      },
      cancel_,
      [weak_self](std::optional<qwen3_asr::TranscriptionResult> result,
                  std::exception_ptr error) {
        const std::shared_ptr<NemotronRealtimeSession> self = weak_self.lock();
        if (!self) {
          return;
        } else {
        }
        std::unique_lock<std::mutex> lock(self->mutex_);
        self->decoding_ = false;
        std::optional<std::string> preview;
        if (error) {
          self->failed_ = true;
        } else if (!result->text.empty() && result->text != self->shown_text_) {
          self->shown_text_ = result->text;
          preview = result->text;
        } else {
        }
        if (!self->failed_ && !self->finished_ &&
            !self->pending_samples_.empty()) {
          self->SubmitDecode();
        } else {
        }
        lock.unlock();
        self->decode_done_.notify_all();
        if (error) {
          self->ReportDecodeFailure(error);
        } else if (preview.has_value()) {
          self->Send({{"type", "transcription.segment"},
                      {"segment_id", 0},
                      {"text", *preview},
                      {"is_final", false}});
        } else {
        }
      });
}

void NemotronRealtimeSession::Finish() {
  std::unique_lock<std::mutex> lock(mutex_);
  decode_done_.wait(lock, [&] { return !decoding_; });
  if (finished_ || failed_) {
    finished_ = true;
    return;
  } else {
  }
  finished_ = true;
  if (stream_ == nullptr) {
    stream_ = std::make_shared<NemotronStream>(transcriber_, options_);
  } else {
  }
  std::vector<float> samples = std::move(pending_samples_);
  pending_samples_.clear();
  lock.unlock();
  try {
    const qwen3_asr::TranscriptionResult result = worker_.Transcribe(
        [stream = stream_,
         samples = std::move(samples)](const std::atomic<bool> &cancel) {
          stream->Finish(samples, cancel);
          return stream->Result();
        },
        cancel_);
    lock.lock();
    final_result_ = result;
    lock.unlock();
    Send({{"type", "transcription.segment"},
          {"segment_id", 0},
          {"text", result.text},
          {"is_final", true},
          {"segments", qwen3_asr::SpeakerSegmentsJson(result.segments)}});
  } catch (...) {
    lock.lock();
    failed_ = true;
    lock.unlock();
    ReportDecodeFailure(std::current_exception());
  }
}

} // namespace nemotron
