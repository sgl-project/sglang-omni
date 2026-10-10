// SPDX-License-Identifier: Apache-2.0
#include "realtime.h"

#include <algorithm>
#include <iostream>
#include <random>

#include "audio.h"

namespace qwen3_asr {

namespace {

// Note (Jiaxin Deng): after two full re-decodes a refresh continues from the
// shown text minus its last tokens, which may still change.
constexpr int kPrefixAfterRefreshCount = 2;
constexpr int kPrefixRollbackTokenCount = 5;
constexpr float kSilentPeak = 1e-3f;
constexpr float kPcm16FullScale = 32768.0f;

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

std::optional<std::vector<uint8_t>> DecodeBase64(const std::string &text) {
  // Note (Jiaxin Deng): matches Python's b64decode(validate=False), which
  // drops characters outside the alphabet but still checks the padding.
  std::string clean;
  for (const char c : text) {
    if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
        (c >= '0' && c <= '9') || c == '+' || c == '/' || c == '=') {
      clean.push_back(c);
    } else {
    }
  }
  size_t padding = 0;
  while (!clean.empty() && clean.back() == '=') {
    clean.pop_back();
    ++padding;
  }
  const size_t remainder = clean.size() % 4;
  if (clean.find('=') != std::string::npos || remainder == 1 ||
      (remainder != 0 && padding < 4 - remainder)) {
    return std::nullopt;
  } else {
  }
  const auto value = [](char c) -> uint32_t {
    if (c >= 'A' && c <= 'Z')
      return c - 'A';
    if (c >= 'a' && c <= 'z')
      return c - 'a' + 26;
    if (c >= '0' && c <= '9')
      return c - '0' + 52;
    return c == '+' ? 62 : 63;
  };
  std::vector<uint8_t> bytes;
  uint32_t buffer = 0;
  int bits = 0;
  for (const char c : clean) {
    buffer = (buffer << 6) | value(c);
    bits += 6;
    if (bits >= 8) {
      bits -= 8;
      bytes.push_back(static_cast<uint8_t>((buffer >> bits) & 0xFF));
    } else {
    }
  }
  return bytes;
}

std::string NewEventId() {
  static thread_local std::mt19937_64 generator{std::random_device{}()};
  static constexpr char kHex[] = "0123456789abcdef";
  std::string id = "evt_";
  for (int i = 0; i < 32; ++i)
    id.push_back(kHex[generator() % 16]);
  return id;
}

} // namespace

RealtimeSettings MakeRealtimeSettings(int decode_interval_ms,
                                      int first_decode_ms,
                                      double max_segment_seconds) {
  return {decode_interval_ms * kSampleRate / 1000,
          first_decode_ms * kSampleRate / 1000,
          static_cast<int>(max_segment_seconds * kSampleRate)};
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

RealtimeConnection::RealtimeConnection(Sender sender)
    : sender_(std::move(sender)) {}

void RealtimeConnection::Send(nlohmann::ordered_json event) {
  std::lock_guard<std::mutex> lock(send_mutex_);
  if (closed_) {
    return;
  } else {
  }
  event_index_ += 1;
  event["event_id"] = NewEventId();
  event["event_index"] = event_index_;
  if (!sender_(event.dump())) {
    closed_ = true;
  } else {
  }
}

void RealtimeConnection::SendError(const std::string &type,
                                   const std::string &code,
                                   const std::string &message) {
  Send({{"type", "error"},
        {"error", {{"type", type}, {"code", code}, {"message", message}}}});
}

void RealtimeConnection::Close() {
  cancel_->store(true);
  std::lock_guard<std::mutex> lock(send_mutex_);
  closed_ = true;
}

std::optional<std::vector<float>>
RealtimeConnection::AppendedSamples(const nlohmann::json &audio) {
  const std::optional<std::vector<uint8_t>> pcm =
      audio.is_string() ? DecodeBase64(audio.get<std::string>()) : std::nullopt;
  if (!pcm.has_value() || pcm->empty() || pcm->size() % 2 != 0) {
    SendError("invalid_request_error", "invalid_audio",
              "Audio must be base64 PCM16.");
    return std::nullopt;
  } else {
  }
  std::vector<float> samples;
  samples.reserve(pcm->size() / 2);
  for (size_t i = 0; i + 1 < pcm->size(); i += 2) {
    const int16_t value =
        static_cast<int16_t>((*pcm)[i] | ((*pcm)[i + 1] << 8));
    samples.push_back(static_cast<float>(value) / kPcm16FullScale);
  }
  return samples;
}

std::optional<nlohmann::json>
RealtimeConnection::ManualTurnSession(const nlohmann::json &message) {
  const nlohmann::json session = message.value("session", nlohmann::json());
  if (!session.is_object() || (session.contains("turn_detection") &&
                               !session["turn_detection"].is_null())) {
    SendError("invalid_request_error", "unsupported_session",
              "Only manual turns are supported.");
    return std::nullopt;
  } else {
    return session;
  }
}

void RealtimeConnection::ReportDecodeFailure(std::exception_ptr error) {
  try {
    std::rethrow_exception(error);
  } catch (const TranscriptionCancelled &) {
    return;
  } catch (const std::exception &failure) {
    // Note (Jiaxin Deng): log the type alone, never the audio or text.
    std::cerr << "realtime decode failed: " << typeid(failure).name() << "\n";
  } catch (...) {
    std::cerr << "realtime decode failed\n";
  }
  SendError("server_error", "transcription_failed", "Transcription failed.");
}

RealtimeSession::RealtimeSession(TranscriptionWorker &worker,
                                 const Qwen3ASRTranscriber &transcriber,
                                 RealtimeSettings settings, Sender sender)
    : RealtimeConnection(std::move(sender)), worker_(worker),
      transcriber_(transcriber), settings_(settings) {}

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
  segment_ = std::move(segment);
}

std::vector<float> RealtimeSession::SegmentSamples(const Segment &segment,
                                                   long end_sample) const {
  const long begin = segment.start_sample - buffer_start_sample_;
  const long end = end_sample - buffer_start_sample_;
  return std::vector<float>(samples_.begin() + begin, samples_.begin() + end);
}

TranscriptionOptions RealtimeSession::DecodeOptions(Segment &segment) {
  TranscriptionOptions options;
  options.language = segment.language;
  const bool use_prefix = segment.decode_count >= kPrefixAfterRefreshCount &&
                          !segment.transcript.empty() &&
                          segment.language.has_value();
  if (use_prefix) {
    auto [ids, text] = transcriber_.RetainedPrefix(segment.transcript,
                                                   kPrefixRollbackTokenCount);
    options.prefix_token_ids = std::move(ids);
    options.prefix_text = std::move(text);
  } else {
  }
  segment.decode_count += 1;
  return options;
}

Transcription RealtimeSession::Decode(std::vector<float> samples,
                                      TranscriptionOptions options) const {
  return [&transcriber = transcriber_, samples = std::move(samples),
          options = std::move(options)](const std::atomic<bool> &cancel) {
    return transcriber.Transcribe(samples, options, cancel);
  };
}

void RealtimeSession::ApplyResult(Segment &segment,
                                  const TranscriptionResult &result) {
  if (result.language.has_value() && !result.language->empty()) {
    segment.language = result.language;
  } else {
  }
  segment.transcript = result.text;
}

void RealtimeSession::MaybeStartRefresh() {
  std::vector<float> samples;
  TranscriptionOptions options;
  int segment_id = 0;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (refreshing_ || finalizing_ || !segment_.has_value() ||
        EndSample() < segment_->next_refresh_sample) {
      return;
    } else {
    }
    Segment &segment = *segment_;
    const long end_sample = EndSample();
    samples = SegmentSamples(segment, end_sample);
    const bool silent =
        PeakIsSilent(samples.data(), samples.size(), kSilentPeak);
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
    options = DecodeOptions(segment);
    segment_id = segment.segment_id;
    refreshing_ = true;
  }
  std::weak_ptr<RealtimeSession> weak_self =
      std::static_pointer_cast<RealtimeSession>(shared_from_this());
  worker_.Submit(
      Decode(std::move(samples), std::move(options)), cancel_,
      [weak_self, segment_id](std::optional<TranscriptionResult> result,
                              std::exception_ptr error) {
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
            self->ApplyResult(*self->segment_, *result);
            if (result->text != self->segment_->last_text) {
              self->segment_->last_text = result->text;
              preview = result->text;
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
    if (PeakIsSilent(samples.data(), samples.size(), kSilentPeak)) {
      text = std::string();
    } else {
      TranscriptionOptions options;
      {
        std::lock_guard<std::mutex> lock(mutex_);
        options = DecodeOptions(segment);
      }
      try {
        const TranscriptionResult result = worker_.Transcribe(
            Decode(std::move(samples), std::move(options)), cancel_);
        ApplyResult(segment, result);
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
