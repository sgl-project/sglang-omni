// SPDX-License-Identifier: Apache-2.0
#include "common/realtime_connection.h"

#include <cstdint>
#include <iostream>
#include <random>
#include <utility>

#include "common/generation.h"
#include "common/text.h"

namespace omni_server {

using omni_common::DecodeBase64;
using omni_common::GenerationCancelled;

namespace {

constexpr float kPcm16FullScale = 32768.0f;

std::string NewEventId() {
  static thread_local std::mt19937_64 generator{std::random_device{}()};
  static constexpr char kHex[] = "0123456789abcdef";
  std::string id = "evt_";
  for (int i = 0; i < 32; ++i)
    id.push_back(kHex[generator() % 16]);
  return id;
}

} // namespace

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
  } catch (const GenerationCancelled &) {
    return;
  } catch (const std::exception &failure) {
    // Note (Jiaxin Deng): log the type alone, never the audio or text.
    std::cerr << "realtime decode failed: " << typeid(failure).name() << "\n";
  } catch (...) {
    std::cerr << "realtime decode failed\n";
  }
  SendError("server_error", "transcription_failed", "Transcription failed.");
}

} // namespace omni_server
