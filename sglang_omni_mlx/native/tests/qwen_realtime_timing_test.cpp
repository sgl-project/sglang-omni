// SPDX-License-Identifier: Apache-2.0
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "common/generation.h"
#include "common/transcription.h"
#include "qwen3_asr/realtime.h"

namespace {

bool Check(bool condition, const char *expression) {
  if (condition) {
    return true;
  }
  std::fprintf(stderr, "check failed: %s\n", expression);
  return false;
}

std::string AudioBase64(int sample_count) {
  static constexpr char alphabet[] =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::vector<uint8_t> bytes;
  bytes.reserve(static_cast<size_t>(sample_count) * 2);
  for (int i = 0; i < sample_count; ++i) {
    bytes.push_back(0xe8);
    bytes.push_back(0x03);
  }
  std::string encoded;
  for (size_t i = 0; i < bytes.size(); i += 3) {
    const uint32_t a = bytes[i];
    const uint32_t b = i + 1 < bytes.size() ? bytes[i + 1] : 0;
    const uint32_t c = i + 2 < bytes.size() ? bytes[i + 2] : 0;
    const uint32_t value = (a << 16) | (b << 8) | c;
    encoded += alphabet[(value >> 18) & 63];
    encoded += alphabet[(value >> 12) & 63];
    encoded += i + 1 < bytes.size() ? alphabet[(value >> 6) & 63] : '=';
    encoded += i + 2 < bytes.size() ? alphabet[value & 63] : '=';
  }
  return encoded;
}

bool WaitFor(const std::atomic<int> &value, int target) {
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (value.load(std::memory_order_acquire) < target) {
    if (std::chrono::steady_clock::now() >= deadline) {
      return false;
    }
    std::this_thread::yield();
  }
  return true;
}

struct SessionOutput {
  std::string log;
  int previews = 0;
};

SessionOutput RunSession(bool profile) {
  omni_server::TranscriptionWorker worker([] {}, [] {}, 1);
  qwen3_asr::Qwen3ASRTranscriber transcriber;
  qwen3_asr::RealtimeSettings settings =
      qwen3_asr::MakeRealtimeSettings(1000, 100, 30);
  settings.profile = profile;
  std::atomic<int> previews{0};
  std::ostringstream captured;
  const auto old = std::cerr.rdbuf(captured.rdbuf());
  const auto session = std::make_shared<qwen3_asr::RealtimeSession>(
      worker, transcriber, settings, [&previews](const std::string &raw) {
        const auto event = nlohmann::json::parse(raw);
        if (event.value("type", "") == "transcription.segment" &&
            !event.value("is_final", true)) {
          previews.fetch_add(1, std::memory_order_release);
        }
        return true;
      });
  session->Handle({{"type", "session.update"}, {"session", {}}});
  session->Handle(
      {{"type", "input_audio_buffer.append"}, {"audio", AudioBase64(1600)}});
  const bool previewed = WaitFor(previews, 1);
  session->Handle({{"type", "transcription.done"}});
  session->Close();
  std::cerr.rdbuf(old);
  return {captured.str(),
          previews.load(std::memory_order_acquire) + (previewed ? 0 : 100)};
}

std::vector<nlohmann::json> Timings(const std::string &log) {
  std::vector<nlohmann::json> timings;
  std::istringstream lines(log);
  std::string line;
  while (std::getline(lines, line)) {
    if (line.find("qwen_realtime_timing") == std::string::npos) {
      continue;
    }
    timings.push_back(nlohmann::json::parse(line));
  }
  return timings;
}

} // namespace

int main() {
  qwen3_asr::RealtimeDecodeTiming timing;
  timing.session_id = 3;
  timing.segment_id = 4;
  timing.decode_index = 5;
  timing.kind = "preview";
  timing.audio_sample_count = 1600;
  timing.end_sample = 1600;
  timing.submitted_at_ms = 100;
  timing.started_at_ms = 110;
  timing.finished_at_ms = 160;
  timing.succeeded = true;
  const auto json = qwen3_asr::RealtimeDecodeTimingJson(timing);
  const std::vector<std::string> schema = {
      "event",         "session_id",         "segment_id", "decode_index",
      "kind",          "audio_sample_count", "end_sample", "submitted_at_ms",
      "started_at_ms", "finished_at_ms",     "succeeded"};
  if (!Check(json.size() == schema.size(), "exact schema size") ||
      !Check(json["event"] == "qwen_realtime_timing", "event") ||
      !Check(json["session_id"] == 3, "session_id") ||
      !Check(json["segment_id"] == 4, "segment_id") ||
      !Check(json["decode_index"] == 5, "decode_index") ||
      !Check(json["kind"] == "preview", "kind") ||
      !Check(json["audio_sample_count"] == 1600, "audio_sample_count") ||
      !Check(json["end_sample"] == 1600, "end_sample") ||
      !Check(json["submitted_at_ms"] == 100, "submitted_at_ms") ||
      !Check(json["started_at_ms"] == 110, "started_at_ms") ||
      !Check(json["finished_at_ms"] == 160, "finished_at_ms") ||
      !Check(json["succeeded"], "succeeded")) {
    return 1;
  }
  for (const std::string &key : schema) {
    if (!Check(json.contains(key), key.c_str())) {
      return 1;
    }
  }
  for (const std::string &key :
       std::vector<std::string>{"text", "prompt", "audio", "speaker"}) {
    if (!Check(!json.contains(key), key.c_str())) {
      return 1;
    }
  }

  const SessionOutput profiled = RunSession(true);
  const auto events = Timings(profiled.log);
  if (!Check(profiled.previews == 1, "preview completed") ||
      !Check(events.size() == 2, "preview and final timing") ||
      !Check(events[0]["kind"] == "preview", "preview kind") ||
      !Check(events[1]["kind"] == "final", "final kind")) {
    return 1;
  }
  for (size_t i = 0; i < events.size(); ++i) {
    const auto &event = events[i];
    if (!Check(event["session_id"].get<uint64_t>() > 0, "session id") ||
        !Check(event["segment_id"] == 0, "segment id") ||
        !Check(event["decode_index"] == static_cast<int>(i + 1),
               "decode index") ||
        !Check(event["audio_sample_count"] == 1600, "sample count") ||
        !Check(event["end_sample"] == 1600, "end sample") ||
        !Check(event["succeeded"], "decode succeeded") ||
        !Check(event["submitted_at_ms"].get<int64_t>() > 0,
               "submitted timestamp") ||
        !Check(event["submitted_at_ms"] <= event["started_at_ms"],
               "queue order") ||
        !Check(event["started_at_ms"] <= event["finished_at_ms"],
               "decode order")) {
      return 1;
    }
  }

  const SessionOutput unprofiled = RunSession(false);
  if (!Check(unprofiled.previews == 1, "unprofiled preview completed") ||
      !Check(Timings(unprofiled.log).empty(), "profiling is off by default")) {
    return 1;
  }
  std::puts("ok: preview and final timings, opt-in only, content-free schema");
  return 0;
}
