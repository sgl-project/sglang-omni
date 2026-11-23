// SPDX-License-Identifier: Apache-2.0
// MossRealtimeSession scheduling without a checkpoint: the 0.75 s
// first-preview gate, and a stop or a full window preempting the preview in
// flight so stop-to-final is not postponed. The transcriber is a scripted
// 400 ms decode that ends early once its cancel flag is set.
#include "moss/moss_realtime.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "common/audio.h"
#include "common/generation.h"
#include "common/worker.h"

namespace {

using namespace std::chrono_literals;

constexpr auto kDecodeTime = 400ms;
// A cancelled preview is dropped within a few poll intervals, so a preempted
// stop-to-final is one decode plus slack; waiting the preview out is two.
constexpr auto kPreemptedStopToFinalLimit = kDecodeTime * 3 / 2;
constexpr int kGateSamples = omni_common::kSampleRate * 3 / 4;
constexpr int kWindowSamples = 4 * omni_common::kSampleRate;

void Require(bool condition, std::string_view message) {
  if (!condition) {
    throw std::runtime_error(std::string(message));
  } else {
  }
}

std::string Base64(const std::vector<uint8_t> &bytes) {
  static constexpr char kAlphabet[] =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string out;
  out.reserve((bytes.size() + 2) / 3 * 4);
  for (size_t i = 0; i < bytes.size(); i += 3) {
    const uint32_t a = bytes[i];
    const uint32_t b = i + 1 < bytes.size() ? bytes[i + 1] : 0;
    const uint32_t c = i + 2 < bytes.size() ? bytes[i + 2] : 0;
    const uint32_t triple = (a << 16) | (b << 8) | c;
    out.push_back(kAlphabet[(triple >> 18) & 63]);
    out.push_back(kAlphabet[(triple >> 12) & 63]);
    out.push_back(i + 1 < bytes.size() ? kAlphabet[(triple >> 6) & 63] : '=');
    out.push_back(i + 2 < bytes.size() ? kAlphabet[triple & 63] : '=');
  }
  return out;
}

nlohmann::json AppendMessage(int sample_count) {
  const std::vector<uint8_t> pcm(static_cast<size_t>(sample_count) * 2, 0);
  return {{"type", "input_audio_buffer.append"}, {"audio", Base64(pcm)}};
}

// The session's events, appended on the socket thread and the worker thread.
class Events {
public:
  omni_server::RealtimeConnection::Sender Sender() {
    return [this](const std::string &event) {
      std::lock_guard<std::mutex> lock(mutex_);
      events_.push_back(nlohmann::json::parse(event));
      return true;
    };
  }

  int Segments(bool is_final) const {
    std::lock_guard<std::mutex> lock(mutex_);
    int count = 0;
    for (const nlohmann::json &event : events_) {
      if (event.value("type", "") == "transcription.segment" &&
          event.value("is_final", false) == is_final) {
        ++count;
      } else {
      }
    }
    return count;
  }

  bool WaitForSegments(bool is_final, int count,
                       std::chrono::milliseconds timeout) const {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (std::chrono::steady_clock::now() < deadline) {
      if (Segments(is_final) >= count) {
        return true;
      } else {
      }
      std::this_thread::sleep_for(1ms);
    }
    return false;
  }

private:
  mutable std::mutex mutex_;
  std::vector<nlohmann::json> events_;
};

// A 400 ms decode that stops within a poll interval once cancelled.
class FakeTranscriber {
public:
  moss::MossRealtimeSession::TranscribeFn Fn() {
    return [this](const std::vector<float> &, const moss::MossOptions &,
                  const std::atomic<bool> &cancel) {
      starts_.fetch_add(1);
      const auto deadline = std::chrono::steady_clock::now() + kDecodeTime;
      while (std::chrono::steady_clock::now() < deadline) {
        if (cancel.load()) {
          cancels_.fetch_add(1);
          throw omni_common::GenerationCancelled();
        } else {
        }
        std::this_thread::sleep_for(1ms);
      }
      omni_common::TranscriptionResult result;
      result.text = "text";
      return result;
    };
  }

  bool WaitForStarts(int count, std::chrono::milliseconds timeout) const {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (std::chrono::steady_clock::now() < deadline) {
      if (starts_.load() >= count) {
        return true;
      } else {
      }
      std::this_thread::sleep_for(1ms);
    }
    return false;
  }

  std::atomic<int> starts_{0};
  std::atomic<int> cancels_{0};
};

// One gate below the gate stays silent; the gate itself starts a preview.
void TestFirstPreviewGate() {
  omni_server::TranscriptionWorker worker([] {}, [] {});
  FakeTranscriber fake;
  Events events;
  const auto session = std::make_shared<moss::MossRealtimeSession>(
      worker, fake.Fn(), events.Sender());
  session->Handle(AppendMessage(kGateSamples - 1));
  std::this_thread::sleep_for(100ms);
  Require(fake.starts_.load() == 0,
          "one sample below the gate must not start a decode");
  Require(events.Segments(false) == 0,
          "one sample below the gate must not publish a preview");
  session->Handle(AppendMessage(1));
  Require(events.WaitForSegments(false, 1, 2s),
          "reaching the gate must publish a preview");
  Require(!session->Handle({{"type", "transcription.done"}}),
          "done ends the session");
  Require(events.Segments(true) == 1, "done must flush one final window");
  std::cout << "TestFirstPreviewGate ok\n";
}

// A stop 100 ms into an in-flight preview: the final is not postponed by the
// preview it replaces, and the preview is never published.
void TestShortStopPreemptsPreview() {
  omni_server::TranscriptionWorker worker([] {}, [] {});
  FakeTranscriber fake;
  Events events;
  const auto session = std::make_shared<moss::MossRealtimeSession>(
      worker, fake.Fn(), events.Sender());
  session->Handle(AppendMessage(kGateSamples));
  Require(fake.WaitForStarts(1, 2s), "the gate preview must start");
  std::this_thread::sleep_for(100ms);
  const auto stop = std::chrono::steady_clock::now();
  session->Handle({{"type", "transcription.done"}});
  const auto stop_to_final = std::chrono::steady_clock::now() - stop;
  Require(events.Segments(false) == 0,
          "a preview cancelled by the stop must not be published");
  Require(fake.cancels_.load() >= 1,
          "the stop must cancel the preview in flight");
  Require(events.Segments(true) == 1, "the stop must flush one final window");
  Require(stop_to_final < kPreemptedStopToFinalLimit,
          "stop-to-final must not wait out the preview it replaces");
  std::cout << "TestShortStopPreemptsPreview ok\n";
}

// A preview in flight as the 4 s window fills, then a stop: the window's
// final is not postponed by the extra preview.
void TestFullWindowPreemptsPreview() {
  omni_server::TranscriptionWorker worker([] {}, [] {});
  FakeTranscriber fake;
  Events events;
  const auto session = std::make_shared<moss::MossRealtimeSession>(
      worker, fake.Fn(), events.Sender());
  session->Handle(AppendMessage(kGateSamples));
  session->Handle(AppendMessage(kWindowSamples - kGateSamples));
  const auto stop = std::chrono::steady_clock::now();
  session->Handle({{"type", "transcription.done"}});
  const auto stop_to_final = std::chrono::steady_clock::now() - stop;
  Require(events.Segments(false) == 0,
          "a preview cancelled by the full window must not be published");
  Require(events.Segments(true) == 1,
          "a full window must flush as one final window");
  Require(stop_to_final < kPreemptedStopToFinalLimit,
          "stop-to-final must not wait out the preview the window replaces");
  std::cout << "TestFullWindowPreemptsPreview ok\n";
}

} // namespace

int main() {
  try {
    TestFirstPreviewGate();
    TestShortStopPreemptsPreview();
    TestFullWindowPreemptsPreview();
  } catch (const std::exception &error) {
    std::cerr << "moss_realtime_test failed: " << error.what() << "\n";
    return 1;
  }
  std::cout << "moss_realtime_test passed\n";
  return 0;
}
