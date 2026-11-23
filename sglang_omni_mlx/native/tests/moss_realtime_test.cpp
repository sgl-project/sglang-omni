// SPDX-License-Identifier: Apache-2.0
// MossRealtimeSession scheduling without a checkpoint: the 0.75 s
// first-preview gate; a stop or a full window preempting the preview in
// flight so stop-to-final is not postponed; a full window's final scheduled
// by the cancelled preview's completion, without a stop; and identical-input
// short-stop and full-window A/Bs against a frozen old-gate control arm. The
// transcriber is a scripted 400 ms decode that ends early once its cancel
// flag is set.
#include "moss/moss_realtime.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <optional>
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
// The new gate may end a final this much later than the old gate at most;
// waiting out an in-flight preview costs 300-400 ms in these scenarios.
constexpr auto kAbSlack = 250ms;
constexpr int kGateSamples = omni_common::kSampleRate * 3 / 4;
constexpr int kOldGateSamples = omni_common::kSampleRate * 5 / 4;
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

// The pre-PR scheduling, frozen as the old-gate control arm: the 1.25 s
// first-preview gate and no preemption, so a stop waits out the preview in
// flight. Only what the A/B scenarios exercise: the cadence, a full window's
// final on append, and a blocking stop. The scenarios wait every decode out,
// so callbacks never outlive the session.
class OldGateSession {
public:
  OldGateSession(omni_server::TranscriptionWorker &worker,
                 moss::MossRealtimeSession::TranscribeFn transcribe)
      : worker_(worker), transcribe_(std::move(transcribe)) {}

  int previews() const { return previews_.load(); }
  int finals() const { return finals_.load(); }

  void Append(int sample_count) {
    bool final = false;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      pending_ += sample_count;
      const auto now = std::chrono::steady_clock::now();
      if (decoding_ || pending_ < kOldGateSamples ||
          (last_decode_.has_value() && now - *last_decode_ < 1s)) {
        return;
      } else {
      }
      decoding_ = true;
      last_decode_ = now;
      final = pending_ >= kWindowSamples;
      if (final) {
        pending_ -= kWindowSamples;
      } else {
        previews_.fetch_add(1);
      }
    }
    worker_.Submit(
        [this](const std::atomic<bool> &cancel) {
          return transcribe_({}, moss::MossOptions{}, cancel);
        },
        cancel_,
        [this, final](std::optional<omni_common::TranscriptionResult>,
                      std::exception_ptr) {
          if (final) {
            finals_.fetch_add(1);
          } else {
          }
          {
            std::lock_guard<std::mutex> lock(mutex_);
            decoding_ = false;
          }
          done_cv_.notify_all();
        });
  }

  // The old Flush: wait out the decode in flight, then decode what is left.
  void Done() {
    bool flush = false;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      done_cv_.wait(lock, [&] { return !decoding_; });
      flush = pending_ > 0;
      pending_ = 0;
    }
    if (flush) {
      worker_.Transcribe(
          [this](const std::atomic<bool> &cancel) {
            return transcribe_({}, moss::MossOptions{}, cancel);
          },
          cancel_);
      finals_.fetch_add(1);
    } else {
    }
  }

private:
  omni_server::TranscriptionWorker &worker_;
  moss::MossRealtimeSession::TranscribeFn transcribe_;
  omni_server::CancelFlag cancel_ = omni_server::NewCancelFlag();
  std::mutex mutex_;
  std::condition_variable done_cv_;
  int pending_ = 0;
  bool decoding_ = false;
  std::optional<std::chrono::steady_clock::time_point> last_decode_;
  std::atomic<int> previews_{0};
  std::atomic<int> finals_{0};
};

std::string Ms(std::chrono::steady_clock::duration duration) {
  return std::to_string(
             std::chrono::duration_cast<std::chrono::milliseconds>(duration)
                 .count()) +
         " ms";
}

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

// A preview in flight as the 4 s window fills, and no further event at all:
// the cancelled preview's completion must schedule the window's final itself,
// not leave it to the next append or stop.
void TestFullWindowSchedulesFinal() {
  omni_server::TranscriptionWorker worker([] {}, [] {});
  FakeTranscriber fake;
  Events events;
  const auto session = std::make_shared<moss::MossRealtimeSession>(
      worker, fake.Fn(), events.Sender());
  session->Handle(AppendMessage(kGateSamples));
  Require(fake.WaitForStarts(1, 2s), "the gate preview must start");
  session->Handle(AppendMessage(kWindowSamples - kGateSamples));
  Require(events.WaitForSegments(true, 1, 2s),
          "the full window must schedule its final after cancelling the "
          "preview");
  Require(events.Segments(false) == 0,
          "a preview cancelled by the full window must not be published");
  Require(fake.cancels_.load() >= 1,
          "the full window must cancel the preview in flight");
  std::cout << "TestFullWindowSchedulesFinal ok\n";
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

// Identical-input short-stop A/B against the old gate: 0.75 s of audio, and
// the stop lands the moment the preview starts, the worst case for the new
// gate. The old gate never starts a preview here; the new gate must end its
// final no later.
void TestShortStopNotLaterThanOldGate() {
  omni_server::TranscriptionWorker new_worker([] {}, [] {});
  FakeTranscriber new_fake;
  Events new_events;
  const auto new_session = std::make_shared<moss::MossRealtimeSession>(
      new_worker, new_fake.Fn(), new_events.Sender());
  new_session->Handle(AppendMessage(kGateSamples));
  Require(new_fake.WaitForStarts(1, 2s), "the gate preview must start");
  const auto new_stop = std::chrono::steady_clock::now();
  new_session->Handle({{"type", "transcription.done"}});
  const auto new_stop_to_final = std::chrono::steady_clock::now() - new_stop;

  omni_server::TranscriptionWorker old_worker([] {}, [] {});
  FakeTranscriber old_fake;
  OldGateSession old_session(old_worker, old_fake.Fn());
  old_session.Append(kGateSamples);
  const auto old_stop = std::chrono::steady_clock::now();
  old_session.Done();
  const auto old_stop_to_final = std::chrono::steady_clock::now() - old_stop;

  Require(old_session.previews() == 0,
          "the old gate must not preview 0.75 s of pending audio");
  Require(new_events.Segments(true) == 1 && old_session.finals() == 1,
          "both arms must end with one final");
  Require(new_stop_to_final <= old_stop_to_final + kAbSlack,
          "short stop: the new gate must not end the final later than the "
          "old gate");
  std::cout << "TestShortStopNotLaterThanOldGate ok (new "
            << Ms(new_stop_to_final) << ", old " << Ms(old_stop_to_final)
            << ")\n";
}

// Identical-input full-window A/B against the old gate: a preview in flight
// as the 4 s window fills, then stop at once. The old gate never previewed
// this audio (its final started when the window filled); the new gate must
// end its final no later.
void TestFullWindowNotLaterThanOldGate() {
  omni_server::TranscriptionWorker new_worker([] {}, [] {});
  FakeTranscriber new_fake;
  Events new_events;
  const auto new_session = std::make_shared<moss::MossRealtimeSession>(
      new_worker, new_fake.Fn(), new_events.Sender());
  new_session->Handle(AppendMessage(kGateSamples));
  Require(new_fake.WaitForStarts(1, 2s), "the gate preview must start");
  new_session->Handle(AppendMessage(kWindowSamples - kGateSamples));
  const auto new_stop = std::chrono::steady_clock::now();
  new_session->Handle({{"type", "transcription.done"}});
  const auto new_stop_to_final = std::chrono::steady_clock::now() - new_stop;

  omni_server::TranscriptionWorker old_worker([] {}, [] {});
  FakeTranscriber old_fake;
  OldGateSession old_session(old_worker, old_fake.Fn());
  old_session.Append(kGateSamples);
  old_session.Append(kWindowSamples - kGateSamples);
  const auto old_stop = std::chrono::steady_clock::now();
  old_session.Done();
  const auto old_stop_to_final = std::chrono::steady_clock::now() - old_stop;

  Require(old_session.previews() == 0,
          "the old gate must not preview before the window fills");
  Require(new_events.Segments(true) == 1 && old_session.finals() == 1,
          "both arms must end with one final");
  Require(new_stop_to_final <= old_stop_to_final + kAbSlack,
          "full window: the new gate must not end the final later than the "
          "old gate");
  std::cout << "TestFullWindowNotLaterThanOldGate ok (new "
            << Ms(new_stop_to_final) << ", old " << Ms(old_stop_to_final)
            << ")\n";
}

} // namespace

int main() {
  try {
    TestFirstPreviewGate();
    TestShortStopPreemptsPreview();
    TestFullWindowSchedulesFinal();
    TestFullWindowPreemptsPreview();
    TestShortStopNotLaterThanOldGate();
    TestFullWindowNotLaterThanOldGate();
  } catch (const std::exception &error) {
    std::cerr << "moss_realtime_test failed: " << error.what() << "\n";
    return 1;
  }
  std::cout << "moss_realtime_test passed\n";
  return 0;
}
