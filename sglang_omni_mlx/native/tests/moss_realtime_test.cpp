// SPDX-License-Identifier: Apache-2.0
// MossRealtimeSession scheduling without a checkpoint: the 0.75 s
// first-preview gate; a stop or a full window preempting the preview in
// flight; a full window's final scheduled by the cancelled preview's
// completion, without a stop; and each decode's timing line. Every scenario
// runs on a virtual clock the test advances by hand, with barriers on
// transcriber starts, cancels and worker drains instead of timed waits.
#include "moss/moss_realtime.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <iostream>
#include <mutex>
#include <optional>
#include <sstream>
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
// Stop-to-final on the virtual clock: one 400 ms final plus up to 50 ms of
// cancel scheduling; waiting the preview out costs 300-580 ms more.
constexpr auto kStopToFinalLimit = kDecodeTime + 50ms;
constexpr int kGateSamples = omni_common::kSampleRate * 3 / 4;
constexpr int kWindowSamples = 4 * omni_common::kSampleRate;
constexpr std::chrono::steady_clock::time_point kT0{};

void Require(bool condition, std::string_view message) {
  if (!condition) {
    throw std::runtime_error(std::string(message));
  } else {
  }
}

// Waits for an event, not a duration; the wall timeout only catches a hang.
template <typename Cond> void WaitFor(Cond &&condition, std::string_view what) {
  const auto deadline = std::chrono::steady_clock::now() + 10s;
  while (!condition()) {
    if (std::chrono::steady_clock::now() > deadline) {
      throw std::runtime_error(std::string("timed out waiting for ") +
                               std::string(what));
    } else {
    }
    std::this_thread::sleep_for(50us);
  }
}

// Waits for an event that a broken build may never produce; reports whether
// it came so the scenario can still be driven to completion and every check
// can run in one run. The wall timeout only bounds a lost race.
template <typename Cond> bool TryWaitFor(Cond &&condition) {
  const auto deadline = std::chrono::steady_clock::now() + 2s;
  while (!condition()) {
    if (std::chrono::steady_clock::now() > deadline) {
      return false;
    } else {
    }
    std::this_thread::sleep_for(50us);
  }
  return true;
}

// Collects failed checks so one run reports every layer that broke, then
// throws them all.
class Checks {
public:
  void operator()(bool condition, std::string_view message) {
    if (!condition) {
      failures_.push_back(std::string(message));
    } else {
    }
  }

  void ThrowIfAny() const {
    if (failures_.empty()) {
      return;
    } else {
    }
    std::string all;
    for (const std::string &failure : failures_) {
      all += (all.empty() ? "" : "; ") + failure;
    }
    throw std::runtime_error(all);
  }

private:
  std::vector<std::string> failures_;
};

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

  std::string FinalText() const {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const nlohmann::json &event : events_) {
      if (event.value("type", "") == "transcription.segment" &&
          event.value("is_final", false)) {
        return event.value("text", "");
      } else {
      }
    }
    return "";
  }

private:
  mutable std::mutex mutex_;
  std::vector<nlohmann::json> events_;
};

// A clock the test advances by hand; a decode waits on it and polls its
// cancel flag, so a cancel lands at exactly the virtual time it was set.
class VirtualClock {
public:
  std::chrono::steady_clock::time_point Now() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return now_;
  }

  void Advance(std::chrono::milliseconds delta) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      now_ += delta;
    }
    cv_.notify_all();
  }

  void WaitUntil(std::chrono::steady_clock::time_point deadline,
                 const std::atomic<bool> &cancel) {
    std::unique_lock<std::mutex> lock(mutex_);
    while (now_ < deadline && !cancel.load()) {
      cv_.wait_for(lock, 200us);
    }
  }

private:
  mutable std::mutex mutex_;
  std::condition_variable cv_;
  std::chrono::steady_clock::time_point now_{};
};

// The 400 ms decode on the virtual clock, in 50 ms steps so a cancel lands
// mid-decode exactly when the scenario sets it. LastEnd is the virtual
// completion time of the latest finished decode.
class ScriptedTranscriber {
public:
  // return_on_cancel plays a transcriber that passes its last cancellation
  // check and returns a normal result after the cancel lands.
  explicit ScriptedTranscriber(VirtualClock &clock,
                               bool return_on_cancel = false)
      : clock_(clock), return_on_cancel_(return_on_cancel) {}

  moss::MossRealtimeSession::TranscribeFn Fn() {
    return [this](const std::vector<float> &, const moss::MossOptions &,
                  const std::atomic<bool> &cancel) {
      starts_.fetch_add(1);
      const auto start = clock_.Now();
      for (int step = 1; step <= 8; ++step) {
        clock_.WaitUntil(start + step * kDecodeTime / 8, cancel);
        if (cancel.load()) {
          cancels_.fetch_add(1);
          if (return_on_cancel_) {
            omni_common::TranscriptionResult result;
            result.text = "stale";
            return result;
          } else {
          }
          throw omni_common::GenerationCancelled();
        } else {
        }
      }
      ends_.store(clock_.Now());
      omni_common::TranscriptionResult result;
      result.text = "text";
      return result;
    };
  }

  std::chrono::steady_clock::time_point LastEnd() const { return ends_.load(); }

  std::atomic<int> starts_{0};
  std::atomic<int> cancels_{0};

private:
  VirtualClock &clock_;
  bool return_on_cancel_;
  std::atomic<std::chrono::steady_clock::time_point> ends_{};
};

// Advances the clock until the scenario thread sets done; bounded only as a
// hang detector.
void DriveToDone(VirtualClock &clock, const std::atomic<bool> &done) {
  for (int step = 0; step < 1000 && !done.load(); ++step) {
    clock.Advance(10ms);
    std::this_thread::sleep_for(50us);
  }
  Require(done.load(), "the scenario must complete within the virtual drive");
}

// Runs a no-op through the worker; the worker runs jobs in queue order and
// a job's session callback finishes before the next job starts, so the
// no-op completes only after every earlier decode's callback. RequestStates
// cannot be this barrier: the worker clears its running count before
// invoking the callback.
void WaitForDrain(omni_server::TranscriptionWorker &worker) {
  worker.Transcribe(
      [](const std::atomic<bool> &) {
        return omni_common::TranscriptionResult{};
      },
      omni_server::NewCancelFlag());
}

std::string Ms(std::chrono::steady_clock::duration duration) {
  return std::to_string(
             std::chrono::duration_cast<std::chrono::milliseconds>(duration)
                 .count()) +
         " ms";
}

// One gate below the gate stays silent; the gate itself starts a preview.
void TestFirstPreviewGate() {
  VirtualClock clock;
  omni_server::TranscriptionWorker worker([] {}, [] {});
  ScriptedTranscriber fake(clock);
  Events events;
  const auto session = std::make_shared<moss::MossRealtimeSession>(
      worker, fake.Fn(), events.Sender(), [&clock] { return clock.Now(); });
  session->Handle(AppendMessage(kGateSamples - 1));
  Require(fake.starts_.load() == 0,
          "one sample below the gate must not start a decode");
  Require(events.Segments(false) == 0,
          "one sample below the gate must not publish a preview");
  session->Handle(AppendMessage(1));
  WaitFor([&] { return fake.starts_.load() == 1; }, "the gate preview");
  clock.Advance(kDecodeTime);
  WaitForDrain(worker);
  Require(events.Segments(false) == 1,
          "reaching the gate must publish a preview");
  std::atomic<bool> done{false};
  std::thread stop_thread([&] {
    session->Handle({{"type", "transcription.done"}});
    done.store(true);
  });
  DriveToDone(clock, done);
  stop_thread.join();
  Require(events.Segments(true) == 1, "done must flush one final window");
  std::cout << "TestFirstPreviewGate ok\n";
}

// A stop 100 ms into an in-flight preview: the final is not postponed by the
// preview it replaces, and the preview is never published.
void TestShortStopPreemptsPreview() {
  VirtualClock clock;
  omni_server::TranscriptionWorker worker([] {}, [] {});
  ScriptedTranscriber fake(clock);
  Events events;
  const auto session = std::make_shared<moss::MossRealtimeSession>(
      worker, fake.Fn(), events.Sender(), [&clock] { return clock.Now(); });
  session->Handle(AppendMessage(kGateSamples));
  WaitFor([&] { return fake.starts_.load() == 1; }, "the gate preview");
  clock.Advance(100ms);
  std::atomic<bool> done{false};
  std::thread stop_thread([&] {
    session->Handle({{"type", "transcription.done"}});
    done.store(true);
  });
  const bool cancelled = TryWaitFor([&] { return fake.cancels_.load() == 1; });
  TryWaitFor([&] { return fake.starts_.load() == 2; });
  DriveToDone(clock, done);
  stop_thread.join();
  const auto stop_to_final = fake.LastEnd() - (kT0 + 100ms);
  Checks checks;
  checks(cancelled, "the stop must cancel the preview in flight");
  checks(fake.starts_.load() == 2, "the stop must decode the final once");
  checks(events.Segments(false) == 0,
         "a preview cancelled by the stop must not be published");
  checks(events.Segments(true) == 1, "the stop must flush one final window");
  checks(stop_to_final <= kStopToFinalLimit,
         "stop-to-final must not wait out the preview it replaces");
  checks.ThrowIfAny();
  std::cout << "TestShortStopPreemptsPreview ok (stop-to-final "
            << Ms(stop_to_final) << ")\n";
}

// A preview in flight as the 4 s window fills, and no further event at all:
// the cancelled preview's completion must schedule the window's final itself,
// not leave it to the next append or stop.
void TestFullWindowSchedulesFinal() {
  VirtualClock clock;
  omni_server::TranscriptionWorker worker([] {}, [] {});
  ScriptedTranscriber fake(clock);
  Events events;
  const auto session = std::make_shared<moss::MossRealtimeSession>(
      worker, fake.Fn(), events.Sender(), [&clock] { return clock.Now(); });
  session->Handle(AppendMessage(kGateSamples));
  WaitFor([&] { return fake.starts_.load() == 1; }, "the gate preview");
  session->Handle(AppendMessage(kWindowSamples - kGateSamples));
  const bool cancelled = TryWaitFor([&] { return fake.cancels_.load() == 1; });
  TryWaitFor([&] { return fake.starts_.load() == 2; });
  for (int step = 0; step < 200 && events.Segments(true) == 0; ++step) {
    clock.Advance(10ms);
    std::this_thread::sleep_for(50us);
  }
  Checks checks;
  checks(cancelled, "the full window must cancel the preview in flight");
  checks(events.Segments(true) == 1,
         "the full window must schedule its final after cancelling the "
         "preview");
  checks(events.Segments(false) == 0,
         "a preview cancelled by the full window must not be published");
  checks.ThrowIfAny();
  std::cout << "TestFullWindowSchedulesFinal ok\n";
}

// A preview in flight as the 4 s window fills, then a stop: the window's
// final is not postponed by the extra preview.
void TestFullWindowPreemptsPreview() {
  VirtualClock clock;
  omni_server::TranscriptionWorker worker([] {}, [] {});
  ScriptedTranscriber fake(clock);
  Events events;
  const auto session = std::make_shared<moss::MossRealtimeSession>(
      worker, fake.Fn(), events.Sender(), [&clock] { return clock.Now(); });
  session->Handle(AppendMessage(kGateSamples));
  WaitFor([&] { return fake.starts_.load() == 1; }, "the gate preview");
  session->Handle(AppendMessage(kWindowSamples - kGateSamples));
  const bool cancelled = TryWaitFor([&] { return fake.cancels_.load() == 1; });
  std::atomic<bool> done{false};
  std::thread stop_thread([&] {
    session->Handle({{"type", "transcription.done"}});
    done.store(true);
  });
  TryWaitFor([&] { return fake.starts_.load() == 2; });
  DriveToDone(clock, done);
  stop_thread.join();
  const auto stop_to_final = fake.LastEnd() - kT0;
  Checks checks;
  checks(cancelled, "the full window must cancel the preview in flight");
  checks(fake.starts_.load() == 2, "the stop must decode the final once");
  checks(events.Segments(false) == 0,
         "a preview cancelled by the full window must not be published");
  checks(events.Segments(true) == 1,
         "a full window must flush as one final window");
  checks(stop_to_final <= kStopToFinalLimit,
         "stop-to-final must not wait out the preview the window replaces");
  checks.ThrowIfAny();
  std::cout << "TestFullWindowPreemptsPreview ok (stop-to-final "
            << Ms(stop_to_final) << ")\n";
}

// A stop 100 ms into an in-flight preview whose transcriber returns a
// normal result after the cancel lands (the real transcriber can return
// after its last cancellation check): the stale result must be dropped, not
// published.
void TestShortStopDropsCancelledPreviewResult() {
  VirtualClock clock;
  omni_server::TranscriptionWorker worker([] {}, [] {});
  ScriptedTranscriber fake(clock, /*return_on_cancel=*/true);
  Events events;
  const auto session = std::make_shared<moss::MossRealtimeSession>(
      worker, fake.Fn(), events.Sender(), [&clock] { return clock.Now(); });
  session->Handle(AppendMessage(kGateSamples));
  WaitFor([&] { return fake.starts_.load() == 1; }, "the gate preview");
  clock.Advance(100ms);
  std::atomic<bool> done{false};
  std::thread stop_thread([&] {
    session->Handle({{"type", "transcription.done"}});
    done.store(true);
  });
  const bool cancelled = TryWaitFor([&] { return fake.cancels_.load() == 1; });
  TryWaitFor([&] { return fake.starts_.load() == 2; });
  DriveToDone(clock, done);
  stop_thread.join();
  const auto stop_to_final = fake.LastEnd() - (kT0 + 100ms);
  Checks checks;
  checks(cancelled, "the stop must cancel the preview in flight");
  checks(events.Segments(false) == 0,
         "a cancelled preview's successful result must be dropped, not "
         "published");
  checks(events.Segments(true) == 1, "the stop must flush one final window");
  checks(events.FinalText() == "text",
         "the final must publish its own result, not the stale preview");
  checks(stop_to_final <= kStopToFinalLimit,
         "stop-to-final must not wait out the preview it replaces");
  checks.ThrowIfAny();
  std::cout << "TestShortStopDropsCancelledPreviewResult ok\n";
}

// A preview in flight as the 4 s window fills whose transcriber returns a
// normal result after the cancel lands: the stale result must be dropped,
// and the completion still schedules the window's final.
void TestFullWindowDropsCancelledPreviewResult() {
  VirtualClock clock;
  omni_server::TranscriptionWorker worker([] {}, [] {});
  ScriptedTranscriber fake(clock, /*return_on_cancel=*/true);
  Events events;
  const auto session = std::make_shared<moss::MossRealtimeSession>(
      worker, fake.Fn(), events.Sender(), [&clock] { return clock.Now(); });
  session->Handle(AppendMessage(kGateSamples));
  WaitFor([&] { return fake.starts_.load() == 1; }, "the gate preview");
  session->Handle(AppendMessage(kWindowSamples - kGateSamples));
  const bool cancelled = TryWaitFor([&] { return fake.cancels_.load() == 1; });
  TryWaitFor([&] { return fake.starts_.load() == 2; });
  for (int step = 0; step < 200 && events.Segments(true) == 0; ++step) {
    clock.Advance(10ms);
    std::this_thread::sleep_for(50us);
  }
  Checks checks;
  checks(cancelled, "the full window must cancel the preview in flight");
  checks(events.Segments(false) == 0,
         "a cancelled preview's successful result must be dropped, not "
         "published");
  checks(events.Segments(true) == 1,
         "the full window must schedule its final after cancelling the "
         "preview");
  checks(events.FinalText() == "text",
         "the final must publish its own result, not the stale preview");
  checks.ThrowIfAny();
  std::cout << "TestFullWindowDropsCancelledPreviewResult ok\n";
}

// Routes std::cerr into a buffer for its lifetime.
class CerrCapture {
public:
  CerrCapture() : original_(std::cerr.rdbuf(buffer_.rdbuf())) {}
  ~CerrCapture() { std::cerr.rdbuf(original_); }
  std::vector<nlohmann::json> Lines() const {
    std::vector<nlohmann::json> lines;
    std::istringstream stream(buffer_.str());
    std::string line;
    while (std::getline(stream, line)) {
      lines.push_back(nlohmann::json::parse(line));
    }
    return lines;
  }

private:
  std::ostringstream buffer_;
  std::streambuf *original_;
};

// A preview, then the stop's final: one timing line each, stamps in stage
// order, the gate Voxt's latency harness reads, and no transcript.
void TestTimingLines() {
  VirtualClock clock;
  omni_server::TranscriptionWorker worker([] {}, [] {});
  ScriptedTranscriber fake(clock);
  Events events;
  const auto session = std::make_shared<moss::MossRealtimeSession>(
      worker, fake.Fn(), events.Sender(), [&clock] { return clock.Now(); });
  std::vector<nlohmann::json> lines;
  {
    const CerrCapture capture;
    session->Handle(AppendMessage(kGateSamples));
    WaitFor([&] { return fake.starts_.load() == 1; }, "the gate preview");
    clock.Advance(kDecodeTime);
    WaitForDrain(worker);
    std::atomic<bool> done{false};
    std::thread stop_thread([&] {
      session->Handle({{"type", "transcription.done"}});
      done.store(true);
    });
    DriveToDone(clock, done);
    stop_thread.join();
    lines = capture.Lines();
  }
  Require(lines.size() == 2, "one timing line per decode");
  const char *const stamps[] = {"append_at_ms",     "decision_at_ms",
                                "submit_at_ms",     "transcribe_started_at_ms",
                                "transcribe_at_ms", "publish_at_ms"};
  for (size_t i = 0; i < lines.size(); ++i) {
    const nlohmann::json &line = lines[i];
    Require(line.at("event") == "moss_preview_timing", "the event name");
    Require(line.at("kind") == (i == 0 ? "preview" : "final"),
            "the preview's line, then the stop's final's");
    Require(line.at("window_id") == 0, "both decode window 0");
    Require(line.at("audio_sample_count") == kGateSamples,
            "both decode the pending audio");
    Require(line.at("min_preview_samples") == kGateSamples,
            "the line carries the first-preview gate");
    Require(!line.contains("text") && !line.contains("segments"),
            "a timing line carries no transcript");
    for (size_t stage = 1; stage < std::size(stamps); ++stage) {
      Require(line.at(stamps[stage - 1]).get<int64_t>() <=
                  line.at(stamps[stage]).get<int64_t>(),
              std::string(stamps[stage]) + " must not precede " +
                  stamps[stage - 1]);
    }
  }
  std::cout << "TestTimingLines ok\n";
}

} // namespace

int main() {
  try {
    TestFirstPreviewGate();
    TestShortStopPreemptsPreview();
    TestFullWindowSchedulesFinal();
    TestFullWindowPreemptsPreview();
    TestShortStopDropsCancelledPreviewResult();
    TestFullWindowDropsCancelledPreviewResult();
    TestTimingLines();
  } catch (const std::exception &error) {
    std::cerr << "moss_realtime_test failed: " << error.what() << "\n";
    return 1;
  }
  std::cout << "moss_realtime_test passed\n";
  return 0;
}
