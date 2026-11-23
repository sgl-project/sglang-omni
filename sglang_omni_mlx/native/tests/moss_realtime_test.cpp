// SPDX-License-Identifier: Apache-2.0
// MossRealtimeSession scheduling without a checkpoint: the 0.75 s
// first-preview gate; a stop or a full window preempting the preview in
// flight; a full window's final scheduled by the cancelled preview's
// completion, without a stop; and the verification comment's published
// short-stop and 4 s window reproductions as old-gate A/Bs. Every scenario
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
// The old-gate A/B no-later allowance; the published regressions were
// 190-297 ms.
constexpr auto kAbVirtualSlack = 50ms;
constexpr int kGateSamples = omni_common::kSampleRate * 3 / 4;
constexpr int kOldGateSamples = omni_common::kSampleRate * 5 / 4;
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

// The pre-PR scheduling, frozen as the old-gate control arm: the 1.25 s
// first-preview gate and no preemption, so a stop waits out the preview in
// flight. Only what the A/B scenarios exercise: a full window's final on
// append (before the cadence, as the pre-PR NextDecode ordered it), the
// cadence, and a blocking stop. The scenarios wait every decode out, so
// callbacks never outlive the session.
class OldGateSession {
public:
  OldGateSession(omni_server::TranscriptionWorker &worker,
                 moss::MossRealtimeSession::TranscribeFn transcribe,
                 moss::MossRealtimeSession::ClockFn clock)
      : worker_(worker), transcribe_(std::move(transcribe)),
        clock_(std::move(clock)) {}

  int previews() const { return previews_.load(); }
  int finals() const { return finals_.load(); }
  const std::string &final_text() const { return final_text_; }

  void Append(int sample_count) {
    bool final = false;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      pending_ += sample_count;
      const auto now = clock_();
      if (decoding_ || pending_ < kOldGateSamples) {
        return;
      } else {
      }
      final = pending_ >= kWindowSamples;
      if (!final && last_decode_.has_value() && now - *last_decode_ < 1s) {
        return;
      } else {
      }
      decoding_ = true;
      last_decode_ = now;
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
        [this, final](std::optional<omni_common::TranscriptionResult> result,
                      std::exception_ptr) {
          if (final) {
            finals_.fetch_add(1);
            final_text_ = result->text;
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
      final_text_ =
          worker_
              .Transcribe(
                  [this](const std::atomic<bool> &cancel) {
                    return transcribe_({}, moss::MossOptions{}, cancel);
                  },
                  cancel_)
              .text;
      finals_.fetch_add(1);
    } else {
    }
  }

private:
  omni_server::TranscriptionWorker &worker_;
  moss::MossRealtimeSession::TranscribeFn transcribe_;
  moss::MossRealtimeSession::ClockFn clock_;
  omni_server::CancelFlag cancel_ = omni_server::NewCancelFlag();
  std::mutex mutex_;
  std::condition_variable done_cv_;
  int pending_ = 0;
  bool decoding_ = false;
  std::optional<std::chrono::steady_clock::time_point> last_decode_;
  std::atomic<int> previews_{0};
  std::atomic<int> finals_{0};
  std::string final_text_;
};

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

// The published short-stop reproduction as a committed regression
// (verification evidence on the PR): append 12000 samples, done at 100 ms,
// controlled 400 ms decode. The old gate decodes only the final; the broken
// new gate decoded preview+final and ended 293-297 ms later. Both arms run
// on virtual clocks with explicit barriers, so the comparison is
// deterministic.
void TestShortStopNotLaterThanOldGate() {
  VirtualClock clock;
  omni_server::TranscriptionWorker new_worker([] {}, [] {});
  ScriptedTranscriber new_fake(clock);
  Events new_events;
  const auto new_session = std::make_shared<moss::MossRealtimeSession>(
      new_worker, new_fake.Fn(), new_events.Sender(),
      [&clock] { return clock.Now(); });
  new_session->Handle(AppendMessage(12000));
  WaitFor([&] { return new_fake.starts_.load() == 1; }, "the gate preview");
  clock.Advance(100ms);
  std::atomic<bool> done{false};
  std::thread stop_thread([&] {
    new_session->Handle({{"type", "transcription.done"}});
    done.store(true);
  });
  const bool cancelled =
      TryWaitFor([&] { return new_fake.cancels_.load() == 1; });
  TryWaitFor([&] { return new_fake.starts_.load() == 2; });
  DriveToDone(clock, done);
  stop_thread.join();
  const auto new_stop_to_final = new_fake.LastEnd() - (kT0 + 100ms);

  VirtualClock old_clock;
  omni_server::TranscriptionWorker old_worker([] {}, [] {});
  ScriptedTranscriber old_fake(old_clock);
  OldGateSession old_session(old_worker, old_fake.Fn(),
                             [&old_clock] { return old_clock.Now(); });
  old_session.Append(12000);
  old_clock.Advance(100ms);
  std::atomic<bool> old_done{false};
  std::thread old_stop_thread([&] {
    old_session.Done();
    old_done.store(true);
  });
  WaitFor([&] { return old_fake.starts_.load() == 1; }, "the old final");
  DriveToDone(old_clock, old_done);
  old_stop_thread.join();
  const auto old_stop_to_final = old_fake.LastEnd() - (kT0 + 100ms);

  Checks checks;
  checks(cancelled, "the short stop must cancel the gate preview");
  checks(new_fake.starts_.load() == 2,
         "the short stop must decode preview work and the final once each");
  checks(new_events.Segments(false) == 0,
         "a preview cancelled by the stop must not be published");
  checks(old_fake.starts_.load() == 1 && old_session.previews() == 0,
         "the old gate must not preview 0.75 s of pending audio");
  checks(new_events.Segments(true) == 1 && old_session.finals() == 1,
         "both arms must end with one final");
  checks(new_events.FinalText() == old_session.final_text(),
         "both arms must end with the same final text");
  checks(new_stop_to_final <= old_stop_to_final + kAbVirtualSlack,
         "short stop: the new gate must not end the final later than the "
         "old gate");
  checks.ThrowIfAny();
  std::cout << "TestShortStopNotLaterThanOldGate ok (new "
            << Ms(new_stop_to_final) << ", old " << Ms(old_stop_to_final)
            << ")\n";
}

// The published 4 s window reproduction as a committed regression
// (verification evidence on the PR): identical 64000 samples per arm with
// each arm's appends on its own preview cadence, then done. New-gate appends
// at 0/1010/2020/3030/3250 ms (12000/16000/16000/16000/4000), old-gate at
// 0/500/1510/2520/3250 ms (12000/8000/16000/16000/12000). The broken new
// gate decoded 4 previews and ended 190-193 ms later; the old gate decoded
// 3. Both arms run on virtual clocks with explicit barriers.
void TestFullWindowNotLaterThanOldGate() {
  VirtualClock clock;
  omni_server::TranscriptionWorker new_worker([] {}, [] {});
  ScriptedTranscriber new_fake(clock);
  Events new_events;
  const auto new_session = std::make_shared<moss::MossRealtimeSession>(
      new_worker, new_fake.Fn(), new_events.Sender(),
      [&clock] { return clock.Now(); });
  new_session->Handle(AppendMessage(12000));
  WaitFor([&] { return new_fake.starts_.load() == 1; }, "preview 1");
  clock.Advance(1010ms);
  WaitForDrain(new_worker);
  new_session->Handle(AppendMessage(16000));
  WaitFor([&] { return new_fake.starts_.load() == 2; }, "preview 2");
  clock.Advance(1010ms);
  WaitForDrain(new_worker);
  new_session->Handle(AppendMessage(16000));
  WaitFor([&] { return new_fake.starts_.load() == 3; }, "preview 3");
  clock.Advance(1010ms);
  WaitForDrain(new_worker);
  new_session->Handle(AppendMessage(16000));
  WaitFor([&] { return new_fake.starts_.load() == 4; }, "preview 4");
  clock.Advance(220ms);
  new_session->Handle(AppendMessage(4000));
  const bool cancelled =
      TryWaitFor([&] { return new_fake.cancels_.load() == 1; });
  std::atomic<bool> done{false};
  std::thread stop_thread([&] {
    new_session->Handle({{"type", "transcription.done"}});
    done.store(true);
  });
  TryWaitFor([&] { return new_fake.starts_.load() == 5; });
  DriveToDone(clock, done);
  stop_thread.join();
  const auto new_stop_to_final = new_fake.LastEnd() - (kT0 + 3250ms);

  VirtualClock old_clock;
  omni_server::TranscriptionWorker old_worker([] {}, [] {});
  ScriptedTranscriber old_fake(old_clock);
  OldGateSession old_session(old_worker, old_fake.Fn(),
                             [&old_clock] { return old_clock.Now(); });
  old_session.Append(12000);
  old_clock.Advance(500ms);
  old_session.Append(8000);
  WaitFor([&] { return old_fake.starts_.load() == 1; }, "old preview 1");
  old_clock.Advance(1010ms);
  WaitForDrain(old_worker);
  old_session.Append(16000);
  WaitFor([&] { return old_fake.starts_.load() == 2; }, "old preview 2");
  old_clock.Advance(1010ms);
  WaitForDrain(old_worker);
  old_session.Append(16000);
  WaitFor([&] { return old_fake.starts_.load() == 3; }, "old preview 3");
  old_clock.Advance(730ms);
  WaitForDrain(old_worker);
  old_session.Append(12000);
  WaitFor([&] { return old_fake.starts_.load() == 4; },
          "the old arm's window final, taken before the cadence");
  std::atomic<bool> old_done{false};
  std::thread old_stop_thread([&] {
    old_session.Done();
    old_done.store(true);
  });
  DriveToDone(old_clock, old_done);
  old_stop_thread.join();
  const auto old_stop_to_final = old_fake.LastEnd() - (kT0 + 3250ms);

  Checks checks;
  checks(cancelled, "the filled window must cancel the fourth preview");
  checks(new_fake.starts_.load() == 5,
         "the new gate decodes four previews plus the final, one preview "
         "more than the old gate, as disclosed");
  checks(new_events.Segments(false) == 3,
         "the cancelled fourth preview must not be published");
  checks(old_session.previews() == 3,
         "the old gate must decode three previews over the window");
  checks(new_events.Segments(true) == 1 && old_session.finals() == 1,
         "both arms must end with one final");
  checks(new_events.FinalText() == old_session.final_text(),
         "both arms must end with the same final text");
  checks(new_stop_to_final <= old_stop_to_final + kAbVirtualSlack,
         "full window: the new gate must not end the final later than the "
         "old gate");
  checks.ThrowIfAny();
  std::cout << "TestFullWindowNotLaterThanOldGate ok (new "
            << Ms(new_stop_to_final) << ", old " << Ms(old_stop_to_final)
            << ")\n";
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

} // namespace

int main() {
  try {
    TestFirstPreviewGate();
    TestShortStopPreemptsPreview();
    TestFullWindowSchedulesFinal();
    TestFullWindowPreemptsPreview();
    TestShortStopDropsCancelledPreviewResult();
    TestFullWindowDropsCancelledPreviewResult();
    TestShortStopNotLaterThanOldGate();
    TestFullWindowNotLaterThanOldGate();
  } catch (const std::exception &error) {
    std::cerr << "moss_realtime_test failed: " << error.what() << "\n";
    return 1;
  }
  std::cout << "moss_realtime_test passed\n";
  return 0;
}
