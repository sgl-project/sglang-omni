// SPDX-License-Identifier: Apache-2.0
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <execinfo.h>
#include <functional>
#include <future>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <streambuf>
#include <string>
#include <tuple>
#include <vector>

#include "common/generation.h"
#include "common/http.h"
#include "common/worker.h"
#include "pause_refresh_test_support.h"
#include "qwen3_asr/realtime.h"

namespace qwen3_asr::tests {
struct RealtimeSessionTestAccess {
  static void SetBeforeRolloverFinalize(RealtimeSession &session,
                                        std::function<void()> callback) {
    session.before_rollover_finalize_for_test_ = std::move(callback);
  }
};
} // namespace qwen3_asr::tests

namespace {

using omni_common::FinishReason;
using omni_common::GenerationCancelled;
using omni_common::TranscriptionResult;
using omni_server::NewCancelFlag;
using omni_server::Transcription;
using omni_server::TranscriptionWorker;
using qwen3_asr::FinalReuse;
using qwen3_asr::PauseRefreshHint;
using qwen3_asr::RealtimeSession;
using qwen3_asr::RefreshReuse;
using qwen3_asr::tests::MakePauseRefreshSession;
using qwen3_asr::tests::PauseRefreshTestDecoder;
using qwen3_asr::tests::Require;
using qwen3_asr::tests::TestGate;
using qwen3_asr::tests::TestSignal;
using qwen3_asr::tests::WaitFor;

constexpr long kSampleRate = 16000;
constexpr long kMillisecondsPerSecond = 1000;
constexpr long kSpeechEndSample = 600 * kSampleRate / kMillisecondsPerSecond;
constexpr int kPauseReuseMarker = 867;

class TestLogCapture final : public std::streambuf {
public:
  TestLogCapture() : previous_(std::clog.rdbuf(this)) {}

  ~TestLogCapture() override { std::clog.rdbuf(previous_); }

  bool WaitFor(const std::string &needle) {
    std::unique_lock<std::mutex> lock(mutex_);
    return changed_.wait_for(lock, std::chrono::seconds(2), [&] {
      return output_.find(needle) != std::string::npos;
    });
  }

  size_t Count(const std::string &needle) {
    std::lock_guard<std::mutex> lock(mutex_);
    size_t count = 0;
    for (size_t position = 0;
         (position = output_.find(needle, position)) != std::string::npos;
         position += needle.size()) {
      ++count;
    }
    return count;
  }

protected:
  int_type overflow(int_type character) override {
    if (traits_type::eq_int_type(character, traits_type::eof())) {
      return traits_type::not_eof(character);
    }
    {
      std::lock_guard<std::mutex> lock(mutex_);
      output_.push_back(traits_type::to_char_type(character));
    }
    changed_.notify_all();
    return character;
  }

  std::streamsize xsputn(const char *characters,
                         std::streamsize count) override {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      output_.append(characters, static_cast<size_t>(count));
    }
    changed_.notify_all();
    return count;
  }

private:
  std::streambuf *previous_;
  std::condition_variable changed_;
  std::mutex mutex_;
  std::string output_;
};

struct ReleaseGatesOnExit {
  TestGate &first;
  TestGate &second;

  ~ReleaseGatesOnExit() {
    first.Release();
    second.Release();
  }
};

std::string AudioBase64(int sample_count) {
  std::string pcm;
  pcm.reserve(sample_count * 2);
  for (int sample_index = 0; sample_index < sample_count; ++sample_index) {
    const int16_t sample = 6000;
    pcm.push_back(static_cast<char>(sample & 0xff));
    pcm.push_back(static_cast<char>((sample >> 8) & 0xff));
  }
  return omni_server::EncodeBase64(pcm);
}

nlohmann::json AppendMessage(
    int sample_count, const std::optional<nlohmann::json> &hint,
    const std::optional<nlohmann::json> &speech_epoch = std::nullopt) {
  nlohmann::json message = {{"type", "input_audio_buffer.append"},
                            {"audio", AudioBase64(sample_count)}};
  if (hint.has_value()) {
    message["preview_refresh"] = *hint;
  } else {
  }
  if (speech_epoch.has_value()) {
    message["speech_epoch"] = *speech_epoch;
  } else {
  }
  return message;
}

void UpdateSession(const std::shared_ptr<RealtimeSession> &session,
                   bool enable_pause_refresh, bool enable_deferred = false,
                   bool enable_attribution = false) {
  nlohmann::json settings = {{"turn_detection", nullptr}, {"final_prompt", ""}};
  if (enable_pause_refresh) {
    settings["preview_refresh_enabled"] = true;
  } else {
  }
  if (enable_deferred) {
    settings["preview_refresh_defer_busy_enabled"] = true;
  } else {
  }
  if (enable_attribution) {
    settings["preview_refresh_attribution_enabled"] = true;
  } else {
  }
  Require(session->Handle({{"type", "session.update"}, {"session", settings}}),
          "manual session update was rejected");
}

TranscriptionResult PreviewResult() {
  TranscriptionResult result;
  result.text = "preview";
  result.language = "en";
  result.finish_reason = FinishReason::kStop;
  return result;
}

PauseRefreshTestDecoder PauseDecoder(TestSignal &started,
                                     std::shared_future<void> release,
                                     std::atomic<int> &decoded_samples) {
  return [&started, release = std::move(release), &decoded_samples](
             const std::vector<float> &samples, const std::atomic<bool> &cancel,
             RefreshReuse *reuse) -> TranscriptionResult {
    decoded_samples.store(static_cast<int>(samples.size()));
    started.Notify();
    WaitFor(release, "pause decoder release timed out");
    if (cancel.load()) {
      throw GenerationCancelled();
    } else {
    }
    if (reuse != nullptr) {
      reuse->head_ids.push_back(kPauseReuseMarker);
    } else {
    }
    return PreviewResult();
  };
}

void DrainWorker(TranscriptionWorker &worker, TestSignal &drained) {
  // Marker one is queued after the active request. If that request's completion
  // admits one follow-up, FIFO puts it behind marker one. Marker one's callback
  // queues marker two behind that follow-up, so marker two cannot complete
  // until the follow-up and its completion callback have run. Each negative
  // scenario uses a cadence deadline beyond this boundary, so that callback
  // cannot admit a third request.
  worker.Submit(
      [](const std::atomic<bool> &) { return TranscriptionResult{}; },
      NewCancelFlag(),
      [&worker, &drained](std::optional<TranscriptionResult>,
                          std::exception_ptr error) {
        if (error != nullptr) {
          drained.Notify(error);
          return;
        }
        worker.Submit(
            [](const std::atomic<bool> &) { return TranscriptionResult{}; },
            NewCancelFlag(),
            [&drained](std::optional<TranscriptionResult>,
                       std::exception_ptr marker_error) {
              drained.Notify(marker_error);
            });
      });
  drained.Wait("worker did not drain after the pause-refresh scenario");
}

struct PauseRefreshFixture {
  TranscriptionWorker worker{[] {}, [] {}};
  std::shared_ptr<RealtimeSession> session;

  PauseRefreshFixture(
      PauseRefreshTestDecoder decoder, qwen3_asr::RealtimeSettings settings,
      omni_server::RealtimeConnection::Sender sender =
          [](const std::string &) { return true; })
      : session(MakePauseRefreshSession(worker, std::move(decoder), settings,
                                        std::move(sender))) {}

  void Update(bool pause_enabled, bool deferred_enabled = false,
              bool attribution_enabled = false) {
    UpdateSession(session, pause_enabled, deferred_enabled,
                  attribution_enabled);
  }

  bool AppendSamples(
      int sample_count,
      const std::optional<nlohmann::json> &hint = std::nullopt,
      const std::optional<nlohmann::json> &speech_epoch = std::nullopt) {
    return session->Handle(AppendMessage(sample_count, hint, speech_epoch));
  }

  bool AppendMilliseconds(
      int duration_ms, const std::optional<nlohmann::json> &hint = std::nullopt,
      const std::optional<nlohmann::json> &speech_epoch = std::nullopt) {
    return AppendSamples(duration_ms * static_cast<int>(kSampleRate) / 1000,
                         hint, speech_epoch);
  }

  void Drain() {
    TestSignal drained;
    DrainWorker(worker, drained);
  }
};

void TestInvalidHintsKeepAudioAndDoNotSchedule() {
  const long audio_end_sample = 800 * kSampleRate / kMillisecondsPerSecond;
  const std::vector<std::pair<std::string, nlohmann::json>> invalid_hints = {
      {"negative",
       {{"speech_start_sample", -1}, {"speech_end_sample", kSpeechEndSample}}},
      {"reversed",
       {{"speech_start_sample", 500 * kSampleRate / 1000},
        {"speech_end_sample", 400 * kSampleRate / 1000}}},
      {"out of range",
       {{"speech_start_sample", 0},
        {"speech_end_sample", audio_end_sample + 1}}},
      {"non integer",
       {{"speech_start_sample", 0.5}, {"speech_end_sample", kSpeechEndSample}}},
      {"oversized",
       {{"speech_start_sample", 0},
        {"speech_end_sample", std::numeric_limits<uint64_t>::max()}}}};
  for (const auto &[name, invalid_hint] : invalid_hints) {
    std::atomic<int> decoded_samples = 0;
    TestSignal decoded;
    PauseRefreshTestDecoder decoder =
        [&decoded_samples, &decoded](const std::vector<float> &samples,
                                     const std::atomic<bool> &,
                                     RefreshReuse *) -> TranscriptionResult {
      decoded_samples.store(static_cast<int>(samples.size()));
      decoded.Notify();
      return PreviewResult();
    };
    PauseRefreshFixture fixture(
        std::move(decoder), qwen3_asr::MakeRealtimeSettings(1000, 10000, 30.0));
    fixture.Update(true);
    Require(
        fixture.AppendSamples(static_cast<int>(audio_end_sample), invalid_hint),
        name + " invalid hint rejected valid audio");
    Require(fixture.worker.RequestStates().empty(),
            name + " malformed pause hint scheduled work");

    Require(fixture.AppendMilliseconds(
                100, nlohmann::json{{"speech_start_sample", 0},
                                    {"speech_end_sample", kSpeechEndSample}}),
            name + " valid follow-up append was rejected");
    decoded.Wait(name + " follow-up decode timed out");
    Require(decoded_samples.load() ==
                900 * kSampleRate / kMillisecondsPerSecond,
            name + " malformed hint dropped previously accepted audio");
  }
}

void TestRepeatedPauseHintIsNotAdmitted() {
  TestSignal preview_sent;
  std::atomic<int> decode_count = 0;
  PauseRefreshTestDecoder decoder =
      [&decode_count](const std::vector<float> &, const std::atomic<bool> &,
                      RefreshReuse *) -> TranscriptionResult {
    decode_count.fetch_add(1);
    return PreviewResult();
  };
  PauseRefreshFixture fixture(
      std::move(decoder), qwen3_asr::MakeRealtimeSettings(1000, 900, 30.0),
      [&preview_sent](const std::string &serialized) {
        if (nlohmann::json::parse(serialized).value("type", std::string()) ==
            "transcription.segment") {
          preview_sent.Notify();
        } else {
        }
        return true;
      });
  fixture.Update(true);
  const nlohmann::json repeated_hint = {
      {"speech_start_sample", 0}, {"speech_end_sample", kSpeechEndSample}};
  Require(fixture.AppendMilliseconds(800, repeated_hint),
          "initial pause append was rejected");
  preview_sent.Wait("initial pause refresh did not finish");
  Require(fixture.AppendMilliseconds(100, repeated_hint),
          "repeated pause append was rejected");
  fixture.Drain();
  Require(decode_count.load() == 1,
          "the same pause hint admitted more than one refresh");
}

void TestBusyPauseHintAdmission(bool enable_deferred, bool close_pending) {
  TestLogCapture log_capture;
  TestSignal first_started;
  TestSignal second_started;
  std::atomic<int> decode_count = 0;
  std::vector<int> decoded_samples;
  TestGate release_first;
  PauseRefreshFixture fixture(
      [&first_started, &second_started, release = release_first.Future(),
       &decode_count, &decoded_samples](const std::vector<float> &samples,
                                        const std::atomic<bool> &cancel,
                                        RefreshReuse *) -> TranscriptionResult {
        const int call = decode_count.fetch_add(1) + 1;
        decoded_samples.push_back(static_cast<int>(samples.size()));
        if (call == 1) {
          first_started.Notify();
          WaitFor(release, "first refresh release timed out");
        } else if (call == 2) {
          second_started.Notify();
        }
        if (cancel.load()) {
          throw GenerationCancelled();
        }
        TranscriptionResult result = PreviewResult();
        result.text += std::to_string(call);
        return result;
      },
      qwen3_asr::MakeRealtimeSettings(1000, 100, 30.0));
  fixture.Update(true, enable_deferred, true);
  Require(fixture.AppendMilliseconds(100, std::nullopt, 0),
          "initial cadence append was rejected");
  first_started.Wait("initial cadence refresh did not start");
  const nlohmann::json hint = {
      {"speech_start_sample", 100 * kSampleRate / 1000},
      {"speech_end_sample", 600 * kSampleRate / 1000}};
  Require(fixture.AppendMilliseconds(700, hint, 1),
          "busy pause append was rejected");
  Require(!fixture.worker.RequestStates().contains("queued"),
          "busy pause created a second worker request");
  Require(fixture.AppendMilliseconds(100, hint, 1),
          "duplicate busy pause append was rejected");
  Require(log_capture.WaitFor(enable_deferred
                                  ? "\"disposition\":\"deferred\""
                                  : "\"disposition\":\"refresh_busy\""),
          "busy pause outcome was not attributed");

  if (close_pending) {
    fixture.session->Close();
  }
  release_first.Release();
  if (enable_deferred && !close_pending) {
    second_started.Wait("useful busy pause was not admitted after completion");
  }
  fixture.Drain();
  const int expected_decodes = enable_deferred && !close_pending ? 2 : 1;
  Require(decode_count.load() == expected_decodes,
          "busy-pause outcome submitted the wrong number of refreshes");
  Require(log_capture.Count("\"event\":\"refresh\"") ==
                  static_cast<size_t>(expected_decodes) &&
              log_capture.Count("\"queue_wait_us\"") ==
                  static_cast<size_t>(expected_decodes) &&
              log_capture.Count("\"decode_execution_wall_us\"") ==
                  static_cast<size_t>(expected_decodes) &&
              log_capture.Count("\"callback_finished_at_us\"") ==
                  static_cast<size_t>(expected_decodes),
          "refresh timing trace was incomplete");
  if (enable_deferred && !close_pending) {
    Require(decoded_samples.size() == 2 &&
                decoded_samples[1] == 900 * kSampleRate / 1000,
            "deferred refresh did not use the latest uncovered audio");
  }
  if (!close_pending) {
    fixture.session->Close();
  }

  const std::string terminal_disposition =
      close_pending     ? "\"disposition\":\"session_close\""
      : enable_deferred ? "\"disposition\":\"admitted_deferred\""
                        : "\"disposition\":\"refresh_busy\"";
  Require(log_capture.WaitFor("\"hints_received\":2") &&
              log_capture.Count("\"disposition\":\"duplicate\"") == 1 &&
              log_capture.Count(terminal_disposition) == 1,
          "received hints did not reconcile with terminal attribution");
  if (enable_deferred) {
    Require(log_capture.Count("\"disposition\":\"deferred\"") == 1,
            "deferred admission was not recorded as an intermediate state");
  }
}

void TestBusyPauseCoveredAndInsufficientHintsAreNotDeferred() {
  for (const auto &[name, initial_ms, hint_start_ms, hint_end_ms] :
       std::vector<std::tuple<std::string, int, int, int>>{
           {"covered", 800, 0, 600}, {"insufficient", 400, 300, 450}}) {
    TestSignal first_started;
    std::atomic<int> decode_count = 0;
    TestGate release_first;
    PauseRefreshFixture fixture(
        [&first_started, release = release_first.Future(),
         &decode_count](const std::vector<float> &, const std::atomic<bool> &,
                        RefreshReuse *) -> TranscriptionResult {
          ++decode_count;
          first_started.Notify();
          WaitFor(release, "busy-refresh release timed out");
          return PreviewResult();
        },
        qwen3_asr::MakeRealtimeSettings(1000, 100, 30.0));
    fixture.Update(true, true);
    Require(fixture.AppendMilliseconds(initial_ms, std::nullopt, 0),
            name + " initial append was rejected");
    first_started.Wait(name + " initial refresh did not start");
    const nlohmann::json hint = {
        {"speech_start_sample", hint_start_ms * kSampleRate / 1000},
        {"speech_end_sample", hint_end_ms * kSampleRate / 1000}};
    const int added_ms = name == "covered" ? 100 : 400;
    Require(fixture.AppendMilliseconds(added_ms, hint, 1),
            name + " busy append was rejected");
    release_first.Release();
    fixture.Drain();
    Require(decode_count.load() == 1,
            name + " busy pause admitted a second refresh");
  }
}

void TestDeferredPauseIsInvalidatedBySpeechResumeOrMalformedEpoch() {
  for (const bool malformed_epoch : {false, true}) {
    TestSignal first_started;
    std::atomic<int> decode_count = 0;
    TestGate release_first;
    PauseRefreshFixture fixture(
        [&first_started, release = release_first.Future(),
         &decode_count](const std::vector<float> &, const std::atomic<bool> &,
                        RefreshReuse *) -> TranscriptionResult {
          decode_count.fetch_add(1);
          first_started.Notify();
          WaitFor(release, "resume-test refresh release timed out");
          return PreviewResult();
        },
        qwen3_asr::MakeRealtimeSettings(1000, 100, 30.0));
    fixture.Update(true, true);
    Require(fixture.AppendMilliseconds(100, std::nullopt, 0),
            "resume-test initial append was rejected");
    first_started.Wait("resume-test refresh did not start");
    const nlohmann::json hint = {
        {"speech_start_sample", 100 * kSampleRate / 1000},
        {"speech_end_sample", 600 * kSampleRate / 1000}};
    Require(fixture.AppendMilliseconds(700, hint, 1),
            "resume-test hint append was rejected");
    const nlohmann::json epoch =
        malformed_epoch ? nlohmann::json("bad") : nlohmann::json(2);
    Require(fixture.AppendMilliseconds(100, std::nullopt, epoch),
            "resume-test resumption append was rejected");
    release_first.Release();
    fixture.Drain();
    Require(decode_count.load() == 1,
            "invalidated pending pause admitted a stale refresh");
  }
}

void TestLaterValidPauseReplacesResumedHint() {
  TestSignal first_started;
  TestSignal second_started;
  std::atomic<int> decode_count = 0;
  std::vector<int> decoded_samples;
  TestGate release_first;
  PauseRefreshFixture fixture(
      [&first_started, &second_started, release = release_first.Future(),
       &decode_count, &decoded_samples](const std::vector<float> &samples,
                                        const std::atomic<bool> &cancel,
                                        RefreshReuse *) -> TranscriptionResult {
        const int call = decode_count.fetch_add(1) + 1;
        decoded_samples.push_back(static_cast<int>(samples.size()));
        if (call == 1) {
          first_started.Notify();
          WaitFor(release, "later-pause refresh release timed out");
        } else if (call == 2) {
          second_started.Notify();
        } else {
        }
        if (cancel.load()) {
          throw GenerationCancelled();
        } else {
        }
        return PreviewResult();
      },
      qwen3_asr::MakeRealtimeSettings(1000, 100, 30.0));
  fixture.Update(true, true);
  Require(fixture.AppendMilliseconds(100, std::nullopt, 0),
          "later-pause initial append was rejected");
  first_started.Wait("later-pause initial refresh did not start");
  const nlohmann::json first_hint = {
      {"speech_start_sample", 100 * kSampleRate / 1000},
      {"speech_end_sample", 300 * kSampleRate / 1000}};
  Require(fixture.AppendMilliseconds(400, first_hint, 1),
          "first pause hint append was rejected");
  const nlohmann::json resumed_hint = {
      {"speech_start_sample", 560 * kSampleRate / 1000},
      {"speech_end_sample", 760 * kSampleRate / 1000}};
  Require(fixture.AppendMilliseconds(460, resumed_hint, 2),
          "resumed pause hint append was rejected");
  const std::map<std::string, int> request_states =
      fixture.worker.RequestStates();
  Require(request_states.find("queued") == request_states.end(),
          "resumed pause queued work before the active refresh completed");
  release_first.Release();
  second_started.Wait("later valid pause was not admitted after resumption");
  fixture.Drain();
  Require(decode_count.load() == 2 && decoded_samples.size() == 2 &&
              decoded_samples[0] ==
                  100 * static_cast<int>(kSampleRate) / 1000 &&
              decoded_samples[1] == 960 * static_cast<int>(kSampleRate) / 1000,
          "speech resumption kept the stale hint or lost the newer pause");
}

void TestCadenceAndFinalHandoffSupersedePendingPause() {
  for (const bool final_handoff : {false, true}) {
    TestSignal first_started;
    TestSignal second_started;
    std::atomic<int> decode_count = 0;
    TestGate release_first;
    PauseRefreshFixture fixture(
        [&first_started, &second_started, release = release_first.Future(),
         &decode_count](const std::vector<float> &, const std::atomic<bool> &,
                        RefreshReuse *) -> TranscriptionResult {
          const int call = decode_count.fetch_add(1) + 1;
          if (call == 1) {
            first_started.Notify();
            WaitFor(release, "cadence/final refresh release timed out");
          } else if (call == 2) {
            second_started.Notify();
          } else {
          }
          return PreviewResult();
        },
        qwen3_asr::MakeRealtimeSettings(1000, 100, 30.0));
    fixture.Update(true, true);
    Require(fixture.AppendMilliseconds(100, std::nullopt, 0),
            "cadence/final initial append was rejected");
    first_started.Wait("cadence/final initial refresh did not start");
    const nlohmann::json hint = {
        {"speech_start_sample", 100 * kSampleRate / 1000},
        {"speech_end_sample", 600 * kSampleRate / 1000}};
    const int added_ms = final_handoff ? 700 : 1100;
    Require(fixture.AppendMilliseconds(added_ms, hint, 1),
            "cadence/final busy append was rejected");
    if (final_handoff) {
      static_cast<void>(
          fixture.session->TakeFinalReuse(std::string(""), false, true, true));
    } else {
    }
    release_first.Release();
    if (final_handoff) {
      fixture.Drain();
      Require(decode_count.load() == 1,
              "Final handoff did not fence deferred pause admission");
    } else {
      second_started.Wait("due cadence refresh did not start");
      fixture.Drain();
      Require(decode_count.load() == 2,
              "cadence and pending pause caused multiple refreshes");
    }
  }
}

void TestIntermediatePieceDoesNotFenceDeferredPause() {
  TestSignal first_started;
  TestSignal second_started;
  std::atomic<int> decode_count = 0;
  TestGate release_first;
  PauseRefreshFixture fixture(
      [&first_started, &second_started, release = release_first.Future(),
       &decode_count](const std::vector<float> &, const std::atomic<bool> &,
                      RefreshReuse *) -> TranscriptionResult {
        const int call = decode_count.fetch_add(1) + 1;
        if (call == 1) {
          first_started.Notify();
          WaitFor(release, "piece-test refresh release timed out");
        } else if (call == 2) {
          second_started.Notify();
        } else {
        }
        return PreviewResult();
      },
      qwen3_asr::MakeRealtimeSettings(1000, 100, 30.0));
  fixture.Update(true, true);
  Require(fixture.AppendMilliseconds(100, std::nullopt, 0),
          "piece-test initial append was rejected");
  first_started.Wait("piece-test initial refresh did not start");
  const nlohmann::json hint = {
      {"speech_start_sample", 100 * kSampleRate / 1000},
      {"speech_end_sample", 600 * kSampleRate / 1000}};
  Require(fixture.AppendMilliseconds(700, hint, 1),
          "piece-test pause append was rejected");
  static_cast<void>(fixture.session->TakeFinalReuse(
      std::string(""), false, true, false, 200 * kSampleRate / 1000));
  release_first.Release();
  second_started.Wait("intermediate piece permanently fenced the pause");
  fixture.Drain();
  Require(decode_count.load() == 2,
          "intermediate piece caused an incorrect pause-refresh count");
}

void TestDeferredPauseClearedOnSegmentRollover() {
  TestSignal first_started;
  TestSignal overflow_detected;
  TestSignal worker_drained;
  std::vector<int> decoded_samples;
  TestGate release_first;
  TestGate release_overflow;
  PauseRefreshFixture fixture(
      [&first_started, release = release_first.Future(), &decoded_samples](
          const std::vector<float> &samples, const std::atomic<bool> &cancel,
          RefreshReuse *) -> TranscriptionResult {
        decoded_samples.push_back(static_cast<int>(samples.size()));
        if (decoded_samples.size() == 1) {
          first_started.Notify();
          WaitFor(release, "rollover-test refresh release timed out");
        } else {
        }
        if (cancel.load()) {
          throw GenerationCancelled();
        } else {
        }
        return PreviewResult();
      },
      qwen3_asr::MakeRealtimeSettings(1000, 100, 1.0));
  qwen3_asr::tests::RealtimeSessionTestAccess::SetBeforeRolloverFinalize(
      *fixture.session, [&] {
        overflow_detected.Notify();
        WaitFor(release_overflow.Future(),
                "rollover finalization gate release timed out");
      });
  fixture.Update(true, true);
  Require(fixture.AppendMilliseconds(100, std::nullopt, 0),
          "rollover-test initial append was rejected");
  first_started.Wait("rollover-test refresh did not start");
  const nlohmann::json hint = {
      {"speech_start_sample", 100 * kSampleRate / 1000},
      {"speech_end_sample", 600 * kSampleRate / 1000}};
  Require(fixture.AppendMilliseconds(700, hint, 1),
          "rollover-test busy pause append was rejected");

  std::promise<void> append_started;
  std::future<void> append_started_future = append_started.get_future();
  auto rollover = std::async(std::launch::async, [&] {
    append_started.set_value();
    return fixture.AppendMilliseconds(201, hint, 1);
  });
  ReleaseGatesOnExit release_gates{release_first, release_overflow};
  WaitFor(append_started_future, "rollover append thread did not start");
  overflow_detected.Wait(
      "rollover was not detected while holding the session lock");
  release_first.Release();
  try {
    DrainWorker(fixture.worker, worker_drained);
  } catch (...) {
    release_overflow.Release();
    throw;
  }
  const size_t refreshes_before_finalize = decoded_samples.size();
  release_overflow.Release();
  Require(refreshes_before_finalize == 1,
          "completion in the rollover gap admitted the stale deferred pause");
  Require(rollover.wait_for(std::chrono::seconds(2)) ==
              std::future_status::ready,
          "segment rollover did not finish after releasing the preview");
  Require(rollover.get(), "segment rollover append was rejected");
  fixture.Drain();
  Require(decoded_samples.size() == 2 &&
              decoded_samples[0] ==
                  100 * static_cast<int>(kSampleRate) / 1000 &&
              decoded_samples[1] == static_cast<int>(kSampleRate),
          "segment rollover submitted a stale deferred preview");
}

void TestPauseRefreshFinalOrdering(bool queue_behind_blocker,
                                   bool close_during_pause) {
  TestSignal blocker_started;
  TestSignal pause_started;
  TestSignal final_finished;
  TestSignal worker_drained;
  std::atomic<int> decoded_samples = 0;
  std::atomic<int> segment_events = 0;
  std::atomic<bool> saw_pause_reuse = false;
  TranscriptionWorker worker([] {}, [] {});
  TestGate release_blocker;
  TestGate release_pause;
  if (queue_behind_blocker) {
    const std::shared_future<void> blocker_release = release_blocker.Future();
    worker.Submit(
        [&blocker_started, blocker_release](const std::atomic<bool> &) {
          blocker_started.Notify();
          WaitFor(blocker_release, "worker blocker release timed out");
          return TranscriptionResult{};
        },
        NewCancelFlag(),
        [](std::optional<TranscriptionResult>, std::exception_ptr) {});
    blocker_started.Wait("worker blocker did not start");
  } else {
  }

  auto session = MakePauseRefreshSession(
      worker,
      PauseDecoder(pause_started, release_pause.Future(), decoded_samples),
      qwen3_asr::MakeRealtimeSettings(1000, 900, 30.0),
      [&segment_events](const std::string &serialized) {
        if (nlohmann::json::parse(serialized).value("type", std::string()) ==
            "transcription.segment") {
          segment_events.fetch_add(1);
        } else {
        }
        return true;
      });
  UpdateSession(session, true, false);
  Require(session->Handle(AppendMessage(
              800 * static_cast<int>(kSampleRate) / 1000,
              nlohmann::json{{"speech_start_sample", 0},
                             {"speech_end_sample", kSpeechEndSample}})),
          "pause append was rejected");
  if (!queue_behind_blocker) {
    pause_started.Wait("pause refresh did not start");
  } else {
    Require(worker.RequestStates().contains("queued"),
            "pause refresh did not queue behind the worker blocker");
  }
  if (close_during_pause) {
    session->Close();
  } else {
    const FinalReuse final_reuse =
        session->TakeFinalReuse(std::string(""), false, true, true);
    Require(final_reuse.reuse != nullptr,
            "Final preparation did not find the session reuse");
    worker.Submit(
        [final_reuse, &saw_pause_reuse](const std::atomic<bool> &) {
          const RefreshReuse snapshot = final_reuse.is_taken
                                            ? std::move(*final_reuse.reuse)
                                            : *final_reuse.reuse;
          saw_pause_reuse =
              std::find(snapshot.head_ids.begin(), snapshot.head_ids.end(),
                        kPauseReuseMarker) != snapshot.head_ids.end();
          return TranscriptionResult{};
        },
        NewCancelFlag(),
        [&final_finished](std::optional<TranscriptionResult>,
                          std::exception_ptr error) {
          final_finished.Notify(error);
        });
  }
  if (queue_behind_blocker) {
    release_blocker.Release();
  } else {
  }
  if (!queue_behind_blocker || !close_during_pause) {
    pause_started.Wait("pause refresh did not reach its barrier");
    release_pause.Release();
  } else {
  }
  if (close_during_pause) {
    worker.Submit(
        [](const std::atomic<bool> &) { return TranscriptionResult{}; },
        NewCancelFlag(),
        [&worker_drained](std::optional<TranscriptionResult>,
                          std::exception_ptr error) {
          worker_drained.Notify(error);
        });
    worker_drained.Wait("worker did not drain after close");
    Require(decoded_samples.load() ==
                    (queue_behind_blocker
                         ? 0
                         : 800 * kSampleRate / kMillisecondsPerSecond) &&
                segment_events.load() == 0,
            "close did not cancel the running pause refresh");
  } else {
    final_finished.Wait("Final decode did not finish");
    Require(saw_pause_reuse.load(),
            "Final ran before an admitted pause refresh updated reuse");
    Require(decoded_samples.load() ==
                800 * kSampleRate / kMillisecondsPerSecond,
            "pause refresh decoded the wrong audio endpoint");
  }
}

} // namespace

namespace qwen3_asr::tests {

void TestAdmissionPolicy() {
  constexpr long sample_rate = 16000;
  constexpr long first_refresh_sample = sample_rate / 10;
  const auto base = [] {
    RefreshAdmissionInput input;
    input.audio_end_sample = first_refresh_sample;
    input.next_refresh_sample = first_refresh_sample;
    input.decode_interval_samples = sample_rate;
    input.minimum_new_speech_samples = sample_rate / 5;
    return input;
  };
  auto pause_case = base();
  pause_case.audio_end_sample = 800 * sample_rate / 1000;
  pause_case.next_refresh_sample = 1400 * sample_rate / 1000;
  pause_case.last_submitted_refresh_end_sample = 400 * sample_rate / 1000;
  pause_case.pause_hint = PauseRefreshHint{0, 600 * sample_rate / 1000};

  struct Case {
    const char *name;
    RefreshAdmissionInput input;
    RefreshSource source;
    PauseHintDisposition disposition;
    long next_refresh_sample;
  };
  std::vector<Case> cases;
  cases.push_back({"first cadence", base(), RefreshSource::kCadence,
                   PauseHintDisposition::kNone,
                   first_refresh_sample + sample_rate});
  auto input = base();
  input.silent = true;
  cases.push_back({"leading silence", input, RefreshSource::kNone,
                   PauseHintDisposition::kNone, first_refresh_sample});
  input.has_decoded = true;
  cases.push_back({"post-preview silence", input, RefreshSource::kNone,
                   PauseHintDisposition::kNone,
                   input.audio_end_sample + sample_rate});
  cases.push_back({"immediate pause", pause_case, RefreshSource::kPause,
                   PauseHintDisposition::kAdmittedImmediate,
                   pause_case.audio_end_sample + sample_rate});
  input = pause_case;
  input.pause_hint_deferred = true;
  cases.push_back({"deferred pause", input, RefreshSource::kDeferredPause,
                   PauseHintDisposition::kAdmittedDeferred,
                   input.audio_end_sample + sample_rate});
  input = pause_case;
  input.last_submitted_refresh_end_sample = 600 * sample_rate / 1000;
  cases.push_back({"covered speech", input, RefreshSource::kNone,
                   PauseHintDisposition::kCovered, input.next_refresh_sample});
  input = pause_case;
  input.pause_hint =
      PauseRefreshHint{300 * sample_rate / 1000, 590 * sample_rate / 1000};
  cases.push_back({"insufficient new speech", input, RefreshSource::kNone,
                   PauseHintDisposition::kInsufficientNewSpeech,
                   input.next_refresh_sample});
  input = pause_case;
  input.refreshing = true;
  input.last_submitted_refresh_end_sample = 100 * sample_rate / 1000;
  input.pause_hint =
      PauseRefreshHint{100 * sample_rate / 1000, 600 * sample_rate / 1000};
  cases.push_back({"busy useful hint", input, RefreshSource::kNone,
                   PauseHintDisposition::kRefreshBusy,
                   input.next_refresh_sample});
  input = pause_case;
  input.cancelled = true;
  cases.push_back({"cancelled pause", input, RefreshSource::kNone,
                   PauseHintDisposition::kCancelled,
                   input.next_refresh_sample});
  input = pause_case;
  input.final_handoff = true;
  cases.push_back({"Final handoff pause", input, RefreshSource::kNone,
                   PauseHintDisposition::kFinalHandoff,
                   input.next_refresh_sample});
  input.audio_end_sample = input.next_refresh_sample;
  cases.push_back({"cadence after handoff", input, RefreshSource::kCadence,
                   PauseHintDisposition::kFinalHandoff,
                   input.audio_end_sample + sample_rate});
  input = base();
  input.finalizing = true;
  cases.push_back({"segment finalization", input, RefreshSource::kNone,
                   PauseHintDisposition::kNone, input.next_refresh_sample});

  for (const Case &test : cases) {
    const RefreshAdmission admission = SelectRefreshAdmission(test.input);
    Require(admission.source == test.source &&
                admission.disposition == test.disposition &&
                admission.next_refresh_sample == test.next_refresh_sample,
            std::string(test.name) + " selected the wrong refresh decision");
  }

  const std::vector<
      std::pair<std::optional<PauseRefreshHint>, PauseHintDisposition>>
      consumption_cases = {
          {std::nullopt, PauseHintDisposition::kNone},
          {PauseRefreshHint{-1, 400}, PauseHintDisposition::kInvalid},
          {PauseRefreshHint{400, 300}, PauseHintDisposition::kInvalid},
          {PauseRefreshHint{0, 1200}, PauseHintDisposition::kInvalid},
          {PauseRefreshHint{100, 600}, PauseHintDisposition::kDuplicate},
          {PauseRefreshHint{100, 800}, PauseHintDisposition::kNone}};
  for (const auto &[hint, disposition] : consumption_cases) {
    const PauseHintConsumption consumed =
        ConsumePauseRefreshHint(hint, 0, 1000, 600);
    Require(consumed.disposition == disposition,
            "hint validation or deduplication changed");
    if (disposition == PauseHintDisposition::kNone && hint.has_value()) {
      Require(consumed.hint.has_value() &&
                  consumed.hint->speech_start_sample ==
                      hint->speech_start_sample &&
                  consumed.hint->speech_end_sample == hint->speech_end_sample &&
                  consumed.last_speech_end_sample == 800,
              "a valid hint did not advance the consumed speech endpoint");
    }
  }
}

void TestPauseRefreshRequiresSessionOptIn() {
  const std::vector<std::pair<std::optional<nlohmann::json>, bool>> cases = {
      {std::nullopt, false},
      {nlohmann::json(false), false},
      {nlohmann::json("true"), false},
      {nlohmann::json(true), true}};
  for (const auto &[opt_in_value, should_decode] : cases) {
    TestSignal decoded;
    TestSignal cadence_decoded;
    TestSignal worker_drained;
    std::atomic<int> decode_count = 0;
    omni_server::TranscriptionWorker worker([] {}, [] {});
    auto session = MakePauseRefreshSession(
        worker,
        [&decoded, &cadence_decoded,
         &decode_count](const std::vector<float> &, const std::atomic<bool> &,
                        RefreshReuse *) -> TranscriptionResult {
          const int count = ++decode_count;
          decoded.Notify();
          if (count == 2) {
            cadence_decoded.Notify();
          } else {
          }
          return PreviewResult();
        },
        qwen3_asr::MakeRealtimeSettings(1000, should_decode ? 500 : 1000, 30.0),
        [](const std::string &) { return true; });
    nlohmann::json settings = {{"turn_detection", nullptr},
                               {"final_prompt", ""}};
    if (opt_in_value.has_value()) {
      settings["preview_refresh_enabled"] = *opt_in_value;
    }
    Require(
        session->Handle({{"type", "session.update"}, {"session", settings}}),
        "session opt-in update was rejected");
    const int first_append_ms = should_decode ? 400 : 800;
    const nlohmann::json hint = {
        {"speech_start_sample", 0},
        {"speech_end_sample",
         (should_decode ? 300 : 600) * kSampleRate / 1000}};
    Require(session->Handle(AppendMessage(
                first_append_ms * static_cast<int>(kSampleRate) / 1000, hint)),
            "pause append was rejected");
    if (should_decode) {
      decoded.Wait("enabled session pause refresh timed out");
      Require(decode_count.load() == 1,
              "enabled session did not admit one pause refresh");
      DrainWorker(worker, worker_drained);
      session->TakeFinalReuse(std::string(""), false, true, true);
      const nlohmann::json later_hint = {
          {"speech_start_sample", 400 * kSampleRate / 1000},
          {"speech_end_sample", 900 * kSampleRate / 1000}};
      Require(session->Handle(AppendMessage(
                  600 * static_cast<int>(kSampleRate) / 1000, later_hint)),
              "pause append after Final preparation was rejected");
      Require(worker.RequestStates().empty(),
              "pause hint after Final preparation queued a refresh");
      Require(session->Handle(AppendMessage(
                  400 * static_cast<int>(kSampleRate) / 1000, std::nullopt)),
              "ordinary cadence append after Final preparation was rejected");
      cadence_decoded.Wait("ordinary cadence preview timed out");
      Require(decode_count.load() == 2,
              "Final fence suppressed the ordinary cadence preview");
    } else {
      Require(worker.RequestStates().empty() && decode_count.load() == 0,
              "session without boolean opt-in admitted a pause refresh");
    }
    session->Close();
  }
}

} // namespace qwen3_asr::tests

int main() {
  std::set_terminate([] {
    void *frames[64];
    const int frame_count = backtrace(frames, 64);
    backtrace_symbols_fd(frames, frame_count, 2);
    std::abort();
  });
  try {
    std::cerr << "TestAdmissionPolicy\n";
    qwen3_asr::tests::TestAdmissionPolicy();
    std::cerr << "TestInvalidHintsKeepAudioAndDoNotSchedule\n";
    TestInvalidHintsKeepAudioAndDoNotSchedule();
    std::cerr << "TestRepeatedPauseHintIsNotAdmitted\n";
    TestRepeatedPauseHintIsNotAdmitted();
    std::cerr << "TestBusyPauseHintAdmission(false, false)\n";
    TestBusyPauseHintAdmission(false, false);
    std::cerr << "TestBusyPauseHintAdmission(true, false)\n";
    TestBusyPauseHintAdmission(true, false);
    std::cerr << "TestBusyPauseHintAdmission(true, true)\n";
    TestBusyPauseHintAdmission(true, true);
    std::cerr << "TestBusyPauseCoveredAndInsufficientHintsAreNotDeferred\n";
    TestBusyPauseCoveredAndInsufficientHintsAreNotDeferred();
    std::cerr
        << "TestDeferredPauseIsInvalidatedBySpeechResumeOrMalformedEpoch\n";
    TestDeferredPauseIsInvalidatedBySpeechResumeOrMalformedEpoch();
    std::cerr << "TestLaterValidPauseReplacesResumedHint\n";
    TestLaterValidPauseReplacesResumedHint();
    std::cerr << "TestCadenceAndFinalHandoffSupersedePendingPause\n";
    TestCadenceAndFinalHandoffSupersedePendingPause();
    std::cerr << "TestIntermediatePieceDoesNotFenceDeferredPause\n";
    TestIntermediatePieceDoesNotFenceDeferredPause();
    std::cerr << "TestDeferredPauseClearedOnSegmentRollover\n";
    TestDeferredPauseClearedOnSegmentRollover();
    std::cerr << "TestPauseRefreshRequiresSessionOptIn\n";
    qwen3_asr::tests::TestPauseRefreshRequiresSessionOptIn();
    std::cerr << "TestPauseRefreshFinalOrdering(true, false)\n";
    TestPauseRefreshFinalOrdering(true, false);
    std::cerr << "TestPauseRefreshFinalOrdering(false, false)\n";
    TestPauseRefreshFinalOrdering(false, false);
    std::cerr << "TestPauseRefreshFinalOrdering(false, true)\n";
    TestPauseRefreshFinalOrdering(false, true);
    std::cerr << "TestPauseRefreshFinalOrdering(true, true)\n";
    TestPauseRefreshFinalOrdering(true, true);
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  return 0;
}
