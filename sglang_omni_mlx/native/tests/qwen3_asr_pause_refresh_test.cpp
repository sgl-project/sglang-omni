// SPDX-License-Identifier: Apache-2.0
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <future>
#include <iostream>
#include <limits>
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

RealtimeSession::DecodeOverride
PauseDecoder(TestSignal &started, std::shared_future<void> release,
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
  worker.Submit(
      [](const std::atomic<bool> &) { return TranscriptionResult{}; },
      NewCancelFlag(),
      [&drained](std::optional<TranscriptionResult>, std::exception_ptr error) {
        drained.Notify(error);
      });
  drained.Wait("worker did not drain after the pause-refresh scenario");
}

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
    TranscriptionWorker worker([] {}, [] {});
    RealtimeSession::DecodeOverride decoder =
        [&decoded_samples, &decoded](const std::vector<float> &samples,
                                     const std::atomic<bool> &,
                                     RefreshReuse *) -> TranscriptionResult {
      decoded_samples.store(static_cast<int>(samples.size()));
      decoded.Notify();
      return PreviewResult();
    };
    auto session = std::make_shared<RealtimeSession>(
        worker, std::move(decoder),
        qwen3_asr::MakeRealtimeSettings(1000, 10000, 30.0),
        [](const std::string &) { return true; });
    UpdateSession(session, true);
    Require(session->Handle(AppendMessage(static_cast<int>(audio_end_sample),
                                          invalid_hint)),
            name + " invalid hint rejected valid audio");
    Require(worker.RequestStates().empty(),
            name + " malformed pause hint scheduled work");

    Require(session->Handle(AppendMessage(
                100 * static_cast<int>(kSampleRate) / 1000,
                nlohmann::json{{"speech_start_sample", 0},
                               {"speech_end_sample", kSpeechEndSample}})),
            name + " valid follow-up append was rejected");
    decoded.Wait(name + " follow-up decode timed out");
    Require(decoded_samples.load() ==
                900 * kSampleRate / kMillisecondsPerSecond,
            name + " malformed hint dropped previously accepted audio");
  }
}

void TestRepeatedPauseHintIsNotAdmitted() {
  TestSignal preview_sent;
  TestSignal worker_drained;
  std::atomic<int> decode_count = 0;
  TranscriptionWorker worker([] {}, [] {});
  RealtimeSession::DecodeOverride decoder =
      [&decode_count](const std::vector<float> &, const std::atomic<bool> &,
                      RefreshReuse *) -> TranscriptionResult {
    decode_count.fetch_add(1);
    return PreviewResult();
  };
  auto session = std::make_shared<RealtimeSession>(
      worker, std::move(decoder),
      qwen3_asr::MakeRealtimeSettings(1000, 900, 30.0),
      [&preview_sent](const std::string &serialized) {
        if (nlohmann::json::parse(serialized).value("type", std::string()) ==
            "transcription.segment") {
          preview_sent.Notify();
        } else {
        }
        return true;
      });
  UpdateSession(session, true);
  const nlohmann::json repeated_hint = {
      {"speech_start_sample", 0}, {"speech_end_sample", kSpeechEndSample}};
  Require(session->Handle(AppendMessage(
              800 * static_cast<int>(kSampleRate) / 1000, repeated_hint)),
          "initial pause append was rejected");
  preview_sent.Wait("initial pause refresh did not finish");
  Require(session->Handle(AppendMessage(
              100 * static_cast<int>(kSampleRate) / 1000, repeated_hint)),
          "repeated pause append was rejected");
  worker.Submit([](const std::atomic<bool> &) { return TranscriptionResult{}; },
                NewCancelFlag(),
                [&worker_drained](std::optional<TranscriptionResult>,
                                  std::exception_ptr error) {
                  worker_drained.Notify(error);
                });
  worker_drained.Wait("worker did not drain after the repeated hint");
  Require(decode_count.load() == 1,
          "the same pause hint admitted more than one refresh");
}

void TestBusyPauseHintDeferredAdmission(bool enable_deferred) {
  TestSignal first_started;
  TestSignal second_started;
  TestSignal worker_drained;
  TestGate release_first;
  std::atomic<int> decode_count = 0;
  std::vector<int> decoded_samples;
  TranscriptionWorker worker([] {}, [] {});
  auto session = std::make_shared<RealtimeSession>(
      worker,
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
        } else {
        }
        if (cancel.load()) {
          throw GenerationCancelled();
        } else {
        }
        TranscriptionResult result = PreviewResult();
        result.text += std::to_string(call);
        return result;
      },
      qwen3_asr::MakeRealtimeSettings(1000, 100, 30.0),
      [](const std::string &) { return true; });
  UpdateSession(session, true, enable_deferred);
  Require(session->Handle(AppendMessage(
              100 * static_cast<int>(kSampleRate) / 1000, std::nullopt, 0)),
          "initial cadence append was rejected");
  first_started.Wait("initial cadence refresh did not start");
  const nlohmann::json hint = {
      {"speech_start_sample", 100 * kSampleRate / 1000},
      {"speech_end_sample", 600 * kSampleRate / 1000}};
  Require(session->Handle(AppendMessage(
              700 * static_cast<int>(kSampleRate) / 1000, hint, 1)),
          "busy pause append was rejected");
  const std::map<std::string, int> request_states = worker.RequestStates();
  Require(request_states.find("queued") == request_states.end(),
          "busy pause created a second worker request");
  Require(session->Handle(AppendMessage(
              100 * static_cast<int>(kSampleRate) / 1000, hint, 1)),
          "duplicate busy pause append was rejected");
  release_first.Release();
  if (enable_deferred) {
    second_started.Wait("useful busy pause was not admitted after completion");
  } else {
  }
  DrainWorker(worker, worker_drained);
  Require(decode_count.load() == (enable_deferred ? 2 : 1),
          "busy-pause arm submitted the wrong number of refreshes");
  if (enable_deferred) {
    Require(decoded_samples.size() == 2 &&
                decoded_samples[1] ==
                    900 * kSampleRate / kMillisecondsPerSecond,
            "deferred refresh did not snapshot the newest uncovered audio");
  } else {
  }
}

void TestBusyPauseCoveredAndInsufficientHintsAreNotDeferred() {
  for (const auto &[name, initial_ms, hint_start_ms, hint_end_ms] :
       std::vector<std::tuple<std::string, int, int, int>>{
           {"covered", 800, 0, 600}, {"insufficient", 400, 300, 450}}) {
    TestSignal first_started;
    TestSignal worker_drained;
    TestGate release_first;
    std::atomic<int> decode_count = 0;
    TranscriptionWorker worker([] {}, [] {});
    auto session = std::make_shared<RealtimeSession>(
        worker,
        [&first_started, release = release_first.Future(),
         &decode_count](const std::vector<float> &, const std::atomic<bool> &,
                        RefreshReuse *) -> TranscriptionResult {
          decode_count.fetch_add(1);
          first_started.Notify();
          WaitFor(release, "busy-refresh release timed out");
          return PreviewResult();
        },
        qwen3_asr::MakeRealtimeSettings(1000, 100, 30.0),
        [](const std::string &) { return true; });
    UpdateSession(session, true, true);
    Require(session->Handle(
                AppendMessage(initial_ms * static_cast<int>(kSampleRate) / 1000,
                              std::nullopt, 0)),
            name + " initial append was rejected");
    first_started.Wait(name + " initial refresh did not start");
    const nlohmann::json hint = {
        {"speech_start_sample", hint_start_ms * kSampleRate / 1000},
        {"speech_end_sample", hint_end_ms * kSampleRate / 1000}};
    const int added_ms = name == "covered" ? 100 : 400;
    Require(session->Handle(AppendMessage(
                added_ms * static_cast<int>(kSampleRate) / 1000, hint, 1)),
            name + " busy append was rejected");
    release_first.Release();
    DrainWorker(worker, worker_drained);
    Require(decode_count.load() == 1,
            name + " busy pause admitted a second refresh");
  }
}

void TestDeferredPauseIsInvalidatedBySpeechResumeOrMalformedEpoch() {
  for (const bool malformed_epoch : {false, true}) {
    TestSignal first_started;
    TestSignal worker_drained;
    TestGate release_first;
    std::atomic<int> decode_count = 0;
    TranscriptionWorker worker([] {}, [] {});
    auto session = std::make_shared<RealtimeSession>(
        worker,
        [&first_started, release = release_first.Future(),
         &decode_count](const std::vector<float> &, const std::atomic<bool> &,
                        RefreshReuse *) -> TranscriptionResult {
          decode_count.fetch_add(1);
          first_started.Notify();
          WaitFor(release, "resume-test refresh release timed out");
          return PreviewResult();
        },
        qwen3_asr::MakeRealtimeSettings(1000, 100, 30.0),
        [](const std::string &) { return true; });
    UpdateSession(session, true, true);
    Require(session->Handle(AppendMessage(
                100 * static_cast<int>(kSampleRate) / 1000, std::nullopt, 0)),
            "resume-test initial append was rejected");
    first_started.Wait("resume-test refresh did not start");
    const nlohmann::json hint = {
        {"speech_start_sample", 100 * kSampleRate / 1000},
        {"speech_end_sample", 600 * kSampleRate / 1000}};
    Require(session->Handle(AppendMessage(
                700 * static_cast<int>(kSampleRate) / 1000, hint, 1)),
            "resume-test hint append was rejected");
    const nlohmann::json epoch =
        malformed_epoch ? nlohmann::json("bad") : nlohmann::json(2);
    Require(
        session->Handle(AppendMessage(
            100 * static_cast<int>(kSampleRate) / 1000, std::nullopt, epoch)),
        "resume-test resumption append was rejected");
    release_first.Release();
    DrainWorker(worker, worker_drained);
    Require(decode_count.load() == 1,
            "invalidated pending pause admitted a stale refresh");
  }
}

void TestLaterValidPauseReplacesResumedHint() {
  TestSignal first_started;
  TestSignal second_started;
  TestSignal worker_drained;
  TestGate release_first;
  std::atomic<int> decode_count = 0;
  std::vector<int> decoded_samples;
  TranscriptionWorker worker([] {}, [] {});
  auto session = std::make_shared<RealtimeSession>(
      worker,
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
      qwen3_asr::MakeRealtimeSettings(1000, 100, 30.0),
      [](const std::string &) { return true; });
  UpdateSession(session, true, true);
  Require(session->Handle(AppendMessage(
              100 * static_cast<int>(kSampleRate) / 1000, std::nullopt, 0)),
          "later-pause initial append was rejected");
  first_started.Wait("later-pause initial refresh did not start");
  const nlohmann::json first_hint = {
      {"speech_start_sample", 100 * kSampleRate / 1000},
      {"speech_end_sample", 300 * kSampleRate / 1000}};
  Require(session->Handle(AppendMessage(
              400 * static_cast<int>(kSampleRate) / 1000, first_hint, 1)),
          "first pause hint append was rejected");
  const nlohmann::json resumed_hint = {
      {"speech_start_sample", 560 * kSampleRate / 1000},
      {"speech_end_sample", 760 * kSampleRate / 1000}};
  Require(session->Handle(AppendMessage(
              460 * static_cast<int>(kSampleRate) / 1000, resumed_hint, 2)),
          "resumed pause hint append was rejected");
  const std::map<std::string, int> request_states = worker.RequestStates();
  Require(request_states.find("queued") == request_states.end(),
          "resumed pause queued work before the active refresh completed");
  release_first.Release();
  second_started.Wait("later valid pause was not admitted after resumption");
  DrainWorker(worker, worker_drained);
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
    TestSignal worker_drained;
    TestGate release_first;
    std::atomic<int> decode_count = 0;
    TranscriptionWorker worker([] {}, [] {});
    auto session = std::make_shared<RealtimeSession>(
        worker,
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
        qwen3_asr::MakeRealtimeSettings(1000, 100, 30.0),
        [](const std::string &) { return true; });
    UpdateSession(session, true, true);
    Require(session->Handle(AppendMessage(
                100 * static_cast<int>(kSampleRate) / 1000, std::nullopt, 0)),
            "cadence/final initial append was rejected");
    first_started.Wait("cadence/final initial refresh did not start");
    const nlohmann::json hint = {
        {"speech_start_sample", 100 * kSampleRate / 1000},
        {"speech_end_sample", 600 * kSampleRate / 1000}};
    const int added_ms = final_handoff ? 700 : 1100;
    Require(session->Handle(AppendMessage(
                added_ms * static_cast<int>(kSampleRate) / 1000, hint, 1)),
            "cadence/final busy append was rejected");
    if (final_handoff) {
      static_cast<void>(
          session->TakeFinalReuse(std::string(""), false, true, true));
    } else {
    }
    release_first.Release();
    if (final_handoff) {
      DrainWorker(worker, worker_drained);
      Require(decode_count.load() == 1,
              "Final handoff did not fence deferred pause admission");
    } else {
      second_started.Wait("due cadence refresh did not start");
      DrainWorker(worker, worker_drained);
      Require(decode_count.load() == 2,
              "cadence and pending pause caused multiple refreshes");
    }
  }
}

void TestIntermediatePieceDoesNotFenceDeferredPause() {
  TestSignal first_started;
  TestSignal second_started;
  TestSignal worker_drained;
  TestGate release_first;
  std::atomic<int> decode_count = 0;
  TranscriptionWorker worker([] {}, [] {});
  auto session = std::make_shared<RealtimeSession>(
      worker,
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
      qwen3_asr::MakeRealtimeSettings(1000, 100, 30.0),
      [](const std::string &) { return true; });
  UpdateSession(session, true, true);
  Require(session->Handle(AppendMessage(
              100 * static_cast<int>(kSampleRate) / 1000, std::nullopt, 0)),
          "piece-test initial append was rejected");
  first_started.Wait("piece-test initial refresh did not start");
  const nlohmann::json hint = {
      {"speech_start_sample", 100 * kSampleRate / 1000},
      {"speech_end_sample", 600 * kSampleRate / 1000}};
  Require(session->Handle(AppendMessage(
              700 * static_cast<int>(kSampleRate) / 1000, hint, 1)),
          "piece-test pause append was rejected");
  static_cast<void>(session->TakeFinalReuse(std::string(""), false, true, false,
                                            200 * kSampleRate / 1000));
  release_first.Release();
  second_started.Wait("intermediate piece permanently fenced the pause");
  DrainWorker(worker, worker_drained);
  Require(decode_count.load() == 2,
          "intermediate piece caused an incorrect pause-refresh count");
}

void TestCloseCancelsDeferredPause() {
  TestSignal first_started;
  TestSignal worker_drained;
  TestGate release_first;
  std::atomic<int> decode_count = 0;
  TranscriptionWorker worker([] {}, [] {});
  auto session = std::make_shared<RealtimeSession>(
      worker,
      [&first_started, release = release_first.Future(), &decode_count](
          const std::vector<float> &, const std::atomic<bool> &cancel,
          RefreshReuse *) -> TranscriptionResult {
        decode_count.fetch_add(1);
        first_started.Notify();
        WaitFor(release, "close-test refresh release timed out");
        if (cancel.load()) {
          throw GenerationCancelled();
        } else {
        }
        return PreviewResult();
      },
      qwen3_asr::MakeRealtimeSettings(1000, 100, 30.0),
      [](const std::string &) { return true; });
  UpdateSession(session, true, true);
  Require(session->Handle(AppendMessage(
              100 * static_cast<int>(kSampleRate) / 1000, std::nullopt, 0)),
          "close-test initial append was rejected");
  first_started.Wait("close-test initial refresh did not start");
  const nlohmann::json hint = {
      {"speech_start_sample", 100 * kSampleRate / 1000},
      {"speech_end_sample", 600 * kSampleRate / 1000}};
  Require(session->Handle(AppendMessage(
              700 * static_cast<int>(kSampleRate) / 1000, hint, 1)),
          "close-test pause append was rejected");
  session->Close();
  release_first.Release();
  DrainWorker(worker, worker_drained);
  Require(decode_count.load() == 1,
          "closed session admitted a stale deferred refresh");
}

void TestDeferredPauseClearedOnSegmentRollover() {
  TestLogCapture log_capture;
  TestSignal first_started;
  TestSignal worker_drained;
  TestGate release_first;
  std::vector<int> decoded_samples;
  TranscriptionWorker worker([] {}, [] {});
  auto session = std::make_shared<RealtimeSession>(
      worker,
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
      qwen3_asr::MakeRealtimeSettings(1000, 100, 1.0),
      [](const std::string &) { return true; });
  UpdateSession(session, true, true, true);
  Require(session->Handle(AppendMessage(
              100 * static_cast<int>(kSampleRate) / 1000, std::nullopt, 0)),
          "rollover-test initial append was rejected");
  first_started.Wait("rollover-test refresh did not start");
  const nlohmann::json hint = {
      {"speech_start_sample", 100 * kSampleRate / 1000},
      {"speech_end_sample", 600 * kSampleRate / 1000}};
  Require(session->Handle(AppendMessage(
              700 * static_cast<int>(kSampleRate) / 1000, hint, 1)),
          "rollover-test busy pause append was rejected");

  std::promise<void> append_started;
  std::future<void> append_started_future = append_started.get_future();
  auto rollover = std::async(std::launch::async, [&] {
    append_started.set_value();
    return session->Handle(AppendMessage(
        201 * static_cast<int>(kSampleRate) / 1000, std::nullopt, 1));
  });
  const bool append_started_on_time =
      append_started_future.wait_for(std::chrono::seconds(2)) ==
      std::future_status::ready;
  const bool rollover_logged =
      log_capture.WaitFor("\"disposition\":\"segment_rollover\"");
  const bool still_waiting_for_refresh =
      rollover.wait_for(std::chrono::milliseconds(0)) ==
      std::future_status::timeout;
  release_first.Release();
  Require(append_started_on_time,
          "rollover append thread did not start in time");
  Require(rollover_logged,
          "segment rollover did not invalidate the pending pause hint");
  Require(still_waiting_for_refresh,
          "rollover did not wait for the in-flight preview barrier");
  Require(rollover.wait_for(std::chrono::seconds(2)) ==
              std::future_status::ready,
          "segment rollover did not finish after releasing the preview");
  Require(rollover.get(), "segment rollover append was rejected");
  DrainWorker(worker, worker_drained);
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

  auto session = std::make_shared<RealtimeSession>(
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
  UpdateSession(session, true);
  Require(session->Handle(AppendMessage(
              800 * static_cast<int>(kSampleRate) / 1000,
              nlohmann::json{{"speech_start_sample", 0},
                             {"speech_end_sample", kSpeechEndSample}})),
          "pause append was rejected");
  if (!queue_behind_blocker) {
    pause_started.Wait("pause refresh did not start");
  } else {
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
  pause_started.Wait("pause refresh did not reach its barrier");
  release_pause.Release();
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
                    800 * kSampleRate / kMillisecondsPerSecond &&
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

void TestFinalFenceBlocksPauseAndPreservesCadence() {
  TestSignal preview_sent;
  std::atomic<int> decoded_samples = 0;
  std::atomic<int> preview_count = 0;
  TranscriptionWorker worker([] {}, [] {});
  auto session = std::make_shared<RealtimeSession>(
      worker,
      [&decoded_samples](const std::vector<float> &samples,
                         const std::atomic<bool> &,
                         RefreshReuse *) -> TranscriptionResult {
        decoded_samples.store(static_cast<int>(samples.size()));
        return PreviewResult();
      },
      qwen3_asr::MakeRealtimeSettings(1000, 500, 30.0),
      [&preview_sent, &preview_count](const std::string &serialized) {
        const auto event = nlohmann::json::parse(serialized);
        if (event.value("type", std::string()) == "transcription.segment") {
          preview_count.fetch_add(1);
          preview_sent.Notify();
        } else {
        }
        return true;
      });
  UpdateSession(session, true);
  session->TakeFinalReuse(std::string(""), false, true, true);
  Require(session->Handle(AppendMessage(
              400 * static_cast<int>(kSampleRate) / 1000,
              nlohmann::json{{"speech_start_sample", 0},
                             {"speech_end_sample", 300 * kSampleRate / 1000}})),
          "pause append after Final preparation was rejected");
  Require(worker.RequestStates().empty(),
          "pause hint after Final preparation queued a refresh");
  Require(session->Handle(AppendMessage(
              100 * static_cast<int>(kSampleRate) / 1000, std::nullopt)),
          "ordinary cadence append after Final preparation was rejected");
  preview_sent.Wait("ordinary cadence preview timed out");
  Require(decoded_samples.load() == 500 * kSampleRate / 1000 &&
              preview_count.load() == 1,
          "Final fence suppressed the ordinary preview path");
}

} // namespace

namespace qwen3_asr::tests {
void TestAdmissionPolicy();
void TestDeferredHintState();
void TestPauseRefreshRequiresSessionOptIn();
} // namespace qwen3_asr::tests

int main() {
  try {
    qwen3_asr::tests::TestAdmissionPolicy();
    qwen3_asr::tests::TestDeferredHintState();
    TestInvalidHintsKeepAudioAndDoNotSchedule();
    TestRepeatedPauseHintIsNotAdmitted();
    TestBusyPauseHintDeferredAdmission(false);
    TestBusyPauseHintDeferredAdmission(true);
    TestBusyPauseCoveredAndInsufficientHintsAreNotDeferred();
    TestDeferredPauseIsInvalidatedBySpeechResumeOrMalformedEpoch();
    TestLaterValidPauseReplacesResumedHint();
    TestCadenceAndFinalHandoffSupersedePendingPause();
    TestIntermediatePieceDoesNotFenceDeferredPause();
    TestCloseCancelsDeferredPause();
    TestDeferredPauseClearedOnSegmentRollover();
    qwen3_asr::tests::TestPauseRefreshRequiresSessionOptIn();
    TestPauseRefreshFinalOrdering(true, false);
    TestPauseRefreshFinalOrdering(false, false);
    TestPauseRefreshFinalOrdering(false, true);
    TestFinalFenceBlocksPauseAndPreservesCadence();
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  return 0;
}
