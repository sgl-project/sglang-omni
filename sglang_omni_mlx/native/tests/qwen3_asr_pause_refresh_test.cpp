// SPDX-License-Identifier: Apache-2.0
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <future>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/generation.h"
#include "common/http.h"
#include "common/worker.h"
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

constexpr long kSampleRate = 16000;
constexpr long kMillisecondsPerSecond = 1000;
constexpr long kSpeechEndSample = 600 * kSampleRate / kMillisecondsPerSecond;
constexpr int kPauseReuseMarker = 867;

void Require(bool condition, const std::string &message) {
  if (!condition) {
    throw std::runtime_error(message);
  } else {
  }
}

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

nlohmann::json AppendMessage(int sample_count,
                             const std::optional<nlohmann::json> &hint) {
  nlohmann::json message = {{"type", "input_audio_buffer.append"},
                            {"audio", AudioBase64(sample_count)}};
  if (hint.has_value()) {
    message["preview_refresh"] = *hint;
  } else {
  }
  return message;
}

void UpdateSession(const std::shared_ptr<RealtimeSession> &session,
                   bool enable_pause_refresh) {
  nlohmann::json settings = {{"turn_detection", nullptr}, {"final_prompt", ""}};
  if (enable_pause_refresh) {
    settings["preview_refresh_enabled"] = true;
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
PauseDecoder(std::promise<void> *started, std::shared_future<void> release,
             std::atomic<int> *decoded_samples) {
  return [started, release = std::move(release), decoded_samples](
             const std::vector<float> &samples, const std::atomic<bool> &,
             RefreshReuse *reuse) -> Transcription {
    decoded_samples->store(static_cast<int>(samples.size()));
    return [started, release, reuse](const std::atomic<bool> &cancel) {
      if (started != nullptr) {
        started->set_value();
      } else {
      }
      if (release.valid()) {
        release.wait();
      } else {
      }
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
  };
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
    TranscriptionWorker worker([] {}, [] {});
    std::atomic<int> decoded_samples = 0;
    std::promise<void> decoded;
    RealtimeSession::DecodeOverride decoder =
        [&decoded_samples, &decoded](const std::vector<float> &samples,
                                     const std::atomic<bool> &,
                                     RefreshReuse *) -> Transcription {
      decoded_samples.store(static_cast<int>(samples.size()));
      return [&decoded](const std::atomic<bool> &) {
        decoded.set_value();
        return PreviewResult();
      };
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
    decoded.get_future().wait();
    Require(decoded_samples.load() ==
                900 * kSampleRate / kMillisecondsPerSecond,
            name + " malformed hint dropped previously accepted audio");
  }
}

void TestPauseRefreshFinalOrdering(bool queue_behind_blocker,
                                   bool close_during_pause) {
  TranscriptionWorker worker([] {}, [] {});
  std::promise<void> blocker_started;
  std::promise<void> release_blocker;
  const std::shared_future<void> blocker_release =
      release_blocker.get_future().share();
  if (queue_behind_blocker) {
    worker.Submit(
        [&blocker_started, blocker_release](const std::atomic<bool> &) {
          blocker_started.set_value();
          blocker_release.wait();
          return TranscriptionResult{};
        },
        NewCancelFlag(),
        [](std::optional<TranscriptionResult>, std::exception_ptr) {});
    blocker_started.get_future().wait();
  } else {
  }

  std::promise<void> pause_started;
  std::future<void> pause_started_future = pause_started.get_future();
  std::promise<void> release_pause;
  const std::shared_future<void> pause_release =
      release_pause.get_future().share();
  std::atomic<int> decoded_samples = 0;
  std::atomic<int> segment_events = 0;
  auto session = std::make_shared<RealtimeSession>(
      worker, PauseDecoder(&pause_started, pause_release, &decoded_samples),
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
  bool saw_pause_reuse = false;
  std::promise<void> final_finished;
  if (!queue_behind_blocker) {
    pause_started_future.wait();
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
          if (error != nullptr) {
            final_finished.set_exception(error);
          } else {
            final_finished.set_value();
          }
        });
  }
  if (queue_behind_blocker) {
    release_blocker.set_value();
  } else {
  }
  pause_started_future.wait();
  release_pause.set_value();
  if (close_during_pause) {
    std::promise<void> worker_drained;
    worker.Submit(
        [](const std::atomic<bool> &) { return TranscriptionResult{}; },
        NewCancelFlag(),
        [&worker_drained](std::optional<TranscriptionResult>,
                          std::exception_ptr error) {
          if (error != nullptr) {
            worker_drained.set_exception(error);
          } else {
            worker_drained.set_value();
          }
        });
    worker_drained.get_future().get();
    Require(decoded_samples.load() ==
                    800 * kSampleRate / kMillisecondsPerSecond &&
                segment_events.load() == 0,
            "close did not cancel the running pause refresh");
  } else {
    final_finished.get_future().get();
    Require(saw_pause_reuse,
            "Final ran before an admitted pause refresh updated reuse");
    Require(decoded_samples.load() ==
                800 * kSampleRate / kMillisecondsPerSecond,
            "pause refresh decoded the wrong audio endpoint");
  }
}

void TestFinalFenceBlocksPauseAndPreservesCadence() {
  TranscriptionWorker worker([] {}, [] {});
  std::atomic<int> decoded_samples = 0;
  std::promise<void> preview_sent;
  std::atomic<int> preview_count = 0;
  auto session = std::make_shared<RealtimeSession>(
      worker,
      [&decoded_samples](const std::vector<float> &samples,
                         const std::atomic<bool> &,
                         RefreshReuse *) -> Transcription {
        decoded_samples.store(static_cast<int>(samples.size()));
        return [](const std::atomic<bool> &) { return PreviewResult(); };
      },
      qwen3_asr::MakeRealtimeSettings(1000, 500, 30.0),
      [&preview_sent, &preview_count](const std::string &serialized) {
        const auto event = nlohmann::json::parse(serialized);
        if (event.value("type", std::string()) == "transcription.segment") {
          preview_count.fetch_add(1);
          preview_sent.set_value();
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
  preview_sent.get_future().wait();
  Require(decoded_samples.load() == 500 * kSampleRate / 1000 &&
              preview_count.load() == 1,
          "Final fence suppressed the ordinary preview path");
}

} // namespace

namespace qwen3_asr::tests {
void TestAdmissionPolicy();
void TestPauseRefreshRequiresSessionOptIn();
} // namespace qwen3_asr::tests

int main() {
  try {
    qwen3_asr::tests::TestAdmissionPolicy();
    TestInvalidHintsKeepAudioAndDoNotSchedule();
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
