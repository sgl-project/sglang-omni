// SPDX-License-Identifier: Apache-2.0
#include "qwen3_asr/realtime_refresh.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string_view>
#include <vector>

#include "common/transcription.h"
#include "common/worker.h"

namespace {

constexpr long kSampleRate = 16000;
constexpr long kFirstRefreshSamples = kSampleRate / 10;
constexpr long kRefreshIntervalSamples = kSampleRate;
constexpr long kMinimumNewSpeechSamples = kSampleRate / 5;
constexpr long kSpeechEndSample = 600 * kSampleRate / 1000;

void Require(bool condition, std::string_view message) {
  if (!condition)
    throw std::runtime_error(std::string(message));
}

qwen3_asr::RefreshAdmissionInput AdmissionInput() {
  qwen3_asr::RefreshAdmissionInput input;
  input.next_refresh_sample = kFirstRefreshSamples;
  input.decode_interval_samples = kRefreshIntervalSamples;
  input.minimum_new_speech_samples = kMinimumNewSpeechSamples;
  return input;
}

qwen3_asr::RefreshAdmissionInput AfterFirstRefreshInput() {
  auto input = AdmissionInput();
  input.next_refresh_sample = kFirstRefreshSamples + kRefreshIntervalSamples;
  input.last_submitted_refresh_end_sample = kFirstRefreshSamples;
  return input;
}

void RequireSource(const qwen3_asr::RefreshAdmissionInput &input,
                   qwen3_asr::RefreshSource source, std::string_view message) {
  Require(qwen3_asr::SelectRefreshAdmission(input).source == source, message);
}

void TestRefreshAdmission() {
  using qwen3_asr::RefreshSource;

  auto input = AdmissionInput();
  input.audio_end_sample = kFirstRefreshSamples - 1;
  RequireSource(input, RefreshSource::kNone,
                "refresh started before the first sample deadline");
  input.audio_end_sample = kFirstRefreshSamples;
  auto admission = qwen3_asr::SelectRefreshAdmission(input);
  Require(admission.source == RefreshSource::kCadence &&
              admission.end_sample == kFirstRefreshSamples &&
              admission.next_refresh_sample ==
                  kFirstRefreshSamples + kRefreshIntervalSamples,
          "first refresh did not keep its sample cadence");

  input = AfterFirstRefreshInput();
  input.audio_end_sample = 800 * kSampleRate / 1000;
  input.pause_hint = qwen3_asr::PauseRefreshHint{0, kSpeechEndSample};
  admission = qwen3_asr::SelectRefreshAdmission(input);
  Require(admission.source == RefreshSource::kPause &&
              admission.end_sample == input.audio_end_sample &&
              admission.next_refresh_sample ==
                  input.audio_end_sample + kRefreshIntervalSamples,
          "pause did not replace the next cadence point");
  input.next_refresh_sample = admission.next_refresh_sample;
  input.last_pause_speech_end_sample = admission.last_pause_speech_end_sample;
  RequireSource(input, RefreshSource::kNone,
                "long silence repeated the same pause hint");

  input.last_submitted_refresh_end_sample = admission.end_sample;
  input.audio_end_sample = 1400 * kSampleRate / 1000;
  input.pause_hint = qwen3_asr::PauseRefreshHint{900 * kSampleRate / 1000,
                                                 1200 * kSampleRate / 1000};
  admission = qwen3_asr::SelectRefreshAdmission(input);
  Require(admission.source == RefreshSource::kPause,
          "resumed speech did not produce its own eligible pause");
  input.last_pause_speech_end_sample = admission.last_pause_speech_end_sample;
  RequireSource(input, RefreshSource::kNone,
                "resumed speech pause was admitted more than once");

  input = AfterFirstRefreshInput();
  input.audio_end_sample = 800 * kSampleRate / 1000;
  input.last_submitted_refresh_end_sample = 400 * kSampleRate / 1000;
  input.pause_hint = qwen3_asr::PauseRefreshHint{300 * kSampleRate / 1000,
                                                 590 * kSampleRate / 1000};
  RequireSource(input, RefreshSource::kNone,
                "insufficient new speech was admitted");

  input = AfterFirstRefreshInput();
  input.audio_end_sample = 800 * kSampleRate / 1000;
  input.pause_hint = qwen3_asr::PauseRefreshHint{0, kSpeechEndSample};
  input.refreshing = true;
  admission = qwen3_asr::SelectRefreshAdmission(input);
  Require(admission.source == RefreshSource::kNone &&
              admission.last_pause_speech_end_sample == kSpeechEndSample,
          "active refresh queued or deferred a pause hint");
  input.refreshing = false;
  input.last_pause_speech_end_sample = admission.last_pause_speech_end_sample;
  RequireSource(input, RefreshSource::kNone,
                "a skipped pause hint was retried after refresh completion");

  input = AfterFirstRefreshInput();
  input.audio_end_sample = 900 * kSampleRate / 1000;
  RequireSource(input, RefreshSource::kNone,
                "continuous speech refreshed before cadence");
  input.pause_hint = qwen3_asr::PauseRefreshHint{200 * kSampleRate / 1000,
                                                 300 * kSampleRate / 1000};
  RequireSource(input, RefreshSource::kNone,
                "a short hesitation refreshed early");
  input.pause_hint = qwen3_asr::PauseRefreshHint{500 * kSampleRate / 1000,
                                                 600 * kSampleRate / 1000};
  RequireSource(input, RefreshSource::kNone,
                "repeated short hesitations amplified refresh work");

  input = AfterFirstRefreshInput();
  input.audio_end_sample = 1090 * kSampleRate / 1000;
  input.pause_hint = qwen3_asr::PauseRefreshHint{0, 1050 * kSampleRate / 1000};
  admission = qwen3_asr::SelectRefreshAdmission(input);
  Require(admission.source == RefreshSource::kPause &&
              admission.next_refresh_sample ==
                  input.audio_end_sample + kRefreshIntervalSamples,
          "early refresh did not reschedule from its sample endpoint");
  input.audio_end_sample = 1200 * kSampleRate / 1000;
  RequireSource(input, RefreshSource::kCadence,
                "due cadence and pause produced more than one decode");

  input = AdmissionInput();
  input.audio_end_sample = 800 * kSampleRate / 1000;
  input.pause_hint = qwen3_asr::PauseRefreshHint{0, kSpeechEndSample};
  input.finalizing = true;
  RequireSource(input, RefreshSource::kNone,
                "finalizing session admitted a pause refresh");
  input.finalizing = false;
  input.cancelled = true;
  RequireSource(input, RefreshSource::kNone,
                "cancelled session admitted a pause refresh");

  input = AfterFirstRefreshInput();
  input.audio_end_sample = 800 * kSampleRate / 1000;
  input.pause_hint = qwen3_asr::PauseRefreshHint{0, 801 * kSampleRate / 1000};
  RequireSource(input, RefreshSource::kNone,
                "future speech endpoint was admitted");
  input.segment_start_sample = 30 * kSampleRate;
  input.last_submitted_refresh_end_sample = 30 * kSampleRate;
  input.next_refresh_sample = 31 * kSampleRate;
  input.audio_end_sample = 30500 * kSampleRate / 1000;
  input.pause_hint = qwen3_asr::PauseRefreshHint{0, kSpeechEndSample};
  RequireSource(input, RefreshSource::kNone,
                "speech from before the segment boundary was admitted");

  input = AdmissionInput();
  input.audio_end_sample = kFirstRefreshSamples;
  input.silent = true;
  admission = qwen3_asr::SelectRefreshAdmission(input);
  Require(admission.source == RefreshSource::kNone &&
              admission.next_refresh_sample == kFirstRefreshSamples,
          "leading silence advanced the initial cadence");
  input.has_decoded = true;
  Require(qwen3_asr::SelectRefreshAdmission(input).next_refresh_sample ==
              input.audio_end_sample + kRefreshIntervalSamples,
          "post-decode silence did not advance cadence as before");
}

struct SyntheticRun {
  int refresh_count = 0;
  int pause_refresh_count = 0;
  int normal_refresh_count = 0;
  int last_submitted_ms = 0;
  int last_completed_ms = 0;
  int last_completed_wall_ms = -1;
  int pause_refresh_completion_wall_ms = -1;
  int refresh_completion_to_stop_ms = 0;
  int final_queue_wait_ms = 0;
  int unprocessed_submitted_speech_ms = 0;
  int unprocessed_completed_speech_ms = 0;
  bool speech_preview_before_headstart = false;
};

void PrintSyntheticArm(std::string_view arm, const SyntheticRun &result,
                       const SyntheticRun &original) {
  std::cout << " arm=" << arm << " submitted_ms=" << result.last_submitted_ms
            << " completed_ms=" << result.last_completed_ms
            << " completion_to_stop_ms=" << result.refresh_completion_to_stop_ms
            << " refreshes=" << result.refresh_count << " replaced="
            << std::max(0, original.normal_refresh_count -
                               result.normal_refresh_count)
            << " additional="
            << std::max(0, result.refresh_count - original.refresh_count)
            << " unprocessed_submitted_ms="
            << result.unprocessed_submitted_speech_ms
            << " unprocessed_completed_ms="
            << result.unprocessed_completed_speech_ms
            << " final_wait_ms=" << result.final_queue_wait_ms
            << " headstart_preview=" << result.speech_preview_before_headstart;
}

class SyntheticTimeline {
public:
  SyntheticTimeline(int pause_ms, int packet_ms, int inference_ms)
      : pause_ms_(pause_ms), packet_ms_(packet_ms),
        inference_ms_(inference_ms) {}

  SyntheticRun Run(int stop_after_speech_ms) {
    const int stop_ms = 600 + stop_after_speech_ms;
    for (int now_ms = std::min(packet_ms_, stop_ms);;
         now_ms = std::min(now_ms + packet_ms_, stop_ms)) {
      Complete(now_ms);
      const long audio_end_sample = now_ms * kSampleRate / 1000;
      std::optional<qwen3_asr::PauseRefreshHint> pause_hint;
      const int pause_ready_ms = 600 + pause_ms_;
      if (pause_ms_ > 0 && !pause_hint_sent_ && now_ms >= pause_ready_ms) {
        pause_hint = qwen3_asr::PauseRefreshHint{0, kSpeechEndSample};
        pause_hint_sent_ = true;
      }
      Admit(now_ms, audio_end_sample, pause_hint);
      if (now_ms == stop_ms) {
        break;
      }
    }
    Complete(stop_ms);
    if (active_) {
      result_.final_queue_wait_ms = completion_ms_ - stop_ms;
      result_.refresh_completion_to_stop_ms = stop_ms - completion_ms_;
    } else if (result_.last_completed_wall_ms >= 0) {
      result_.refresh_completion_to_stop_ms =
          stop_ms - result_.last_completed_wall_ms;
    } else {
    }
    int pause_preview_completion_ms = result_.pause_refresh_completion_wall_ms;
    if (active_ && active_is_pause_ &&
        (pause_preview_completion_ms < 0 ||
         completion_ms_ < pause_preview_completion_ms)) {
      pause_preview_completion_ms = completion_ms_;
    } else {
    }
    result_.speech_preview_before_headstart =
        pause_preview_completion_ms >= 0 &&
        pause_preview_completion_ms < stop_ms + 500;
    result_.unprocessed_submitted_speech_ms =
        std::max(0, 600 - result_.last_submitted_ms);
    result_.unprocessed_completed_speech_ms =
        std::max(0, 600 - result_.last_completed_ms);
    return result_;
  }

private:
  void Complete(int now_ms) {
    if (!active_ || completion_ms_ > now_ms) {
      return;
    }
    active_ = false;
    result_.last_completed_ms = active_endpoint_ms_;
    result_.last_completed_wall_ms = completion_ms_;
    if (active_is_pause_ && result_.pause_refresh_completion_wall_ms < 0) {
      result_.pause_refresh_completion_wall_ms = completion_ms_;
    } else {
    }
    Admit(now_ms, audio_end_sample_, std::nullopt);
  }

  void Admit(int now_ms, long audio_end_sample,
             std::optional<qwen3_asr::PauseRefreshHint> pause_hint) {
    auto input = AdmissionInput();
    input.audio_end_sample = audio_end_sample;
    input.next_refresh_sample = next_refresh_sample_;
    input.last_submitted_refresh_end_sample = last_submitted_sample_;
    input.last_pause_speech_end_sample = last_pause_speech_end_sample_;
    input.pause_hint = pause_hint;
    input.refreshing = active_;
    const auto admission = qwen3_asr::SelectRefreshAdmission(input);
    last_pause_speech_end_sample_ = admission.last_pause_speech_end_sample;
    next_refresh_sample_ = admission.next_refresh_sample;
    audio_end_sample_ = audio_end_sample;
    if (admission.source == qwen3_asr::RefreshSource::kNone) {
      return;
    }
    active_ = true;
    active_is_pause_ = admission.source == qwen3_asr::RefreshSource::kPause;
    completion_ms_ = now_ms + inference_ms_;
    active_endpoint_ms_ =
        static_cast<int>(admission.end_sample * 1000 / kSampleRate);
    result_.speech_preview_before_headstart |=
        active_endpoint_ms_ >= 600 && completion_ms_ < 1100;
    last_submitted_sample_ = admission.end_sample;
    result_.last_submitted_ms = active_endpoint_ms_;
    ++result_.refresh_count;
    if (admission.source == qwen3_asr::RefreshSource::kPause) {
      ++result_.pause_refresh_count;
    } else {
      ++result_.normal_refresh_count;
    }
  }

  const int pause_ms_;
  const int packet_ms_;
  const int inference_ms_;
  int completion_ms_ = 0;
  int active_endpoint_ms_ = 0;
  bool active_is_pause_ = false;
  long audio_end_sample_ = 0;
  long next_refresh_sample_ = kFirstRefreshSamples;
  long last_submitted_sample_ = 0;
  long last_pause_speech_end_sample_ = 0;
  bool active_ = false;
  bool pause_hint_sent_ = false;
  SyntheticRun result_;
};

void TestPairedSyntheticExperiment() {
  for (const int inference_ms : {200, 400}) {
    for (const int packet_ms : {100, 200}) {
      for (const int stop_after_speech_ms : {0, 100, 300, 500, 800}) {
        const SyntheticRun original =
            SyntheticTimeline(0, packet_ms, inference_ms)
                .Run(stop_after_speech_ms);
        const SyntheticRun disabled =
            SyntheticTimeline(0, packet_ms, inference_ms)
                .Run(stop_after_speech_ms);
        const SyntheticRun arm_200 =
            SyntheticTimeline(200, packet_ms, inference_ms)
                .Run(stop_after_speech_ms);
        const SyntheticRun arm_300 =
            SyntheticTimeline(300, packet_ms, inference_ms)
                .Run(stop_after_speech_ms);
        Require(original.refresh_count == disabled.refresh_count &&
                    original.last_submitted_ms == disabled.last_submitted_ms &&
                    original.last_completed_ms == disabled.last_completed_ms,
                "disabled candidate changed the original schedule");
        Require(arm_200.pause_refresh_count <= 1 &&
                    arm_300.pause_refresh_count <= 1,
                "one speech pause caused repeated early refreshes");
        Require(arm_200.last_submitted_ms <= 600 + stop_after_speech_ms &&
                    arm_300.last_submitted_ms <= 600 + stop_after_speech_ms,
                "synthetic inference submitted future audio");
        std::cout << "packet_ms=" << packet_ms
                  << " inference_ms=" << inference_ms
                  << " stop_after_speech_ms=" << stop_after_speech_ms;
        PrintSyntheticArm("A", original, original);
        PrintSyntheticArm("B", disabled, original);
        PrintSyntheticArm("C", arm_200, original);
        PrintSyntheticArm("D", arm_300, original);
        std::cout << '\n';
      }
    }
  }
}

void TestFinalAndPauseRefreshKeepWorkerFifo() {
  using namespace std::chrono_literals;
  constexpr int kPauseRefreshJob = 1;
  constexpr int kFinalJob = 2;
  for (const bool pause_refresh_first : {true, false}) {
    omni_server::TranscriptionWorker worker([] {}, [] {});
    std::mutex mutex;
    std::condition_variable changed;
    std::vector<int> started_jobs;
    std::vector<int> completed_jobs;
    bool release_first_job = false;
    const auto submit_job = [&](int job_id, bool should_block) {
      const omni_server::Transcription transcribe =
          [&, job_id, should_block](const std::atomic<bool> &) {
            std::unique_lock<std::mutex> lock(mutex);
            started_jobs.push_back(job_id);
            changed.notify_all();
            if (should_block) {
              changed.wait(lock, [&] { return release_first_job; });
            }
            return omni_common::TranscriptionResult{};
          };
      worker.Submit(transcribe, omni_server::NewCancelFlag(),
                    [&, job_id](std::optional<omni_common::TranscriptionResult>,
                                std::exception_ptr) {
                      std::lock_guard<std::mutex> lock(mutex);
                      completed_jobs.push_back(job_id);
                      changed.notify_all();
                    });
    };

    const int first_job = pause_refresh_first ? kPauseRefreshJob : kFinalJob;
    const int second_job = pause_refresh_first ? kFinalJob : kPauseRefreshJob;
    submit_job(first_job, true);
    {
      std::unique_lock<std::mutex> lock(mutex);
      Require(
          changed.wait_for(lock, 10s, [&] { return started_jobs.size() == 1; }),
          "the first worker job did not start");
    }
    submit_job(second_job, false);
    const auto states = worker.RequestStates();
    Require(states.contains("queued") && states.at("queued") == 1,
            "the second worker job was not queued behind the first");
    {
      std::lock_guard<std::mutex> lock(mutex);
      release_first_job = true;
    }
    changed.notify_all();
    {
      std::unique_lock<std::mutex> lock(mutex);
      Require(changed.wait_for(lock, 10s,
                               [&] { return completed_jobs.size() == 2; }),
              "the worker jobs did not finish");
    }
    Require(started_jobs == std::vector<int>{first_job, second_job} &&
                completed_jobs == std::vector<int>{first_job, second_job},
            "Final and pause refresh did not preserve FIFO order");
  }
}

} // namespace

int main() {
  try {
    TestRefreshAdmission();
    TestPairedSyntheticExperiment();
    TestFinalAndPauseRefreshKeepWorkerFifo();
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  return 0;
}
