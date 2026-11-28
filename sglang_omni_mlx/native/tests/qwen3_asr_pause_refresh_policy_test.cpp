// SPDX-License-Identifier: Apache-2.0
#include <stdexcept>
#include <string>

#include "qwen3_asr/realtime_refresh.h"

namespace qwen3_asr::tests {
namespace {

constexpr long kSampleRate = 16000;
constexpr long kMillisecondsPerSecond = 1000;
constexpr long kFirstRefreshSample = kSampleRate / 10;
constexpr long kRefreshIntervalSamples = kSampleRate;

void Require(bool condition, const std::string &message) {
  if (!condition) {
    throw std::runtime_error(message);
  } else {
  }
}

RefreshAdmissionInput AdmissionInput() {
  RefreshAdmissionInput input;
  input.audio_end_sample = kFirstRefreshSample;
  input.next_refresh_sample = kFirstRefreshSample;
  input.decode_interval_samples = kRefreshIntervalSamples;
  input.minimum_new_speech_samples = kSampleRate / 5;
  return input;
}

} // namespace

void TestAdmissionPolicy() {
  auto input = AdmissionInput();
  const auto expect_source = [&](RefreshSource source,
                                 const std::string &message) {
    Require(SelectRefreshAdmission(input).source == source, message);
  };
  expect_source(RefreshSource::kCadence,
                "initial cadence refresh was not admitted");

  input.silent = true;
  auto admission = SelectRefreshAdmission(input);
  Require(admission.source == RefreshSource::kNone &&
              admission.next_refresh_sample == input.next_refresh_sample,
          "leading silence moved the first cadence deadline");
  input.has_decoded = true;
  admission = SelectRefreshAdmission(input);
  Require(admission.source == RefreshSource::kNone &&
              admission.next_refresh_sample ==
                  input.audio_end_sample + input.decode_interval_samples,
          "post-preview silence did not move the cadence deadline");

  input = AdmissionInput();
  input.next_refresh_sample = 1400 * kSampleRate / kMillisecondsPerSecond;
  input.audio_end_sample = 800 * kSampleRate / kMillisecondsPerSecond;
  input.last_submitted_refresh_end_sample =
      400 * kSampleRate / kMillisecondsPerSecond;
  input.has_decoded = true;
  input.pause_hint = PauseRefreshHint{0, 600 * kSampleRate / 1000};
  expect_source(RefreshSource::kPause,
                "qualifying pause did not admit one early refresh");

  input.pause_hint_deferred = true;
  expect_source(RefreshSource::kDeferredPause,
                "a valid deferred pause did not retain its source");
  input.pause_hint_deferred = false;

  input.final_handoff = true;
  expect_source(RefreshSource::kNone, "Final handoff admitted a pause refresh");
  input.audio_end_sample = input.next_refresh_sample;
  expect_source(RefreshSource::kCadence, "Final handoff suppressed cadence");

  input = AdmissionInput();
  input.audio_end_sample = 800 * kSampleRate / kMillisecondsPerSecond;
  input.next_refresh_sample = 1400 * kSampleRate / kMillisecondsPerSecond;
  input.last_submitted_refresh_end_sample =
      400 * kSampleRate / kMillisecondsPerSecond;
  input.pause_hint =
      PauseRefreshHint{300 * kSampleRate / 1000, 590 * kSampleRate / 1000};
  admission = SelectRefreshAdmission(input);
  Require(admission.source == RefreshSource::kNone &&
              admission.disposition ==
                  PauseHintDisposition::kInsufficientNewSpeech,
          "idle insufficient-speech hint was not attributed by the selector");
  input.pause_hint_deferred = true;
  admission = SelectRefreshAdmission(input);
  Require(admission.source == RefreshSource::kNone &&
              admission.disposition ==
                  PauseHintDisposition::kInsufficientNewSpeech,
          "deferred insufficient-speech hint bypassed selector attribution");

  input.pause_hint = PauseRefreshHint{0, 600 * kSampleRate / 1000};
  input.last_submitted_refresh_end_sample = 600 * kSampleRate / 1000;
  admission = SelectRefreshAdmission(input);
  Require(admission.source == RefreshSource::kNone &&
              admission.disposition == PauseHintDisposition::kCovered,
          "idle covered hint was not attributed by the selector");
  input.pause_hint_deferred = true;
  admission = SelectRefreshAdmission(input);
  Require(admission.source == RefreshSource::kNone &&
              admission.disposition == PauseHintDisposition::kCovered,
          "deferred covered hint bypassed selector attribution");

  input = AdmissionInput();
  input.next_refresh_sample = 1400 * kSampleRate / 1000;
  input.audio_end_sample = 800 * kSampleRate / 1000;
  input.last_submitted_refresh_end_sample = 100 * kSampleRate / 1000;
  input.pause_hint =
      PauseRefreshHint{100 * kSampleRate / 1000, 600 * kSampleRate / 1000};
  input.refreshing = true;
  admission = SelectRefreshAdmission(input);
  Require(admission.source == RefreshSource::kNone &&
              admission.disposition == PauseHintDisposition::kRefreshBusy,
          "busy useful hint was not attributed by the selector");

  input.refreshing = false;
  input.cancelled = true;
  expect_source(RefreshSource::kNone,
                "cancelled session admitted a pause refresh");
  input.audio_end_sample = input.next_refresh_sample;
  expect_source(RefreshSource::kCadence, "cancellation changed cadence policy");

  input = AdmissionInput();
  input.finalizing = true;
  expect_source(RefreshSource::kNone,
                "realtime finalization admitted a refresh");
}

void TestDeferredHintState() {
  const PauseRefreshHint hint{300 * kSampleRate / 1000,
                              600 * kSampleRate / 1000};
  const auto state = [&](int segment_id, long speech_epoch,
                         long segment_start) {
    return CheckDeferredHintState(hint, 4, segment_id, 7, speech_epoch,
                                  segment_start);
  };
  Require(state(4, 7, 0) == DeferredHintState::kValid,
          "a current deferred hint did not remain eligible");
  Require(state(5, 7, 0) == DeferredHintState::kSegmentChanged,
          "a segment rollover kept a deferred hint alive");
  Require(state(4, 8, 0) == DeferredHintState::kSpeechResumed,
          "a speech epoch change kept a deferred hint alive");
  Require(state(4, 7, 400 * kSampleRate / 1000) == DeferredHintState::kInvalid,
          "a hint before the new segment remained eligible");
}

} // namespace qwen3_asr::tests
