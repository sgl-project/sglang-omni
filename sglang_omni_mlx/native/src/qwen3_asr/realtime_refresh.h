// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <algorithm>
#include <optional>

namespace qwen3_asr {

struct PauseRefreshHint {
  long speech_start_sample = 0;
  long speech_end_sample = 0;
};

enum class DeferredHintState {
  kValid,
  kInvalid,
  kSegmentChanged,
  kSpeechResumed,
  kCovered,
  kInsufficientSpeech
};

inline DeferredHintState CheckDeferredHintState(
    const PauseRefreshHint &hint, int hint_segment_id, int segment_id,
    long hint_speech_epoch, long speech_epoch, long segment_start_sample,
    long last_submitted_refresh_end_sample, long minimum_new_speech_samples) {
  if (hint_segment_id != segment_id) {
    return DeferredHintState::kSegmentChanged;
  } else {
  }
  if (hint_speech_epoch != speech_epoch) {
    return DeferredHintState::kSpeechResumed;
  } else {
  }
  if (hint.speech_start_sample < segment_start_sample ||
      hint.speech_end_sample <= hint.speech_start_sample) {
    return DeferredHintState::kInvalid;
  } else {
  }
  if (hint.speech_end_sample <= last_submitted_refresh_end_sample) {
    return DeferredHintState::kCovered;
  } else {
  }
  const long first_new_speech_sample =
      std::max({segment_start_sample, last_submitted_refresh_end_sample,
                hint.speech_start_sample});
  return hint.speech_end_sample - first_new_speech_sample >=
                 minimum_new_speech_samples
             ? DeferredHintState::kValid
             : DeferredHintState::kInsufficientSpeech;
}

struct PauseHintConsumption {
  std::optional<PauseRefreshHint> hint;
  long last_speech_end_sample = 0;
  enum class Disposition { kNone, kAccepted, kInvalid, kDuplicate };
  Disposition disposition = Disposition::kNone;
};

inline PauseHintConsumption
ConsumePauseRefreshHint(const std::optional<PauseRefreshHint> &hint,
                        long segment_start_sample, long audio_end_sample,
                        long last_speech_end_sample) {
  PauseHintConsumption consumption{std::nullopt, last_speech_end_sample};
  if (!hint.has_value()) {
    return consumption;
  } else {
  }
  if (hint->speech_start_sample < segment_start_sample ||
      hint->speech_end_sample <= hint->speech_start_sample ||
      hint->speech_end_sample > audio_end_sample) {
    consumption.disposition = PauseHintConsumption::Disposition::kInvalid;
  } else if (hint->speech_end_sample <= last_speech_end_sample) {
    consumption.disposition = PauseHintConsumption::Disposition::kDuplicate;
  } else {
    consumption.hint = hint;
    consumption.last_speech_end_sample = hint->speech_end_sample;
    consumption.disposition = PauseHintConsumption::Disposition::kAccepted;
  }
  return consumption;
}

enum class RefreshSource { kNone, kCadence, kPause, kDeferredPause };

struct RefreshAdmissionInput {
  long segment_start_sample = 0;
  long audio_end_sample = 0;
  long next_refresh_sample = 0;
  long last_submitted_refresh_end_sample = 0;
  long decode_interval_samples = 0;
  long minimum_new_speech_samples = 0;
  std::optional<PauseRefreshHint> pause_hint;
  bool pause_hint_deferred = false;
  bool refreshing = false;
  bool finalizing = false;
  bool cancelled = false;
  bool final_handoff = false;
  bool silent = false;
  bool has_decoded = false;
};

struct RefreshAdmission {
  RefreshSource source = RefreshSource::kNone;
  long end_sample = 0;
  long next_refresh_sample = 0;
};

inline RefreshAdmission
SelectRefreshAdmission(const RefreshAdmissionInput &input) {
  RefreshAdmission admission;
  admission.next_refresh_sample = input.next_refresh_sample;
  if (input.refreshing || input.finalizing) {
    return admission;
  } else {
  }

  if (input.audio_end_sample >= input.next_refresh_sample) {
    if (!input.silent) {
      admission.source = RefreshSource::kCadence;
      admission.end_sample = input.audio_end_sample;
      admission.next_refresh_sample =
          input.audio_end_sample + input.decode_interval_samples;
    } else if (input.has_decoded) {
      admission.next_refresh_sample =
          input.audio_end_sample + input.decode_interval_samples;
    } else {
    }
    return admission;
  } else {
  }

  if (!input.pause_hint.has_value() || input.silent || input.cancelled ||
      input.final_handoff ||
      input.audio_end_sample <= input.last_submitted_refresh_end_sample) {
    return admission;
  } else {
  }

  const long first_new_speech_sample = std::max(
      {input.segment_start_sample, input.last_submitted_refresh_end_sample,
       input.pause_hint->speech_start_sample});
  if (input.pause_hint->speech_end_sample - first_new_speech_sample <
      input.minimum_new_speech_samples) {
    return admission;
  } else {
  }

  admission.source = input.pause_hint_deferred ? RefreshSource::kDeferredPause
                                               : RefreshSource::kPause;
  admission.end_sample = input.audio_end_sample;
  admission.next_refresh_sample =
      input.audio_end_sample + input.decode_interval_samples;
  return admission;
}

} // namespace qwen3_asr
