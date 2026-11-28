// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <algorithm>
#include <cstddef>
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
  kSpeechResumed
};

inline DeferredHintState
CheckDeferredHintState(const PauseRefreshHint &hint, int hint_segment_id,
                       int segment_id, long hint_speech_epoch,
                       long speech_epoch, long segment_start_sample) {
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
  return DeferredHintState::kValid;
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

enum class PauseHintDisposition {
  kNone,
  kInvalid,
  kDuplicate,
  kDisabled,
  kInvalidDeferredMetadata,
  kRefreshBusy,
  kCovered,
  kInsufficientNewSpeech,
  kSupersededByCadence,
  kSilent,
  kFinalHandoff,
  kSegmentFinalization,
  kCancelled,
  kSpeechResumption,
  kSegmentRollover,
  kDeferred,
  kSuperseded,
  kAdmittedImmediate,
  kAdmittedDeferred,
  kSessionClose,
  kCount,
};

inline constexpr size_t kPauseHintDispositionCount =
    static_cast<size_t>(PauseHintDisposition::kCount);

inline PauseHintDisposition DeferredHintDisposition(DeferredHintState state) {
  switch (state) {
  case DeferredHintState::kInvalid:
    return PauseHintDisposition::kInvalidDeferredMetadata;
  case DeferredHintState::kSegmentChanged:
    return PauseHintDisposition::kSegmentRollover;
  case DeferredHintState::kSpeechResumed:
    return PauseHintDisposition::kSpeechResumption;
  case DeferredHintState::kValid:
    return PauseHintDisposition::kNone;
  }
  return PauseHintDisposition::kInvalidDeferredMetadata;
}

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
  PauseHintDisposition disposition = PauseHintDisposition::kNone;
  long end_sample = 0;
  long next_refresh_sample = 0;
};

inline PauseHintDisposition
PauseHintCoverageDisposition(const RefreshAdmissionInput &input) {
  if (input.pause_hint->speech_end_sample <=
      input.last_submitted_refresh_end_sample) {
    return PauseHintDisposition::kCovered;
  } else {
  }
  const long first_new_speech_sample = std::max(
      {input.segment_start_sample, input.last_submitted_refresh_end_sample,
       input.pause_hint->speech_start_sample});
  if (input.pause_hint->speech_end_sample - first_new_speech_sample <
      input.minimum_new_speech_samples) {
    return PauseHintDisposition::kInsufficientNewSpeech;
  } else {
  }
  return PauseHintDisposition::kNone;
}

inline RefreshAdmission
SelectRefreshAdmission(const RefreshAdmissionInput &input) {
  RefreshAdmission admission;
  admission.next_refresh_sample = input.next_refresh_sample;
  if (input.finalizing) {
    admission.disposition = input.pause_hint.has_value()
                                ? PauseHintDisposition::kSegmentFinalization
                                : PauseHintDisposition::kNone;
    return admission;
  }
  if (input.refreshing) {
    if (!input.pause_hint.has_value()) {
      return admission;
    }
    if (input.cancelled) {
      admission.disposition = PauseHintDisposition::kCancelled;
    } else if (input.final_handoff) {
      admission.disposition = PauseHintDisposition::kFinalHandoff;
    } else {
      admission.disposition = PauseHintCoverageDisposition(input);
      if (admission.disposition == PauseHintDisposition::kNone) {
        admission.disposition =
            input.audio_end_sample >= input.next_refresh_sample
                ? PauseHintDisposition::kSupersededByCadence
                : PauseHintDisposition::kRefreshBusy;
      } else {
      }
    }
    return admission;
  }

  const bool cadence_due = input.audio_end_sample >= input.next_refresh_sample;
  if (cadence_due) {
    if (input.pause_hint.has_value()) {
      admission.disposition = input.cancelled ? PauseHintDisposition::kCancelled
                              : input.final_handoff
                                  ? PauseHintDisposition::kFinalHandoff
                                  : PauseHintDisposition::kSupersededByCadence;
    }
    if (!input.silent) {
      admission.source = RefreshSource::kCadence;
      admission.end_sample = input.audio_end_sample;
      admission.next_refresh_sample =
          input.audio_end_sample + input.decode_interval_samples;
    } else if (input.has_decoded) {
      admission.next_refresh_sample =
          input.audio_end_sample + input.decode_interval_samples;
    }
    return admission;
  }
  if (!input.pause_hint.has_value()) {
    return admission;
  }
  if (input.cancelled) {
    admission.disposition = PauseHintDisposition::kCancelled;
    return admission;
  }
  if (input.final_handoff) {
    admission.disposition = PauseHintDisposition::kFinalHandoff;
    return admission;
  }
  if (input.silent) {
    admission.disposition = PauseHintDisposition::kSilent;
    return admission;
  }
  admission.disposition = PauseHintCoverageDisposition(input);
  if (admission.disposition != PauseHintDisposition::kNone) {
    return admission;
  }
  admission.source = input.pause_hint_deferred ? RefreshSource::kDeferredPause
                                               : RefreshSource::kPause;
  admission.disposition = input.pause_hint_deferred
                              ? PauseHintDisposition::kAdmittedDeferred
                              : PauseHintDisposition::kAdmittedImmediate;
  admission.end_sample = input.audio_end_sample;
  admission.next_refresh_sample =
      input.audio_end_sample + input.decode_interval_samples;
  return admission;
}

} // namespace qwen3_asr
