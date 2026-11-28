// SPDX-License-Identifier: Apache-2.0
#include <algorithm>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

#include "qwen3_asr/realtime_refresh.h"

namespace {

constexpr long kRate = 16000;
constexpr long kFirstRefresh = kRate / 10;
constexpr long kInterval = kRate;
constexpr long kMinimumSpeech = kRate / 5;
constexpr int kOnsetMilliseconds = 60;

struct Speech {
  int start_ms;
  int end_ms;
};

struct Trace {
  std::string name;
  std::vector<Speech> speech;
  int audio_ms;
  bool buffered_burst = false;
  int segment_ms = 30000;
  int stop_delay_ms = 0;
};

struct Packet {
  int wall_ms;
  long end_sample;
};

struct HintEvent {
  qwen3_asr::PauseRefreshHint hint;
  int notification_ms;
  long speech_epoch;
  int segment_id;
};

struct RefreshJob {
  qwen3_asr::RefreshSource source;
  long end_sample;
  int admission_ms;
  int start_ms;
  int completion_ms;
  int hint_notification_ms = -1;
};

struct Transition {
  int wall_ms;
  long audio_end_sample;
  long segment_start_sample;
  long next_refresh_sample;
  qwen3_asr::RefreshSource source;
  bool refreshing;
  bool has_decoded;
  bool operator==(const Transition &) const = default;
};

struct Metrics {
  int cadence_at_stop = 0;
  int pause_at_stop = 0;
  int completed_at_stop = 0;
  int cadence_total = 0;
  int pause_total = 0;
  int pause_skipped_while_refreshing = 0;
  int hints_received = 0;
  int hints_valid = 0;
  int hints_invalid_or_duplicate = 0;
  int hints_rejected_busy = 0;
  int hints_rejected_covered = 0;
  int hints_rejected_insufficient = 0;
  int hints_deferred = 0;
  int deferred_superseded = 0;
  int deferred_admitted = 0;
  int deferred_invalidated_resume = 0;
  int deferred_invalidated_rollover = 0;
  int deferred_invalidated_final = 0;
  int hints_superseded_by_cadence = 0;
  int submitted_coverage_at_stop_ms = 0;
  int completed_coverage_at_stop_ms = 0;
  int residual_refresh_ms = 0;
  int final_queue_wait_ms = 0;
  int refresh_busy_ms = 0;
  int worker_occupancy_ms = 0;
  int stop_ms = 0;
  int pause_trigger_ms = 0;
  int final_completion_ms = 0;
  bool speech_covered_by_completed_preview_at_stop = false;
  bool preview_ready_before_stop = false;
  bool ready_before_pause_trigger = false;
  bool ready_before_first_headstart = false;
  int first_headstart_opportunity_ms = 0;
  int pause_notification_ms = -1;
  int pause_admission_ms = -1;
  int pause_start_ms = -1;
  int pause_completion_ms = -1;
  std::vector<RefreshJob> submitted;
  std::vector<RefreshJob> completed;
  std::vector<Transition> transitions;
};

long Samples(int milliseconds) { return milliseconds * kRate / 1000; }

std::vector<Packet> MakePackets(const Trace &trace, int packet_ms) {
  std::vector<Packet> packets;
  for (int end_ms = std::min(packet_ms, trace.audio_ms);;
       end_ms = std::min(end_ms + packet_ms, trace.audio_ms)) {
    packets.push_back({trace.buffered_burst ? 0 : end_ms, Samples(end_ms)});
    if (end_ms == trace.audio_ms) {
      return packets;
    }
  }
}

bool IsSilent(const Trace &trace, long segment_start_sample,
              long audio_end_sample) {
  for (const Speech &speech : trace.speech) {
    const long speech_start_sample = Samples(speech.start_ms);
    const long speech_end_sample = Samples(speech.end_ms);
    if (speech_start_sample < audio_end_sample &&
        speech_end_sample > segment_start_sample) {
      return false;
    }
  }
  return true;
}

std::optional<HintEvent> FindPauseHint(const Trace &trace, int packet_wall_ms,
                                       long segment_start_sample,
                                       long audio_end_sample, int pause_ms,
                                       long last_pause_speech_end_sample,
                                       long speech_epoch, int segment_id) {
  std::optional<HintEvent> latest;
  for (const Speech &speech : trace.speech) {
    const long speech_start_sample = Samples(speech.start_ms);
    const long speech_end_sample = Samples(speech.end_ms);
    if (speech_start_sample < segment_start_sample ||
        speech_end_sample + Samples(pause_ms) > audio_end_sample ||
        speech_end_sample <= last_pause_speech_end_sample) {
      continue;
    }
    bool detector_speaking = false;
    for (const Speech &current : trace.speech) {
      const long onset_sample = Samples(current.start_ms + kOnsetMilliseconds);
      const long close_sample = Samples(current.end_ms + pause_ms);
      if (audio_end_sample >= onset_sample && audio_end_sample < close_sample) {
        detector_speaking = true;
      }
    }
    if (!detector_speaking) {
      latest = HintEvent{{speech_start_sample, speech_end_sample},
                         packet_wall_ms,
                         speech_epoch,
                         segment_id};
    } else {
      latest.reset();
    }
  }
  return latest;
}

qwen3_asr::RefreshAdmission
OriginalCadenceReference(const qwen3_asr::RefreshAdmissionInput &input) {
  qwen3_asr::RefreshAdmission admission;
  admission.next_refresh_sample = input.next_refresh_sample;
  if (input.refreshing || input.finalizing ||
      input.audio_end_sample < input.next_refresh_sample) {
    return admission;
  }
  if (!(input.silent && !input.has_decoded)) {
    admission.next_refresh_sample =
        input.audio_end_sample + input.decode_interval_samples;
  }
  if (input.silent) {
    return admission;
  }
  admission.source = qwen3_asr::RefreshSource::kCadence;
  admission.end_sample = input.audio_end_sample;
  return admission;
}

class Timeline {
public:
  Timeline(const Trace &trace, int packet_ms, int inference_ms, int pause_ms)
      : trace_(trace), packets_(MakePackets(trace, packet_ms)),
        inference_ms_(inference_ms), pause_ms_(pause_ms),
        segment_ms_(trace.segment_ms) {
    next_refresh_sample_ = kFirstRefresh;
  }

  Timeline(const Trace &trace, int packet_ms, int inference_ms, int pause_ms,
           bool use_original_reference, bool defer_busy_pause = false)
      : trace_(trace), packets_(MakePackets(trace, packet_ms)),
        inference_ms_(inference_ms), pause_ms_(pause_ms),
        segment_ms_(trace.segment_ms), arm_original_(use_original_reference),
        defer_busy_pause_(defer_busy_pause) {
    next_refresh_sample_ = kFirstRefresh;
  }

  Metrics Run() {
    for (const Packet &packet : packets_) {
      CompleteThrough(packet.wall_ms);
      audio_end_sample_ = packet.end_sample;
      RolloverIfNeeded();
      InvalidateResumedHint();
      std::optional<HintEvent> hint;
      if (pause_ms_ > 0) {
        hint = FindPauseHint(trace_, packet.wall_ms, segment_start_sample_,
                             audio_end_sample_, pause_ms_,
                             last_pause_speech_end_sample_,
                             SpeechEpochAt(audio_end_sample_), segment_id_);
      }
      TryAdmit(packet.wall_ms, hint);
    }
    stop_ms_ = packets_.back().wall_ms + trace_.stop_delay_ms;
    CompleteThrough(stop_ms_);
    RecordAtStop();
    EnqueueFinal();
    metrics_.ready_before_pause_trigger = HasCompletedPreview(
        trace_.speech.front().end_ms, metrics_.pause_trigger_ms, stop_ms_);
    return metrics_;
  }

private:
  qwen3_asr::RefreshAdmissionInput
  Input(const std::optional<qwen3_asr::PauseRefreshHint> &hint) const {
    return {.segment_start_sample = segment_start_sample_,
            .audio_end_sample = audio_end_sample_,
            .next_refresh_sample = next_refresh_sample_,
            .last_submitted_refresh_end_sample = last_submitted_sample_,
            .decode_interval_samples = kInterval,
            .minimum_new_speech_samples = kMinimumSpeech,
            .pause_hint = hint,
            .refreshing = active_.has_value(),
            .finalizing = finalizing_,
            .cancelled = false,
            .final_handoff = final_handoff_,
            .silent =
                IsSilent(trace_, segment_start_sample_, audio_end_sample_),
            .has_decoded = decode_count_ > 0};
  }

  qwen3_asr::PauseHintConsumption
  ConsumeHint(const std::optional<HintEvent> &hint) {
    if (!hint.has_value()) {
      return {std::nullopt, last_pause_speech_end_sample_};
    }
    const auto consumed = qwen3_asr::ConsumePauseRefreshHint(
        hint->hint, segment_start_sample_, audio_end_sample_,
        last_pause_speech_end_sample_);
    last_pause_speech_end_sample_ = consumed.last_speech_end_sample;
    ++metrics_.hints_received;
    if (consumed.hint.has_value()) {
      ++metrics_.hints_valid;
    } else {
      ++metrics_.hints_invalid_or_duplicate;
    }
    return consumed;
  }

  void TryAdmit(int wall_ms, const std::optional<HintEvent> &hint) {
    const auto consumed = ConsumeHint(hint);
    std::optional<HintEvent> candidate =
        consumed.hint.has_value() ? hint : std::nullopt;
    bool candidate_is_deferred = false;
    if (active_.has_value() && candidate.has_value()) {
      const auto state = qwen3_asr::CheckDeferredHintState(
          candidate->hint, candidate->segment_id, segment_id_,
          candidate->speech_epoch, SpeechEpochAt(audio_end_sample_),
          segment_start_sample_);
      if (state != qwen3_asr::DeferredHintState::kValid) {
        candidate.reset();
      } else {
      }
    } else if (!candidate.has_value() && !active_.has_value() &&
               pending_hint_.has_value()) {
      const auto state = qwen3_asr::CheckDeferredHintState(
          pending_hint_->hint, pending_hint_->segment_id, segment_id_,
          pending_hint_->speech_epoch, SpeechEpochAt(audio_end_sample_),
          segment_start_sample_);
      if (state == qwen3_asr::DeferredHintState::kValid) {
        candidate = pending_hint_;
        candidate_is_deferred = true;
      } else {
        pending_hint_.reset();
      }
    }
    auto input =
        Input(candidate.has_value()
                  ? std::optional<qwen3_asr::PauseRefreshHint>(candidate->hint)
                  : std::nullopt);
    input.pause_hint_deferred = candidate_is_deferred;
    const auto admission = arm_original_
                               ? OriginalCadenceReference(input)
                               : qwen3_asr::SelectRefreshAdmission(input);
    next_refresh_sample_ = admission.next_refresh_sample;
    if (candidate.has_value()) {
      if (admission.disposition == qwen3_asr::PauseHintDisposition::kCovered) {
        ++metrics_.hints_rejected_covered;
        if (candidate_is_deferred) {
          pending_hint_.reset();
        } else {
        }
      } else if (admission.disposition ==
                 qwen3_asr::PauseHintDisposition::kInsufficientNewSpeech) {
        ++metrics_.hints_rejected_insufficient;
        if (candidate_is_deferred) {
          pending_hint_.reset();
        } else {
        }
      } else if (admission.disposition ==
                 qwen3_asr::PauseHintDisposition::kRefreshBusy) {
        if (defer_busy_pause_) {
          if (pending_hint_.has_value()) {
            ++metrics_.deferred_superseded;
          } else {
          }
          pending_hint_ = candidate;
          ++metrics_.hints_deferred;
        } else {
          ++metrics_.hints_rejected_busy;
          ++metrics_.pause_skipped_while_refreshing;
        }
      } else if (admission.disposition ==
                 qwen3_asr::PauseHintDisposition::kSupersededByCadence) {
        ++metrics_.hints_superseded_by_cadence;
        if (candidate_is_deferred) {
          pending_hint_.reset();
        } else {
        }
      } else {
      }
    } else {
    }
    metrics_.transitions.push_back({wall_ms, audio_end_sample_,
                                    segment_start_sample_, next_refresh_sample_,
                                    admission.source, active_.has_value(),
                                    decode_count_ > 0});
    if (admission.source == qwen3_asr::RefreshSource::kNone) {
      return;
    }
    if (admission.source == qwen3_asr::RefreshSource::kDeferredPause) {
      ++metrics_.deferred_admitted;
      pending_hint_.reset();
    }
    last_submitted_sample_ = admission.end_sample;
    ++decode_count_;
    if (admission.source == qwen3_asr::RefreshSource::kCadence) {
      ++metrics_.cadence_total;
    } else {
      ++metrics_.pause_total;
    }
    const int start_ms = final_handoff_ ? final_completion_ms_ : wall_ms;
    active_ =
        RefreshJob{admission.source,
                   admission.end_sample,
                   wall_ms,
                   start_ms,
                   start_ms + inference_ms_,
                   candidate.has_value() ? candidate->notification_ms : -1};
    metrics_.refresh_busy_ms += inference_ms_;
    metrics_.submitted.push_back(*active_);
    if ((admission.source == qwen3_asr::RefreshSource::kPause ||
         admission.source == qwen3_asr::RefreshSource::kDeferredPause) &&
        metrics_.pause_notification_ms < 0) {
      metrics_.pause_notification_ms = candidate->notification_ms;
      metrics_.pause_admission_ms = wall_ms;
      metrics_.pause_start_ms = start_ms;
      metrics_.pause_completion_ms = start_ms + inference_ms_;
    }
  }

  void CompleteThrough(int wall_ms) {
    while (active_.has_value() && active_->completion_ms <= wall_ms) {
      CompleteActive();
    }
  }

  void CompleteActive() {
    const RefreshJob finished = *active_;
    active_.reset();
    metrics_.completed.push_back(finished);
    TryAdmit(finished.completion_ms, std::nullopt);
  }

  long SpeechEpochAt(long audio_end_sample) const {
    long epoch = 0;
    for (const Speech &speech : trace_.speech) {
      if (Samples(speech.start_ms + kOnsetMilliseconds) <= audio_end_sample) {
        ++epoch;
      }
    }
    return epoch;
  }

  void InvalidateResumedHint() {
    if (pending_hint_.has_value() &&
        pending_hint_->speech_epoch != SpeechEpochAt(audio_end_sample_)) {
      ++metrics_.deferred_invalidated_resume;
      pending_hint_.reset();
    }
  }

  void RolloverIfNeeded() {
    if (segment_ms_ <= 0 ||
        audio_end_sample_ < segment_start_sample_ + Samples(segment_ms_)) {
      return;
    }
    const long completed_segments =
        (audio_end_sample_ - segment_start_sample_) / Samples(segment_ms_);
    segment_start_sample_ += completed_segments * Samples(segment_ms_);
    if (pending_hint_.has_value()) {
      ++metrics_.deferred_invalidated_rollover;
      pending_hint_.reset();
    }
    ++segment_id_;
    next_refresh_sample_ = segment_start_sample_ + kFirstRefresh;
    last_submitted_sample_ = segment_start_sample_;
    last_pause_speech_end_sample_ = segment_start_sample_;
    decode_count_ = 0;
  }

  int TargetSpeechEndAtStop() const {
    int target_ms = 0;
    for (const Speech &speech : trace_.speech) {
      if (Samples(speech.start_ms) < audio_end_sample_) {
        target_ms = std::max(
            target_ms,
            std::min(speech.end_ms,
                     static_cast<int>(audio_end_sample_ * 1000 / kRate)));
      }
    }
    return target_ms;
  }

  int DeliveredWallForSample(int target_sample) const {
    for (const Packet &packet : packets_) {
      if (packet.end_sample >= target_sample) {
        return packet.wall_ms;
      }
    }
    const int first_end_wall =
        DeliveredWallForSample(Samples(trace_.speech.front().end_ms));
    const long remaining_samples =
        std::max(0L, target_sample - Samples(trace_.speech.front().end_ms));
    return first_end_wall + static_cast<int>(remaining_samples * 1000 / kRate);
  }

  bool HasCompletedPreview(int endpoint_ms, int before_wall_ms,
                           int admission_before_wall_ms) const {
    return std::any_of(metrics_.completed.begin(), metrics_.completed.end(),
                       [&](const RefreshJob &job) {
                         return job.end_sample >= Samples(endpoint_ms) &&
                                job.admission_ms <= admission_before_wall_ms &&
                                job.completion_ms <= before_wall_ms;
                       });
  }

  void RecordAtStop() {
    metrics_.stop_ms = stop_ms_;
    const int target_speech_end_ms = TargetSpeechEndAtStop();
    for (const RefreshJob &job : metrics_.submitted) {
      if (job.admission_ms <= stop_ms_) {
        metrics_.submitted_coverage_at_stop_ms =
            std::max(metrics_.submitted_coverage_at_stop_ms,
                     static_cast<int>(job.end_sample * 1000 / kRate));
      }
    }
    for (const RefreshJob &job : metrics_.completed) {
      if (job.completion_ms <= stop_ms_) {
        ++metrics_.completed_at_stop;
        metrics_.completed_coverage_at_stop_ms =
            std::max(metrics_.completed_coverage_at_stop_ms,
                     static_cast<int>(job.end_sample * 1000 / kRate));
      }
    }
    for (const RefreshJob &job : metrics_.submitted) {
      if (job.admission_ms <= stop_ms_) {
        if (job.source == qwen3_asr::RefreshSource::kCadence) {
          ++metrics_.cadence_at_stop;
        } else {
          ++metrics_.pause_at_stop;
        }
      }
    }
    metrics_.residual_refresh_ms =
        active_.has_value() ? active_->completion_ms - stop_ms_ : 0;
    metrics_.speech_covered_by_completed_preview_at_stop =
        metrics_.completed_coverage_at_stop_ms >= target_speech_end_ms;
    metrics_.preview_ready_before_stop =
        HasCompletedPreview(target_speech_end_ms, stop_ms_, stop_ms_);
    const int first_speech_end_ms = trace_.speech.front().end_ms;
    metrics_.pause_trigger_ms =
        DeliveredWallForSample(Samples(first_speech_end_ms + 500));
    const int first_headstart_wall_ms =
        std::min(stop_ms_, metrics_.pause_trigger_ms);
    metrics_.first_headstart_opportunity_ms = first_headstart_wall_ms;
    metrics_.ready_before_first_headstart = HasCompletedPreview(
        first_speech_end_ms, first_headstart_wall_ms, stop_ms_);
  }

  void EnqueueFinal() {
    final_handoff_ = true;
    if (pending_hint_.has_value()) {
      ++metrics_.deferred_invalidated_final;
      pending_hint_.reset();
    }
    const bool refresh_active_at_stop = active_.has_value();
    const int final_start_ms =
        refresh_active_at_stop ? active_->completion_ms : stop_ms_;
    metrics_.final_queue_wait_ms = final_start_ms - stop_ms_;
    final_completion_ms_ = final_start_ms + inference_ms_;
    metrics_.final_completion_ms = final_completion_ms_;
    metrics_.worker_occupancy_ms = metrics_.refresh_busy_ms + inference_ms_;
    if (refresh_active_at_stop) {
      CompleteActive();
      if (active_.has_value()) {
        const int queued_refresh_end = active_->completion_ms;
        metrics_.worker_occupancy_ms +=
            queued_refresh_end - final_completion_ms_;
        CompleteActive();
      }
    } else {
    }
  }

  const Trace &trace_;
  const std::vector<Packet> packets_;
  const int inference_ms_;
  const int pause_ms_;
  long audio_end_sample_ = 0;
  long segment_start_sample_ = 0;
  long next_refresh_sample_ = 0;
  long last_submitted_sample_ = 0;
  long last_pause_speech_end_sample_ = 0;
  int segment_ms_;
  int stop_ms_ = 0;
  int decode_count_ = 0;
  bool finalizing_ = false;
  bool final_handoff_ = false;
  bool arm_original_ = false;
  bool defer_busy_pause_ = false;
  int segment_id_ = 0;
  int final_completion_ms_ = 0;
  std::optional<RefreshJob> active_;
  std::optional<HintEvent> pending_hint_;
  Metrics metrics_;
};

} // namespace

int main() {
  const std::vector<Trace> traces = {
      {"continuous", {{0, 2600}}, 2600},
      {"pause-stop-0", {{0, 600}}, 600},
      {"pause-stop-100", {{0, 600}}, 700},
      {"pause-stop-300", {{0, 600}}, 900},
      {"pause-stop-500", {{0, 600}}, 1100},
      {"pause-stop-800", {{0, 600}}, 1400},
      {"pause-continue", {{0, 600}, {900, 1500}}, 1700},
      {"repeated-pauses",
       {{0, 600}, {900, 1250}, {1550, 1900}, {2200, 2550}},
       2900},
      {"short-pause", {{0, 600}, {720, 1400}}, 1600},
      {"long-trailing-silence", {{0, 600}}, 1800},
      {"buffered-burst", {{0, 600}, {900, 1400}}, 1700, true},
      {"active-pause-stop", {{0, 600}}, 900},
      {"segment-rollover", {{0, 4500}}, 4500, false, 3000},
      {"busy-already-covered", {{0, 50}}, 400, true},
      {"busy-useful-deferred", {{0, 600}}, 900, true, 30000, 800},
      {"busy-covered-by-cadence", {{0, 600}}, 1400, true, 30000, 400},
      {"resume-invalidates", {{0, 600}, {900, 1100}}, 1200, true, 30000, 800},
      {"final-before-deferred", {{0, 600}}, 900, true, 30000, 0},
      {"multiple-pauses-one-refresh",
       {{0, 600}, {900, 1200}, {1500, 1800}},
       2100,
       true,
       30000,
       800}};

  std::cout
      << "model=synthetic_scheduling_only first_refresh_ms=100 cadence_ms=1000 "
         "minimum_new_speech_ms=200 headstart_pause_ms=500\n";
  std::cout
      << "trace packet infer A=B C(cadence,pause,total,net) "
         "D(cadence,pause,total,net) C/D coverage(submitted/completed/target) "
         "residual final_wait skipped_pause worker_busy "
         "C/D_pause_ms(notify/admit/start/done/stop/trigger) "
         "C/D_readiness(covered/stop/trigger/first_headstart/opportunity) "
         "E_deferred(received/valid/invalid_or_duplicate/rejected_covered/"
         "rejected_insufficient/deferred/admitted/superseded_by_cadence/"
         "resume_invalid/final_invalid/additional_refreshes/coverage_completed/"
         "target/ready_first_headstart/opportunity/residual/final_wait)\n";

  qwen3_asr::RefreshAdmissionInput finalizing_input;
  finalizing_input.audio_end_sample = kFirstRefresh;
  finalizing_input.next_refresh_sample = kFirstRefresh;
  finalizing_input.finalizing = true;
  if (OriginalCadenceReference(finalizing_input).source !=
          qwen3_asr::RefreshSource::kNone ||
      qwen3_asr::SelectRefreshAdmission(finalizing_input).source !=
          qwen3_asr::RefreshSource::kNone) {
    std::cerr << "original and candidate finalizing behavior diverged\n";
    return 1;
  }

  for (const Trace &trace : traces) {
    for (const int packet_ms : {100, 200}) {
      for (const int inference_ms : {200, 400}) {
        Timeline original(trace, packet_ms, inference_ms, 0, true);
        Timeline disabled(trace, packet_ms, inference_ms, 0);
        Timeline pause_200(trace, packet_ms, inference_ms, 200);
        Timeline pause_300(trace, packet_ms, inference_ms, 300);
        Timeline deferred(trace, packet_ms, inference_ms, 300, false, true);
        const Metrics a = original.Run();
        const Metrics b = disabled.Run();
        const Metrics c = pause_200.Run();
        const Metrics d = pause_300.Run();
        const Metrics e = deferred.Run();
        if (trace.name == "busy-already-covered" &&
            (e.hints_rejected_covered == 0 || e.deferred_admitted != 0)) {
          std::cerr << "covered busy hint was not rejected\n";
          return 1;
        }
        if (trace.name == "busy-useful-deferred" &&
            (e.hints_deferred == 0 || e.deferred_admitted != 1)) {
          std::cerr << "useful busy hint was not admitted exactly once\n";
          return 1;
        }
        if (trace.name == "busy-covered-by-cadence" &&
            (e.hints_superseded_by_cadence == 0 || e.deferred_admitted != 0)) {
          std::cerr << "cadence did not supersede the deferred hint\n";
          return 1;
        }
        if (trace.name == "resume-invalidates" &&
            (e.deferred_admitted != 0 ||
             (packet_ms == 100 && e.deferred_invalidated_resume == 0))) {
          std::cerr << "speech resumption left a stale deferred hint\n";
          return 1;
        }
        if (trace.name == "final-before-deferred" &&
            (e.hints_deferred == 0 || e.deferred_admitted != 0 ||
             e.deferred_invalidated_final == 0)) {
          std::cerr << "Final handoff did not fence deferred admission\n";
          return 1;
        }
        if (trace.name == "multiple-pauses-one-refresh" &&
            e.deferred_admitted > 1) {
          std::cerr << "multiple pauses admitted more than one deferred "
                       "refresh\n";
          return 1;
        }
        if (a.transitions != b.transitions ||
            a.cadence_at_stop != b.cadence_at_stop ||
            a.cadence_total != b.cadence_total ||
            a.completed_coverage_at_stop_ms !=
                b.completed_coverage_at_stop_ms) {
          std::cerr << "candidate-disabled diverged from independent cadence "
                       "reference\n";
          return 1;
        }
        const int target_ms =
            static_cast<int>(std::min(Samples(trace.speech.back().end_ms),
                                      Samples(trace.audio_ms)) *
                             1000 / kRate);
        std::cout << trace.name << ' ' << packet_ms << ' ' << inference_ms
                  << ' ' << a.cadence_at_stop << '/' << b.cadence_at_stop << ' '
                  << c.cadence_total << ',' << c.pause_total << ','
                  << c.cadence_total + c.pause_total << ','
                  << c.cadence_total + c.pause_total - a.cadence_total << ' '
                  << d.cadence_total << ',' << d.pause_total << ','
                  << d.cadence_total + d.pause_total << ','
                  << d.cadence_total + d.pause_total - a.cadence_total << ' '
                  << c.submitted_coverage_at_stop_ms << '/'
                  << c.completed_coverage_at_stop_ms << '/' << target_ms << ':'
                  << d.submitted_coverage_at_stop_ms << '/'
                  << d.completed_coverage_at_stop_ms << '/' << target_ms << ' '
                  << c.residual_refresh_ms << '/' << d.residual_refresh_ms
                  << ' ' << c.final_queue_wait_ms << '/'
                  << d.final_queue_wait_ms << ' '
                  << c.pause_skipped_while_refreshing << '/'
                  << d.pause_skipped_while_refreshing << ' '
                  << c.worker_occupancy_ms << '/' << d.worker_occupancy_ms
                  << ' ' << c.pause_notification_ms << '/'
                  << c.pause_admission_ms << '/' << c.pause_start_ms << '/'
                  << c.pause_completion_ms << '/' << c.stop_ms << '/'
                  << c.pause_trigger_ms << ':' << d.pause_notification_ms << '/'
                  << d.pause_admission_ms << '/' << d.pause_start_ms << '/'
                  << d.pause_completion_ms << '/' << d.stop_ms << '/'
                  << d.pause_trigger_ms << ' '
                  << c.speech_covered_by_completed_preview_at_stop << '/'
                  << c.preview_ready_before_stop << '/'
                  << c.ready_before_pause_trigger << '/'
                  << c.ready_before_first_headstart << ':'
                  << c.first_headstart_opportunity_ms << ' '
                  << d.speech_covered_by_completed_preview_at_stop << '/'
                  << d.preview_ready_before_stop << '/'
                  << d.ready_before_pause_trigger << '/'
                  << d.ready_before_first_headstart << ':'
                  << d.first_headstart_opportunity_ms << ' ' << e.hints_received
                  << '/' << e.hints_valid << '/' << e.hints_invalid_or_duplicate
                  << '/' << e.hints_rejected_covered << '/'
                  << e.hints_rejected_insufficient << '/' << e.hints_deferred
                  << '/' << e.deferred_admitted << '/'
                  << e.hints_superseded_by_cadence << '/'
                  << e.deferred_invalidated_resume << '/'
                  << e.deferred_invalidated_final << '/'
                  << (e.cadence_total + e.pause_total - b.cadence_total) << '/'
                  << e.completed_coverage_at_stop_ms << '/' << target_ms << '/'
                  << e.ready_before_first_headstart << '/'
                  << e.first_headstart_opportunity_ms << '/'
                  << e.residual_refresh_ms << '/' << e.final_queue_wait_ms
                  << '\n';
      }
    }
  }

  constexpr int counterexample_speech_end_ms = 600;
  constexpr int counterexample_pause_notification_ms = 800;
  constexpr int counterexample_pause_start_ms = 800;
  constexpr int counterexample_inference_ms = 400;
  constexpr int counterexample_stop_ms = 900;
  constexpr int counterexample_completion_ms =
      counterexample_pause_start_ms + counterexample_inference_ms;
  constexpr int counterexample_pause_trigger_ms =
      counterexample_speech_end_ms + 500;
  constexpr int counterexample_endpoint_ms = 600;
  const bool ready_before_stop =
      counterexample_completion_ms <= counterexample_stop_ms &&
      counterexample_endpoint_ms >= counterexample_speech_end_ms;
  const bool ready_before_trigger =
      counterexample_completion_ms <= counterexample_pause_trigger_ms &&
      counterexample_endpoint_ms >= counterexample_speech_end_ms;
  const int first_headstart_opportunity_ms =
      std::min(counterexample_stop_ms, counterexample_pause_trigger_ms);
  const bool ready_before_first_headstart =
      counterexample_completion_ms <= first_headstart_opportunity_ms &&
      counterexample_endpoint_ms >= counterexample_speech_end_ms;
  const int refresh_residual_ms =
      counterexample_completion_ms - counterexample_stop_ms;
  if (ready_before_stop || ready_before_trigger ||
      ready_before_first_headstart || refresh_residual_ms != 300) {
    std::cerr << "required HeadStart counterexample assertion failed\n";
    return 1;
  }
  std::cout
      << "counterexample speech_end=600 notification="
      << counterexample_pause_notification_ms
      << " admission=800 start=800 "
         "stop=900 completion=1200 pause_trigger=1100 first_headstart=900 "
         "ready_stop="
      << ready_before_stop << " ready_trigger=" << ready_before_trigger
      << " ready_first_headstart=" << ready_before_first_headstart
      << " residual=" << refresh_residual_ms << '\n';
  std::cout << "preview_ready_during_final_handoff=not_modeled; "
               "late_candidate_readiness=not_modeled; "
               "final_queue_wait_is_synthetic_fifo_only\n";
  return 0;
}
