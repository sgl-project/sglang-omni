// SPDX-License-Identifier: Apache-2.0
// MOSS-Transcribe-Diarize realtime over one socket: fixed windows finalized as
// audio arrives, with previews of the audio still pending, as Voxt's Swift
// streaming session runs the model.
#pragma once

#include <chrono>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "common/realtime_connection.h"
#include "common/transcription.h"
#include "common/worker.h"
#include "moss/moss_transcriber.h"
#include "nlohmann/json.hpp"

namespace moss {

// One decode's steady-clock stamps, logged as a moss_preview_timing event.
struct DecodeTiming;

class MossRealtimeSession : public omni_server::RealtimeConnection {
public:
  // One window's decode, as MossTranscriber::Transcribe runs it; a seam so
  // tests can drive scheduling without a checkpoint.
  using TranscribeFn = std::function<omni_common::TranscriptionResult(
      const std::vector<float> &, const MossOptions &,
      const std::atomic<bool> &)>;
  // steady_clock::now in production; tests inject a virtual clock so timed
  // scenarios are deterministic.
  using ClockFn = std::function<std::chrono::steady_clock::time_point()>;

  MossRealtimeSession(omni_server::TranscriptionWorker &worker,
                      TranscribeFn transcribe, Sender sender,
                      ClockFn clock = std::chrono::steady_clock::now);

  bool Handle(const nlohmann::json &message) override;
  void Close() override;

private:
  enum class DecodeKind { kPreview, kFinal };

  struct Decode {
    DecodeKind kind = DecodeKind::kPreview;
    int window_id = 0;
    std::vector<float> samples;
    double offset_seconds = 0.0;
  };

  void Append(const nlohmann::json &audio);
  // Note (Dayuxiaoshui): returns nothing while a decode is running, which keeps
  // a second decode of this session off the worker.
  std::optional<Decode> NextDecode();
  // Note (Dayuxiaoshui): takes its samples off the pending buffer, so the next
  // window starts where this one ended.
  Decode TakeFinalWindow(int sample_count);
  // Queues decode on the worker with its kind's flag; the completion publishes
  // and runs OnDecodeDone. due_at is the event that made the decode due.
  void SubmitDecode(Decode decode, MossOptions options,
                    omni_server::CancelFlag flag,
                    std::chrono::steady_clock::time_point due_at,
                    std::chrono::steady_clock::time_point decided_at);
  // Clears the in-flight flags and starts the final of a window that filled
  // while the decode ran; runs on the worker thread.
  void OnDecodeDone(DecodeKind kind);
  MossOptions DecodeOptions(const Decode &decode) const;
  static std::shared_ptr<DecodeTiming>
  StartDecodeTiming(const Decode &decode,
                    std::chrono::steady_clock::time_point due_at,
                    std::chrono::steady_clock::time_point decided_at);
  // The worker's job for decode; stamps when the transcription starts.
  omni_server::Transcription
  DecodeJob(const Decode &decode, MossOptions options,
            std::shared_ptr<DecodeTiming> timing) const;
  // Note (Dayuxiaoshui): a final window is committed before its event is sent,
  // so a client that has seen every window has seen everything committed.
  void Publish(const Decode &decode,
               const omni_common::TranscriptionResult &result);
  // Publish, then logs the decode's timing line.
  void PublishWithTiming(const Decode &decode,
                         const omni_common::TranscriptionResult &result,
                         DecodeTiming &timing);
  // Note (Dayuxiaoshui): waits out a decode in flight first, since it reads
  // the pending audio this finalizes.
  void Flush();
  // Stops the preview in flight, if any: a final decode of the same audio is
  // due, so the preview's result is already replaced. Caller holds mutex_.
  void CancelPreviewLocked();

  omni_server::TranscriptionWorker &worker_;
  TranscribeFn transcribe_;
  ClockFn clock_;

  // Session state; decodes finish on the worker thread.
  std::mutex mutex_;
  std::condition_variable decode_done_;
  std::optional<std::string> prompt_;
  std::vector<float> pending_samples_;
  long pending_start_sample_ = 0;
  int next_window_id_ = 0;
  bool decoding_ = false;
  // decoding_ is a preview, and that preview's flag; a final decode keeps the
  // session flag so only previews are ever preempted.
  bool preview_in_flight_ = false;
  omni_server::CancelFlag preview_cancel_ = omni_server::NewCancelFlag();
  std::optional<std::chrono::steady_clock::time_point> last_decode_time_;
  std::vector<std::string> committed_texts_;
  std::vector<omni_common::SpeakerSegment> committed_segments_;
};

} // namespace moss
