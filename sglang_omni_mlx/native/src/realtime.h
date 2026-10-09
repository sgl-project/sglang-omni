// SPDX-License-Identifier: Apache-2.0
// Realtime transcription over one socket: segments refreshed on a cadence,
// then finalized.
#pragma once

#include <condition_variable>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "nlohmann/json.hpp"
#include "worker.h"

namespace qwen3_asr {

struct RealtimeSettings {
  int decode_interval_samples = 0;
  int first_decode_samples = 0;
  int max_segment_samples = 0;
};

RealtimeSettings MakeRealtimeSettings(int decode_interval_ms,
                                      int first_decode_ms,
                                      double max_segment_seconds);

// Join segment texts, with a space only between two spaced scripts.
std::string JoinTranscriptParts(const std::vector<std::string> &parts);

// Manual-turn session: audio becomes one segment, cut every max segment
// length. Handle runs on the socket's thread; preview decodes run on the
// worker and report back through the sender.
class RealtimeSession : public std::enable_shared_from_this<RealtimeSession> {
public:
  // Writes one serialized event; false once the socket is gone.
  using Sender = std::function<bool(const std::string &)>;

  RealtimeSession(TranscriptionWorker &worker, RealtimeSettings settings,
                  Sender sender);

  // Applies one client event; false once the session has completed.
  bool Handle(const nlohmann::json &message);
  void SendError(const std::string &type, const std::string &code,
                 const std::string &message);
  // The client left: stop any decode in flight and send nothing more.
  void Close();

private:
  struct Segment {
    int segment_id = 0;
    long start_sample = 0;
    long next_refresh_sample = 0;
    std::optional<std::string> language;
    int decode_count = 0;
    std::string transcript;
    std::string last_text;
  };

  void Send(nlohmann::ordered_json event);
  void Append(const nlohmann::json &audio);
  void StartSegment(long start_sample);
  long EndSample() const {
    return buffer_start_sample_ + static_cast<long>(samples_.size());
  }
  long LockedEndSample();
  std::vector<float> SegmentSamples(const Segment &segment,
                                    long end_sample) const;
  // Builds the request for a decode of segment and counts it.
  TranscriptionOptions DecodeOptions(Segment &segment);
  void ApplyResult(Segment &segment, const TranscriptionResult &result);
  void MaybeStartRefresh();
  void FinalizeThrough(long end_sample);
  void ReportDecodeFailure(std::exception_ptr error);

  TranscriptionWorker &worker_;
  const RealtimeSettings settings_;
  Sender sender_;
  CancelFlag cancel_ = NewCancelFlag();

  // Session state; refreshes finish on the worker thread.
  std::mutex mutex_;
  std::condition_variable refresh_done_;
  std::optional<std::string> language_;
  std::vector<float> samples_;
  long buffer_start_sample_ = 0;
  std::optional<Segment> segment_;
  int next_segment_id_ = 0;
  std::vector<std::pair<int, std::string>> committed_;
  bool refreshing_ = false;
  bool finalizing_ = false;

  // Event order and the socket's lifetime.
  std::mutex send_mutex_;
  int event_index_ = 0;
  bool closed_ = false;
};

} // namespace qwen3_asr
