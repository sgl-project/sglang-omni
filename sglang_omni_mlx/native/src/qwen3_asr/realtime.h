// SPDX-License-Identifier: Apache-2.0
// Realtime transcription over one socket: segments refreshed on a cadence,
// then finalized.
#pragma once

#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "common/realtime_connection.h"
#include "common/transcription.h"
#include "common/worker.h"
#include "nlohmann/json.hpp"
#include "qwen3_asr/qwen3_asr_transcriber.h"

namespace qwen3_asr {

struct RealtimeSettings {
  int decode_interval_samples = 0;
  int first_decode_samples = 0;
  int max_segment_samples = 0;
  // Content-free decode timings are opt-in for the profiling harness.
  bool profile = false;
};

RealtimeSettings MakeRealtimeSettings(int decode_interval_ms,
                                      int first_decode_ms,
                                      double max_segment_seconds);

// Join segment texts, with a space only between two spaced scripts.
std::string JoinTranscriptParts(const std::vector<std::string> &parts);

// Qwen3-ASR manual-turn session: audio becomes one segment, cut every max
// segment length, with preview decodes on a cadence.
class RealtimeSession : public omni_server::RealtimeConnection {
public:
  RealtimeSession(omni_server::TranscriptionWorker &worker,
                  const Qwen3ASRTranscriber &transcriber,
                  RealtimeSettings settings, Sender sender);

  bool Handle(const nlohmann::json &message) override;

private:
  struct Segment {
    int segment_id = 0;
    long start_sample = 0;
    long next_refresh_sample = 0;
    std::optional<std::string> language;
    int decode_count = 0;
    std::string transcript;
    std::string last_text;
    // The last decode ran out of tokens, a loop the guard missed, likely on
    // the prefix: its text was not kept, and the next decode has no prefix.
    bool ran_out_of_tokens = false;
    // Where the audio of the decode the transcript came from ended.
    long transcript_end_sample = 0;
    // What the segment's refreshes reuse.
    std::shared_ptr<RefreshReuse> reuse;
  };

  void Append(const nlohmann::json &audio);
  void StartSegment(long start_sample);
  long EndSample() const {
    return buffer_start_sample_ + static_cast<long>(samples_.size());
  }
  long LockedEndSample();
  std::vector<float> SegmentSamples(const Segment &segment,
                                    long end_sample) const;
  // Builds the request for a decode of segment through end_sample, a refresh
  // or its final decode, and counts it.
  TranscriptionOptions DecodeOptions(Segment &segment, bool is_refresh,
                                     long end_sample);
  std::shared_ptr<RealtimeDecodeTiming>
  NewDecodeTiming(const Segment &segment, std::string kind, long end_sample,
                  int audio_sample_count) const;
  // A refresh passes its segment's reuse; the final decode passes none.
  omni_server::Transcription
  Decode(std::vector<float> samples, TranscriptionOptions options,
         std::shared_ptr<RefreshReuse> reuse,
         std::shared_ptr<RealtimeDecodeTiming> timing = nullptr) const;
  void ApplyResult(Segment &segment,
                   const omni_common::TranscriptionResult &result,
                   long end_sample);
  void MaybeStartRefresh();
  void FinalizeThrough(long end_sample);

  omni_server::TranscriptionWorker &worker_;
  const Qwen3ASRTranscriber &transcriber_;
  const RealtimeSettings settings_;
  const uint64_t profile_session_id_ = 0;

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
};

} // namespace qwen3_asr
