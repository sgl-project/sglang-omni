// SPDX-License-Identifier: Apache-2.0
// MOSS-Transcribe-Diarize realtime over one socket: fixed windows finalized as
// audio arrives, with previews of the audio still pending, as Voxt's Swift
// streaming session runs the model.
#pragma once

#include <chrono>
#include <condition_variable>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "moss_transcriber.h"
#include "nlohmann/json.hpp"
#include "realtime.h"
#include "worker.h"

namespace moss {

class MossRealtimeSession : public qwen3_asr::RealtimeConnection {
public:
  MossRealtimeSession(qwen3_asr::TranscriptionWorker &worker,
                      const MossTranscriber &transcriber, Sender sender);

  bool Handle(const nlohmann::json &message) override;

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
  MossOptions DecodeOptions(const Decode &decode) const;
  // The worker's job for decode.
  qwen3_asr::Transcription DecodeJob(const Decode &decode,
                                     MossOptions options) const;
  // Note (Dayuxiaoshui): a final window is committed before its event is sent,
  // so a client that has seen every window has seen everything committed.
  void Publish(const Decode &decode,
               const qwen3_asr::TranscriptionResult &result);
  // Note (Dayuxiaoshui): waits out a decode in flight first, since it reads
  // the pending audio this finalizes.
  void Flush();

  qwen3_asr::TranscriptionWorker &worker_;
  const MossTranscriber &transcriber_;

  // Session state; decodes finish on the worker thread.
  std::mutex mutex_;
  std::condition_variable decode_done_;
  std::optional<std::string> prompt_;
  std::vector<float> pending_samples_;
  long pending_start_sample_ = 0;
  int next_window_id_ = 0;
  bool decoding_ = false;
  std::optional<std::chrono::steady_clock::time_point> last_decode_time_;
  std::vector<std::string> committed_texts_;
  std::vector<qwen3_asr::SpeakerSegment> committed_segments_;
};

} // namespace moss
