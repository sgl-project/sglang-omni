// SPDX-License-Identifier: Apache-2.0
// Nemotron 3.5 ASR realtime over one socket: one cache-aware stream per
// session, so each decode encodes only the audio that arrived since the last,
// as Voxt's Swift live session runs the model.
#pragma once

#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "nemotron_transcriber.h"
#include "nlohmann/json.hpp"
#include "realtime.h"
#include "worker.h"

namespace nemotron {

class NemotronRealtimeSession : public qwen3_asr::RealtimeConnection {
public:
  NemotronRealtimeSession(qwen3_asr::TranscriptionWorker &worker,
                          const NemotronTranscriber &transcriber,
                          Sender sender);

  bool Handle(const nlohmann::json &message) override;

private:
  void Append(const nlohmann::json &audio);
  // Note (Dayuxiaoshui): called with mutex_ held and no decode in flight, so
  // at most one decode of this stream is ever queued.
  void SubmitDecode();
  // Decodes the audio still pending and flushes the stream's tail, once.
  void Finish();

  qwen3_asr::TranscriptionWorker &worker_;
  const NemotronTranscriber &transcriber_;

  std::mutex mutex_;
  std::condition_variable decode_done_;
  NemotronOptions options_;
  // Note (Dayuxiaoshui): shared with the worker's job, which may still be
  // running when the client leaves and the session is destroyed.
  std::shared_ptr<NemotronStream> stream_;
  std::vector<float> pending_samples_;
  int64_t received_samples_ = 0;
  bool decoding_ = false;
  bool finished_ = false;
  // A failed decode leaves the stream half advanced, so nothing decodes after.
  bool failed_ = false;
  std::string shown_text_;
  qwen3_asr::TranscriptionResult final_result_;
};

} // namespace nemotron
