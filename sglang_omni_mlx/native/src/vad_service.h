// SPDX-License-Identifier: Apache-2.0
// Silero VAD for Voxt's detectors: POST /v1/vad/speech_timestamps (speech
// ranges in samples) and WS /v1/vad/stream (16 kHz float32 LE samples in,
// {"probability": p} of each message's last whole chunk or null out).
#pragma once

#include <filesystem>
#include <map>
#include <mutex>
#include <string>

#include "civetweb.h"
#include "silero_vad.h"

namespace silero_vad {

class VADService {
public:
  explicit VADService(const std::filesystem::path &model_directory);

  void Register(mg_context *context);
  // Request counts for /health.
  std::map<std::string, int> RequestStates() const;

  // Speech probability of each whole chunk in samples, in order.
  std::vector<float> FeedSamples(const std::vector<float> &samples,
                                 StreamState &state);
  std::vector<Timestamp> SpeechTimestamps(const std::vector<float> &samples,
                                          const TimestampOptions &options);
  const TimestampOptions &defaults() const { return vad_.defaults(); }

  void StreamOpened();
  void StreamClosed();

private:
  SileroVAD vad_;
  // One model, many callers: inference runs one call at a time.
  std::mutex inference_mutex_;
  mutable std::mutex states_mutex_;
  int open_streams_ = 0;
  int running_requests_ = 0;
};

} // namespace silero_vad
