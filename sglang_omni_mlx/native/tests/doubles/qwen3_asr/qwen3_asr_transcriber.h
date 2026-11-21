// SPDX-License-Identifier: Apache-2.0
// Checkpoint-free Qwen3ASRTranscriber double for realtime session tests.
#pragma once

#include <atomic>
#include <chrono>
#include <iostream>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "common/transcription.h"

namespace qwen3_asr {

struct TranscriptionOptions {
  std::optional<std::string> language;
  bool stop_on_token_loop = false;
  std::vector<int> prefix_token_ids;
  std::string prefix_text;
  std::vector<int> guessed_token_ids;
  std::optional<int> max_new_tokens;
};

struct RefreshReuse {};

struct RealtimeDecodeTiming {
  uint64_t session_id = 0;
  int segment_id = 0;
  int decode_index = 0;
  std::string kind;
  int audio_sample_count = 0;
  long end_sample = 0;
  int64_t submitted_at_ms = 0;
  int64_t started_at_ms = 0;
  int64_t finished_at_ms = 0;
  bool succeeded = false;
};

inline nlohmann::ordered_json
RealtimeDecodeTimingJson(const RealtimeDecodeTiming &timing) {
  return {{"event", "qwen_realtime_timing"},
          {"session_id", timing.session_id},
          {"segment_id", timing.segment_id},
          {"decode_index", timing.decode_index},
          {"kind", timing.kind},
          {"audio_sample_count", timing.audio_sample_count},
          {"end_sample", timing.end_sample},
          {"submitted_at_ms", timing.submitted_at_ms},
          {"started_at_ms", timing.started_at_ms},
          {"finished_at_ms", timing.finished_at_ms},
          {"succeeded", timing.succeeded}};
}

inline int64_t NowRealtimeMilliseconds() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

inline uint64_t NextRealtimeProfileSessionId() {
  static std::atomic<uint64_t> next_id{0};
  return next_id.fetch_add(1, std::memory_order_relaxed) + 1;
}

inline void LogRealtimeDecodeTiming(const RealtimeDecodeTiming &timing) {
  static std::mutex log_mutex;
  const std::string line = RealtimeDecodeTimingJson(timing).dump();
  std::lock_guard<std::mutex> lock(log_mutex);
  std::cerr << line << "\n";
}

template <typename Callback>
omni_common::TranscriptionResult
RunRealtimeDecodeTiming(std::shared_ptr<RealtimeDecodeTiming> timing,
                        Callback &&callback) {
  if (timing) {
    timing->started_at_ms = NowRealtimeMilliseconds();
  } else {
  }
  try {
    omni_common::TranscriptionResult result = callback();
    if (timing) {
      timing->finished_at_ms = NowRealtimeMilliseconds();
      timing->succeeded = true;
      LogRealtimeDecodeTiming(*timing);
    } else {
    }
    return result;
  } catch (...) {
    if (timing) {
      timing->finished_at_ms = NowRealtimeMilliseconds();
      LogRealtimeDecodeTiming(*timing);
    } else {
    }
    throw;
  }
}

struct RetainedText {
  std::vector<int> prefix_ids;
  std::string prefix_text;
  std::vector<int> rest_ids;
};

inline std::optional<std::string>
NormalizeLanguage(const std::string &language) {
  if (language.empty()) {
    return std::nullopt;
  }
  return language;
}

class Qwen3ASRTranscriber {
public:
  RetainedText RetainedPrefix(const std::string &, int) const { return {}; }

  omni_common::TranscriptionResult Transcribe(const std::vector<float> &,
                                              const TranscriptionOptions &,
                                              const std::atomic<bool> &,
                                              RefreshReuse * = nullptr) const {
    omni_common::TranscriptionResult result;
    result.text = "ok";
    result.finish_reason = omni_common::FinishReason::kStop;
    return result;
  }
};

struct BatchableTranscription {
  const Qwen3ASRTranscriber *transcriber;
  std::vector<float> samples;
  TranscriptionOptions options;
  std::shared_ptr<RealtimeDecodeTiming> timing;

  omni_common::TranscriptionResult
  operator()(const std::atomic<bool> &cancel) const {
    return RunRealtimeDecodeTiming(timing, [&] {
      return transcriber->Transcribe(samples, options, cancel);
    });
  }
};

} // namespace qwen3_asr
