// SPDX-License-Identifier: Apache-2.0
// Test double of moss/moss_transcriber.h for moss_preview_timing_test: the
// real header pulls in the checkpoint loader, so no realtime session could be
// built in a test. MossOptions mirrors the real struct field for field and
// must track it; MossTranscriber loads nothing and answers an empty result.
#pragma once

#include <atomic>
#include <filesystem>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "common/transcription.h"

namespace moss {

// While false, the double's Transcribe waits, so a test can hold a decode in
// flight instead of racing its completion.
inline std::atomic<bool> g_transcribe_gate{true};

struct MossOptions {
  std::optional<std::string> prompt;
  std::optional<int> max_new_tokens;
  bool stop_at_end_of_text = false;
  bool stop_on_token_loop = false;
  std::optional<double> window_offset_seconds;
};

class MossTranscriber {
public:
  MossTranscriber(const std::filesystem::path &, const std::filesystem::path &,
                  int) {}

  omni_common::TranscriptionResult Transcribe(const std::vector<float> &,
                                              const MossOptions &,
                                              const std::atomic<bool> &) const {
    while (!g_transcribe_gate.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }
    return {};
  }
};

inline void ValidatePrompt(const std::optional<std::string> &) {}

} // namespace moss
