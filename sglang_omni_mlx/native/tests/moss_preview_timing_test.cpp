// SPDX-License-Identifier: Apache-2.0
#include <cstdio>
#include <string>
#include <vector>

#include "common/transcription.h"

bool Check(bool condition, const char *expression) {
  if (condition) {
    return true;
  }
  std::fprintf(stderr, "check failed: %s\n", expression);
  return false;
}

int main() {
  omni_common::MossPreviewTiming timing;
  timing.window_id = 7;
  timing.kind = "preview";
  timing.audio_sample_count = 40000;
  timing.append_at_ms = 1000;
  timing.decision_at_ms = 1002;
  timing.submit_at_ms = 1004;
  timing.transcribe_at_ms = 1250;
  timing.publish_at_ms = 1252;

  const auto json = omni_common::MossPreviewTimingJson(timing);
  if (!Check(json["event"] == "moss_preview_timing", "event") ||
      !Check(json["window_id"] == 7, "window_id") ||
      !Check(json["kind"] == "preview", "kind") ||
      !Check(json["audio_sample_count"] == 40000, "audio_sample_count") ||
      !Check(json["append_at_ms"] == 1000, "append_at_ms") ||
      !Check(json["decision_at_ms"] == 1002, "decision_at_ms") ||
      !Check(json["submit_at_ms"] == 1004, "submit_at_ms") ||
      !Check(json["transcribe_at_ms"] == 1250, "transcribe_at_ms") ||
      !Check(json["publish_at_ms"] == 1252, "publish_at_ms")) {
    return 1;
  }
  for (const std::string &key :
       std::vector<std::string>{"text", "prompt", "speaker", "segments"}) {
    if (!Check(!json.contains(key), key.c_str())) {
      return 1;
    }
  }
  return 0;
}
