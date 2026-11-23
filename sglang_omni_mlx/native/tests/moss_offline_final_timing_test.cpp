// SPDX-License-Identifier: Apache-2.0
#include <cstdio>
#include <string>
#include <vector>

#include "common/transcription.h"

namespace {

bool Check(bool condition, const char *expression) {
  if (condition) {
    return true;
  }
  std::fprintf(stderr, "check failed: %s\n", expression);
  return false;
}

} // namespace

int main() {
  omni_common::MossOfflineFinalTiming timing;
  timing.window_id = 3;
  timing.audio_sample_count = 64000;
  timing.transcribe_started_at_ms = 1200;
  timing.transcribe_at_ms = 2200;
  timing.verify_round_count = 4;
  timing.proposed_draft_token_count = 28;
  timing.accepted_draft_token_count = 17;
  timing.prompt_embeddings_ms = 800.5;
  timing.prefill_ms = 100.25;
  timing.decode_rounds_ms = 60.125;
  timing.chunk_result_ms = 0.5;

  const auto json = omni_common::MossOfflineFinalTimingJson(timing);
  const std::vector<std::string> schema = {"event",
                                           "window_id",
                                           "audio_sample_count",
                                           "transcribe_started_at_ms",
                                           "transcribe_at_ms",
                                           "verify_round_count",
                                           "proposed_draft_token_count",
                                           "accepted_draft_token_count",
                                           "prompt_embeddings_ms",
                                           "prefill_ms",
                                           "decode_rounds_ms",
                                           "chunk_result_ms"};
  if (!Check(json.size() == schema.size(), "exact schema size") ||
      !Check(json["event"] == "moss_offline_final_timing", "event") ||
      !Check(json["window_id"] == 3, "window_id") ||
      !Check(json["audio_sample_count"] == 64000, "audio_sample_count") ||
      !Check(json["transcribe_started_at_ms"] == 1200,
             "transcribe_started_at_ms") ||
      !Check(json["transcribe_at_ms"] == 2200, "transcribe_at_ms") ||
      !Check(json["verify_round_count"] == 4, "verify_round_count") ||
      !Check(json["proposed_draft_token_count"] == 28,
             "proposed_draft_token_count") ||
      !Check(json["accepted_draft_token_count"] == 17,
             "accepted_draft_token_count") ||
      !Check(json["prompt_embeddings_ms"] == 800.5, "prompt_embeddings_ms") ||
      !Check(json["prefill_ms"] == 100.25, "prefill_ms") ||
      !Check(json["decode_rounds_ms"] == 60.125, "decode_rounds_ms") ||
      !Check(json["chunk_result_ms"] == 0.5, "chunk_result_ms")) {
    return 1;
  }
  for (const std::string &key : schema) {
    if (!Check(json.contains(key), key.c_str())) {
      return 1;
    }
  }
  for (const std::string &key :
       std::vector<std::string>{"text", "prompt", "speaker", "segments", "kind",
                                "min_preview_samples"}) {
    if (!Check(!json.contains(key), key.c_str())) {
      return 1;
    }
  }
  return 0;
}
