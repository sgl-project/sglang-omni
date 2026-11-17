// SPDX-License-Identifier: Apache-2.0
#include <cstdio>
#include <string>
#include <vector>

#include "moss/moss_transcriber.h"

bool Check(bool condition, const char *expression) {
  if (condition) {
    return true;
  }
  std::fprintf(stderr, "check failed: %s\n", expression);
  return false;
}

int main() {
  omni_common::TranscriptionResult result;
  result.native_split_ms = 1.0;
  result.frontend_build_ms = 2.0;
  result.frontend_eval_ms = 3.0;
  result.decoder_prefill_ms = 4.0;
  result.autoregressive_decode_ms = 5.0;
  result.result_rebuild_ms = 6.0;
  result.cache_clear_ms = 7.0;
  result.native_total_ms = 28.0;
  result.generated_token_count = 41;
  result.verify_round_count = 7;
  result.proposed_draft_token_count = 49;
  result.accepted_draft_token_count = 23;
  result.text = "transcript content must not enter timing logs";
  const auto timing = moss::MossNativeTimingJson(result, 120240, 1);
  if (!Check(timing["event"] == "moss_native_timing", "event") ||
      !Check(timing["audio_ms"] == 120240, "audio_ms") ||
      !Check(timing["native_chunk_count"] == 1, "native_chunk_count") ||
      !Check(timing["native_split_ms"].get<double>() == 1.0,
             "native_split_ms") ||
      !Check(timing["frontend_build_ms"].get<double>() == 2.0,
             "frontend_build_ms") ||
      !Check(timing["frontend_eval_ms"].get<double>() == 3.0,
             "frontend_eval_ms") ||
      !Check(timing["decoder_prefill_ms"].get<double>() == 4.0,
             "decoder_prefill_ms") ||
      !Check(timing["autoregressive_decode_ms"].get<double>() == 5.0,
             "autoregressive_decode_ms") ||
      !Check(timing["result_rebuild_ms"].get<double>() == 6.0,
             "result_rebuild_ms") ||
      !Check(timing["cache_clear_ms"].get<double>() == 7.0, "cache_clear_ms") ||
      !Check(timing["native_total_ms"].get<double>() == 28.0,
             "native_total_ms") ||
      !Check(timing["generated_token_count"] == 41, "generated_token_count") ||
      !Check(timing["verify_round_count"] == 7, "verify_round_count") ||
      !Check(timing["proposed_draft_token_count"] == 49,
             "proposed_draft_token_count") ||
      !Check(timing["accepted_draft_token_count"] == 23,
             "accepted_draft_token_count")) {
    return 1;
  }
  for (const std::string &key :
       std::vector<std::string>{"text", "prompt", "speaker", "segments"}) {
    if (!Check(!timing.contains(key), key.c_str())) {
      return 1;
    }
  }
  return 0;
}
