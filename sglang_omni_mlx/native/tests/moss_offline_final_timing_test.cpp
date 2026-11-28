// SPDX-License-Identifier: Apache-2.0
#include <atomic>
#include <cstdio>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "common/dflash_draft.h"
#include "common/transcription.h"
#include "moss/moss_transcriber.h"

namespace {

bool Check(bool condition, const char *expression) {
  if (condition) {
    return true;
  }
  std::fprintf(stderr, "check failed: %s\n", expression);
  return false;
}

class CerrCapture {
public:
  CerrCapture() : original_(std::cerr.rdbuf(buffer_.rdbuf())) {}
  ~CerrCapture() { std::cerr.rdbuf(original_); }

  std::vector<nlohmann::json> Lines() const {
    std::vector<nlohmann::json> lines;
    std::istringstream stream(buffer_.str());
    std::string line;
    while (std::getline(stream, line)) {
      lines.push_back(nlohmann::json::parse(line));
    }
    return lines;
  }

private:
  std::ostringstream buffer_;
  std::streambuf *original_;
};

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

  moss::MossTranscriber transcriber("batch-double",
                                    omni_common::DraftOptions{});
  moss::MossOptions first_options;
  first_options.max_new_tokens = 1;
  moss::MossOptions second_options;
  second_options.max_new_tokens = 2;
  omni_server::Transcription first = moss::BatchableTranscription{
      &transcriber, std::vector<float>(1, 0.0f), first_options};
  omni_server::Transcription second = moss::BatchableTranscription{
      &transcriber, std::vector<float>(16001, 0.0f), second_options};
  const auto first_cancel = omni_server::NewCancelFlag();
  const auto second_cancel = omni_server::NewCancelFlag();
  const std::vector<omni_server::BatchItem> requests = {
      {&first, first_cancel.get()}, {&second, second_cancel.get()}};
  std::vector<std::optional<omni_common::TranscriptionResult>> results(2);
  std::vector<std::exception_ptr> errors(2);
  std::vector<size_t> completion_order;
  std::vector<nlohmann::json> lines;
  {
    CerrCapture capture;
    moss::BatchableTranscriptionRunner().run(
        requests, [&](size_t index,
                      std::optional<omni_common::TranscriptionResult> result,
                      std::exception_ptr error) {
          completion_order.push_back(index);
          results[index] = std::move(result);
          errors[index] = error;
        });
    lines = capture.Lines();
  }
  if (!Check(completion_order == std::vector<size_t>{0, 1},
             "each batch row finishes in its own round") ||
      !Check(lines.size() == 2, "one offline timing event per batch row")) {
    return 1;
  }
  const int expected_sample_counts[] = {16000, 16001};
  for (size_t index = 0; index < lines.size(); ++index) {
    if (!Check(!errors[index], "batch row succeeds") ||
        !Check(results[index].has_value(), "batch row returns a result")) {
      return 1;
    }
    const auto &line = lines[index];
    const auto &result = *results[index];
    if (!Check(result.prompt_embeddings_ms > 0.0,
               "batch measures prompt embeddings") ||
        !Check(result.prefill_ms > 0.0, "batch measures prefill") ||
        !Check(result.decode_rounds_ms > 0.0, "batch measures decode rounds") ||
        !Check(result.chunk_result_ms > 0.0,
               "batch measures result assembly") ||
        !Check(line.at("event") == "moss_offline_final_timing",
               "batched offline event") ||
        !Check(line.at("window_id") == 0, "batched offline window") ||
        !Check(line.at("audio_sample_count") == expected_sample_counts[index],
               "batched audio sample count") ||
        !Check(line.at("transcribe_started_at_ms") <=
                   line.at("transcribe_at_ms"),
               "batched timestamps are ordered") ||
        !Check(line.at("prompt_embeddings_ms") == result.prompt_embeddings_ms,
               "batched prompt embedding timing") ||
        !Check(line.at("prefill_ms") == result.prefill_ms,
               "batched prefill timing") ||
        !Check(line.at("decode_rounds_ms") == result.decode_rounds_ms,
               "batched decode timing") ||
        !Check(line.at("chunk_result_ms") == result.chunk_result_ms,
               "batched result timing") ||
        !Check(!line.contains("text") && !line.contains("prompt") &&
                   !line.contains("speaker") && !line.contains("segments"),
               "batched event has no transcript content")) {
      return 1;
    }
  }
  return 0;
}
