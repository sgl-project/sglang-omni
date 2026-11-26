// SPDX-License-Identifier: Apache-2.0
// Two offline requests decoded together each log one moss_native_timing
// event with that row's real stage timings.
#include <atomic>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "common/audio.h"
#include "common/generation.h"
#include "moss/moss_transcriber.h"
#include "nlohmann/json.hpp"

namespace {

namespace mx = mlx::core;

void Require(bool condition, const char *message) {
  if (!condition) {
    throw std::runtime_error(message);
  } else {
  }
}

// Builds one fake batch step set: real per-row clocks, no model. Decoding two
// rows takes two commits each, so every timing span that the batch path
// measures is strictly positive.
moss::BatchSteps FakeSteps() {
  moss::BatchSteps steps;
  steps.build = [](const std::vector<float> &, const moss::MossOptions &) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    return mx::zeros({1, 1, 1}, mx::float32);
  };
  steps.prefill = [](const mx::array &, const std::atomic<bool> &) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    return omni_common::PrefilledSequence{{}, 0, mx::zeros({1}, mx::float32)};
  };
  steps.decode = [](std::vector<omni_common::PrefilledSequence> &sequences,
                    const std::function<bool(size_t, int)> &commit) {
    for (size_t i = 0; i < sequences.size(); ++i) {
      if (!commit(i, 10 + static_cast<int>(i))) {
        commit(i, 20 + static_cast<int>(i));
      } else {
      }
    }
  };
  steps.commit = [](moss::MossTranscriber::ChunkOutput &output, int token_id,
                    const moss::MossOptions &, int) {
    output.output_ids.push_back(token_id);
    output.finish_reason = omni_common::FinishReason::kStop;
    return output.output_ids.size() >= 2;
  };
  steps.finish = [](const moss::MossTranscriber::ChunkOutput &output, int,
                    double) {
    omni_common::TranscriptionResult result;
    result.text = "batch row text must not enter timing logs";
    result.generated_token_count = static_cast<int>(output.output_ids.size());
    result.finish_reason = output.finish_reason;
    return result;
  };
  return steps;
}

std::vector<nlohmann::json> ParseTimingLines(const std::string &log) {
  std::vector<nlohmann::json> events;
  std::istringstream lines(log);
  std::string line;
  while (std::getline(lines, line)) {
    if (line.find("moss_native_timing") == std::string::npos) {
      continue;
    } else {
    }
    events.push_back(nlohmann::json::parse(line));
  }
  return events;
}

} // namespace

int main() {
  constexpr int kSampleRate = omni_common::kSampleRate;
  std::vector<float> short_samples(static_cast<size_t>(kSampleRate) / 2, 0.0f);
  std::vector<float> long_samples(static_cast<size_t>(kSampleRate) * 2, 0.0f);
  moss::MossOptions short_options;
  moss::MossOptions long_options;
  std::atomic<bool> short_cancel{false};
  std::atomic<bool> long_cancel{false};
  const std::vector<moss::BatchRequest> requests = {
      {&short_samples, &short_options, &short_cancel},
      {&long_samples, &long_options, &long_cancel}};

  std::vector<moss::BatchOutcome> outcomes(requests.size());
  std::ostringstream captured;
  std::streambuf *saved = std::cerr.rdbuf(captured.rdbuf());
  try {
    moss::RunTranscriptionBatch(requests, FakeSteps(),
                                [&](size_t index, moss::BatchOutcome outcome) {
                                  outcomes[index] = std::move(outcome);
                                });
  } catch (...) {
    std::cerr.rdbuf(saved);
    throw;
  }
  std::cerr.rdbuf(saved);

  Require(outcomes.size() == 2, "two requests must finish");
  for (const moss::BatchOutcome &outcome : outcomes) {
    Require(outcome.result.has_value() && outcome.error == nullptr,
            "each batched request must succeed");
  }
  const std::vector<nlohmann::json> events = ParseTimingLines(captured.str());
  Require(events.size() == 2,
          "two offline requests decoded together must log one event each");

  Require(events[0]["audio_ms"] == 500,
          "the first row logs its own audio length");
  Require(events[1]["audio_ms"] == 2000,
          "the second row logs its own audio length");
  for (const nlohmann::json &event : events) {
    Require(event["event"] == "moss_native_timing", "event name");
    Require(event["native_chunk_count"] == 1, "one chunk per batched row");
    Require(event["frontend_build_ms"].get<double>() > 0.5,
            "the row's frontend build is timed, not zero");
    Require(event["decoder_prefill_ms"].get<double>() > 0.5,
            "the row's prefill is timed, not zero");
    Require(event["native_total_ms"].get<double>() >=
                event["decoder_prefill_ms"].get<double>(),
            "the row's total covers its stages");
    Require(event["autoregressive_decode_ms"].get<double>() >= 0.0,
            "the row's decode span is measured");
    Require(event["result_rebuild_ms"].get<double>() >= 0.0,
            "the row's result rebuild is measured");
    Require(!event.contains("text"),
            "the event must not carry transcript text");
  }
  return 0;
}
