// SPDX-License-Identifier: Apache-2.0
// The batched row loop: one moss_native_timing event per finished offline
// request, with the first blocking GPU wait charged to prefill and the result
// rebuild kept out of the decode span. Prefill only queues work here, so the
// fake decode blocks before its first callback, as production does.
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

struct FakeConfig {
  std::chrono::milliseconds prefill_sleep{0};
  // Blocks before the first decode callback, where production's first
  // mx::eval materializes the encoder, prefill and first token.
  std::chrono::milliseconds decode_wait{0};
  std::chrono::milliseconds finish_sleep{0};
};

// A request whose first sample is -1 fails during build, standing in for a
// row that errors before decoding.
moss::BatchSteps FakeSteps(const FakeConfig &config) {
  moss::BatchSteps steps;
  steps.build = [](const std::vector<float> &samples,
                   const moss::MossOptions &) {
    if (!samples.empty() && samples[0] == -1.0f) {
      throw std::runtime_error("fake build failure");
    } else {
    }
    return mx::zeros({1, 1, 1}, mx::float32);
  };
  steps.prefill = [config](const mx::array &, const std::atomic<bool> &) {
    std::this_thread::sleep_for(config.prefill_sleep);
    return omni_common::PrefilledSequence{{}, 0, mx::zeros({1}, mx::float32)};
  };
  steps.decode =
      [config](std::vector<omni_common::PrefilledSequence> &sequences,
               const std::function<bool(size_t, int)> &commit) {
        std::this_thread::sleep_for(config.decode_wait);
        for (size_t i = 0; i < sequences.size(); ++i) {
          commit(i, 10 + static_cast<int>(i));
        }
        for (size_t i = 0; i < sequences.size(); ++i) {
          commit(i, 20 + static_cast<int>(i));
        }
      };
  steps.commit = [](moss::MossTranscriber::ChunkOutput &output, int token_id,
                    const moss::MossOptions &, int) {
    output.output_ids.push_back(token_id);
    output.finish_reason = omni_common::FinishReason::kStop;
    return output.output_ids.size() >= 2;
  };
  steps.finish = [config](const moss::MossTranscriber::ChunkOutput &output, int,
                          double) {
    std::this_thread::sleep_for(config.finish_sleep);
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

struct RunResult {
  std::vector<moss::BatchOutcome> outcomes;
  std::vector<nlohmann::json> events;
};

RunResult Drive(const std::vector<moss::BatchRequest> &requests,
                const FakeConfig &config) {
  RunResult run;
  run.outcomes.resize(requests.size());
  std::ostringstream captured;
  std::streambuf *saved = std::cerr.rdbuf(captured.rdbuf());
  try {
    moss::RunTranscriptionBatch(requests, FakeSteps(config),
                                [&](size_t index, moss::BatchOutcome outcome) {
                                  run.outcomes[index] = std::move(outcome);
                                });
  } catch (...) {
    std::cerr.rdbuf(saved);
    throw;
  }
  std::cerr.rdbuf(saved);
  run.events = ParseTimingLines(captured.str());
  return run;
}

std::vector<float> Samples(double seconds, float first = 0.0f) {
  std::vector<float> samples(
      static_cast<size_t>(omni_common::kSampleRate * seconds), 0.0f);
  if (!samples.empty()) {
    samples[0] = first;
  } else {
  }
  return samples;
}

void CheckEventShape(const nlohmann::json &event) {
  Require(event["event"] == "moss_native_timing", "event name");
  Require(event["native_chunk_count"] == 1, "one chunk per batched row");
  Require(!event.contains("text"), "the event must not carry transcript text");
}

// Prefill returns immediately; the shared decode blocks before its first
// callback. That wait must land in prefill, not vanish. A delayed result
// rebuild must stay out of the decode span.
void TestTwoOfflineRows() {
  std::vector<float> short_samples = Samples(0.5);
  std::vector<float> long_samples = Samples(2.0);
  moss::MossOptions short_options;
  moss::MossOptions long_options;
  std::atomic<bool> short_cancel{false};
  std::atomic<bool> long_cancel{false};
  const std::vector<moss::BatchRequest> requests = {
      {&short_samples, &short_options, &short_cancel},
      {&long_samples, &long_options, &long_cancel}};

  FakeConfig config;
  config.decode_wait = std::chrono::milliseconds(30);
  config.finish_sleep = std::chrono::milliseconds(20);
  const RunResult run = Drive(requests, config);

  Require(run.outcomes.size() == 2, "two requests must finish");
  for (const moss::BatchOutcome &outcome : run.outcomes) {
    Require(outcome.result.has_value() && outcome.error == nullptr,
            "each batched request must succeed");
  }
  Require(run.events.size() == 2,
          "two offline requests decoded together must log one event each");
  Require(run.events[0]["audio_ms"] == 500,
          "the first row logs its own audio length");
  Require(run.events[1]["audio_ms"] == 2000,
          "the second row logs its own audio length");
  for (const nlohmann::json &event : run.events) {
    CheckEventShape(event);
    Require(event["decoder_prefill_ms"].get<double>() >= 20.0,
            "the first GPU wait must be charged to prefill");
    Require(event["frontend_eval_ms"].get<double>() >= 20.0,
            "the frontend wait must not be dropped");
    Require(event["autoregressive_decode_ms"].get<double>() < 15.0,
            "the result rebuild must stay out of the decode span");
    Require(event["result_rebuild_ms"].get<double>() >= 15.0,
            "the result rebuild is timed");
  }
}

// A realtime window rides the same batch path without an event; a row that
// fails before decoding logs none either.
void TestRealtimeAndFailedRowsAreSilent() {
  std::vector<float> window_samples = Samples(2.0);
  std::vector<float> offline_samples = Samples(1.0);
  std::vector<float> failed_samples = Samples(3.0, -1.0f);
  moss::MossOptions window_options;
  window_options.window_offset_seconds = 1.0;
  window_options.max_new_tokens = 8;
  moss::MossOptions offline_options;
  moss::MossOptions failed_options;
  std::atomic<bool> window_cancel{false};
  std::atomic<bool> offline_cancel{false};
  std::atomic<bool> failed_cancel{false};
  const std::vector<moss::BatchRequest> requests = {
      {&window_samples, &window_options, &window_cancel},
      {&offline_samples, &offline_options, &offline_cancel},
      {&failed_samples, &failed_options, &failed_cancel}};

  FakeConfig config;
  config.decode_wait = std::chrono::milliseconds(10);
  const RunResult run = Drive(requests, config);

  Require(run.outcomes.size() == 3, "three requests must report");
  Require(run.outcomes[0].result.has_value(),
          "a realtime window still decodes");
  Require(run.outcomes[1].result.has_value(), "the offline row still decodes");
  Require(run.outcomes[2].error != nullptr,
          "a row that fails before decoding reports its error");
  Require(run.events.size() == 1, "only the offline row logs an event");
  CheckEventShape(run.events[0]);
  Require(run.events[0]["audio_ms"] == 1000,
          "the logged event belongs to the offline row");
}

} // namespace

int main() {
  TestTwoOfflineRows();
  TestRealtimeAndFailedRowsAreSilent();
  return 0;
}
