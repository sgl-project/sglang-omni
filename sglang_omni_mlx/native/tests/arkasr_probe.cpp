// SPDX-License-Identifier: Apache-2.0
// Checks ARK-ASR's request budget, windowing and text joining against small
// checkpoints: an explicit max_new_tokens is one budget for the whole
// request, unset or 0 is the default, also one budget, and window texts join
// with a space between two spaced scripts (the shared JoinTranscriptParts),
// not the old byte rule that glued a non-ASCII ending to the next window.
//
//   arkasr_probe LOOP_CHECKPOINT STOP_CHECKPOINT
//
// The loop checkpoint decodes one repeated token, so every window runs to
// its limit and the decoded window count shows where the budget stopped it.
// The stop checkpoint decodes two tokens ("éx") and <|im_end|> per window,
// so several windows produce text and their join is observable.
#include <atomic>
#include <iostream>
#include <optional>
#include <vector>

#include "arkasr/arkasr_transcriber.h"
#include "common/transcription.h"
#include "nlohmann/json.hpp"

namespace {

bool Require(bool holds, const char *message) {
  if (!holds) {
    std::cerr << "arkasr_probe: " << message << '\n';
  } else {
  }
  return holds;
}

// 75 s of low-amplitude noise: three windows of at most 30 s.
std::vector<float> Noise() {
  std::vector<float> samples(75 * 16000);
  uint32_t state = 1;
  for (float &sample : samples) {
    state = state * 1664525 + 1013904223;
    sample =
        static_cast<float>((state >> 8) & 0xFFFF) / 65535.0f * 0.02f - 0.01f;
  }
  return samples;
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 3) {
    std::cerr << "usage: arkasr_probe LOOP_CHECKPOINT STOP_CHECKPOINT\n";
    return 2;
  } else {
  }
  const std::vector<float> samples = Noise();
  const std::atomic<bool> cancel(false);
  const auto run = [&](const arkasr::ArkTranscriber &transcriber,
                       std::optional<int> max_new_tokens) {
    arkasr::ArkOptions options;
    options.max_new_tokens = max_new_tokens;
    return transcriber.Transcribe(samples, options, cancel);
  };
  bool ok = true;

  const arkasr::ArkTranscriber loop(argv[1]);
  const omni_common::TranscriptionResult limited = run(loop, 2);
  const omni_common::TranscriptionResult defaulted = run(loop, std::nullopt);
  const omni_common::TranscriptionResult zero = run(loop, 0);
  ok &=
      Require(limited.generated_token_count == 2 &&
                  limited.finish_reason == omni_common::FinishReason::kLength &&
                  limited.window_texts.size() == 1,
              "an explicit budget must cover the whole request, not each "
              "window (a 75 s request with 2 tokens must decode one window "
              "of 2 tokens, not three)");
  ok &= Require(defaulted.generated_token_count == 3 * 256 &&
                    defaulted.window_texts.size() == 1,
                "an unset budget must be one budget for the whole request "
                "(the first window may spend it all), not 256 per window");
  ok &= Require(zero.generated_token_count == 3 * 256 &&
                    zero.window_texts.size() == 1,
                "a zero budget must mean the default, like the other ASR "
                "servers");

  const arkasr::ArkTranscriber stop(argv[2]);
  const omni_common::TranscriptionResult joined = run(stop, std::nullopt);
  const omni_common::TranscriptionResult capped = run(stop, 5);
  ok &= Require(joined.generated_token_count == 3 * 2 &&
                    joined.finish_reason == omni_common::FinishReason::kStop &&
                    joined.window_texts.size() == 3 &&
                    joined.text == "\xc3\xa9x \xc3\xa9x \xc3\xa9x",
                "window texts must join with a space between two spaced "
                "scripts (the shared JoinTranscriptParts), not the byte rule "
                "that glued a non-ASCII ending to the next window");
  ok &=
      Require(capped.generated_token_count == 5 &&
                  capped.finish_reason == omni_common::FinishReason::kLength &&
                  capped.window_texts.size() == 3,
              "an explicit budget must be spent across windows in order, "
              "not renewed per window (5 tokens over three 2-token windows "
              "must stop the third at 1)");

  const nlohmann::json rows = {
      {{"checkpoint", "loop"},
       {"budget", 2},
       {"tokens", limited.generated_token_count},
       {"windows", limited.window_texts.size()}},
      {{"checkpoint", "loop"},
       {"budget", nullptr},
       {"tokens", defaulted.generated_token_count},
       {"windows", defaulted.window_texts.size()}},
      {{"checkpoint", "loop"},
       {"budget", 0},
       {"tokens", zero.generated_token_count},
       {"windows", zero.window_texts.size()}},
      {{"checkpoint", "stop"},
       {"budget", nullptr},
       {"tokens", joined.generated_token_count},
       {"windows", joined.window_texts.size()},
       {"text", joined.text}},
      {{"checkpoint", "stop"},
       {"budget", 5},
       {"tokens", capped.generated_token_count},
       {"windows", capped.window_texts.size()}},
  };
  std::cout << rows.dump() << '\n';
  return ok ? 0 : 1;
}
