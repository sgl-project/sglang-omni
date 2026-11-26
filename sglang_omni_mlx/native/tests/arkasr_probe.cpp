// SPDX-License-Identifier: Apache-2.0
// Checks ARK-ASR's request budget, windowing and text joining against small
// checkpoints: an explicit max_new_tokens is one budget for the whole
// request, unset or 0 is the default, also one budget, and window texts join
// with a space between two spaced scripts (the shared JoinTranscriptParts),
// not the old byte rule that glued a non-ASCII ending to the next window.
// A split window that is only the request's own zero padding must add no
// text: the real checkpoint reads such a window as silence and hallucinates.
// Decoding with a DFlash draft must give the plain decode's result, also
// when the budget ends inside a draft block.
//
//   arkasr_probe LOOP_CHECKPOINT STOP_CHECKPOINT DRAFT_CHECKPOINT
//
// The loop checkpoint decodes one repeated token, so every window runs to
// its limit and the decoded window count shows where the budget stopped it.
// The stop checkpoint decodes two tokens ("éx") and <|im_end|> per window,
// so several windows produce text and their join is observable. The draft
// always proposes the loop checkpoint's token, so the loop's rounds accept
// every proposal and the stop's reject every one.
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

// Low-amplitude noise: loud enough that no window is taken for silence.
std::vector<float> Noise(size_t sample_count, float amplitude = 0.02f) {
  std::vector<float> samples(sample_count);
  uint32_t state = 1;
  for (float &sample : samples) {
    state = state * 1664525 + 1013904223;
    sample = static_cast<float>((state >> 8) & 0xFFFF) / 65535.0f * amplitude -
             amplitude / 2;
  }
  return samples;
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 4) {
    std::cerr << "usage: arkasr_probe LOOP_CHECKPOINT STOP_CHECKPOINT "
                 "DRAFT_CHECKPOINT\n";
    return 2;
  } else {
  }
  const std::vector<float> samples = Noise(75 * 16000);
  const std::atomic<bool> cancel(false);
  const auto run = [&](const arkasr::ArkTranscriber &transcriber,
                       std::optional<int> max_new_tokens) {
    arkasr::ArkOptions options;
    options.max_new_tokens = max_new_tokens;
    return transcriber.Transcribe(samples, options, cancel);
  };
  bool ok = true;

  const arkasr::ArkTranscriber loop(argv[1], {});
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

  const arkasr::ArkTranscriber stop(argv[2], {});
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

  // Note (AzizbekMuminoff): 20 s of noise zero-padded to one sample over
  // 30 s splits into noise and an all-zero window; the zero window must add
  // no text, at either end of the request.
  std::vector<float> padded = Noise(20 * 16000);
  padded.resize(480001, 0.0f);
  const omni_common::TranscriptionResult trailed =
      stop.Transcribe(padded, arkasr::ArkOptions{}, cancel);
  std::vector<float> prepended(26 * 16000, 0.0f);
  const std::vector<float> tail = Noise(20 * 16000);
  prepended.insert(prepended.end(), tail.begin(), tail.end());
  const omni_common::TranscriptionResult led =
      stop.Transcribe(prepended, arkasr::ArkOptions{}, cancel);
  ok &= Require(trailed.text == "\xc3\xa9x" && trailed.window_texts.size() == 1,
                "a split window that is only the request's zero padding "
                "must add no text after the speech");
  ok &= Require(led.text == "\xc3\xa9x" && led.window_texts.size() == 1,
                "a split window that is only the request's zero padding "
                "must add no text before the speech");

  // Note (AzizbekMuminoff): nonzero audio is never padding, however quiet:
  // 20 s of noise at 1e-5 after 26 s of zeros splits into a skipped zero
  // window and this one, which must still decode.
  std::vector<float> quiet(26 * 16000, 0.0f);
  const std::vector<float> quiet_speech = Noise(20 * 16000, 2e-5f);
  quiet.insert(quiet.end(), quiet_speech.begin(), quiet_speech.end());
  const omni_common::TranscriptionResult quiet_result =
      stop.Transcribe(quiet, arkasr::ArkOptions{}, cancel);
  ok &= Require(quiet_result.text == "\xc3\xa9x" &&
                    quiet_result.window_texts.size() == 1,
                "a nonzero window must decode however quiet it is (only "
                "exact zeros are padding)");

  nlohmann::json drafted_rows = nlohmann::json::array();
  const auto compare = [&](const char *checkpoint,
                           const omni_common::TranscriptionResult &plain,
                           const arkasr::ArkTranscriber &drafted_transcriber,
                           std::optional<int> max_new_tokens) {
    const omni_common::TranscriptionResult drafted =
        run(drafted_transcriber, max_new_tokens);
    ok &= Require(drafted.text == plain.text &&
                      drafted.generated_token_count ==
                          plain.generated_token_count &&
                      drafted.finish_reason == plain.finish_reason &&
                      drafted.window_texts == plain.window_texts,
                  "decoding with the draft must keep the plain decode's "
                  "text, token count, finish reason and windows");
    drafted_rows.push_back(
        {{"checkpoint", checkpoint},
         {"budget", max_new_tokens.has_value() ? nlohmann::json(*max_new_tokens)
                                               : nlohmann::json()},
         {"proposed", drafted.proposed_draft_token_count},
         {"accepted", drafted.accepted_draft_token_count}});
  };
  const arkasr::ArkTranscriber drafted_loop(argv[1], {argv[3]});
  // Budgets 1, 2, 8 and 9 end around the draft's 8-token block.
  for (const int budget : {1, 8, 9}) {
    compare("loop", run(loop, budget), drafted_loop, budget);
  }
  compare("loop", limited, drafted_loop, 2);
  compare("loop", defaulted, drafted_loop, std::nullopt);
  const arkasr::ArkTranscriber drafted_stop(argv[2], {argv[3]});
  compare("stop", joined, drafted_stop, std::nullopt);
  compare("stop", capped, drafted_stop, 5);

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
  std::cout << nlohmann::json{{"plain", rows}, {"drafted", drafted_rows}}.dump()
            << '\n';
  return ok ? 0 : 1;
}
