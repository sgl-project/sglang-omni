// SPDX-License-Identifier: Apache-2.0
#include <atomic>
#include <cstdint>
#include <future>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "common/generation.h"
#include "common/http.h"
#include "common/worker.h"
#include "pause_refresh_test_support.h"
#include "qwen3_asr/realtime.h"

namespace qwen3_asr::tests {
namespace {

constexpr int kSampleRate = 16000;
constexpr int kSpeechEndSample = 600 * kSampleRate / 1000;

std::string AudioBase64(int sample_count) {
  std::string pcm;
  pcm.reserve(sample_count * 2);
  for (int sample_index = 0; sample_index < sample_count; ++sample_index) {
    const int16_t sample = 6000;
    pcm.push_back(static_cast<char>(sample & 0xff));
    pcm.push_back(static_cast<char>((sample >> 8) & 0xff));
  }
  return omni_server::EncodeBase64(pcm);
}

omni_common::TranscriptionResult PreviewResult() {
  omni_common::TranscriptionResult result;
  result.text = "preview";
  result.language = "en";
  result.finish_reason = omni_common::FinishReason::kStop;
  return result;
}

} // namespace

void TestPauseRefreshRequiresSessionOptIn() {
  const std::vector<std::pair<std::optional<nlohmann::json>, bool>>
      session_opt_in_cases = {{std::nullopt, false},
                              {nlohmann::json(false), false},
                              {nlohmann::json("true"), false},
                              {nlohmann::json(true), true}};
  for (const auto &[opt_in_value, expect_pause_admission] :
       session_opt_in_cases) {
    TestSignal decode_finished;
    std::atomic<int> decode_count = 0;
    omni_server::TranscriptionWorker worker([] {}, [] {});
    RealtimeSession::DecodeOverride decoder =
        [&decode_finished,
         &decode_count](const std::vector<float> &, const std::atomic<bool> &,
                        RefreshReuse *) -> omni_common::TranscriptionResult {
      decode_count.fetch_add(1);
      decode_finished.Notify();
      return PreviewResult();
    };
    auto session = std::make_shared<RealtimeSession>(
        worker, std::move(decoder), MakeRealtimeSettings(1000, 1000, 30.0),
        [](const std::string &) { return true; });
    nlohmann::json session_settings = {{"turn_detection", nullptr},
                                       {"final_prompt", ""}};
    if (opt_in_value.has_value()) {
      session_settings["preview_refresh_enabled"] = *opt_in_value;
    } else {
    }
    Require(session->Handle(
                {{"type", "session.update"}, {"session", session_settings}}),
            "session opt-in update was rejected");
    if (expect_pause_admission) {
      session->TakeFinalReuse(std::string(""), false, true, false);
    } else {
    }
    const nlohmann::json append_message = {
        {"type", "input_audio_buffer.append"},
        {"audio", AudioBase64(800 * kSampleRate / 1000)},
        {"preview_refresh",
         {{"speech_start_sample", 0},
          {"speech_end_sample", kSpeechEndSample}}}};
    Require(session->Handle(append_message), "pause append was rejected");
    if (expect_pause_admission) {
      decode_finished.Wait("enabled session pause refresh timed out");
      Require(decode_count.load() == 1,
              "enabled session did not admit exactly one pause refresh");
    } else {
      Require(worker.RequestStates().empty() && decode_count.load() == 0,
              "session without boolean opt-in admitted a pause refresh");
    }
    session->Close();
  }
}

} // namespace qwen3_asr::tests
