// SPDX-License-Identifier: Apache-2.0
#include "qwen3_asr/realtime.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "common/audio.h"
#include "common/http.h"
#include "common/worker.h"

namespace {

using omni_common::TranscriptionResult;
using omni_server::NewCancelFlag;
using omni_server::TranscriptionWorker;
using qwen3_asr::AudioLayout;
using qwen3_asr::FinalReuse;
using qwen3_asr::MakeRealtimeSettings;
using qwen3_asr::Qwen3ASRTranscriber;
using qwen3_asr::RealtimeSession;
using qwen3_asr::RefreshReuse;
using qwen3_asr::RetainedText;
using qwen3_asr::TranscriptionOptions;

constexpr size_t kPreviewAudioSampleCount = 9 * omni_common::kSampleRate;
constexpr size_t kFollowupAudioSampleCount = 2 * omni_common::kSampleRate;
constexpr int kPrefixedTokenSlack = 8;
constexpr auto kSessionEventTimeout = std::chrono::seconds(180);

void Require(bool condition, const std::string &message) {
  if (!condition) {
    throw std::runtime_error(message);
  } else {
  }
}

std::vector<float> ReadAudioSamples(const std::filesystem::path &wav_path) {
  std::ifstream wav_file(wav_path, std::ios::binary);
  if (!wav_file) {
    throw std::runtime_error("cannot open audio corpus file");
  } else {
  }
  const std::string wav_bytes((std::istreambuf_iterator<char>(wav_file)),
                              std::istreambuf_iterator<char>());
  return omni_common::DecodeWav(wav_bytes);
}

TranscriptionResult DecodeFinalTranscription(
    TranscriptionWorker &worker, const Qwen3ASRTranscriber &transcriber,
    const std::vector<float> &audio_samples, TranscriptionOptions options) {
  return worker.Transcribe(
      [&transcriber, &audio_samples,
       options = std::move(options)](const std::atomic<bool> &cancel) {
        return transcriber.Transcribe(audio_samples, options, cancel);
      },
      NewCancelFlag());
}

TranscriptionResult DecodeWithReuse(TranscriptionWorker &worker,
                                    const Qwen3ASRTranscriber &transcriber,
                                    const std::vector<float> &audio_samples,
                                    TranscriptionOptions options,
                                    RefreshReuse reuse) {
  return worker.Transcribe(
      [&transcriber, &audio_samples, options = std::move(options),
       reuse = std::move(reuse)](const std::atomic<bool> &cancel) mutable {
        return transcriber.Transcribe(audio_samples, options, cancel, &reuse);
      },
      NewCancelFlag());
}

void RequireSameFinalOutput(const TranscriptionResult &expected,
                            const TranscriptionResult &actual,
                            const std::string &guess_kind) {
  Require(actual.text == expected.text,
          guess_kind + " changed the prefixed Final text");
  Require(actual.language == expected.language,
          guess_kind + " changed detected language");
  Require(actual.finish_reason == expected.finish_reason,
          guess_kind + " changed finish reason");
  Require(actual.generated_token_count == expected.generated_token_count,
          guess_kind + " changed generated token count");
}

struct SessionEventLog {
  std::mutex mutex;
  std::condition_variable ready;
  std::vector<nlohmann::json> events;
};

class RealtimeSessionCloseGuard {
public:
  explicit RealtimeSessionCloseGuard(std::shared_ptr<RealtimeSession> session)
      : session_(std::move(session)) {}

  ~RealtimeSessionCloseGuard() { session_->Close(); }

private:
  std::shared_ptr<RealtimeSession> session_;
};

std::shared_ptr<RealtimeSession>
CreateRealtimeSession(TranscriptionWorker &worker,
                      const Qwen3ASRTranscriber &transcriber,
                      const std::shared_ptr<SessionEventLog> &event_log) {
  return std::make_shared<RealtimeSession>(
      worker, transcriber, MakeRealtimeSettings(100, 100, 120.0),
      [event_log](const std::string &serialized_event) {
        {
          std::lock_guard<std::mutex> lock(event_log->mutex);
          event_log->events.push_back(nlohmann::json::parse(serialized_event));
        }
        event_log->ready.notify_all();
        return true;
      });
}

nlohmann::json WaitForSegment(const std::shared_ptr<SessionEventLog> &event_log,
                              int after_event_index, bool is_final) {
  const auto matches = [after_event_index,
                        is_final](const nlohmann::json &event) {
    return event.value("type", std::string()) == "transcription.segment" &&
           event.value("event_index", -1) > after_event_index &&
           event.value("is_final", false) == is_final;
  };
  std::unique_lock<std::mutex> lock(event_log->mutex);
  const bool received = event_log->ready.wait_for(
      lock, kSessionEventTimeout, [event_log, &matches] {
        return std::any_of(event_log->events.begin(), event_log->events.end(),
                           matches);
      });
  Require(received, "timed out waiting for a realtime segment event");
  const auto event =
      std::find_if(event_log->events.begin(), event_log->events.end(), matches);
  return *event;
}

void ConfigureFinalPrompt(RealtimeSession &session) {
  const bool handled = session.Handle(
      {{"type", "session.update"},
       {"session",
        {{"turn_detection", nullptr}, {"final_prompt", "GLOSSARY"}}}});
  Require(handled, "realtime session closed during its update");
}

void AppendAudio(RealtimeSession &session,
                 const std::vector<float> &audio_samples) {
  const std::string pcm_bytes = omni_common::Pcm16Bytes(audio_samples);
  const bool handled =
      session.Handle({{"type", "input_audio_buffer.append"},
                      {"audio", omni_server::EncodeBase64(pcm_bytes)}});
  Require(handled, "realtime session closed during audio append");
}

void TestWrongAutoLanguageGuess(TranscriptionWorker &worker,
                                const Qwen3ASRTranscriber &transcriber,
                                const std::vector<float> &final_audio_samples,
                                bool uses_draft) {
  const std::string guessed_language = "Chinese";
  const std::string guessed_text = "这是故意错误的转录内容。";
  TranscriptionOptions baseline_options;
  baseline_options.context = "GLOSSARY";
  baseline_options.max_new_tokens = 1024;
  baseline_options.stop_at_end_of_text = true;
  baseline_options.stop_on_token_loop = true;
  baseline_options.layout = AudioLayout::kReference;

  TranscriptionOptions candidate_options = baseline_options;
  std::vector<int> guessed_token_ids;
  worker.Transcribe(
      [&transcriber, &guessed_token_ids, &guessed_language,
       &guessed_text](const std::atomic<bool> &) {
        guessed_token_ids =
            transcriber.AutoLanguageGuesses(guessed_language, guessed_text);
        return TranscriptionResult();
      },
      NewCancelFlag());
  Require(!guessed_token_ids.empty(),
          "production auto-language guess construction returned no tokens");
  candidate_options.guessed_token_ids = std::move(guessed_token_ids);
  Require(!baseline_options.language.has_value() &&
              !candidate_options.language.has_value(),
          "automatic-language comparison set an explicit language");
  Require(!candidate_options.guessed_token_ids.empty(),
          "candidate transcription received no guessed token IDs");

  const TranscriptionResult baseline_result = DecodeFinalTranscription(
      worker, transcriber, final_audio_samples, baseline_options);
  const TranscriptionResult candidate_result = DecodeFinalTranscription(
      worker, transcriber, final_audio_samples, candidate_options);
  Require(!baseline_result.text.empty() && baseline_result.language.has_value(),
          "baseline did not produce an authoritative transcript and language");
  Require(*baseline_result.language != guessed_language &&
              baseline_result.text != guessed_text,
          "the test's language and transcript guess was not incorrect");
  Require(candidate_result.text == baseline_result.text,
          "incorrect guessed language or transcript changed the final text");
  Require(candidate_result.language == baseline_result.language,
          "incorrect guessed language or transcript changed detected language");
  Require(
      candidate_result.generated_token_count ==
          baseline_result.generated_token_count,
      "incorrect guessed language or transcript changed generated token count");
  Require(candidate_result.verified_guess_token_count ==
              static_cast<int>(candidate_options.guessed_token_ids.size()),
          "the target did not verify every token in the wrong guess");
  Require(candidate_result.accepted_guess_token_count <
              candidate_result.verified_guess_token_count,
          "the target accepted the deliberately incorrect guess in full");
  Require(candidate_result.finish_reason == baseline_result.finish_reason,
          "incorrect guessed language or transcript changed finish reason");
  if (uses_draft) {
    Require(candidate_result.verify_round_count > 0 &&
                candidate_result.proposed_draft_token_count > 0,
            "DFlash comparison did not verify draft proposals");
  } else {
  }
}

void TestRollbackGuessParity(TranscriptionWorker &worker,
                             const Qwen3ASRTranscriber &transcriber,
                             const std::vector<float> &first_audio_samples,
                             const std::vector<float> &followup_audio_samples) {
  Require(first_audio_samples.size() >= kPreviewAudioSampleCount,
          "rollback parity clip is shorter than the preview input");
  Require(followup_audio_samples.size() >= kFollowupAudioSampleCount,
          "rollback parity follow-up clip is too short");

  const std::vector<float> first_preview_samples(first_audio_samples.begin(),
                                                 first_audio_samples.begin() +
                                                     kPreviewAudioSampleCount);
  const std::vector<float> followup_preview_samples(
      followup_audio_samples.begin(),
      followup_audio_samples.begin() + kFollowupAudioSampleCount);
  std::vector<float> final_input_samples = first_preview_samples;
  final_input_samples.insert(final_input_samples.end(),
                             followup_preview_samples.begin(),
                             followup_preview_samples.end());
  // note (chenye): Preserve the session's PCM16 samples for cache reuse.
  const std::vector<float> final_audio_samples = omni_common::DecodeWav(
      omni_common::WavBytes(omni_common::Pcm16Bytes(final_input_samples),
                            omni_common::kSampleRate));

  const std::shared_ptr<SessionEventLog> event_log =
      std::make_shared<SessionEventLog>();
  const std::shared_ptr<RealtimeSession> session =
      CreateRealtimeSession(worker, transcriber, event_log);
  RealtimeSessionCloseGuard close_session(session);
  const bool handled = session->Handle({{"type", "session.update"},
                                        {"session",
                                         {{"turn_detection", nullptr},
                                          {"language", "en"},
                                          {"final_prompt", ""}}}});
  Require(handled, "realtime session closed during its update");
  AppendAudio(session, first_preview_samples);
  const nlohmann::json first_preview = WaitForSegment(event_log, 0, false);
  AppendAudio(session, followup_preview_samples);
  const int first_preview_index = first_preview.at("event_index").get<int>();
  const nlohmann::json last_preview =
      WaitForSegment(event_log, first_preview_index, false);
  const std::string preview_text = last_preview.at("text").get<std::string>();
  Require(!preview_text.empty(), "rollback parity preview was empty");

  const FinalReuse final_reuse = session->TakeFinalReuse(std::string(), false);
  Require(final_reuse.reuse != nullptr && !final_reuse.is_taken,
          "rollback parity did not capture the segment refresh cache");
  // note (chenye): Each decode gets this unchanged cache state by value.
  const RefreshReuse initial_reuse = *final_reuse.reuse;

  const RetainedText retained = transcriber.RetainedPrefix(preview_text, 5);
  Require(!retained.prefix_ids.empty() && !retained.rest_ids.empty(),
          "rollback parity preview did not produce a retained prefix and tail");

  TranscriptionOptions baseline_options;
  baseline_options.language = "English";
  baseline_options.max_new_tokens =
      static_cast<int>(retained.rest_ids.size()) + kPrefixedTokenSlack;
  baseline_options.stop_on_token_loop = true;
  baseline_options.layout = AudioLayout::kReference;
  baseline_options.prefix_token_ids = retained.prefix_ids;
  baseline_options.prefix_text = retained.prefix_text;

  TranscriptionOptions preview_guess_options = baseline_options;
  preview_guess_options.guessed_token_ids = retained.rest_ids;
  const TranscriptionResult baseline_result =
      DecodeWithReuse(worker, transcriber, final_audio_samples,
                      baseline_options, initial_reuse);
  const TranscriptionResult preview_guess_result =
      DecodeWithReuse(worker, transcriber, final_audio_samples,
                      preview_guess_options, initial_reuse);
  Require(!baseline_result.text.empty() && baseline_result.language.has_value(),
          "no-guess prefixed Final did not produce text and language");
  Require(baseline_result.reused_audio_token_count > 0 &&
              preview_guess_result.reused_audio_token_count > 0,
          "rollback parity did not reuse cached audio rows");
  RequireSameFinalOutput(baseline_result, preview_guess_result,
                         "valid rollback guesses");
  Require(preview_guess_result.verified_guess_token_count ==
              static_cast<int>(retained.rest_ids.size()),
          "the target did not verify every rollback guess");
  Require(preview_guess_result.accepted_guess_token_count >= 0 &&
              preview_guess_result.accepted_guess_token_count <=
                  preview_guess_result.verified_guess_token_count,
          "invalid rollback guess acceptance count");

  std::vector<int> incorrect_guess_ids;
  for (const std::string &incorrect_text :
       {"A deliberately unrelated transcript.",
        "The stale rollback prediction is incorrect."}) {
    std::vector<int> candidate_ids =
        transcriber.RetainedPrefix(incorrect_text, 0).prefix_ids;
    if (!candidate_ids.empty() &&
        candidate_ids.front() != retained.rest_ids.front()) {
      incorrect_guess_ids = std::move(candidate_ids);
      break;
    } else {
    }
  }
  Require(!incorrect_guess_ids.empty(),
          "could not construct a wrong rollback guess");
  incorrect_guess_ids.resize(retained.rest_ids.size(),
                             incorrect_guess_ids.back());
  TranscriptionOptions incorrect_guess_options = baseline_options;
  incorrect_guess_options.guessed_token_ids = incorrect_guess_ids;
  const TranscriptionResult incorrect_guess_result =
      DecodeWithReuse(worker, transcriber, final_audio_samples,
                      incorrect_guess_options, initial_reuse);
  RequireSameFinalOutput(baseline_result, incorrect_guess_result,
                         "a rejected rollback guess");
  Require(incorrect_guess_result.verified_guess_token_count ==
              static_cast<int>(incorrect_guess_ids.size()),
          "the target did not verify every deliberately wrong guess");
  Require(incorrect_guess_result.accepted_guess_token_count <
              incorrect_guess_result.verified_guess_token_count,
          "the target accepted the deliberately wrong rollback guess in full");

  TranscriptionResult expected_final_result = preview_guess_result;
  if (preview_guess_result.finish_reason ==
      omni_common::FinishReason::kLength) {
    TranscriptionOptions retry_options = baseline_options;
    retry_options.prefix_token_ids.clear();
    retry_options.prefix_text.clear();
    retry_options.guessed_token_ids.clear();
    retry_options.max_new_tokens.reset();
    expected_final_result = DecodeWithReuse(
        worker, transcriber, final_audio_samples, retry_options, initial_reuse);
  } else {
  }

  const bool finalized =
      session->Handle({{"type", "input_audio_buffer.commit"}});
  Require(finalized, "realtime session closed during Segment Final");
  const nlohmann::json final_event = WaitForSegment(
      event_log, last_preview.at("event_index").get<int>(), true);
  Require(final_event.at("text").get<std::string>() ==
              expected_final_result.text,
          "Segment Final changed the expected prefixed or retry output");
  const nlohmann::json generation_metadata =
      final_event.at("generation_metadata");
  Require(generation_metadata.at("reused_audio_token_count").get<int>() > 0,
          "Segment Final did not reuse cached audio rows");
  Require(generation_metadata.at("verified_guess_token_count").get<int>() ==
              expected_final_result.verified_guess_token_count,
          "Segment Final verified a different number of rollback guesses");
  Require(generation_metadata.at("accepted_guess_token_count").get<int>() ==
              expected_final_result.accepted_guess_token_count,
          "Segment Final accepted a different number of rollback guesses");
}

void TestPositiveEligibilityControl(
    TranscriptionWorker &worker, const Qwen3ASRTranscriber &transcriber,
    const std::vector<float> &preview_audio_samples) {
  const std::shared_ptr<SessionEventLog> event_log =
      std::make_shared<SessionEventLog>();
  const std::shared_ptr<RealtimeSession> session =
      CreateRealtimeSession(worker, transcriber, event_log);
  RealtimeSessionCloseGuard close_session(session);
  ConfigureFinalPrompt(*session);
  AppendAudio(*session, preview_audio_samples);
  const nlohmann::json preview = WaitForSegment(event_log, 0, false);
  const std::string preview_text = preview.at("text").get<std::string>();
  Require(!preview_text.empty(), "positive control produced an empty preview");

  const FinalReuse final_reuse =
      session->TakeFinalReuse(std::string("GLOSSARY"), true);
  Require(final_reuse.auto_language_guess.has_value(),
          "eligible auto-language preview did not produce a Final guess");
  const qwen3_asr::FinalPreviewGuess &preview_guess =
      *final_reuse.auto_language_guess;
  Require(preview_guess.language == "English",
          "positive control did not detect English speech");
  Require(!preview_guess.text.empty() && preview_guess.text == preview_text,
          "positive control did not retain the displayed preview text");
}

void TestNegativeEligibilityControl(
    TranscriptionWorker &worker, const Qwen3ASRTranscriber &transcriber,
    const std::vector<float> &preview_audio_samples) {
  const std::shared_ptr<SessionEventLog> event_log =
      std::make_shared<SessionEventLog>();
  const std::shared_ptr<RealtimeSession> session =
      CreateRealtimeSession(worker, transcriber, event_log);
  RealtimeSessionCloseGuard close_session(session);
  ConfigureFinalPrompt(*session);
  AppendAudio(*session, preview_audio_samples);
  const nlohmann::json first_preview = WaitForSegment(event_log, 0, false);
  Require(!first_preview.at("text").get<std::string>().empty(),
          "negative control's first preview was empty");

  const bool handled = session->Handle({{"type", "input_audio_buffer.commit"}});
  Require(handled, "realtime session closed during segment commit");
  const int first_preview_index = first_preview.at("event_index").get<int>();
  const nlohmann::json committed_segment =
      WaitForSegment(event_log, first_preview_index, true);
  Require(!committed_segment.at("text").get<std::string>().empty(),
          "negative control did not commit its first segment");
  Require(committed_segment.at("segment_id").get<int>() ==
              first_preview.at("segment_id").get<int>(),
          "negative control committed a different segment than its preview");

  const int committed_event_index =
      committed_segment.at("event_index").get<int>();
  AppendAudio(*session, preview_audio_samples);
  const nlohmann::json later_preview =
      WaitForSegment(event_log, committed_event_index, false);
  Require(!later_preview.at("text").get<std::string>().empty(),
          "negative control did not produce a later preview");
  Require(later_preview.at("segment_id").get<int>() >
              committed_segment.at("segment_id").get<int>(),
          "negative control's preview did not follow the committed segment");

  const FinalReuse final_reuse =
      session->TakeFinalReuse(std::string("GLOSSARY"), true);
  Require(!final_reuse.auto_language_guess.has_value(),
          "committed-session preview incorrectly produced a Final guess");
}

void TestEligibility(TranscriptionWorker &worker,
                     const Qwen3ASRTranscriber &transcriber,
                     const std::vector<float> &corpus_audio_samples) {
  Require(corpus_audio_samples.size() >= kPreviewAudioSampleCount,
          "provisioned clip is shorter than the realtime preview input");
  const std::vector<float> preview_audio_samples(corpus_audio_samples.begin(),
                                                 corpus_audio_samples.begin() +
                                                     kPreviewAudioSampleCount);
  TestPositiveEligibilityControl(worker, transcriber, preview_audio_samples);
  TestNegativeEligibilityControl(worker, transcriber, preview_audio_samples);
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 5 && argc != 6) {
      throw std::invalid_argument("usage: qwen3_asr_auto_language_test TARGET "
                                  "DRAFT WAV MODE [FOLLOWUP_WAV]");
    } else {
    }
    const std::filesystem::path model_directory(argv[1]);
    const std::filesystem::path draft_directory =
        std::string(argv[2]) == "-" ? std::filesystem::path()
                                    : std::filesystem::path(argv[2]);
    const std::vector<float> audio_samples = ReadAudioSamples(argv[3]);
    const std::string test_mode(argv[4]);
    std::unique_ptr<Qwen3ASRTranscriber> transcriber;
    TranscriptionWorker worker(
        [&transcriber, &model_directory, &draft_directory] {
          transcriber = std::make_unique<Qwen3ASRTranscriber>(model_directory,
                                                              draft_directory);
        },
        [] {});
    Require(transcriber != nullptr, "Qwen3-ASR transcriber did not load");
    if (test_mode == "wrong_guess") {
      if (argc != 5) {
        throw std::invalid_argument(
            "wrong_guess does not accept a follow-up WAV");
      } else {
      }
      TestWrongAutoLanguageGuess(worker, *transcriber, audio_samples,
                                 !draft_directory.empty());
    } else if (test_mode == "rollback_guess_parity") {
      if (argc != 6) {
        throw std::invalid_argument(
            "rollback_guess_parity requires a follow-up WAV");
      } else {
      }
      const std::vector<float> followup_audio_samples =
          ReadAudioSamples(argv[5]);
      TestRollbackGuessParity(worker, *transcriber, audio_samples,
                              followup_audio_samples);
    } else if (test_mode == "eligibility") {
      if (argc != 5) {
        throw std::invalid_argument(
            "eligibility does not accept a follow-up WAV");
      } else {
      }
      TestEligibility(worker, *transcriber, audio_samples);
    } else {
      throw std::invalid_argument("unknown auto-language test mode");
    }
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
