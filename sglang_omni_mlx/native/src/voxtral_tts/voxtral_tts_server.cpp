// SPDX-License-Identifier: Apache-2.0
// Native Voxtral TTS server: POST /v1/audio/speech with the CUDA server's
// request fields, answering a WAV or PCM file, or 16-bit PCM streamed as it
// is decoded, and Voxt's supervisor protocol.
//
//   voxtral_tts_server --model-path DIR [--model-name NAME] [--host H]
//       [--port P] [--max-queued-requests N] [--request-timeout-s S]
//   voxtral_tts_server --supervised --model-kind voxtral_tts
//       --model-directory DIR [--max-queued-requests N]
//       [--request-timeout-s S]
//
// Request fields: input, model (the served name, when given), voice (or
// speaker; "default" is cheerful_female), response_format (wav or pcm),
// stream (pcm only), max_new_tokens (audio frames, 12.5 a second) and speed
// (1 only). One request runs at a time and N more (2 by default) may wait; the
// rest get 503. A request not done S seconds (600 by default) after it
// arrives is cancelled, with 504 or an early end to its stream, and so is a
// request whose client disconnects. A request body over 1 MiB gets 413.
#include <algorithm>
#include <array>
#include <filesystem>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/text.h"
#include "qwen3_asr/asr_service.h"
#include "qwen3_asr/speech_service.h"
#include "voxtral_tts/voxtral_tts_model.h"

namespace {

using speech_service::FieldOr;
using speech_service::Json;
using speech_service::PresentFieldOr;

constexpr const char *kDefaultVoice = "cheerful_female";
constexpr size_t kMaxInputCharacters = 4096;
// Far above the largest valid request: kMaxInputCharacters characters, each
// JSON-escaped as a 12-byte surrogate pair, take 48 KiB.
constexpr size_t kMaxRequestBodyBytes = 1024 * 1024;
constexpr int kMaxAudioFrames = 4096;
// A collection field of CUDA's CreateSpeechRequest and the JSON type it takes.
struct CollectionField {
  const char *name;
  Json::value_t type;
  const char *type_description;
};
// Note (Minyi Tang): the fields the CUDA server's Voxtral stages reject
// (validate_voxtral_speech_params), each counted as given by the same rule:
// sampling fields when not null, text fields when not null or "", and
// collections when not empty. CUDA's request model first rejects a collection
// of another type, even an empty one.
constexpr std::array<const char *, 5> kUnsupportedSamplingFields = {
    "repetition_penalty", "seed", "temperature", "top_k", "top_p"};
constexpr std::array<const char *, 5> kUnsupportedTextFields = {
    "instructions", "language", "ref_audio", "ref_text", "task_type"};
constexpr std::array<CollectionField, 2> kUnsupportedCollectionFields = {{
    {"references", Json::value_t::array, "a list"},
    {"stage_params", Json::value_t::object, "an object"},
}};

// Throws naming every unsupported field body gives, in sorted order.
void RejectUnsupportedFields(const Json &body) {
  std::vector<std::string> given;
  for (const char *field : kUnsupportedSamplingFields) {
    if (!FieldOr(body, field, Json()).is_null()) {
      given.push_back(field);
    } else {
    }
  }
  for (const char *field : kUnsupportedTextFields) {
    const Json value = FieldOr(body, field, Json());
    if (!value.is_null() && value != "") {
      given.push_back(field);
    } else {
    }
  }
  for (const CollectionField &field : kUnsupportedCollectionFields) {
    const Json value = FieldOr(body, field.name, Json());
    if (!value.is_null() && value.type() != field.type) {
      throw std::invalid_argument(std::string(field.name) + " must be " +
                                  field.type_description);
    } else if (!value.is_null() && !value.empty()) {
      given.push_back(field.name);
    } else {
    }
  }
  if (given.empty()) {
    return;
  } else {
  }
  std::sort(given.begin(), given.end());
  std::string fields;
  for (const std::string &field : given) {
    fields += (fields.empty() ? "" : ", ") + field;
  }
  throw std::invalid_argument(
      "Voxtral TTS does not support these /v1/audio/speech fields: " + fields +
      ". Supported model-specific fields are voice and max_new_tokens.");
}

class VoxtralTtsService : public speech_service::SpeechServedModel {
public:
  VoxtralTtsService(const std::filesystem::path &model_directory,
                    int max_queued_requests, double request_timeout_seconds)
      : SpeechServedModel(kMaxRequestBodyBytes, max_queued_requests,
                          request_timeout_seconds),
        model_(model_directory) {}

private:
  int SampleRate() const override { return model_.codec().sample_rate(); }

  speech_service::SpeechJob ParseSpeech(const Json &body) override {
    RejectUnsupportedFields(body);
    const Json input = FieldOr(body, "input", Json());
    if (!input.is_string() ||
        omni_common::StripUnicodeWhitespace(input.get<std::string>()).empty()) {
      throw std::invalid_argument("input must be a non-empty string");
    } else {
    }
    const std::string text = input.get<std::string>();
    if (omni_common::CodePointCount(text) > kMaxInputCharacters) {
      throw std::invalid_argument("input must be at most " +
                                  std::to_string(kMaxInputCharacters) +
                                  " characters");
    } else {
    }
    const std::optional<std::string> model = speech_service::ParseModel(body);
    // Note (Minyi Tang): as CUDA's AliasChoices("voice", "speaker"), voice
    // wins whenever it is given, even as null.
    const Json voice_field = PresentFieldOr(
        body, "voice", PresentFieldOr(body, "speaker", Json("default")));
    if (!voice_field.is_string()) {
      throw std::invalid_argument("voice must be a string");
    } else {
    }
    std::string voice = voice_field.get<std::string>();
    const std::map<std::string, int> &voices =
        model_.tokenizer().voice_frame_counts();
    if (voice.empty() || voice == "default") {
      voice = kDefaultVoice;
    } else if (voices.count(voice) == 0) {
      std::string supported;
      for (const auto &[name, frame_count] : voices) {
        supported += (supported.empty() ? "" : ", ") + name;
      }
      throw std::invalid_argument("Unknown voice '" + voice +
                                  "'. Supported voices: " + supported);
    } else {
    }
    speech_service::SpeechJob job = speech_service::ParseCommonFields(body);
    job.model = model;
    const Json speed = PresentFieldOr(body, "speed", Json(1.0));
    if (!speed.is_number() || speed.get<double>() != 1.0) {
      throw std::invalid_argument("speed must be 1.0 on this model");
    } else {
    }
    const Json max_new_tokens =
        FieldOr(body, "max_new_tokens", Json(kMaxAudioFrames));
    if (!max_new_tokens.is_number_integer()) {
      throw std::invalid_argument("max_new_tokens must be an integer");
    } else if (max_new_tokens.get<int64_t>() <= 0) {
      throw std::invalid_argument("max_new_tokens must be greater than 0");
    } else if (max_new_tokens.get<int64_t>() > kMaxAudioFrames) {
      throw std::invalid_argument("max_new_tokens must be at most " +
                                  std::to_string(kMaxAudioFrames));
    } else {
    }
    const int max_frames = max_new_tokens.get<int>();
    const std::vector<int> prompt_ids =
        model_.tokenizer().EncodeSpeechRequest(text, voice);
    job.synthesize = [this, prompt_ids, voice, max_frames](
                         const speech_service::SpeechContext &context) {
      voxtral_tts::CodecStream stream(model_.codec());
      speech_service::SpeechResult result;
      result.prompt_token_count = prompt_ids.size();
      result.finish_reason =
          model_.Generate(prompt_ids, voice, max_frames, context.cancel,
                          [&](const mlx::core::array &codes) {
                            context.check_deadline();
                            ++result.completion_token_count;
                            context.send(stream.Push(codes));
                          });
      if (result.completion_token_count == 0) {
        // Note (Minyi Tang): the CUDA vocoder raises on an end of audio at the
        // first frame rather than answer silence.
        throw std::runtime_error("Voxtral TTS generated no audio codes");
      } else {
      }
      context.send(stream.Finish());
      return result;
    };
    return job;
  }

  voxtral_tts::VoxtralTtsModel model_;
};

} // namespace

int main(int argc, char **argv) {
  asr_service::ServedKind kind;
  kind.model_kind = "voxtral_tts";
  speech_service::AddSpeechFlags(
      kind, [](const std::filesystem::path &model_directory,
               int max_queued_requests, double request_timeout_seconds) {
        return std::make_unique<VoxtralTtsService>(
            model_directory, max_queued_requests, request_timeout_seconds);
      });
  return asr_service::Serve(argc, argv, kind);
}
