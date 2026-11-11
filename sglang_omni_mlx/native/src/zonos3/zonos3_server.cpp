// SPDX-License-Identifier: Apache-2.0
// Native ZONOS3-Mini server: POST /v1/audio/speech with the CUDA server's
// request fields, answering a WAV or PCM file, or 16-bit PCM streamed as it
// is decoded, and Voxt's supervisor protocol.
//
//   zonos3_server --model-path DIR [--model-name NAME] [--host H] [--port P]
//       [--max-queued-requests N] [--request-timeout-s S]
//   zonos3_server --supervised --model-kind zonos3 --model-directory DIR
//       [--max-queued-requests N] [--request-timeout-s S]
//
// Request fields: input, model (the served name, when given), voice (or
// speaker: a bundled voice, "default" for jett, or "none"), response_format
// (wav or pcm), stream (pcm only), speed, language (English text is
// normalized; another language is read as written), max_new_tokens (delayed
// rows, 15 a second; 2000 by default), temperature, top_k, top_p,
// repetition_penalty, seed, condition_buckets or condition_values (objects
// keyed by loudness, fmax and speaking_rate), prepend_silence, and one voice
// to clone: ref_audio (a base64 WAV or a data: URL) or references, a list of
// one {audio or data} or {speaker_embedding}; transcripts are not needed. One
// request runs at a time and N more (2 by default) may wait; the rest get
// 503. A request not done S seconds (600 by default) after it arrives is
// cancelled, with 504 or an early end to its stream, and so is a request
// whose client disconnects. A request body over 16 MiB gets 413.
#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/reference_audio.h"
#include "common/text.h"
#include "qwen3_asr/asr_service.h"
#include "qwen3_asr/speech_service.h"
#include "zonos3/zonos3_model.h"

namespace {

using speech_service::DecodeReferenceAudio;
using speech_service::FieldOr;
using speech_service::IsAboveInt64;
using speech_service::Json;
using speech_service::NumberOr;
using speech_service::PresentFieldOr;

constexpr size_t kMaxInputCharacters = 4096;
// Room for 60 s of 48 kHz stereo 16-bit reference audio in base64.
constexpr size_t kMaxRequestBodyBytes = 16 * 1024 * 1024;
// Seeds stay below 2^63, as the CUDA server's.
constexpr uint64_t kSeedLimit = uint64_t{1} << 63;
constexpr std::array<const char *, 2> kUnsupportedTextFields = {"instructions",
                                                                "task_type"};
constexpr std::array<const char *, 6> kUnsupportedValueFields = {
    "duration_tokens",     "initial_codec_chunk_frames",
    "stream_codec_output", "suppress_bootstrap_silence",
    "token_count",         "x_vector_only_mode"};

// The voice a request asks for, in plain values so that only the worker
// touches the model.
struct ParsedVoice {
  std::optional<std::string> bundled;
  std::optional<omni_common::DecodedAudio> audio;
  std::optional<std::vector<float>> embedding;
};

void RejectUnsupportedFields(const Json &body) {
  std::vector<std::string> given;
  for (const char *field : kUnsupportedTextFields) {
    const Json value = FieldOr(body, field, Json());
    if (!value.is_null() && value != "") {
      given.push_back(field);
    } else {
    }
  }
  for (const char *field : kUnsupportedValueFields) {
    if (!FieldOr(body, field, Json()).is_null()) {
      given.push_back(field);
    } else {
    }
  }
  const Json stage_params = FieldOr(body, "stage_params", Json());
  if (!stage_params.is_null() && !stage_params.is_object()) {
    throw std::invalid_argument("stage_params must be an object");
  } else if (!stage_params.is_null() && !stage_params.empty()) {
    given.push_back("stage_params");
  } else {
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
      "ZONOS3 does not support these /v1/audio/speech fields: " + fields);
}

std::string AsciiLower(std::string text) {
  for (char &character : text) {
    if (character >= 'A' && character <= 'Z') {
      character = static_cast<char>(character - 'A' + 'a');
    } else {
    }
  }
  return text;
}

std::optional<std::map<std::string, nlohmann::json>>
ConditionField(const Json &body, const char *name) {
  const Json value = FieldOr(body, name, Json());
  if (value.is_null()) {
    return std::nullopt;
  } else if (!value.is_object()) {
    throw std::invalid_argument(std::string(name) + " must be an object");
  } else {
  }
  std::map<std::string, nlohmann::json> entries;
  for (const auto &[key, entry] : value.items()) {
    entries[key] = nlohmann::json::parse(entry.dump());
  }
  return entries;
}

class Zonos3Service : public speech_service::SpeechServedModel {
public:
  Zonos3Service(const std::filesystem::path &model_directory,
                int max_queued_requests, double request_timeout_seconds)
      : SpeechServedModel(kMaxRequestBodyBytes, max_queued_requests,
                          request_timeout_seconds),
        model_(model_directory) {}

private:
  int SampleRate() const override { return zonos3::kSampleRate; }

  ParsedVoice ParseVoice(const Json &body) const {
    const Json listed = FieldOr(body, "references", Json::array());
    const Json reference_audio = FieldOr(body, "ref_audio", Json());
    if (!listed.is_array()) {
      throw std::invalid_argument("references must be a list");
    } else if (!reference_audio.is_null() && !reference_audio.is_string()) {
      throw std::invalid_argument("ref_audio must be a string");
    } else if (listed.size() + (reference_audio.is_null() ? 0 : 1) > 1) {
      throw std::invalid_argument("at most one reference is supported");
    } else {
    }
    ParsedVoice voice;
    if (reference_audio.is_string()) {
      voice.audio = DecodeReferenceAudio(reference_audio.get<std::string>(),
                                         "ref_audio", zonos3::kSampleRate);
      return voice;
    } else if (listed.size() == 1) {
      const Json &item = listed.at(0);
      if (!item.is_object()) {
        throw std::invalid_argument("references[0] must be an object");
      } else if (!FieldOr(item, "vq_codes", Json()).is_null()) {
        throw std::invalid_argument(
            "references[0].vq_codes is not supported by this model; send "
            "audio or a speaker_embedding");
      } else if (!FieldOr(item, "audio_path", Json()).is_null()) {
        throw std::invalid_argument("references[0].audio_path is not "
                                    "supported; send the WAV as data or audio");
      } else {
      }
      const Json embedding = FieldOr(item, "speaker_embedding", Json());
      if (!embedding.is_null()) {
        if (!embedding.is_array() ||
            static_cast<int>(embedding.size()) !=
                model_.config().speaker_embedding_size) {
          throw std::invalid_argument(
              "references[0].speaker_embedding must be " +
              std::to_string(model_.config().speaker_embedding_size) +
              " numbers");
        } else {
        }
        std::vector<float> values;
        for (const Json &value : embedding) {
          if (!value.is_number() || !std::isfinite(value.get<double>())) {
            throw std::invalid_argument(
                "references[0].speaker_embedding must be finite numbers");
          } else {
          }
          values.push_back(value.get<float>());
        }
        voice.embedding = values;
        return voice;
      } else {
      }
      for (const char *name : {"data", "audio", "ref_audio"}) {
        const Json audio = FieldOr(item, name, Json());
        if (audio.is_string()) {
          voice.audio = DecodeReferenceAudio(
              audio.get<std::string>(), std::string("references[0].") + name,
              zonos3::kSampleRate);
          return voice;
        } else if (!audio.is_null()) {
          throw std::invalid_argument(std::string("references[0].") + name +
                                      " must be a string");
        } else {
        }
      }
      throw std::invalid_argument(
          "references[0] has no audio or speaker_embedding");
    } else {
    }
    // Note: as CUDA's AliasChoices("voice", "speaker"), voice wins whenever it
    // is given, even as null.
    const Json name = PresentFieldOr(
        body, "voice", PresentFieldOr(body, "speaker", Json("default")));
    if (!name.is_string()) {
      throw std::invalid_argument("voice must be a string");
    } else {
    }
    const std::string lowered = AsciiLower(name.get<std::string>());
    if (lowered.empty() || lowered == "default") {
      voice.bundled = zonos3::kDefaultVoice;
    } else if (lowered == zonos3::kNoVoice) {
    } else if (model_.HasVoice(lowered)) {
      voice.bundled = lowered;
    } else {
      std::string supported;
      for (const std::string &voice_name : model_.VoiceNames()) {
        supported += (supported.empty() ? "" : ", ") + voice_name;
      }
      throw std::invalid_argument("Unknown voice '" + name.get<std::string>() +
                                  "'. Supported voices: " + supported +
                                  ", none");
    }
    return voice;
  }

  speech_service::SpeechJob ParseSpeech(const Json &body) override {
    RejectUnsupportedFields(body);
    const Json input = FieldOr(body, "input", Json());
    if (!input.is_string() ||
        omni_common::StripUnicodeWhitespace(input.get<std::string>()).empty()) {
      throw std::invalid_argument("input must be a non-empty string");
    } else if (omni_common::CodePointCount(input.get<std::string>()) >
               kMaxInputCharacters) {
      throw std::invalid_argument("input must be at most " +
                                  std::to_string(kMaxInputCharacters) +
                                  " characters");
    } else {
    }
    speech_service::SpeechJob job = speech_service::ParseCommonFields(body);
    job.model = speech_service::ParseModel(body);
    auto request = std::make_shared<zonos3::SynthesisRequest>();
    request->text = input.get<std::string>();
    request->stream = job.stream;

    const Json language = FieldOr(body, "language", Json("en"));
    if (!language.is_string()) {
      throw std::invalid_argument("language must be a string");
    } else {
    }
    request->language = language.get<std::string>();

    const Json max_new_tokens =
        FieldOr(body, "max_new_tokens", Json(zonos3::kDefaultMaxRows));
    if (!max_new_tokens.is_number_integer()) {
      throw std::invalid_argument("max_new_tokens must be an integer");
    } else if (IsAboveInt64(max_new_tokens)) {
      request->max_rows = model_.config().max_rows;
    } else if (max_new_tokens.get<int64_t>() <= 0) {
      throw std::invalid_argument("max_new_tokens must be greater than 0");
    } else {
      request->max_rows = static_cast<int>(std::min<int64_t>(
          max_new_tokens.get<int64_t>(), model_.config().max_rows));
    }

    zonos3::Sampling &sampling = request->sampling;
    sampling.temperature = NumberOr(body, "temperature", sampling.temperature);
    if (sampling.temperature < 0.0) {
      throw std::invalid_argument(
          "temperature must be >= 0 (0 decodes greedily)");
    } else {
    }
    sampling.top_p = NumberOr(body, "top_p", sampling.top_p);
    if (!(sampling.top_p > 0.0 && sampling.top_p <= 1.0)) {
      throw std::invalid_argument("top_p must be in (0, 1]");
    } else {
    }
    const int vocabulary = model_.config().audio_vocabulary_size;
    const Json top_k = FieldOr(body, "top_k", Json(sampling.top_k));
    if (!top_k.is_number_integer() || IsAboveInt64(top_k) ||
        top_k.get<int64_t>() < 1 || top_k.get<int64_t>() > vocabulary) {
      throw std::invalid_argument("top_k must be in [1, " +
                                  std::to_string(vocabulary) + "]");
    } else {
    }
    sampling.top_k = top_k.get<int>();
    sampling.repetition_penalty =
        NumberOr(body, "repetition_penalty", sampling.repetition_penalty);
    if (sampling.repetition_penalty < 1.0) {
      throw std::invalid_argument(
          "repetition_penalty must be >= 1 (1 disables it)");
    } else {
    }
    const Json seed = FieldOr(body, "seed", Json());
    if (seed.is_null()) {
      static thread_local std::mt19937_64 generator{std::random_device{}()};
      sampling.seed = generator() % kSeedLimit;
    } else if (!seed.is_number_integer() ||
               (!seed.is_number_unsigned() && seed.get<int64_t>() < 0) ||
               seed.get<uint64_t>() >= kSeedLimit) {
      throw std::invalid_argument("seed must be an integer in [0, 2**63)");
    } else {
      sampling.seed = seed.get<uint64_t>();
    }

    const Json prepend_silence = FieldOr(body, "prepend_silence", Json(true));
    if (!prepend_silence.is_boolean()) {
      throw std::invalid_argument("prepend_silence must be a boolean");
    } else {
    }
    request->silence_prefill_frames =
        prepend_silence.get<bool>() ? zonos3::kDefaultSilencePrefillFrames : 0;

    const auto buckets = ConditionField(body, "condition_buckets");
    const auto values = ConditionField(body, "condition_values");
    const double speed = NumberOr(body, "speed", 1.0);
    if (!(speed > 0.0)) {
      throw std::invalid_argument("speed must be positive");
    } else if (speed != 1.0 && (buckets.has_value() || values.has_value())) {
      throw std::invalid_argument(
          "speed cannot be combined with condition_buckets or "
          "condition_values");
    } else if (speed != 1.0) {
      request->condition_buckets = model_.SpeedConditionIds(speed);
    } else {
      request->condition_buckets = model_.ConditionIds(buckets, values);
    }

    const auto voice = std::make_shared<const ParsedVoice>(ParseVoice(body));
    // Prompt length errors are a 400, so the prompt is built before the
    // request is admitted.
    zonos3::SynthesisRequest probe = *request;
    if (voice->bundled.has_value() || voice->audio.has_value() ||
        voice->embedding.has_value()) {
      probe.speaker_embedding =
          std::vector<float>(model_.config().speaker_embedding_size, 0.0f);
    } else {
    }
    model_.BuildPrompt(probe);

    job.synthesize =
        [this, voice,
         request = std::shared_ptr<const zonos3::SynthesisRequest>(request)](
            const speech_service::SpeechContext &context) {
          return Synthesize(*request, *voice, context);
        };
    return job;
  }

  speech_service::SpeechResult
  Synthesize(const zonos3::SynthesisRequest &parsed, const ParsedVoice &voice,
             const speech_service::SpeechContext &context) {
    zonos3::SynthesisRequest request = parsed;
    if (voice.embedding.has_value()) {
      request.speaker_embedding = *voice.embedding;
    } else if (voice.audio.has_value()) {
      context.check_deadline();
      const omni_common::DecodedAudio &audio = *voice.audio;
      request.speaker_embedding = model_.ReferenceEmbedding(
          audio.sample_rate == zonos3::kSampleRate
              ? audio.samples
              : omni_common::Resample(audio.samples, audio.sample_rate,
                                      zonos3::kSampleRate));
    } else if (voice.bundled.has_value()) {
      request.speaker_embedding = model_.VoiceEmbedding(*voice.bundled);
    } else {
    }
    const zonos3::SynthesisResult result = model_.Synthesize(
        request, context.cancel,
        [&](const mlx::core::array &) { context.check_deadline(); },
        [&](std::vector<float> audio) { context.send(std::move(audio)); });
    if (result.frame_count == 0) {
      throw std::runtime_error("ZONOS3 generated no audio frames");
    } else {
    }
    return speech_service::SpeechResult{
        result.finish_reason, static_cast<size_t>(result.prompt_row_count),
        result.generated_row_count};
  }

  zonos3::Zonos3Model model_;
};

} // namespace

int main(int argc, char **argv) {
  asr_service::ServedKind kind;
  kind.model_kind = "zonos3";
  speech_service::AddSpeechFlags(
      kind, [](const std::filesystem::path &model_directory,
               int max_queued_requests, double request_timeout_seconds) {
        return std::make_unique<Zonos3Service>(
            model_directory, max_queued_requests, request_timeout_seconds);
      });
  return asr_service::Serve(argc, argv, kind);
}
