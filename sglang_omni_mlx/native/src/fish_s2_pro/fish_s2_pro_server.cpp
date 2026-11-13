// SPDX-License-Identifier: Apache-2.0
// Native Fish S2-Pro server: POST /v1/audio/speech with the CUDA server's
// request fields, voice cloning from reference audio, answering a WAV or PCM
// file, or 16-bit PCM streamed as it is decoded, and Voxt's supervisor
// protocol.
//
//   fish_s2_pro_server --model-path DIR [--model-name NAME] [--host H]
//       [--port P] [--max-queued-requests N] [--request-timeout-s S]
//   fish_s2_pro_server --supervised --model-kind fish_s2_pro
//       --model-directory DIR [--max-queued-requests N]
//       [--request-timeout-s S]
//
// Request fields: input, model (the served name, when given), voice (or
// speaker; "default" only), response_format (wav or pcm), stream (pcm only),
// speed (1 only), max_new_tokens (audio frames, 21.5 a second; 1024 by
// default, at most 2048), the sampling fields temperature, top_p, top_k,
// repetition_penalty and seed, and the voice to clone: ref_audio (a base64
// WAV or a data: URL) with ref_text, or references, a list of {audio or data,
// text} or {vq_codes, text}; at most 4 references, with 60 s of audio and 2048
// transcript characters in all. One request runs at a time and N more (2 by
// default) may wait; the rest get 503. A request not done S seconds (600 by
// default) after it arrives is cancelled, with 504 or an early end to its
// stream, and so is a request whose client disconnects. A request body over
// 16 MiB gets 413.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "common/reference_audio.h"
#include "common/text.h"
#include "fish_s2_pro/fish_s2_pro_model.h"
#include "fish_s2_pro/fish_s2_pro_stream.h"
#include "qwen3_asr/asr_service.h"
#include "qwen3_asr/speech_service.h"

namespace {

using speech_service::FieldOr;
using speech_service::Json;
using speech_service::PresentFieldOr;

constexpr size_t kMaxInputCharacters = 4096;
constexpr size_t kMaxReferences = 4;
constexpr double kMaxReferenceSeconds = 60.0;
constexpr size_t kMaxReferenceCharacters = 2048;
// Room for 60 s of 48 kHz stereo 16-bit reference audio in base64.
constexpr size_t kMaxRequestBodyBytes = 16 * 1024 * 1024;
constexpr int kMaxTopK = 30;
constexpr double kMaxRepetitionPenalty = 2.0;
// Fields of CUDA's CreateSpeechRequest this model does not use, each counted
// as given by the CUDA server's rule: text fields when not null or "", flags
// and counts when not null, and stage_params when not empty.
constexpr std::array<const char *, 3> kUnsupportedTextFields = {
    "instructions", "language", "task_type"};
constexpr std::array<const char *, 6> kUnsupportedValueFields = {
    "duration_tokens",     "initial_codec_chunk_frames",
    "stream_codec_output", "suppress_bootstrap_silence",
    "token_count",         "x_vector_only_mode"};

// A reference as the request gives it: its WAV at its own rate, or the codes
// the client gave, in plain values so that only the worker makes MLX arrays.
struct ParsedReference {
  std::string text;
  omni_common::DecodedAudio wav;
  std::vector<int32_t> codes;
  int frame_count = 0;
};

struct SpeechRequest {
  std::string text;
  bool stream = false;
  int max_frames = fish_s2_pro::kDefaultMaxFrames;
  fish_s2_pro::SamplingOptions sampling;
  std::vector<ParsedReference> references;
};

// Throws naming every unsupported field body gives, in sorted order.
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
      "Fish S2-Pro does not support these /v1/audio/speech fields: " + fields +
      ". Supported model-specific fields are ref_audio, ref_text, references, "
      "max_new_tokens, temperature, top_p, top_k, repetition_penalty and "
      "seed.");
}

// Whether value is a JSON integer too large for int64, which nlohmann would
// wrap to a negative number.
bool IsAboveInt64(const Json &value) {
  return value.is_number_unsigned() &&
         value.get<uint64_t>() >
             static_cast<uint64_t>(std::numeric_limits<int64_t>::max());
}

// An optional number field: fallback when missing or null, else a finite
// number that stays finite as float32.
double NumberOr(const Json &body, const char *name, double fallback) {
  const Json value = FieldOr(body, name, Json(fallback));
  if (!value.is_number() || !std::isfinite(value.get<double>())) {
    throw std::invalid_argument(std::string(name) + " must be a number");
  } else if (std::abs(value.get<double>()) >
             std::numeric_limits<float>::max()) {
    throw std::invalid_argument(std::string(name) +
                                " must fit in a finite float32 number");
  } else {
  }
  return value.get<double>();
}

// The base64 payload of a reference audio string: a base64 WAV or a data:
// URL of one. Paths and URLs are the CUDA server's and not read.
std::string ReferencePayload(const std::string &audio,
                             const std::string &field) {
  if (audio.rfind("data:", 0) == 0) {
    return omni_common::AudioDataUrlPayload(audio, field);
  } else if (audio.rfind("http://", 0) == 0 ||
             audio.rfind("https://", 0) == 0 ||
             audio.rfind("file://", 0) == 0 || audio.rfind("/", 0) == 0 ||
             audio.rfind("./", 0) == 0 || audio.rfind("../", 0) == 0 ||
             audio.rfind("~", 0) == 0) {
    throw std::invalid_argument(
        field + " must be a WAV file as base64 or a data: URL; this server "
                "reads no paths or URLs");
  } else {
    return audio;
  }
}

// The WAV of a reference audio string, at its own rate, checked to resample;
// the worker resamples it once the request is admitted.
omni_common::DecodedAudio DecodeReference(const std::string &audio,
                                          const std::string &field) {
  const std::optional<std::vector<uint8_t>> bytes =
      omni_common::DecodeBase64(ReferencePayload(audio, field));
  if (!bytes.has_value() || bytes->empty()) {
    throw std::invalid_argument(field + " is not valid base64");
  } else {
  }
  omni_common::DecodedAudio wav;
  try {
    wav = omni_common::DecodeAnyWav(std::string_view(
        reinterpret_cast<const char *>(bytes->data()), bytes->size()));
  } catch (const std::invalid_argument &error) {
    throw std::invalid_argument(field + ": " + error.what());
  }
  if (wav.samples.empty()) {
    throw std::invalid_argument(field + " has no samples");
  } else {
  }
  omni_common::CheckResampleRates(wav.sample_rate, fish_dac::kSampleRate);
  return wav;
}

// A references item: {audio_path, ref_audio, audio, data, media_type, text,
// vq_codes}, as CUDA's SpeechReference.
ParsedReference ParseReference(const Json &item, int index,
                               const fish_s2_pro::FishS2ProModel &model) {
  const std::string field = "references[" + std::to_string(index) + "]";
  if (!item.is_object()) {
    throw std::invalid_argument(field + " must be an object");
  } else if (!FieldOr(item, "speaker_embedding", Json()).is_null()) {
    throw std::invalid_argument(field + ".speaker_embedding is not supported "
                                        "by this model");
  } else {
  }
  ParsedReference reference;
  const Json text = FieldOr(item, "text", Json(""));
  if (!text.is_string()) {
    throw std::invalid_argument(field + ".text must be a string");
  } else {
  }
  reference.text = text.get<std::string>();
  const Json codes = FieldOr(item, "vq_codes", Json());
  if (!codes.is_null()) {
    const int codebook_count = model.codebook_count();
    if (!codes.is_array() || static_cast<int>(codes.size()) != codebook_count) {
      throw std::invalid_argument(field + ".vq_codes must have shape (" +
                                  std::to_string(codebook_count) + ", T)");
    } else {
    }
    for (int codebook = 0; codebook < codebook_count; ++codebook) {
      const Json &row = codes.at(codebook);
      if (!row.is_array() || row.empty() ||
          (codebook > 0 && row.size() != codes.at(0).size())) {
        throw std::invalid_argument(field + ".vq_codes must have shape (" +
                                    std::to_string(codebook_count) + ", T)");
      } else {
      }
      // The semantic codebook is the model's; the residual ones are the
      // codec's.
      const int64_t size = codebook == 0 ? model.codebook_size()
                                         : fish_s2_pro::kResidualCodebookSize;
      for (const Json &code : row) {
        if (!code.is_number_integer() || code.get<int64_t>() < 0 ||
            code.get<int64_t>() >= size) {
          throw std::invalid_argument(
              field + ".vq_codes row " + std::to_string(codebook) +
              " must be integers in [0, " + std::to_string(size) + ")");
        } else {
        }
        reference.codes.push_back(code.get<int32_t>());
      }
    }
    reference.frame_count = static_cast<int>(codes.at(0).size());
    return reference;
  } else {
  }
  if (!FieldOr(item, "audio_path", Json()).is_null()) {
    throw std::invalid_argument(field + ".audio_path is not supported; send "
                                        "the WAV as data or audio");
  } else {
  }
  for (const char *name : {"data", "audio", "ref_audio"}) {
    const Json audio = FieldOr(item, name, Json());
    if (audio.is_string()) {
      reference.wav =
          DecodeReference(audio.get<std::string>(), field + "." + name);
      return reference;
    } else if (!audio.is_null()) {
      throw std::invalid_argument(field + "." + name + " must be a string");
    } else {
    }
  }
  throw std::invalid_argument(field + " has no audio or vq_codes");
}

class FishS2ProService : public speech_service::SpeechServedModel {
public:
  FishS2ProService(const std::filesystem::path &model_directory,
                   int max_queued_requests, double request_timeout_seconds)
      : SpeechServedModel(kMaxRequestBodyBytes, max_queued_requests,
                          request_timeout_seconds),
        model_(model_directory) {
    mlx::core::set_cache_limit(fish_s2_pro::kCacheLimitBytes);
  }

private:
  int SampleRate() const override { return fish_dac::kSampleRate; }

  // Every reference's audio is decoded, and the totals checked, before the
  // request is admitted and any is resampled.
  std::vector<ParsedReference> ParseReferences(const Json &body) const {
    const Json listed = FieldOr(body, "references", Json::array());
    const Json reference_audio = FieldOr(body, "ref_audio", Json());
    const Json reference_text = FieldOr(body, "ref_text", Json());
    if (!listed.is_array()) {
      throw std::invalid_argument("references must be a list");
    } else if (!reference_text.is_null() && !reference_text.is_string()) {
      throw std::invalid_argument("ref_text must be a string");
    } else if (!reference_audio.is_null() && !reference_audio.is_string()) {
      throw std::invalid_argument("ref_audio must be a string");
    } else if (listed.size() + (reference_audio.is_null() ? 0 : 1) >
               kMaxReferences) {
      throw std::invalid_argument("at most " + std::to_string(kMaxReferences) +
                                  " references are supported");
    } else {
    }
    std::vector<ParsedReference> references;
    double seconds = 0.0;
    size_t characters = 0;
    const auto add = [&](ParsedReference reference) {
      seconds += reference.codes.empty()
                     ? static_cast<double>(reference.wav.samples.size()) /
                           reference.wav.sample_rate
                     : static_cast<double>(reference.frame_count) *
                           fish_dac::kFrameSamples / fish_dac::kSampleRate;
      characters += omni_common::CodePointCount(reference.text);
      if (seconds > kMaxReferenceSeconds) {
        throw std::invalid_argument(
            "the references are longer than " +
            std::to_string(static_cast<int>(kMaxReferenceSeconds)) +
            " s in all");
      } else if (characters > kMaxReferenceCharacters) {
        throw std::invalid_argument(
            "reference transcripts must total at most " +
            std::to_string(kMaxReferenceCharacters) + " characters");
      } else {
      }
      references.push_back(std::move(reference));
    };
    for (size_t index = 0; index < listed.size(); ++index) {
      add(ParseReference(listed.at(index), static_cast<int>(index), model_));
    }
    if (reference_audio.is_string()) {
      add({reference_text.is_string() ? reference_text.get<std::string>() : "",
           DecodeReference(reference_audio.get<std::string>(), "ref_audio"),
           {},
           0});
    } else {
    }
    return references;
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
    auto request = std::make_shared<SpeechRequest>();
    request->text = input.get<std::string>();
    request->stream = job.stream;

    // Note (Minyi Tang): as CUDA's AliasChoices("voice", "speaker"), voice wins
    // whenever it is given, even as null.
    const Json voice = PresentFieldOr(
        body, "voice", PresentFieldOr(body, "speaker", Json("default")));
    if (!voice.is_string()) {
      throw std::invalid_argument("voice must be a string");
    } else if (voice != "" && voice != "default") {
      throw std::invalid_argument("Unknown voice '" + voice.get<std::string>() +
                                  "'. Supported voices: default; clone a voice "
                                  "with ref_audio or references");
    } else {
    }

    const Json speed = PresentFieldOr(body, "speed", Json(1.0));
    if (!speed.is_number() || speed.get<double>() != 1.0) {
      throw std::invalid_argument("speed must be 1.0 on this model");
    } else {
    }
    const Json max_new_tokens =
        FieldOr(body, "max_new_tokens", Json(fish_s2_pro::kDefaultMaxFrames));
    if (!max_new_tokens.is_number_integer()) {
      throw std::invalid_argument("max_new_tokens must be an integer");
    } else if (IsAboveInt64(max_new_tokens)) {
      request->max_frames = fish_s2_pro::kMaxFrames;
    } else if (max_new_tokens.get<int64_t>() <= 0) {
      throw std::invalid_argument("max_new_tokens must be greater than 0");
    } else {
      request->max_frames = static_cast<int>(std::min<int64_t>(
          max_new_tokens.get<int64_t>(), fish_s2_pro::kMaxFrames));
    }

    // Note (Jiaxin Deng): temperature, top_p and top_k take the bounds SGLang's
    // SamplingParams.verify and the CUDA model's top_k check put on them.
    fish_s2_pro::SamplingOptions &sampling = request->sampling;
    const double temperature =
        NumberOr(body, "temperature", sampling.temperature);
    if (temperature < 0.0) {
      throw std::invalid_argument("temperature must be a non-negative number");
    } else {
    }
    sampling.temperature = static_cast<float>(temperature);
    const double top_p = NumberOr(body, "top_p", sampling.top_p);
    if (!(static_cast<float>(top_p) > 0.0f && top_p <= 1.0)) {
      throw std::invalid_argument("top_p must be in (0, 1]");
    } else {
    }
    sampling.top_p = static_cast<float>(top_p);
    const Json top_k = FieldOr(body, "top_k", Json(sampling.top_k));
    if (!top_k.is_number_integer() || IsAboveInt64(top_k) ||
        (top_k.get<int64_t>() != -1 &&
         (top_k.get<int64_t>() < 1 || top_k.get<int64_t>() > kMaxTopK))) {
      throw std::invalid_argument("top_k must be -1 or between 1 and " +
                                  std::to_string(kMaxTopK));
    } else {
    }
    sampling.top_k = top_k.get<int>();
    // CUDA bounds repetition_penalty nowhere, and its SamplingParams get a
    // fixed 1.0; this range is this server's.
    const double repetition_penalty =
        NumberOr(body, "repetition_penalty", sampling.repetition_penalty);
    if (!(static_cast<float>(repetition_penalty) > 0.0f &&
          repetition_penalty <= kMaxRepetitionPenalty)) {
      throw std::invalid_argument("repetition_penalty must be in (0, 2]");
    } else {
    }
    sampling.repetition_penalty = static_cast<float>(repetition_penalty);
    const Json seed = FieldOr(body, "seed", Json());
    if (seed.is_null()) {
      static thread_local std::mt19937 generator{std::random_device{}()};
      sampling.seed =
          static_cast<uint32_t>(generator()) & fish_s2_pro::kSeedMask;
    } else if (!seed.is_number_integer()) {
      throw std::invalid_argument("seed must be an integer");
    } else if (!seed.is_number_unsigned() && seed.get<int64_t>() < 0) {
      // Note (Jiaxin Deng): as the CUDA server's validate_non_negative_int.
      throw std::invalid_argument("seed must be greater than or equal to 0");
    } else {
      sampling.seed =
          static_cast<uint32_t>(seed.get<uint64_t>() & fish_s2_pro::kSeedMask);
    }

    request->references = ParseReferences(body);
    job.synthesize = [this,
                      request = std::shared_ptr<const SpeechRequest>(request)](
                         const speech_service::SpeechContext &context) {
      return Synthesize(*request, context);
    };
    return job;
  }

  speech_service::SpeechResult
  Synthesize(const SpeechRequest &request,
             const speech_service::SpeechContext &context) const {
    std::vector<fish_s2_pro::VoiceReference> references;
    for (const ParsedReference &reference : request.references) {
      context.check_deadline();
      if (context.cancel.load()) {
        throw qwen3_asr::TranscriptionCancelled();
      } else {
      }
      if (reference.codes.empty()) {
        references.push_back(
            {reference.text,
             model_.codec().Encode(omni_common::Resample(
                 reference.wav.samples, reference.wav.sample_rate,
                 fish_dac::kSampleRate))});
      } else {
        references.push_back(
            {reference.text,
             mlx::core::array(reference.codes.begin(),
                              {model_.codebook_count(), reference.frame_count},
                              mlx::core::int32)});
      }
    }
    const fish_s2_pro::SpeechPrompt prompt =
        model_.BuildPrompt(request.text, references);
    fish_s2_pro::VocoderStream stream(model_.codec());
    std::vector<mlx::core::array> frames;
    int frame_count = 0;
    const qwen3_asr::FinishReason finish_reason =
        model_.Generate(prompt, request.max_frames, request.sampling,
                        context.cancel, [&](const mlx::core::array &codes) {
                          context.check_deadline();
                          ++frame_count;
                          if (request.stream) {
                            context.send(stream.Push(codes));
                          } else {
                            frames.push_back(codes);
                          }
                        });
    if (frame_count == 0) {
      // Note (Minyi Tang): the CUDA vocoder raises on an end of audio at the
      // first frame rather than answer silence.
      throw std::runtime_error("Fish S2-Pro generated no audio codes");
    } else if (request.stream) {
      context.send(stream.Finish());
    } else {
      // Note (Jiaxin Deng): as the CUDA vocoder, a whole request decodes in
      // one window.
      context.check_deadline();
      const mlx::core::array audio =
          model_.codec().Decode(mlx::core::stack(frames, 1));
      mlx::core::eval(audio);
      context.send(std::vector<float>(audio.data<float>(),
                                      audio.data<float>() + audio.size()));
    }
    return speech_service::SpeechResult{finish_reason, prompt.ids.size(),
                                        frame_count};
  }

  fish_s2_pro::FishS2ProModel model_;
};

} // namespace

int main(int argc, char **argv) {
  asr_service::ServedKind kind;
  kind.model_kind = "fish_s2_pro";
  speech_service::AddSpeechFlags(
      kind, [](const std::filesystem::path &model_directory,
               int max_queued_requests, double request_timeout_seconds) {
        return std::make_unique<FishS2ProService>(
            model_directory, max_queued_requests, request_timeout_seconds);
      });
  return asr_service::Serve(argc, argv, kind);
}
