// SPDX-License-Identifier: Apache-2.0
// Native Fish Audio Drama 3 server: POST /v1/audio/speech with the CUDA
// server's request fields, answering 44.1 kHz WAV or PCM, or 16-bit PCM
// streamed as it is decoded, and Voxt's supervisor protocol.
//
//   fishaudio_drama3_server --model-path DIR [--model-name NAME] [--host H]
//       [--port P] [--max-queued-requests N] [--request-timeout-s S]
//   fishaudio_drama3_server --supervised --model-kind fishaudio_drama3
//       --model-directory DIR [--max-queued-requests N]
//       [--request-timeout-s S]
//
// Request fields: input (with optional <|speaker:N|> tags and [cues]), model,
// voice (default only), references (each a transcript text and either data,
// base64 mono 44.1 kHz WAV, or vq_codes [10][frames]; together at most 30 s
// and 1024 transcript characters), ref_audio and ref_text (one reference,
// ref_audio a base64 WAV data URL), response_format (wav or pcm), stream (pcm
// only), max_new_tokens (frames per text chunk, 21.5 a second), temperature,
// top_p, top_k, repetition_penalty (1 only) and speed (1 only). Long input is
// generated chunk by chunk. One request runs at a time and N more (2 by
// default) may wait; the rest get 503. A request not done S seconds (600 by
// default) after it arrives is cancelled, with 504 or an early end to its
// stream, and so is a request whose client disconnects.
#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/audio.h"
#include "common/text.h"
#include "fishaudio_drama3/fishaudio_drama3_model.h"
#include "fishaudio_drama3/fishaudio_drama3_prompt.h"
#include "fishaudio_drama3/fishaudio_drama3_text.h"
#include "qwen3_asr/asr_service.h"
#include "qwen3_asr/speech_service.h"

namespace {

using speech_service::FieldOr;
using speech_service::Json;
using speech_service::PresentFieldOr;

constexpr size_t kMaxInputCharacters = 4096;
constexpr int kMaxReferences = 4;
// The references' audio and transcripts together, which the prompt's system
// block holds for every chunk and the voice cache keeps.
constexpr int kMaxReferenceSeconds = 30;
constexpr size_t kMaxReferenceCharacters = 1024;
// The frames of kMaxReferenceSeconds of audio.
constexpr size_t kMaxReferenceFrames =
    (kMaxReferenceSeconds * fishaudio_drama3::kCodecSampleRate +
     fishaudio_drama3::kCodecFrameSamples - 1) /
    fishaudio_drama3::kCodecFrameSamples;
// Room for kMaxReferenceSeconds of float32 WAV as base64.
constexpr size_t kMaxRequestBodyBytes = 16 * 1024 * 1024;
constexpr int kMaxFrames = 4096;
constexpr const char *kDataUrlPrefix = "data:";
// The request fields CUDA's Drama 3 stages reject when given (not null).
constexpr std::array<const char *, 11> kUnsupportedFields = {
    "duration_tokens",
    "initial_codec_chunk_frames",
    "instructions",
    "language",
    "seed",
    "stage_params",
    "stage_sampling",
    "stream_codec_output",
    "suppress_bootstrap_silence",
    "token_count",
    "x_vector_only_mode"};
constexpr std::array<const char *, 3> kReferencePathFields = {
    "audio_path", "ref_audio", "audio"};

// Mono 44.1 kHz samples of base64 WAV bytes, a data URL's included.
std::vector<float> DecodeReferenceAudio(const std::string &encoded,
                                        const std::string &location) {
  std::string base64 = encoded;
  if (base64.starts_with(kDataUrlPrefix)) {
    const size_t comma = base64.find(',');
    if (comma == std::string::npos ||
        base64.substr(0, comma).find(";base64") == std::string::npos) {
      throw std::invalid_argument(location +
                                  " data URL must include base64 media data");
    } else {
    }
    base64 = base64.substr(comma + 1);
  } else {
  }
  const std::optional<std::vector<uint8_t>> bytes =
      omni_common::DecodeBase64(base64);
  if (!bytes.has_value()) {
    throw std::invalid_argument(location + " must be valid base64");
  } else {
  }
  std::vector<float> samples;
  try {
    samples = omni_common::DecodeWav(
        std::string_view(reinterpret_cast<const char *>(bytes->data()),
                         bytes->size()),
        fishaudio_drama3::kCodecSampleRate);
  } catch (const std::invalid_argument &error) {
    throw std::invalid_argument(location + ": " + error.what());
  }
  if (samples.empty()) {
    throw std::invalid_argument(location + " audio is empty");
  } else if (!std::all_of(samples.begin(), samples.end(),
                          [](float sample) { return std::isfinite(sample); })) {
    throw std::invalid_argument(location + " audio must be finite");
  } else if (samples.size() >
             static_cast<size_t>(kMaxReferenceSeconds *
                                 fishaudio_drama3::kCodecSampleRate)) {
    throw std::invalid_argument(location + " audio must be at most " +
                                std::to_string(kMaxReferenceSeconds) +
                                " seconds");
  } else {
  }
  return samples;
}

// A reference's transcript, given in field, checked against the prompt's
// rules.
std::string ReferenceText(const Json &text, const std::string &field) {
  if (!text.is_string()) {
    throw std::invalid_argument(field + " must be a string");
  } else if (omni_common::CodePointCount(text.get<std::string>()) >
             kMaxReferenceCharacters) {
    throw std::invalid_argument(field + " must be at most " +
                                std::to_string(kMaxReferenceCharacters) +
                                " characters");
  } else {
  }
  fishaudio_drama3::ValidateText(text.get<std::string>(), field);
  return text.get<std::string>();
}

class Drama3Service : public speech_service::SpeechServedModel {
public:
  Drama3Service(const std::filesystem::path &model_directory,
                int max_queued_requests, double request_timeout_seconds)
      : SpeechServedModel(kMaxRequestBodyBytes, max_queued_requests,
                          request_timeout_seconds),
        model_(model_directory) {}

private:
  int SampleRate() const override { return fishaudio_drama3::kCodecSampleRate; }

  std::vector<fishaudio_drama3::ReferenceInput>
  ParseReferences(const Json &body) const {
    std::vector<fishaudio_drama3::ReferenceInput> references;
    const Json listed = FieldOr(body, "references", Json::array());
    const Json ref_audio = FieldOr(body, "ref_audio", Json());
    const Json ref_text = FieldOr(body, "ref_text", Json());
    if (!listed.is_array()) {
      throw std::invalid_argument("references must be a list");
    } else if (!listed.empty() && !ref_audio.is_null()) {
      throw std::invalid_argument("give references or ref_audio, not both");
    } else if (listed.size() > static_cast<size_t>(kMaxReferences)) {
      throw std::invalid_argument("at most " + std::to_string(kMaxReferences) +
                                  " references are supported");
    } else if (!ref_audio.is_null()) {
      if (!ref_audio.is_string() ||
          !ref_audio.get<std::string>().starts_with(kDataUrlPrefix)) {
        throw std::invalid_argument(
            "ref_audio must be a base64 WAV data URL on this server");
      } else if (ref_text.is_null()) {
        throw std::invalid_argument("reference transcript is required");
      } else {
      }
      return {{ReferenceText(ref_text, "ref_text"),
               DecodeReferenceAudio(ref_audio.get<std::string>(), "ref_audio"),
               std::nullopt}};
    } else if (!ref_text.is_null()) {
      throw std::invalid_argument("ref_text needs ref_audio");
    } else {
    }
    for (size_t index = 0; index < listed.size(); ++index) {
      const Json &reference = listed.at(index);
      const std::string location = "references." + std::to_string(index);
      if (!reference.is_object()) {
        throw std::invalid_argument(location + " must be an object");
      } else {
      }
      if (!FieldOr(reference, "speaker_embedding", Json()).is_null()) {
        throw std::invalid_argument(location +
                                    ".speaker_embedding is not supported");
      } else {
      }
      fishaudio_drama3::ReferenceInput input;
      input.text =
          ReferenceText(FieldOr(reference, "text", Json()), location + ".text");
      const Json vq_codes = FieldOr(reference, "vq_codes", Json());
      const Json data = FieldOr(reference, "data", Json());
      std::optional<std::string> audio_url;
      for (const char *field : kReferencePathFields) {
        const Json value = FieldOr(reference, field, Json());
        if (value.is_string() &&
            value.get<std::string>().starts_with(kDataUrlPrefix)) {
          audio_url = value.get<std::string>();
        } else if (!value.is_null()) {
          throw std::invalid_argument(
              location + "." + field +
              " must be a base64 WAV data URL on this server");
        } else {
        }
      }
      if (!vq_codes.is_null()) {
        input.codes = fishaudio_drama3::ReferenceCodes(
            vq_codes, model_.codebook_count(), model_.codebook_size(),
            kMaxReferenceFrames, location);
      } else if (data.is_string()) {
        input.samples = DecodeReferenceAudio(data.get<std::string>(), location);
      } else if (audio_url.has_value()) {
        input.samples = DecodeReferenceAudio(*audio_url, location);
      } else {
        throw std::invalid_argument(location + " needs audio or vq_codes");
      }
      references.push_back(std::move(input));
    }
    size_t frame_count = 0;
    size_t character_count = 0;
    for (const fishaudio_drama3::ReferenceInput &reference : references) {
      frame_count += reference.codes.has_value()
                         ? reference.codes->front().size()
                         : (reference.samples.size() +
                            fishaudio_drama3::kCodecFrameSamples - 1) /
                               fishaudio_drama3::kCodecFrameSamples;
      character_count += omni_common::CodePointCount(reference.text);
    }
    // Note (Yitong Li): each reference's audio is padded to whole frames, so
    // references that together hold 30 s may round up to one frame more each.
    if (frame_count > kMaxReferenceFrames + references.size() - 1) {
      throw std::invalid_argument(
          "references must total at most " +
          std::to_string(kMaxReferenceSeconds) + " seconds (" +
          std::to_string(kMaxReferenceFrames) + " frames)");
    } else if (character_count > kMaxReferenceCharacters) {
      throw std::invalid_argument("reference transcripts must total at most " +
                                  std::to_string(kMaxReferenceCharacters) +
                                  " characters");
    } else {
    }
    return references;
  }

  speech_service::SpeechJob ParseSpeech(const Json &body) override {
    std::vector<std::string> given;
    for (const char *field : kUnsupportedFields) {
      if (!FieldOr(body, field, Json()).is_null()) {
        given.push_back(field);
      } else {
      }
    }
    if (!given.empty()) {
      std::string fields;
      for (const std::string &field : given) {
        fields += (fields.empty() ? "" : ", ") + field;
      }
      throw std::invalid_argument(fields + " not supported");
    } else {
    }
    const Json task_type = FieldOr(body, "task_type", Json("Base"));
    if (task_type != "Base") {
      throw std::invalid_argument("task_type " + task_type.dump() +
                                  " is not supported; only Base is");
    } else {
    }
    // Note (Yitong Li): as CUDA's AliasChoices("voice", "speaker"), voice
    // wins whenever it is given, even as null; only the default voice exists.
    const Json voice = PresentFieldOr(
        body, "voice", PresentFieldOr(body, "speaker", Json("default")));
    if (!voice.is_string()) {
      throw std::invalid_argument("voice must be a string");
    } else {
    }
    std::string voice_name;
    for (const char character : voice.get<std::string>()) {
      voice_name.push_back(static_cast<char>(
          std::tolower(static_cast<unsigned char>(character))));
    }
    if (voice_name != "" && voice_name != "default") {
      throw std::invalid_argument("unknown voice " + voice.dump() +
                                  "; use default or references");
    } else {
    }
    const Json input = FieldOr(body, "input", Json());
    if (input.is_string() &&
        omni_common::CodePointCount(input.get<std::string>()) >
            kMaxInputCharacters) {
      throw std::invalid_argument("input must be at most " +
                                  std::to_string(kMaxInputCharacters) +
                                  " characters");
    } else if (!input.is_string() ||
               omni_common::StripUnicodeWhitespace(input.get<std::string>())
                   .empty()) {
      throw std::invalid_argument("input must be a non-empty string");
    } else {
    }
    const std::string text = input.get<std::string>();
    fishaudio_drama3::ValidateText(text, "input");
    if (fishaudio_drama3::SplitText(text).empty()) {
      throw std::invalid_argument("input has nothing to speak");
    } else {
    }
    const std::optional<std::string> model = speech_service::ParseModel(body);
    speech_service::SpeechJob job = speech_service::ParseCommonFields(body);
    job.model = model;
    const Json speed = PresentFieldOr(body, "speed", Json(1.0));
    if (!speed.is_number() || speed.get<double>() != 1.0) {
      throw std::invalid_argument("speed must be 1.0 on this model");
    } else {
    }

    fishaudio_drama3::Sampling sampling;
    const Json temperature = FieldOr(
        body, "temperature", Json(fishaudio_drama3::kDefaultTemperature));
    const Json top_p =
        FieldOr(body, "top_p", Json(fishaudio_drama3::kDefaultTopP));
    const Json top_k = FieldOr(body, "top_k", Json());
    const Json repetition_penalty =
        FieldOr(body, "repetition_penalty", Json(1.0));
    const Json max_new_tokens = FieldOr(
        body, "max_new_tokens", Json(fishaudio_drama3::kDefaultMaxFrames));
    // Sampling runs in float32, so the values are checked as float32.
    if (!temperature.is_number() || !std::isfinite(temperature.get<float>()) ||
        temperature.get<float>() < 0) {
      throw std::invalid_argument("temperature must be a number of at least 0");
    } else if (!top_p.is_number() || !(top_p.get<float>() > 0) ||
               top_p.get<float>() > 1) {
      throw std::invalid_argument("top_p must be a number in (0, 1]");
    } else if (!top_k.is_null() &&
               !(top_k.is_number_unsigned() && top_k.get<uint64_t>() >= 1)) {
      throw std::invalid_argument("top_k must be an integer of at least 1");
    } else if (!repetition_penalty.is_number() ||
               repetition_penalty.get<double>() != 1.0) {
      throw std::invalid_argument("repetition_penalty must be 1.0");
    } else if (!max_new_tokens.is_number_integer() ||
               max_new_tokens.get<int64_t>() <= 0 ||
               max_new_tokens.get<int64_t>() > kMaxFrames) {
      throw std::invalid_argument("max_new_tokens must be an integer in [1, " +
                                  std::to_string(kMaxFrames) + "]");
    } else {
    }
    sampling.temperature = temperature.get<float>();
    sampling.top_p = top_p.get<float>();
    if (!top_k.is_null()) {
      sampling.top_k = static_cast<int>(std::min<uint64_t>(
          top_k.get<uint64_t>(), model_.codebook_size() + 1));
    } else {
    }
    const int max_frames = max_new_tokens.get<int>();
    const std::vector<fishaudio_drama3::ReferenceInput> references =
        ParseReferences(body);
    const bool stream = job.stream;
    job.synthesize = [this, text, references, sampling, max_frames,
                      stream](const speech_service::SpeechContext &context) {
      const std::shared_ptr<fishaudio_drama3::Voice> voice =
          model_.GetVoice(references);
      const fishaudio_drama3::SynthesisResult synthesis = model_.Synthesize(
          text, *voice, sampling, max_frames, stream, context.cancel,
          [&](const mlx::core::array &) { context.check_deadline(); },
          context.send);
      // A chunk may end before its first frame, as in mlx-audio; a request
      // with no frame at all fails, as on CUDA.
      if (synthesis.frame_count == 0) {
        throw std::runtime_error("Drama 3 generated no audio frames");
      } else {
      }
      return speech_service::SpeechResult{
          synthesis.finish_reason,
          static_cast<size_t>(synthesis.prompt_row_count),
          synthesis.frame_count};
    };
    return job;
  }

  fishaudio_drama3::Drama3Model model_;
};

} // namespace

int main(int argc, char **argv) {
  asr_service::ServedKind kind;
  kind.model_kind = "fishaudio_drama3";
  speech_service::AddSpeechFlags(
      kind, [](const std::filesystem::path &model_directory,
               int max_queued_requests, double request_timeout_seconds) {
        return std::make_unique<Drama3Service>(
            model_directory, max_queued_requests, request_timeout_seconds);
      });
  return asr_service::Serve(argc, argv, kind);
}
