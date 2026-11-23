// SPDX-License-Identifier: Apache-2.0
// Native Fun-CosyVoice3 server: POST /v1/audio/speech with the CUDA server's
// request fields, answering a WAV or PCM file, or 16-bit PCM as a stream, and
// Voxt's supervisor protocol.
//
//   cosyvoice3_server --model-path DIR [--model-name NAME] [--host H]
//       [--port P] [--max-queued-requests N] [--request-timeout-s S]
//       [--allowed-local-media-path DIR]
//   cosyvoice3_server --supervised --model-kind cosyvoice3
//       --model-directory DIR [--max-queued-requests N]
//       [--request-timeout-s S] [--allowed-local-media-path DIR]
//
// Request fields: input, model (the served name, when given), voice (or
// speaker; only "default"), instructions, language (accepted, unused, as on
// CUDA), response_format (wav or pcm), stream (pcm only), speed (1 only),
// temperature, top_p, top_k, repetition_penalty, seed and max_new_tokens
// (speech tokens, 25 a second). A voice is cloned from ref_audio with its
// transcript ref_text, or from one references item (audio_path, ref_audio,
// audio or data, and text): 0.5 to 30 s of WAV as a data URL or base64, or a
// path or file URL inside --allowed-local-media-path. Without a transcript
// the reference conditions only the flow, as on CUDA. A stream sends the whole
// utterance's PCM once it is vocoded, since the flow decodes it at once. One
// request runs at a time and N more (2 by default) may wait; the rest get
// 503. A request not done S seconds (600 by default) after it arrives is
// cancelled, with 504 or an early end to its stream, and so is a request whose
// client disconnects. A request body over 32 MiB gets 413.
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <system_error>
#include <tuple>
#include <vector>

#include "common/asr_service.h"
#include "common/generation.h"
#include "common/reference_audio.h"
#include "common/speech_service.h"
#include "common/text.h"
#include "cosyvoice3/cosyvoice3_model.h"

namespace {

using speech_service::FieldOr;
using speech_service::Json;
using speech_service::PresentFieldOr;

constexpr size_t kMaxInputCharacters = 4096;
// Room for 30 s of base64 WAV at 48 kHz in stereo float32 (15 MiB) and
// kMaxInputCharacters characters of text in every text field.
constexpr size_t kMaxRequestBodyBytes = 32 * 1024 * 1024;
constexpr size_t kMaxReferenceFileBytes = 24 * 1024 * 1024;
// The reference speech tokens a prompt can hold: 30 s at 25 a second.
constexpr size_t kMaxReferenceSpeechTokens = 750;
// The most a repetition penalty may be, as SGLang's sampler allows.
constexpr double kMaxRepetitionPenalty = 2.0;
// Note (Minyi Tang): fields this server does not serve, each counted as
// given by the CUDA request model's rule: text and flag fields when not null
// or "", and collections when not empty. CUDA's request model first rejects a
// collection of another type, even an empty one.
const speech_service::UnsupportedFields kUnsupportedFields = {
    {},
    {"duration_tokens", "initial_codec_chunk_frames", "stream_codec_output",
     "suppress_bootstrap_silence", "task_type", "token_count",
     "x_vector_only_mode"},
    {
        {"stage_params", Json::value_t::object, "an object"},
    },
};

struct SpeechRequest {
  cosyvoice3::SpeechText text;
  // The reference recording to clone, decoded and resampled.
  std::optional<cosyvoice3::ReferenceAudio> reference_audio;
  // The prompt's text ids and the target text's, from EncodeText.
  std::vector<int> prompt_ids;
  std::vector<int> target_ids;
  cosyvoice3::SamplingOptions sampling;
  std::optional<std::string> model;
  std::string response_format;
  bool stream = false;
};

bool IsInside(const std::filesystem::path &path,
              const std::filesystem::path &directory) {
  const auto [directory_end, path_at] = std::mismatch(
      directory.begin(), directory.end(), path.begin(), path.end());
  return directory_end == directory.end();
}

// The bytes of a regular file inside allowed_directory.
// Note (Jiaxin Deng): the opened file's own path is checked, so a symlink
// swapped in after the name was resolved cannot lead outside the directory.
std::string ReadAllowedFile(const std::string &path,
                            const std::filesystem::path &allowed_directory) {
  const int file_descriptor = open(path.c_str(), O_RDONLY | O_NONBLOCK);
  if (file_descriptor < 0) {
    throw std::invalid_argument("reference audio file not found");
  } else {
  }
  std::string bytes;
  char opened_path[PATH_MAX] = "";
  struct stat status {};
  const bool is_allowed =
      fcntl(file_descriptor, F_GETPATH, opened_path) == 0 &&
      IsInside(std::filesystem::path(opened_path), allowed_directory);
  const bool is_regular =
      fstat(file_descriptor, &status) == 0 && S_ISREG(status.st_mode);
  const auto size = static_cast<size_t>(is_regular ? status.st_size : 0);
  bool is_read = is_allowed && is_regular && size <= kMaxReferenceFileBytes;
  if (is_read) {
    bytes.resize(size);
    size_t done = 0;
    while (is_read && done < size) {
      const ssize_t count =
          read(file_descriptor, bytes.data() + done, size - done);
      is_read = count > 0;
      done += is_read ? static_cast<size_t>(count) : 0;
    }
  } else {
  }
  close(file_descriptor);
  if (!is_allowed) {
    throw std::invalid_argument(
        "reference audio path is outside --allowed-local-media-path");
  } else if (!is_regular) {
    throw std::invalid_argument("reference audio file not found");
  } else if (size > kMaxReferenceFileBytes) {
    throw std::invalid_argument("reference audio file is larger than " +
                                std::to_string(kMaxReferenceFileBytes) +
                                " bytes");
  } else if (!is_read) {
    throw std::invalid_argument("reference audio file cannot be read");
  } else {
  }
  return bytes;
}

std::string Base64Bytes(const std::string &text) {
  const std::optional<std::vector<uint8_t>> bytes =
      omni_common::DecodeBase64(text);
  if (!bytes.has_value() || bytes->empty()) {
    throw std::invalid_argument("reference audio is not valid base64");
  } else {
  }
  return std::string(bytes->begin(), bytes->end());
}

// The WAV bytes a reference names: a base64 data URL, or a path or file URL
// inside allowed_directory.
std::string
ReferenceBytes(const std::string &source,
               const std::optional<std::filesystem::path> &allowed_directory) {
  if (source.rfind("data:", 0) == 0) {
    return Base64Bytes(
        omni_common::AudioDataUrlPayload(source, "reference audio"));
  } else if (source.rfind("http://", 0) == 0 ||
             source.rfind("https://", 0) == 0) {
    throw std::invalid_argument(
        "reference audio URLs are not supported; send a data URL");
  } else if (!allowed_directory.has_value()) {
    throw std::invalid_argument(
        "local reference audio paths need --allowed-local-media-path; send a "
        "data URL");
  } else {
  }
  std::string path = source;
  if (source.rfind("file://", 0) == 0) {
    // Note (Jiaxin Deng): as CUDA's unquote(urlparse(url).path), with the
    // local host named or left out.
    std::string_view location = std::string_view(source).substr(7);
    if (location.rfind("localhost/", 0) == 0) {
      location.remove_prefix(9);
    } else if (location.empty() || location.front() != '/') {
      throw std::invalid_argument("reference audio file URLs must be local");
    } else {
    }
    path.clear();
    for (size_t i = 0; i < location.size(); ++i) {
      if (location[i] == '%' && i + 2 < location.size() &&
          std::isxdigit(static_cast<unsigned char>(location[i + 1])) &&
          std::isxdigit(static_cast<unsigned char>(location[i + 2]))) {
        int value = 0;
        std::from_chars(location.data() + i + 1, location.data() + i + 3, value,
                        16);
        path.push_back(static_cast<char>(value));
        i += 2;
      } else {
        path.push_back(location[i]);
      }
    }
  } else {
  }
  std::error_code error;
  const std::filesystem::path resolved =
      std::filesystem::weakly_canonical(path, error);
  if (error || !IsInside(resolved, *allowed_directory)) {
    throw std::invalid_argument(
        "reference audio path is outside --allowed-local-media-path");
  } else {
  }
  return ReadAllowedFile(resolved.string(), *allowed_directory);
}

// A text field stripped, or nothing when it is missing, null or blank.
std::optional<std::string> OptionalText(const Json &object, const char *name,
                                        const std::string &label) {
  const Json value = FieldOr(object, name, Json());
  if (value.is_null()) {
    return std::nullopt;
  } else if (!value.is_string()) {
    throw std::invalid_argument(label + " must be a string");
  } else {
  }
  const std::string stripped =
      omni_common::StripUnicodeWhitespace(value.get<std::string>());
  if (omni_common::CodePointCount(stripped) > kMaxInputCharacters) {
    throw std::invalid_argument(label + " must be at most " +
                                std::to_string(kMaxInputCharacters) +
                                " characters");
  } else if (stripped.empty()) {
    return std::nullopt;
  } else {
    return stripped;
  }
}

// body's field, or nullptr when it is missing or null; unlike FieldOr, it
// does not copy a large value such as inline audio.
const Json *FieldPointer(const Json &object, const char *name) {
  const auto found = object.find(name);
  return found == object.end() || found->is_null() ? nullptr : &*found;
}

// Where a request's reference audio is: a data URL, path or file URL, or
// base64 WAV bytes; text points into the request body.
struct ReferenceSource {
  const std::string *text = nullptr;
  bool is_base64 = false;
};

// The WAV bytes a reference source holds.
std::string
SourceBytes(const ReferenceSource &source,
            const std::optional<std::filesystem::path> &allowed_directory) {
  return source.is_base64 ? Base64Bytes(*source.text)
                          : ReferenceBytes(*source.text, allowed_directory);
}

// The reference a request gives, its audio's source if any, and its
// transcript.
// Note (Jiaxin Deng): the CUDA server's fields: ref_audio with ref_text, or
// one references item, whose audio_path, ref_audio, audio or data holds the
// audio and whose text is its transcript, else ref_text.
std::pair<std::optional<ReferenceSource>, std::optional<std::string>>
RequestReference(const Json &body) {
  const Json *references = FieldPointer(body, "references");
  const Json *ref_audio = FieldPointer(body, "ref_audio");
  if (references != nullptr && !references->is_array()) {
    throw std::invalid_argument("references must be a list");
  } else if (ref_audio != nullptr && !ref_audio->is_string()) {
    throw std::invalid_argument("ref_audio must be a string");
  } else if (references != nullptr && references->size() > 1) {
    throw std::invalid_argument("Fun-CosyVoice3 accepts exactly one reference");
  } else {
  }
  const bool has_ref_audio = ref_audio != nullptr &&
                             !ref_audio->get_ref<const std::string &>().empty();
  const bool has_references = references != nullptr && !references->empty();
  const std::optional<std::string> ref_text =
      OptionalText(body, "ref_text", "ref_text");
  if (has_references && has_ref_audio) {
    throw std::invalid_argument("give references or ref_audio, not both");
  } else if (!has_references) {
    return {has_ref_audio ? std::optional<ReferenceSource>(ReferenceSource{
                                &ref_audio->get_ref<const std::string &>()})
                          : std::nullopt,
            ref_text};
  } else {
  }
  const Json &reference = references->front();
  if (!reference.is_object()) {
    throw std::invalid_argument("Fun-CosyVoice3 references must be objects");
  } else if (FieldPointer(reference, "vq_codes") != nullptr) {
    throw std::invalid_argument(
        "Fun-CosyVoice3 does not take vq_codes; give reference audio");
  } else {
  }
  const std::optional<std::string> reference_text =
      OptionalText(reference, "text", "references[0].text");
  const std::optional<std::string> text =
      reference_text.has_value() ? reference_text : ref_text;
  for (const char *field : {"audio_path", "ref_audio", "audio"}) {
    const Json *source = FieldPointer(reference, field);
    if (source != nullptr && !source->is_string()) {
      throw std::invalid_argument(std::string("references[0].") + field +
                                  " must be a string");
    } else if (source != nullptr &&
               !source->get_ref<const std::string &>().empty()) {
      return {ReferenceSource{&source->get_ref<const std::string &>()}, text};
    } else {
    }
  }
  const Json *data = FieldPointer(reference, "data");
  const Json media_type = FieldOr(reference, "media_type", Json("audio/wav"));
  if (data == nullptr || !data->is_string()) {
    throw std::invalid_argument("references[0] has no audio");
  } else if (!media_type.is_string() ||
             (media_type != "audio/wav" && media_type != "audio/x-wav" &&
              media_type != "audio/wave")) {
    throw std::invalid_argument("reference audio must be a WAV file");
  } else {
  }
  return {ReferenceSource{&data->get_ref<const std::string &>(), true}, text};
}

// value as an integer when it is one, or a float with an integral value, as
// CUDA's request model takes 5.0 for an int field.
std::optional<int64_t> IntegralValue(const Json &value) {
  if (value.is_number_unsigned()) {
    return value.get<uint64_t>() >
                   static_cast<uint64_t>(std::numeric_limits<int64_t>::max())
               ? std::nullopt
               : std::optional<int64_t>(value.get<int64_t>());
  } else if (value.is_number_integer()) {
    return value.get<int64_t>();
  } else if (value.is_number_float()) {
    const double number = value.get<double>();
    return std::isfinite(number) && std::trunc(number) == number &&
                   std::fabs(number) < 9.0e18
               ? std::optional<int64_t>(static_cast<int64_t>(number))
               : std::nullopt;
  } else {
    return std::nullopt;
  }
}

// A number field within [low, high], or fallback when it is missing or null.
float NumberField(const Json &body, const char *name, float fallback,
                  double low, double high, const std::string &range) {
  const Json value = FieldOr(body, name, Json(fallback));
  // Note (Jiaxin Deng): checked as the float it is used as, which a large
  // double overflows to infinity.
  if (!value.is_number() || !std::isfinite(value.get<float>()) ||
      value.get<double>() < low || value.get<double>() > high) {
    throw std::invalid_argument(std::string(name) + " must be a number " +
                                range);
  } else {
  }
  return value.get<float>();
}

// The request body's fields; the reference audio is only located, and
// decoded once the request is admitted.
SpeechRequest ParseRequest(const Json &body,
                           std::optional<ReferenceSource> &reference_source) {
  if (!body.is_object()) {
    throw std::invalid_argument("speech request body must be a JSON object");
  } else {
  }
  speech_service::RejectUnsupportedFields(
      body, kUnsupportedFields, "Fun-CosyVoice3 on this server",
      "ref_audio, ref_text, references, instructions, temperature, top_p, "
      "top_k, repetition_penalty, seed and max_new_tokens");
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

  SpeechRequest request;
  request.text.text = input.get<std::string>();
  const Json instructions = FieldOr(body, "instructions", Json());
  if (instructions.is_string()) {
    // Note (Jiaxin Deng): as CUDA, blank instructions are none.
    const std::string stripped =
        omni_common::StripUnicodeWhitespace(instructions.get<std::string>());
    if (omni_common::CodePointCount(stripped) > kMaxInputCharacters) {
      throw std::invalid_argument("instructions must be at most " +
                                  std::to_string(kMaxInputCharacters) +
                                  " characters");
    } else if (!stripped.empty()) {
      request.text.instructions = stripped;
    } else {
    }
  } else if (!instructions.is_null()) {
    throw std::invalid_argument("instructions must be a string");
  } else {
  }
  const auto [source, reference_text] = RequestReference(body);
  if (reference_text.has_value() && request.text.instructions.has_value()) {
    throw std::invalid_argument(
        "Fun-CosyVoice3 accepts either ref_text or instructions, not both");
  } else {
  }
  request.text.reference_text = reference_text;
  reference_source = source;
  const Json language = FieldOr(body, "language", Json());
  if (!language.is_null() && !language.is_string()) {
    throw std::invalid_argument("language must be a string");
  } else {
  }
  const Json model = FieldOr(body, "model", Json());
  if (model.is_string()) {
    request.model = model.get<std::string>();
  } else if (!model.is_null()) {
    throw std::invalid_argument("model must be a string");
  } else {
  }
  // Note (Minyi Tang): as CUDA's AliasChoices("voice", "speaker"), voice
  // wins whenever it is given, even as null.
  const Json voice = PresentFieldOr(
      body, "voice", PresentFieldOr(body, "speaker", Json("default")));
  if (!voice.is_string()) {
    throw std::invalid_argument("voice must be a string");
  } else if (voice != "" && voice != "default") {
    throw std::invalid_argument("Unknown voice '" + voice.get<std::string>() +
                                "'. Supported voices: default");
  } else {
  }

  const Json response_format =
      PresentFieldOr(body, "response_format", Json("wav"));
  if (!response_format.is_string()) {
    throw std::invalid_argument("response_format must be a string");
  } else {
  }
  // Note (Minyi Tang): as CUDA's response_format.strip().lower().
  for (const char character : omni_common::StripUnicodeWhitespace(
           response_format.get<std::string>())) {
    request.response_format.push_back(
        static_cast<char>(std::tolower(static_cast<unsigned char>(character))));
  }
  if (request.response_format != "pcm" && request.response_format != "wav") {
    throw std::invalid_argument("response_format must be one of: pcm, wav");
  } else {
  }

  const Json stream = PresentFieldOr(body, "stream", Json(false));
  if (!stream.is_boolean()) {
    throw std::invalid_argument("stream must be a boolean");
  } else {
  }
  request.stream = stream.get<bool>();
  if (PresentFieldOr(body, "stream_format", Json("audio")) != "audio") {
    throw std::invalid_argument("stream_format must be audio");
  } else {
  }
  if (request.stream && request.response_format != "pcm") {
    throw std::invalid_argument("stream=true requires response_format='pcm'");
  } else {
  }

  const Json speed = PresentFieldOr(body, "speed", Json(1.0));
  if (!speed.is_number() || speed.get<double>() != 1.0) {
    throw std::invalid_argument("speed must be 1.0 on this model");
  } else {
  }
  cosyvoice3::SamplingOptions &sampling = request.sampling;
  sampling.temperature =
      NumberField(body, "temperature", sampling.temperature, 0.0,
                  std::numeric_limits<double>::max(), "of at least 0");
  sampling.top_p =
      NumberField(body, "top_p", sampling.top_p, 0.0, 1.0, "in (0, 1]");
  if (sampling.top_p == 0.0f) {
    throw std::invalid_argument("top_p must be a number in (0, 1]");
  } else {
  }
  sampling.repetition_penalty =
      NumberField(body, "repetition_penalty", sampling.repetition_penalty, 0.0,
                  kMaxRepetitionPenalty, "in (0, 2]");
  if (sampling.repetition_penalty == 0.0f) {
    throw std::invalid_argument(
        "repetition_penalty must be a number in (0, 2]");
  } else {
  }
  const std::optional<int64_t> top_k =
      IntegralValue(FieldOr(body, "top_k", Json(sampling.top_k)));
  if (!top_k.has_value() || (*top_k != -1 && *top_k < 1) ||
      *top_k > cosyvoice3::kTotalSpeechVocabularySize) {
    throw std::invalid_argument(
        "top_k must be -1 or an integer from 1 to " +
        std::to_string(cosyvoice3::kTotalSpeechVocabularySize));
  } else {
  }
  sampling.top_k = static_cast<int>(*top_k);
  const Json seed = FieldOr(body, "seed", Json());
  if (IntegralValue(seed).has_value()) {
    sampling.seed = static_cast<uint64_t>(*IntegralValue(seed));
  } else if (seed.is_null()) {
    static thread_local std::mt19937_64 generator{std::random_device{}()};
    sampling.seed = generator();
  } else {
    throw std::invalid_argument("seed must be an integer");
  }
  const Json max_new_tokens = FieldOr(body, "max_new_tokens", Json());
  const std::optional<int64_t> token_limit = IntegralValue(max_new_tokens);
  if (max_new_tokens.is_null()) {
  } else if (!token_limit.has_value()) {
    throw std::invalid_argument("max_new_tokens must be an integer");
  } else if (*token_limit <= 0) {
    throw std::invalid_argument("max_new_tokens must be greater than 0");
  } else if (*token_limit > cosyvoice3::kMaxSpeechTokens) {
    throw std::invalid_argument("max_new_tokens must be at most " +
                                std::to_string(cosyvoice3::kMaxSpeechTokens));
  } else {
    sampling.max_new_tokens = static_cast<int>(*token_limit);
  }
  return request;
}

class CosyVoice3Service : public speech_service::SpeechServedModel {
public:
  CosyVoice3Service(const std::filesystem::path &model_directory,
                    int max_queued_requests, double request_timeout_seconds,
                    std::optional<std::filesystem::path> allowed_directory)
      : SpeechServedModel(kMaxRequestBodyBytes, max_queued_requests,
                          request_timeout_seconds),
        model_(model_directory),
        allowed_directory_(std::move(allowed_directory)) {}

private:
  int SampleRate() const override { return cosyvoice3::kSampleRate; }

  speech_service::SpeechJob ParseSpeech(const Json &body) override {
    std::optional<ReferenceSource> reference_source;
    auto request =
        std::make_shared<SpeechRequest>(ParseRequest(body, reference_source));
    speech_service::SpeechJob job;
    job.model = request->model;
    job.response_format = request->response_format;
    job.stream = request->stream;
    job.starts_stream_at_admission = true;
    // Note (Jiaxin Deng): the text is tokenized and the reference audio read
    // and resampled only once the request holds a slot, so a burst of long
    // texts or large references cannot hold more than the queue allows.
    job.prepare = [this, request, reference_source]() {
      std::tie(request->prompt_ids, request->target_ids) =
          model_.EncodeText(request->text);
      // Note (Jiaxin Deng): SOS, the text, TASK and every speech token must
      // fit the decoder's context, which text of unmerged bytes can overflow.
      if (request->prompt_ids.size() + request->target_ids.size() + 2 +
              kMaxReferenceSpeechTokens + cosyvoice3::kMaxSpeechTokens >
          static_cast<size_t>(cosyvoice3::kMaxContextTokens)) {
        throw std::invalid_argument(
            "input and instructions take too many text tokens");
      } else if (reference_source.has_value()) {
        request->reference_audio = cosyvoice3::ReadReferenceAudio(
            SourceBytes(*reference_source, allowed_directory_));
      } else {
      }
    };
    job.synthesize = [this,
                      request](const speech_service::SpeechContext &context) {
      return Synthesize(*request, context);
    };
    return job;
  }

  speech_service::SpeechResult
  Synthesize(const SpeechRequest &request,
             const speech_service::SpeechContext &context) const {
    const auto check = [&] {
      if (context.cancel.load()) {
        throw omni_common::GenerationCancelled();
      } else {
        context.check_deadline();
      }
    };
    const std::vector<int> &prompt_ids = request.prompt_ids;
    const std::vector<int> &target_ids = request.target_ids;
    const std::optional<cosyvoice3::Reference> reference =
        request.reference_audio.has_value()
            ? std::optional<cosyvoice3::Reference>(
                  model_.EncodeReference(*request.reference_audio))
            : std::nullopt;
    // Note (Jiaxin Deng): as on CUDA, the LLM's speech prompt is the
    // reference's tokens only when its transcript is given.
    const std::vector<int> prompt_speech_tokens =
        reference.has_value() && request.text.reference_text.has_value()
            ? reference->speech_tokens
            : std::vector<int>();
    check();
    const cosyvoice3::GeneratedSpeech generated = model_.Generate(
        prompt_ids, target_ids, prompt_speech_tokens, request.sampling,
        context.cancel, [&](int) { check(); });
    const std::vector<int> kept =
        cosyvoice3::TrimSilentRuns(generated.speech_tokens);
    if (kept.empty()) {
      // Note (Minyi Tang): the CUDA vocoder raises on no speech tokens
      // rather than answer silence.
      throw std::runtime_error("Fun-CosyVoice3 generated no speech tokens");
    } else {
    }
    const mlx::core::array samples = model_.Vocode(kept, reference, check);
    context.send(std::vector<float>(samples.data<float>(),
                                    samples.data<float>() + samples.size()));
    // Note (Jiaxin Deng): the prompt counts SOS and TASK around the text, and
    // the speech prompt after them.
    return {generated.finish_reason,
            prompt_ids.size() + target_ids.size() +
                prompt_speech_tokens.size() + 2,
            static_cast<int>(generated.speech_tokens.size())};
  }

  cosyvoice3::CosyVoice3Model model_;
  const std::optional<std::filesystem::path> allowed_directory_;
};

} // namespace

int main(int argc, char **argv) {
  std::optional<std::filesystem::path> allowed_directory;
  asr_service::ServedKind kind;
  kind.model_kind = "cosyvoice3";
  speech_service::AddSpeechFlags(
      kind, [&allowed_directory](const std::filesystem::path &model_directory,
                                 int max_queued_requests,
                                 double request_timeout_seconds) {
        return std::make_unique<CosyVoice3Service>(
            model_directory, max_queued_requests, request_timeout_seconds,
            allowed_directory);
      });
  kind.flags["--allowed-local-media-path"] = [&](const std::string &value) {
    std::error_code error;
    allowed_directory = std::filesystem::canonical(value, error);
    if (error || !std::filesystem::is_directory(*allowed_directory)) {
      throw std::logic_error("--allowed-local-media-path must be a directory");
    } else {
    }
  };
  return asr_service::Serve(argc, argv, kind);
}
