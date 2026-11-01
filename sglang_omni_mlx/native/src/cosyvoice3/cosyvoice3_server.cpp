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
#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <filesystem>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <system_error>
#include <tuple>
#include <typeinfo>
#include <vector>

#include "civetweb.h"
#include "common/audio.h"
#include "common/http.h"
#include "common/text.h"
#include "cosyvoice3/cosyvoice3_model.h"
#include "qwen3_asr/asr_service.h"
#include "qwen3_asr/qwen3_asr_transcriber.h"

namespace {

using omni_server::Json;

constexpr size_t kMaxInputCharacters = 4096;
// Room for 30 s of base64 WAV at 48 kHz in stereo float32 (15 MiB) and
// kMaxInputCharacters characters of text in every text field.
constexpr size_t kMaxRequestBodyBytes = 32 * 1024 * 1024;
constexpr size_t kMaxReferenceFileBytes = 24 * 1024 * 1024;
// The reference speech tokens a prompt can hold: 30 s at 25 a second.
constexpr size_t kMaxReferenceSpeechTokens = 750;
constexpr int kRequestIdHexDigits = 32;
constexpr int kMaxRequestTimeoutSeconds = 24 * 60 * 60;
// As often as the CUDA server checks whether a speech client disconnected.
constexpr auto kClientPollInterval = std::chrono::milliseconds(50);
// The status civetweb logs for a request whose client left before its answer.
constexpr int kClientLeftStatus = 499;
// The most one disconnect check reads of what a client sent after its request,
// so a client that keeps sending cannot hold the check.
constexpr size_t kMaxUnreadBytesPerCheck = 64 * 1024;
// The most a repetition penalty may be, as SGLang's sampler allows.
constexpr double kMaxRepetitionPenalty = 2.0;
// A collection field of CUDA's CreateSpeechRequest and the JSON type it takes.
struct CollectionField {
  const char *name;
  Json::value_t type;
  const char *type_description;
};
// Note (Minyi Tang): fields this server does not serve, each counted as
// given by the CUDA request model's rule: text and flag fields when not null
// or "", and collections when not empty. CUDA's request model first rejects a
// collection of another type, even an empty one.
constexpr std::array<const char *, 7> kUnsupportedFields = {
    "duration_tokens",     "initial_codec_chunk_frames",
    "stream_codec_output", "suppress_bootstrap_silence",
    "task_type",           "token_count",
    "x_vector_only_mode"};
constexpr std::array<CollectionField, 1> kUnsupportedCollectionFields = {{
    {"stage_params", Json::value_t::object, "an object"},
}};

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

// Audio one synthesis hands from the worker thread to its request's thread.
struct AudioChannel {
  std::mutex mutex;
  std::condition_variable ready;
  std::optional<std::vector<float>> samples;
  bool done = false;
  std::exception_ptr error;
  int prompt_token_count = 0;
  int completion_token_count = 0;
  qwen3_asr::FinishReason finish_reason = qwen3_asr::FinishReason::kStop;
};

// body's field, or fallback when it is missing or null.
Json FieldOr(const Json &body, const char *name, Json fallback) {
  const auto found = body.find(name);
  if (found == body.end() || found->is_null()) {
    return fallback;
  } else {
    return *found;
  }
}

// body's field, null included, or fallback when it is missing.
// Note (Minyi Tang): CUDA's CreateSpeechRequest takes null as the default
// only for the fields it types as optional, and rejects it for the rest.
Json PresentFieldOr(const Json &body, const char *name, Json fallback) {
  const auto found = body.find(name);
  if (found == body.end()) {
    return fallback;
  } else {
    return *found;
  }
}

size_t CharacterCount(const std::string &text) {
  return static_cast<size_t>(
      std::count_if(text.begin(), text.end(), [](char byte) {
        return (static_cast<unsigned char>(byte) & 0xc0) != 0x80;
      }));
}

// Throws naming every unsupported field body gives, in sorted order.
void RejectUnsupportedFields(const Json &body) {
  std::vector<std::string> given;
  for (const char *field : kUnsupportedFields) {
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
      "Fun-CosyVoice3 on this server does not support these "
      "/v1/audio/speech fields: " +
      fields +
      ". Supported model-specific fields are ref_audio, ref_text, references, "
      "instructions, temperature, top_p, top_k, repetition_penalty, seed and "
      "max_new_tokens.");
}

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
    const size_t comma = source.find(',');
    if (comma == std::string::npos ||
        !std::string_view(source).substr(0, comma).ends_with(";base64")) {
      throw std::invalid_argument("a reference audio data URL must be base64 "
                                  "(data:audio/wav;base64,...)");
    } else {
    }
    return Base64Bytes(source.substr(comma + 1));
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
  if (CharacterCount(stripped) > kMaxInputCharacters) {
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
  RejectUnsupportedFields(body);
  const Json input = FieldOr(body, "input", Json());
  if (!input.is_string() ||
      omni_common::StripUnicodeWhitespace(input.get<std::string>()).empty()) {
    throw std::invalid_argument("input must be a non-empty string");
  } else if (CharacterCount(input.get<std::string>()) > kMaxInputCharacters) {
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
    if (CharacterCount(stripped) > kMaxInputCharacters) {
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

std::string NewRequestId() {
  static thread_local std::mt19937_64 generator{std::random_device{}()};
  static constexpr char kHex[] = "0123456789abcdef";
  std::string id = "speech-";
  for (int index = 0; index < kRequestIdHexDigits; ++index) {
    id.push_back(kHex[generator() % 16]);
  }
  return id;
}

// The socket civetweb accepted for connection's client, found by the client's
// address and port since civetweb does not expose it, or -1.
int ClientSocket(const mg_connection *connection) {
  const mg_request_info *request_info = mg_get_request_info(connection);
  // Note (Minyi Tang): a listing that fails, as it does in a process out of
  // descriptors, ends the search, and the request is then served without
  // watching its client. Nothing here throws, since an exception would unwind
  // through civetweb's C callback and terminate the server.
  std::error_code listing_error;
  for (std::filesystem::directory_iterator listing("/dev/fd", listing_error);
       !listing_error && listing != std::filesystem::directory_iterator();
       listing.increment(listing_error)) {
    const std::string name = listing->path().filename().string();
    int file_descriptor = -1;
    const auto [parse_end, parse_error] = std::from_chars(
        name.data(), name.data() + name.size(), file_descriptor);
    if (parse_error != std::errc() || parse_end != name.data() + name.size()) {
      continue;
    } else {
    }
    sockaddr_storage peer{};
    socklen_t peer_size = sizeof(peer);
    char address[INET6_ADDRSTRLEN] = "";
    int port = -1;
    if (getpeername(file_descriptor, reinterpret_cast<sockaddr *>(&peer),
                    &peer_size) != 0) {
      continue;
    } else if (peer.ss_family == AF_INET) {
      const auto *ipv4 = reinterpret_cast<const sockaddr_in *>(&peer);
      inet_ntop(AF_INET, &ipv4->sin_addr, address, sizeof(address));
      port = ntohs(ipv4->sin_port);
    } else if (peer.ss_family == AF_INET6) {
      const auto *ipv6 = reinterpret_cast<const sockaddr_in6 *>(&peer);
      inet_ntop(AF_INET6, &ipv6->sin6_addr, address, sizeof(address));
      port = ntohs(ipv6->sin6_port);
    } else {
    }
    if (port == request_info->remote_port &&
        request_info->remote_addr == std::string(address)) {
      return file_descriptor;
    } else {
    }
  }
  return -1;
}

// Whether the client of client_socket closed it, which its socket shows by
// reading as ended or failing.
// Note (Minyi Tang): civetweb keeps no connection alive, so it never serves
// what a client sends after its request; that is read and dropped here, since
// bytes left unread would hide the end of the socket from every later check.
// Note (Minyi Tang): a client that only shuts down its sending side also reads
// as ended and is treated as gone, as uvicorn treats it for the CUDA server.
// TCP tells the two apart only once something is written to the socket.
bool ClientLeft(int client_socket) {
  if (client_socket < 0) {
    return false;
  } else {
  }
  std::array<char, 4096> unread{};
  size_t dropped_bytes = 0;
  while (dropped_bytes < kMaxUnreadBytesPerCheck) {
    const ssize_t received =
        recv(client_socket, unread.data(), unread.size(), MSG_DONTWAIT);
    if (received == 0) {
      return true;
    } else if (received < 0) {
      return errno != EAGAIN && errno != EINTR;
    } else {
      dropped_bytes += static_cast<size_t>(received);
    }
  }
  return false;
}

// Thrown by a generation that reaches its request's deadline; a cancellation,
// so it is never logged as a failure.
class DeadlineReached : public qwen3_asr::TranscriptionCancelled {};

// Whether error is a generation stopping at its request's deadline.
bool ReachedDeadline(std::exception_ptr error) {
  if (!error) {
    return false;
  } else {
  }
  try {
    std::rethrow_exception(error);
  } catch (const DeadlineReached &) {
    return true;
  } catch (...) {
    return false;
  }
}

void LogFailure(std::exception_ptr error) {
  try {
    std::rethrow_exception(error);
  } catch (const qwen3_asr::TranscriptionCancelled &) {
  } catch (const std::exception &failure) {
    // Note (Minyi Tang): only the exception type is logged, never the text.
    std::cerr << "speech generation failed: " << typeid(failure).name() << "\n";
  }
}

class CosyVoice3Service : public asr_service::ServedModel {
public:
  CosyVoice3Service(const std::filesystem::path &model_directory,
                    int max_queued_requests, double request_timeout_seconds,
                    std::optional<std::filesystem::path> allowed_directory)
      : model_(model_directory), max_queued_requests_(max_queued_requests),
        request_timeout_(
            std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                std::chrono::duration<double>(request_timeout_seconds))),
        allowed_directory_(std::move(allowed_directory)) {}

  asr_service::Transcription
  Prepare(std::vector<float>, const asr_service::FormFields &) const override {
    throw std::logic_error("cosyvoice3 serves no transcriptions");
  }
  bool Transcribes() const override { return false; }
  // Note (Minyi Tang): /health counts the slots admission holds, which the
  // worker frees only after it marks a request done; once /health shows the
  // server idle, it admits a full queue.
  std::map<std::string, int>
  RequestStates(const qwen3_asr::TranscriptionWorker &) const override {
    std::lock_guard<std::mutex> lock(admission_mutex_);
    std::map<std::string, int> states;
    if (admitted_requests_ > 0) {
      states["running"] = 1;
    } else {
    }
    if (admitted_requests_ > 1) {
      states["queued"] = admitted_requests_ - 1;
    } else {
    }
    return states;
  }
  void AddHandlers(mg_context *context,
                   qwen3_asr::TranscriptionWorker &worker) override {
    worker_ = &worker;
    mg_set_request_handler(context, "/v1/audio/speech$", HandleSpeech, this);
  }

private:
  static int HandleSpeech(mg_connection *connection, void *data) {
    return static_cast<CosyVoice3Service *>(data)->Speak(connection);
  }

  int Speak(mg_connection *connection) {
    if (!omni_server::IsPost(connection)) {
      return omni_server::WriteJson(connection, 405,
                                    {{"detail", "Method Not Allowed"}});
    } else {
    }
    const std::optional<std::string> raw_body =
        omni_server::ReadBody(connection, kMaxRequestBodyBytes);
    if (!raw_body.has_value()) {
      return 413;
    } else {
    }
    const Json body = Json::parse(*raw_body, nullptr, false);
    if (body.is_discarded()) {
      return omni_server::BadRequest(connection,
                                     "speech request body must be valid JSON");
    } else {
    }
    SpeechRequest request;
    std::optional<ReferenceSource> reference_source;
    try {
      request = ParseRequest(body, reference_source);
    } catch (const std::invalid_argument &error) {
      return omni_server::BadRequest(connection, error.what());
    }
    // Note (Minyi Tang): no client can know a generated name, and the CUDA
    // speech API ignores model, so only a chosen name is enforced.
    if (has_chosen_model_name() && request.model.has_value() &&
        *request.model != model_name()) {
      return omni_server::WriteJson(
          connection, 404, {{"detail", "Unknown model: " + *request.model}});
    } else {
    }
    bool admitted = false;
    {
      std::lock_guard<std::mutex> lock(admission_mutex_);
      admitted = admitted_requests_ <= max_queued_requests_;
      admitted_requests_ += admitted ? 1 : 0;
    }
    if (!admitted) {
      return omni_server::WriteJson(connection, 503,
                                    {{"detail", "The request queue is full."}});
    } else {
    }
    // Note (Jiaxin Deng): the text is tokenized and the reference audio read
    // and resampled only once the request holds a slot, so a burst of long
    // texts or large references cannot hold more than the queue allows; a
    // request refused here frees its slot again. Its deadline runs from
    // here, so that work counts against it.
    const auto deadline = std::chrono::steady_clock::now() + request_timeout_;
    const auto release_slot = [this] {
      std::lock_guard<std::mutex> lock(admission_mutex_);
      --admitted_requests_;
    };
    try {
      std::tie(request.prompt_ids, request.target_ids) =
          model_.EncodeText(request.text);
      // Note (Jiaxin Deng): SOS, the text, TASK and every speech token must
      // fit the decoder's context, which text of unmerged bytes can overflow.
      if (request.prompt_ids.size() + request.target_ids.size() + 2 +
              kMaxReferenceSpeechTokens + cosyvoice3::kMaxSpeechTokens >
          static_cast<size_t>(cosyvoice3::kMaxContextTokens)) {
        throw std::invalid_argument(
            "input and instructions take too many text tokens");
      } else {
      }
      if (reference_source.has_value()) {
        request.reference_audio = cosyvoice3::ReadReferenceAudio(
            SourceBytes(*reference_source, allowed_directory_));
      } else {
      }
    } catch (const std::invalid_argument &error) {
      release_slot();
      return omni_server::BadRequest(connection, error.what());
    }
    if (std::chrono::steady_clock::now() >= deadline) {
      release_slot();
      std::cerr << "speech generation timed out\n";
      return omni_server::WriteJson(
          connection, 504, {{"detail", "Speech generation timed out."}});
    } else {
    }
    const std::string request_id = NewRequestId();
    auto channel = std::make_shared<AudioChannel>();
    const qwen3_asr::CancelFlag cancel = qwen3_asr::NewCancelFlag();
    worker_->Submit(
        [this, request, channel, deadline](const std::atomic<bool> &cancelled) {
          return Synthesize(request, *channel, cancelled, deadline);
        },
        cancel,
        [this, channel](std::optional<qwen3_asr::TranscriptionResult>,
                        std::exception_ptr error) {
          // Note (Minyi Tang): the slot frees before the response can end,
          // so a client's next request is never refused for its previous one.
          {
            std::lock_guard<std::mutex> lock(admission_mutex_);
            --admitted_requests_;
          }
          std::lock_guard<std::mutex> lock(channel->mutex);
          channel->done = true;
          channel->error = error;
          channel->ready.notify_all();
        });
    if (request.stream) {
      mg_printf(connection,
                "HTTP/1.1 200 OK\r\nContent-Type: audio/pcm\r\n"
                "X-Request-Id: %s\r\nX-SGLang-Omni-Speech-Id: %s\r\n"
                "X-Sample-Rate: %d\r\nX-Channels: 1\r\nX-Bit-Depth: 16\r\n"
                "Cache-Control: no-cache\r\nConnection: close\r\n\r\n",
                request_id.c_str(), request_id.c_str(),
                cosyvoice3::kSampleRate);
    } else {
    }

    const int client_socket = ClientSocket(connection);
    bool timed_out = false;
    bool client_left = false;
    std::unique_lock<std::mutex> lock(channel->mutex);
    while (true) {
      const bool done = channel->ready.wait_until(
          lock,
          std::min(deadline,
                   std::chrono::steady_clock::now() + kClientPollInterval),
          [&] { return channel->done; });
      if (done) {
        break;
      } else if (std::chrono::steady_clock::now() >= deadline) {
        timed_out = true;
        break;
      } else {
        // Note (Minyi Tang): the client is watched while its request waits
        // in the queue too, and a request cancelled there never starts.
        client_left = ClientLeft(client_socket);
        if (client_left) {
          break;
        } else {
        }
      }
    }
    lock.unlock();
    if (client_left) {
      // Note (Minyi Tang): nothing is read from the channel, which the
      // synthesis may still be writing.
      cancel->store(true);
      return kClientLeftStatus;
    } else {
    }
    // Note (Minyi Tang): the synthesis stops at the deadline itself, and may
    // report that before this thread's wait reaches the deadline. Any other
    // error is a failure, even one that arrives after the deadline.
    if (!timed_out && ReachedDeadline(channel->error)) {
      timed_out = true;
    } else {
    }
    if (timed_out) {
      // Note (Minyi Tang): a request still queued is dropped once the worker
      // reaches it.
      cancel->store(true);
      std::cerr << "speech generation timed out\n";
    } else if (channel->error) {
      LogFailure(channel->error);
    } else {
    }
    if (request.stream && (timed_out || channel->error)) {
      return 200;
    } else if (timed_out) {
      return omni_server::WriteJson(
          connection, 504, {{"detail", "Speech generation timed out."}});
    } else if (channel->error) {
      return omni_server::WriteJson(connection, 500,
                                    {{"detail", "Speech generation failed."}});
    } else {
    }

    const std::string pcm = omni_common::Pcm16Bytes(*channel->samples);
    if (request.stream) {
      mg_write(connection, pcm.data(), pcm.size());
      return 200;
    } else {
    }
    const std::string audio =
        request.response_format == "wav"
            ? omni_common::WavBytes(pcm, cosyvoice3::kSampleRate)
            : pcm;
    const int prompt_tokens = channel->prompt_token_count;
    const int completion_tokens = channel->completion_token_count;
    mg_printf(
        connection,
        "HTTP/1.1 200 OK\r\nContent-Type: audio/%s\r\nContent-Length: %zu\r\n"
        "Content-Disposition: attachment; filename=\"speech.%s\"\r\n"
        "X-Request-Id: %s\r\nX-Finish-Reason: %s\r\n"
        "X-Prompt-Tokens: %d\r\nX-Completion-Tokens: %d\r\n"
        "X-SGLang-Omni-Input-Tokens: %d\r\n"
        "X-SGLang-Omni-Output-Tokens: %d\r\n"
        "X-SGLang-Omni-Total-Tokens: %d\r\nConnection: close\r\n\r\n",
        request.response_format.c_str(), audio.size(),
        request.response_format.c_str(), request_id.c_str(),
        qwen3_asr::FinishReasonName(channel->finish_reason), prompt_tokens,
        completion_tokens, prompt_tokens, completion_tokens,
        prompt_tokens + completion_tokens);
    mg_write(connection, audio.data(), audio.size());
    return 200;
  }

  qwen3_asr::TranscriptionResult
  Synthesize(const SpeechRequest &request, AudioChannel &channel,
             const std::atomic<bool> &cancel,
             std::chrono::steady_clock::time_point deadline) const {
    // Note (Minyi Tang): the synthesis checks its own deadline, since the
    // request's thread may be blocked writing to a slow client.
    const auto check = [&] {
      if (cancel.load()) {
        throw qwen3_asr::TranscriptionCancelled();
      } else if (std::chrono::steady_clock::now() >= deadline) {
        throw DeadlineReached();
      } else {
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
    const cosyvoice3::GeneratedSpeech generated =
        model_.Generate(prompt_ids, target_ids, prompt_speech_tokens,
                        request.sampling, cancel, [&](int) { check(); });
    const std::vector<int> kept =
        cosyvoice3::TrimSilentRuns(generated.speech_tokens);
    if (kept.empty()) {
      // Note (Minyi Tang): the CUDA vocoder raises on no speech tokens
      // rather than answer silence.
      throw std::runtime_error("Fun-CosyVoice3 generated no speech tokens");
    } else {
    }
    const mlx::core::array samples = model_.Vocode(kept, reference, check);
    qwen3_asr::TranscriptionResult result;
    result.generated_token_count =
        static_cast<int>(generated.speech_tokens.size());
    result.finish_reason = generated.finish_reason;
    std::lock_guard<std::mutex> lock(channel.mutex);
    channel.samples.emplace(samples.data<float>(),
                            samples.data<float>() + samples.size());
    // Note (Jiaxin Deng): the prompt counts SOS and TASK around the text, and
    // the speech prompt after them.
    channel.prompt_token_count =
        static_cast<int>(prompt_ids.size() + target_ids.size() +
                         prompt_speech_tokens.size()) +
        2;
    channel.completion_token_count = result.generated_token_count;
    channel.finish_reason = result.finish_reason;
    return result;
  }

  cosyvoice3::CosyVoice3Model model_;
  const int max_queued_requests_;
  const std::chrono::steady_clock::duration request_timeout_;
  mutable std::mutex admission_mutex_;
  // Requests the worker holds: running, queued, or finishing.
  int admitted_requests_ = 0;
  qwen3_asr::TranscriptionWorker *worker_ = nullptr;
  const std::optional<std::filesystem::path> allowed_directory_;
};

} // namespace

int main(int argc, char **argv) {
  int max_queued_requests = 2;
  double request_timeout_seconds = 600;
  std::optional<std::filesystem::path> allowed_directory;
  asr_service::ServedKind kind;
  kind.model_kind = "cosyvoice3";
  kind.flags = {
      {"--max-queued-requests",
       [&](const std::string &value) {
         max_queued_requests = std::stoi(value);
       }},
      {"--request-timeout-s",
       [&](const std::string &value) {
         request_timeout_seconds = std::stod(value);
       }},
      {"--allowed-local-media-path",
       [&](const std::string &value) {
         std::error_code error;
         allowed_directory = std::filesystem::canonical(value, error);
         if (error || !std::filesystem::is_directory(*allowed_directory)) {
           throw std::logic_error(
               "--allowed-local-media-path must be a directory");
         } else {
         }
       }},
  };
  kind.check_flags = [&]() {
    if (max_queued_requests < 0) {
      throw std::invalid_argument("--max-queued-requests must not be negative");
    } else if (!(request_timeout_seconds > 0 &&
                 request_timeout_seconds <= kMaxRequestTimeoutSeconds)) {
      throw std::invalid_argument(
          "--request-timeout-s must be above 0 and at most " +
          std::to_string(kMaxRequestTimeoutSeconds));
    } else {
    }
  };
  kind.load = [&](const std::filesystem::path &model_directory) {
    return std::make_unique<CosyVoice3Service>(
        model_directory, max_queued_requests, request_timeout_seconds,
        allowed_directory);
  };
  return asr_service::Serve(argc, argv, kind);
}
