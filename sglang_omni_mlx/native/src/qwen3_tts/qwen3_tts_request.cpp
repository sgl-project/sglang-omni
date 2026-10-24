// SPDX-License-Identifier: Apache-2.0
#include "qwen3_tts/qwen3_tts_model.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <limits>
#include <random>

#include "common/audio.h"

namespace qwen3_tts {

namespace {

std::string Lower(std::string value) {
  std::transform(
      value.begin(), value.end(), value.begin(),
      [](unsigned char character) { return std::tolower(character); });
  return value;
}

std::string TextField(const nlohmann::json &body, const std::string &name,
                      const std::string &fallback = "") {
  const auto value = body.value(name, nlohmann::json(fallback));
  if (!value.is_string()) {
    throw std::invalid_argument(name + " must be a string");
  } else {
  }
  const auto text = value.get<std::string>();
  if (text.size() > kMaximumInputBytes ||
      text.find("<|") != std::string::npos) {
    throw std::invalid_argument(
        name + " exceeds 16384 bytes or contains reserved token markers");
  } else {
    return text;
  }
}

int IntegerField(const nlohmann::json &body, const std::string &name,
                 int fallback, int minimum, int maximum) {
  const auto value = body.value(name, nlohmann::json(fallback));
  if (!value.is_number_integer() || value < minimum || value > maximum) {
    throw std::invalid_argument(name + " must be an integer in [" +
                                std::to_string(minimum) + "," +
                                std::to_string(maximum) + "]");
  } else {
    return value.get<int>();
  }
}

float NumberField(const nlohmann::json &body, const std::string &name,
                  float fallback, float minimum, float maximum) {
  const auto value = body.value(name, nlohmann::json(fallback));
  if (!value.is_number() || !std::isfinite(value.get<double>()) ||
      value < minimum || value > maximum) {
    throw std::invalid_argument(name + " must be a finite number in range");
  } else {
    return value.get<float>();
  }
}

bool BooleanField(const nlohmann::json &body, const std::string &name,
                  bool fallback) {
  const auto value = body.value(name, nlohmann::json(fallback));
  if (!value.is_boolean()) {
    throw std::invalid_argument(name + " must be a boolean");
  } else {
    return value.get<bool>();
  }
}

std::string DecodeBase64(std::string encoded) {
  const std::string prefix = "data:audio/wav;base64,";
  if (encoded.starts_with(prefix)) {
    encoded.erase(0, prefix.size());
  } else {
  }
  const std::string alphabet =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  if (encoded.empty() || encoded.size() % 4 ||
      encoded.size() > 4 * (kMaximumReferenceSamples * 4 + 4096) / 3 + 4) {
    throw std::invalid_argument("ref_audio must be bounded base64 WAV data");
  } else {
  }
  std::string decoded;
  for (size_t offset = 0; offset < encoded.size(); offset += 4) {
    uint32_t word = 0;
    int padding = 0;
    for (int index = 0; index < 4; ++index) {
      const char character = encoded[offset + index];
      const auto found = alphabet.find(character);
      if (character == '=' && index >= 2 && offset + 4 == encoded.size()) {
        ++padding;
        word <<= 6;
      } else if (found == std::string::npos || padding != 0) {
        throw std::invalid_argument("ref_audio contains invalid base64");
      } else {
        word = (word << 6) | static_cast<uint32_t>(found);
      }
    }
    decoded.push_back(static_cast<char>((word >> 16) & 255));
    if (padding < 2) {
      decoded.push_back(static_cast<char>((word >> 8) & 255));
    } else {
    }
    if (padding == 0) {
      decoded.push_back(static_cast<char>(word & 255));
    } else {
    }
  }
  return decoded;
}

} // namespace

Request Model::ParseRequest(const nlohmann::json &body,
                            const std::string &model_name) const {
  static const std::set<std::string> fields = {"input",
                                               "model",
                                               "voice",
                                               "language",
                                               "instructions",
                                               "task_type",
                                               "ref_audio",
                                               "ref_text",
                                               "response_format",
                                               "stream",
                                               "stream_format",
                                               "max_new_tokens",
                                               "seed",
                                               "speed",
                                               "temperature",
                                               "top_k",
                                               "top_p",
                                               "repetition_penalty",
                                               "subtalker_dosample",
                                               "subtalker_temperature",
                                               "subtalker_top_k",
                                               "subtalker_top_p",
                                               "non_streaming_mode"};
  if (!body.is_object()) {
    throw std::invalid_argument("speech request must be a JSON object");
  } else {
  }
  for (const auto &[name, value] : body.items()) {
    if (!fields.contains(name)) {
      throw std::invalid_argument("unsupported speech field: " + name);
    } else {
    }
  }
  const auto input = TextField(body, "input");
  if (input.empty() ||
      input.find_first_not_of(" \t\r\n") == std::string::npos) {
    throw std::invalid_argument("input must contain speech text");
  } else {
  }
  if (body.contains("model") && !model_name.empty() &&
      TextField(body, "model") != model_name) {
    throw std::invalid_argument("model does not match the served model");
  } else if (body.contains("model")) {
    TextField(body, "model");
  } else {
  }
  const auto kind = config_.at("tts_model_type").get<std::string>();
  if (body.contains("task_type")) {
    auto task = Lower(TextField(body, "task_type"));
    task.erase(std::remove(task.begin(), task.end(), '_'), task.end());
    const auto expected = kind == "custom_voice"   ? "customvoice"
                          : kind == "voice_design" ? "voicedesign"
                                                   : "base";
    if (task != expected) {
      throw std::invalid_argument("task_type does not match checkpoint kind");
    } else {
    }
  } else {
  }
  Request request;
  request.text_ids = tokenizer_.Encode("<|im_start|>assistant\n" + input +
                                       "<|im_end|>\n<|im_start|>assistant\n");
  if (request.text_ids.size() <= 8) {
    throw std::invalid_argument("input has no text tokens");
  } else {
  }
  const auto instructions = TextField(body, "instructions");
  if (!instructions.empty() && config_.at("tts_model_size") == "0b6") {
    throw std::invalid_argument("0.6B checkpoints do not support instructions");
  } else if (!instructions.empty()) {
    request.instruction_ids =
        tokenizer_.Encode("<|im_start|>user\n" + instructions + "<|im_end|>\n");
  } else if (kind == "voice_design") {
    throw std::invalid_argument("VoiceDesign requires instructions");
  } else {
  }
  std::string voice =
      Lower(TextField(body, "voice", kind == "custom_voice" ? "Vivian" : ""));
  const auto language = Lower(TextField(body, "language", "auto"));
  const auto &languages = talker_config_.at("codec_language_id");
  std::optional<int> language_id;
  if (language != "auto") {
    if (!languages.contains(language)) {
      throw std::invalid_argument("unsupported Qwen3-TTS language");
    } else {
      language_id = languages.at(language).get<int>();
    }
  } else if (kind == "custom_voice" &&
             talker_config_.contains("spk_is_dialect") &&
             talker_config_.at("spk_is_dialect").contains(voice) &&
             talker_config_.at("spk_is_dialect").at(voice).is_string()) {
    language_id = languages
                      .at(talker_config_.at("spk_is_dialect")
                              .at(voice)
                              .get<std::string>())
                      .get<int>();
  } else {
  }
  request.control_ids = {talker_config_.at(language_id.has_value()
                                               ? "codec_think_id"
                                               : "codec_nothink_id"),
                         talker_config_.at("codec_think_bos_id")};
  if (language_id.has_value()) {
    request.control_ids.push_back(*language_id);
  } else {
  }
  request.control_ids.push_back(talker_config_.at("codec_think_eos_id"));
  if (kind == "custom_voice") {
    if (!talker_config_.at("spk_id").contains(voice)) {
      throw std::invalid_argument("unsupported Qwen3-TTS voice");
    } else {
      const auto &speaker = talker_config_.at("spk_id").at(voice);
      request.control_ids.push_back(
          speaker.is_array() ? speaker.at(0).get<int>() : speaker.get<int>());
    }
  } else if (!voice.empty()) {
    throw std::invalid_argument("voice is only supported by CustomVoice");
  } else {
  }
  if (kind == "base") {
    if (!body.contains("ref_audio") || !body.at("ref_audio").is_string() ||
        TextField(body, "ref_text").empty()) {
      throw std::invalid_argument(
          "Base requires inline ref_audio and non-empty ref_text");
    } else {
      request.reference_samples = DecodeReferenceWav(
          DecodeBase64(body.at("ref_audio").get<std::string>()));
      const auto reference_ids =
          tokenizer_.Encode("<|im_start|>assistant\n" +
                            TextField(body, "ref_text") + "<|im_end|>\n");
      request.reference_text_ids.assign(reference_ids.begin() + 3,
                                        reference_ids.end() - 2);
    }
  } else if (body.contains("ref_audio") || body.contains("ref_text")) {
    throw std::invalid_argument(
        "reference audio/text require a Base checkpoint");
  } else {
  }
  request.non_streaming_mode = BooleanField(body, "non_streaming_mode", true);
  if (kind == "base") {
    request.non_streaming_mode = true;
  } else {
  }
  const auto format = TextField(body, "response_format", "wav");
  request.stream = BooleanField(body, "stream", false);
  if ((format != "wav" && format != "pcm") ||
      (request.stream && format != "pcm") ||
      TextField(body, "stream_format", "audio") != "audio") {
    throw std::invalid_argument(
        "response_format must be wav or pcm; streaming requires pcm/audio");
  } else {
    request.pcm = format == "pcm";
  }
  NumberField(body, "speed", 1.0f, 1.0f, 1.0f);
  request.max_frames =
      IntegerField(body, "max_new_tokens", 256, 1, kMaximumFrames);
  request.semantic = {NumberField(body, "temperature", 0.9f, 0.0f, 10.0f),
                      IntegerField(body, "top_k", 50, 0, kCodebookSize),
                      NumberField(body, "top_p", 1.0f,
                                  std::numeric_limits<float>::epsilon(), 1.0f),
                      NumberField(body, "repetition_penalty", 1.05f,
                                  std::numeric_limits<float>::epsilon(),
                                  100.0f)};
  request.subtalker = {
      NumberField(body, "subtalker_temperature", 0.9f, 0.0f, 10.0f),
      IntegerField(body, "subtalker_top_k", 50, 0, kCodebookSize),
      NumberField(body, "subtalker_top_p", 1.0f,
                  std::numeric_limits<float>::epsilon(), 1.0f),
      1.0f};
  if (!BooleanField(body, "subtalker_dosample", true)) {
    request.subtalker.temperature = 0.0f;
  } else {
  }
  if (body.contains("seed")) {
    request.seed = static_cast<uint64_t>(
        IntegerField(body, "seed", 0, 0, std::numeric_limits<int>::max()));
  } else {
    request.seed = std::random_device{}();
  }
  for (const auto token : request.control_ids) {
    if (token < 0 || token >= talker_config_.at("vocab_size").get<int>()) {
      throw std::invalid_argument(
          "checkpoint has invalid control or speaker IDs");
    } else {
    }
  }
  for (const auto *tokens : {&request.text_ids, &request.instruction_ids,
                             &request.reference_text_ids}) {
    for (const int token : *tokens) {
      if (token < 0 ||
          token >= talker_config_.at("text_vocab_size").get<int>()) {
        throw std::invalid_argument(
            "tokenizer produced an out-of-range text token");
      } else {
      }
    }
  }
  const size_t reference_frames =
      (request.reference_samples.size() + kSamplesPerFrame - 1) /
      kSamplesPerFrame;
  if (request.text_ids.size() + request.instruction_ids.size() +
          request.reference_text_ids.size() + reference_frames +
          request.max_frames + 16 >
      static_cast<size_t>(
          talker_config_.at("max_position_embeddings").get<int>())) {
    throw std::invalid_argument("speech request exceeds model context");
  } else {
  }
  return request;
}

std::vector<float> DecodeReferenceWav(const std::string &bytes) {
  uint32_t riff_bytes = 0;
  if (bytes.size() < 12) {
    throw std::invalid_argument("reference WAV header is truncated");
  } else {
    std::memcpy(&riff_bytes, bytes.data() + 4, sizeof(riff_bytes));
  }
  if (static_cast<uint64_t>(riff_bytes) + 8 != bytes.size()) {
    throw std::invalid_argument(
        "reference WAV length does not match its RIFF header");
  } else {
  }
  uint16_t alignment = 0;
  size_t offset = 12;
  while (offset + 8 <= bytes.size()) {
    uint32_t count = 0;
    std::memcpy(&count, bytes.data() + offset + 4, sizeof(count));
    if (count > bytes.size() - offset - 8) {
      throw std::invalid_argument("reference WAV chunk is truncated");
    } else if (bytes.compare(offset, 4, "fmt ") == 0 && count >= 16) {
      std::memcpy(&alignment, bytes.data() + offset + 8 + 12,
                  sizeof(alignment));
    } else if (bytes.compare(offset, 4, "data") == 0 &&
               (alignment == 0 || count % alignment != 0)) {
      throw std::invalid_argument(
          "reference WAV has an incomplete audio sample");
    } else {
    }
    offset += 8 + count + (count & 1u);
  }
  if (offset != bytes.size()) {
    throw std::invalid_argument(
        "reference WAV has trailing incomplete chunk data");
  } else {
  }
  auto samples = qwen3_asr::DecodeWavAtRate(bytes, kSampleRate);
  if (samples.size() < 1024 || samples.size() > kMaximumReferenceSamples ||
      !std::all_of(samples.begin(), samples.end(), [](float sample) {
        return std::isfinite(sample) && std::abs(sample) <= 1.0f;
      })) {
    throw std::invalid_argument("reference must contain 1024 samples to 30 "
                                "seconds of finite 24 kHz mono audio");
  } else {
    return samples;
  }
}

std::string EncodeAudio(const std::vector<float> &samples, bool pcm) {
  if (!std::all_of(samples.begin(), samples.end(),
                   [](float sample) { return std::isfinite(sample); })) {
    throw std::runtime_error("Qwen3-TTS produced non-finite audio");
  } else {
  }
  const auto bytes = qwen3_asr::Pcm16Bytes(samples);
  return pcm ? bytes : qwen3_asr::WavBytes(bytes, kSampleRate);
}

} // namespace qwen3_tts
