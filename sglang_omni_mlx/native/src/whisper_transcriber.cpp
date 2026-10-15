// SPDX-License-Identifier: Apache-2.0
#include "whisper_transcriber.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <stdexcept>

#include "nlohmann/json.hpp"
#include "swift_port.h"

namespace whisper {

namespace mx = mlx::core;

namespace {

constexpr float kSuppressedLogit = -1e9f;

// Note (khazic): the English names Swift's WhisperTokenizer maps to codes, as
// it lists them.
const std::map<std::string, std::string> &LanguageNameToCode() {
  static const std::map<std::string, std::string> table = {
      {"english", "en"},    {"chinese", "zh"},    {"mandarin", "zh"},
      {"cantonese", "yue"}, {"japanese", "ja"},   {"korean", "ko"},
      {"french", "fr"},     {"german", "de"},     {"spanish", "es"},
      {"italian", "it"},    {"portuguese", "pt"}, {"russian", "ru"},
      {"polish", "pl"},     {"turkish", "tr"},    {"dutch", "nl"},
      {"arabic", "ar"},     {"hindi", "hi"},      {"indonesian", "id"},
      {"vietnamese", "vi"}, {"thai", "th"},       {"ukrainian", "uk"},
      {"swedish", "sv"},    {"czech", "cs"},      {"romanian", "ro"},
      {"hungarian", "hu"},  {"danish", "da"},     {"finnish", "fi"},
      {"norwegian", "no"},  {"hebrew", "he"},     {"greek", "el"},
      {"tagalog", "tl"},    {"filipino", "tl"},   {"malay", "ms"},
      {"tamil", "ta"},      {"telugu", "te"},     {"bengali", "bn"},
      {"urdu", "ur"},
  };
  return table;
}

nlohmann::json ReadJson(const std::filesystem::path &path) {
  std::ifstream stream(path);
  if (!stream) {
    throw std::runtime_error("cannot read " + path.string());
  } else {
  }
  return nlohmann::json::parse(stream);
}

std::string Lowercase(std::string text) {
  std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return text;
}

// "<|en|>" or "EN" to "en".
std::string LanguageKey(const std::string &text) {
  std::string key = Lowercase(text);
  if (key.size() >= 4 && key.compare(0, 2, "<|") == 0 &&
      key.compare(key.size() - 2, 2, "|>") == 0) {
    return key.substr(2, key.size() - 4);
  } else {
    return key;
  }
}

// The language code of an added token such as <|haw|>: one to three
// lowercase letters between <| and |>.
std::optional<std::string> AddedTokenLanguage(const std::string &content) {
  if (content.size() < 5 || content.size() > 7 ||
      content.compare(0, 2, "<|") != 0 ||
      content.compare(content.size() - 2, 2, "|>") != 0) {
    return std::nullopt;
  } else {
  }
  const std::string body = content.substr(2, content.size() - 4);
  if (std::all_of(body.begin(), body.end(),
                  [](char c) { return c >= 'a' && c <= 'z'; })) {
    return body;
  } else {
    return std::nullopt;
  }
}

// Note (khazic): the Swift tokenizer's cleanup of English tokenization spaces,
// in its order.
std::string CleanUpTokenizationSpaces(std::string text) {
  static const std::pair<std::string, std::string> kReplacements[] = {
      {" .", "."},     {" ?", "?"},     {" !", "!"},   {" ,", ","},
      {" ' ", "'"},    {" n't", "n't"}, {" 'm", "'m"}, {" 's", "'s"},
      {" 've", "'ve"}, {" 're", "'re"},
  };
  for (const auto &[from, to] : kReplacements) {
    std::string replaced;
    size_t start = 0;
    for (size_t found = text.find(from); found != std::string::npos;
         found = text.find(from, start)) {
      replaced.append(text, start, found - start);
      replaced += to;
      start = found + from.size();
    }
    replaced.append(text, start, std::string::npos);
    text = std::move(replaced);
  }
  return text;
}

// -1e9 at the given ids, 0 elsewhere; absent when no id is in range.
std::optional<mx::array> SuppressionMask(const std::vector<int> &token_ids,
                                         int vocabulary_size) {
  std::vector<float> mask(vocabulary_size, 0.0f);
  bool suppresses = false;
  for (const int id : token_ids) {
    if (id >= 0 && id < vocabulary_size) {
      mask[id] = kSuppressedLogit;
      suppresses = true;
    } else {
    }
  }
  if (!suppresses) {
    return std::nullopt;
  } else {
    return mx::array(mask.data(), {vocabulary_size}, mx::float32);
  }
}

} // namespace

WhisperTranscriber::WhisperTranscriber(
    const std::filesystem::path &model_directory)
    : model_(model_directory), tokenizer_(model_directory) {
  const nlohmann::json tokenizer_config =
      ReadJson(model_directory / "tokenizer_config.json");
  clean_up_tokenization_spaces_ =
      tokenizer_config.value("clean_up_tokenization_spaces", true);
  std::map<std::string, int> added_token_ids;
  for (const auto &[id_text, token] :
       tokenizer_config.at("added_tokens_decoder").items()) {
    const int id = std::stoi(id_text);
    const std::string content = token.at("content").get<std::string>();
    added_token_ids.emplace(content, id);
    special_token_ids_.insert(id);
  }
  const nlohmann::json generation =
      ReadJson(model_directory / "generation_config.json");
  const auto token_id = [&](const char *generation_key,
                            const std::string &content) -> std::optional<int> {
    if (generation.contains(generation_key)) {
      return generation.at(generation_key).get<int>();
    } else if (added_token_ids.count(content) > 0) {
      return added_token_ids.at(content);
    } else {
      return std::nullopt;
    }
  };
  start_of_transcript_id_ =
      token_id("decoder_start_token_id", "<|startoftranscript|>").value();
  end_of_text_id_ = token_id("eos_token_id", "<|endoftext|>").value();
  no_timestamps_id_ =
      token_id("no_timestamps_token_id", "<|notimestamps|>").value();
  timestamp_begin_id_ = no_timestamps_id_ + 1;
  if (generation.contains("task_to_id")) {
    transcribe_id_ = generation.at("task_to_id").at("transcribe").get<int>();
  } else if (added_token_ids.count("<|transcribe|>") > 0) {
    transcribe_id_ = added_token_ids.at("<|transcribe|>");
  } else {
  }
  if (generation.contains("lang_to_id")) {
    for (const auto &[key, id] : generation.at("lang_to_id").items()) {
      language_ids_[LanguageKey(key)] = id.get<int>();
    }
  } else {
  }
  for (const auto &[content, id] : added_token_ids) {
    const std::optional<std::string> code = AddedTokenLanguage(content);
    if (code.has_value()) {
      language_ids_[*code] = id;
    } else {
    }
  }
  is_multilingual_ =
      generation.value("is_multilingual", !language_ids_.empty());
  for (const int id :
       {start_of_transcript_id_, end_of_text_id_, no_timestamps_id_}) {
    special_token_ids_.insert(id);
  }
  if (transcribe_id_.has_value()) {
    special_token_ids_.insert(*transcribe_id_);
  } else {
  }
  for (const auto &[code, id] : language_ids_) {
    special_token_ids_.insert(id);
  }

  const int vocabulary_size = model_.config().vocabulary_size;
  begin_suppression_ =
      SuppressionMask(generation.value("begin_suppress_tokens",
                                       std::vector<int>{end_of_text_id_}),
                      vocabulary_size);
  step_suppression_ = SuppressionMask(
      generation.value("suppress_tokens", std::vector<int>{}), vocabulary_size);
  std::vector<int> timestamp_ids;
  for (int id = timestamp_begin_id_; id < vocabulary_size; ++id)
    timestamp_ids.push_back(id);
  timestamp_suppression_ = SuppressionMask(timestamp_ids, vocabulary_size);
}

std::optional<std::string>
WhisperTranscriber::LanguageCode(const std::string &language) const {
  const std::string key =
      LanguageKey(swift_port::TrimWhitespace(language, true));
  const auto name = LanguageNameToCode().find(key);
  if (key.empty()) {
    return std::nullopt;
  } else if (language_ids_.count(key) > 0) {
    return key;
  } else if (name != LanguageNameToCode().end() &&
             language_ids_.count(name->second) > 0) {
    return name->second;
  } else {
    return std::nullopt;
  }
}

std::string
WhisperTranscriber::DecodeText(const std::vector<int> &token_ids) const {
  std::vector<int> text_ids;
  for (const int id : token_ids) {
    if (id >= 0 && id < timestamp_begin_id_ &&
        special_token_ids_.count(id) == 0) {
      text_ids.push_back(id);
    } else {
    }
  }
  const std::string text = tokenizer_.Decode(text_ids, true);
  return clean_up_tokenization_spaces_ ? CleanUpTokenizationSpaces(text) : text;
}

WhisperTranscriber::WindowResult
WhisperTranscriber::TranscribeWindow(const std::vector<float> &window_samples,
                                     const std::vector<int> &prompt_ids,
                                     int max_new_tokens, float temperature,
                                     const std::atomic<bool> &cancel) const {
  const mx::array encoder_states = model_.Encode(window_samples);
  std::vector<DecoderLayerCache> caches(model_.config().text_layer_count);
  const auto next_token = [&](mx::array logits, bool first_step) {
    if (first_step && begin_suppression_.has_value()) {
      logits = mx::add(logits, mx::astype(*begin_suppression_, logits.dtype()));
    } else {
    }
    if (step_suppression_.has_value()) {
      logits = mx::add(logits, mx::astype(*step_suppression_, logits.dtype()));
    } else {
    }
    if (timestamp_suppression_.has_value()) {
      logits =
          mx::add(logits, mx::astype(*timestamp_suppression_, logits.dtype()));
    } else {
    }
    if (temperature <= 0.0f) {
      return mx::argmax(logits, -1);
    } else {
      return mx::reshape(
          mx::random::categorical(mx::expand_dims(
              mx::divide(logits, mx::array(temperature, logits.dtype())), 0)),
          {});
    }
  };
  const mx::array prompt(prompt_ids.data(),
                         {1, static_cast<int>(prompt_ids.size())}, mx::int32);
  mx::array token =
      next_token(model_.Decode(prompt, 0, encoder_states, caches), true);
  mx::async_eval({token});
  WindowResult result;
  std::vector<int> generated_ids;
  for (int step = 0;; ++step) {
    if (cancel.load()) {
      throw qwen3_asr::TranscriptionCancelled();
    } else if (step == max_new_tokens) {
      result.reached_token_limit = true;
      break;
    } else {
    }
    const mx::array current = token;
    // Note (khazic): queue the next step before reading this token, so the GPU
    // decodes while the end of text is checked.
    if (step + 1 < max_new_tokens) {
      token =
          next_token(model_.Decode(mx::reshape(current, {1, 1}),
                                   static_cast<int>(prompt_ids.size()) + step,
                                   encoder_states, caches),
                     false);
      mx::async_eval({token});
    } else {
    }
    const int token_id = static_cast<int>(current.item<uint32_t>());
    if (token_id == end_of_text_id_) {
      break;
    } else {
    }
    generated_ids.push_back(token_id);
  }
  result.text = DecodeText(generated_ids);
  result.generated_token_count = static_cast<int>(generated_ids.size());
  return result;
}

qwen3_asr::TranscriptionResult
WhisperTranscriber::Transcribe(const std::vector<float> &samples,
                               const WhisperOptions &options,
                               const std::atomic<bool> &cancel) const {
  qwen3_asr::TranscriptionResult result;
  result.finish_reason = qwen3_asr::FinishReason::kStop;
  // Note (khazic): start of transcript, then for multilingual checkpoints the
  // language (when it resolves) and the transcribe task, then no timestamps.
  std::vector<int> prompt_ids = {start_of_transcript_id_};
  if (is_multilingual_) {
    result.language = LanguageCode(options.language);
    if (result.language.has_value()) {
      prompt_ids.push_back(language_ids_.at(*result.language));
    } else {
    }
    if (transcribe_id_.has_value()) {
      prompt_ids.push_back(*transcribe_id_);
    } else {
    }
  } else {
  }
  prompt_ids.push_back(no_timestamps_id_);
  const int context_length = model_.config().text_context_length;
  // Note (khazic): the Swift port's default budget leaves 16 positions of the
  // text context unused.
  const int requested_tokens = options.max_new_tokens != 0
                                   ? options.max_new_tokens
                                   : context_length - 16;
  const int max_new_tokens = std::max(
      1, std::min(requested_tokens,
                  context_length - static_cast<int>(prompt_ids.size()) - 1));

  // Note (khazic): audio up to 30 s is one window; longer audio is cut every 30
  // s. Each window is padded with silence, or trimmed, to exactly 30 s and
  // decoded on its own.
  const size_t window_count = std::max<size_t>(
      1, (samples.size() + kWindowSampleCount - 1) / kWindowSampleCount);
  for (size_t window_index = 0; window_index < window_count; ++window_index) {
    // Note (khazic): a request cancelled between windows skips the next encoder
    // pass.
    if (cancel.load()) {
      throw qwen3_asr::TranscriptionCancelled();
    } else {
    }
    const size_t start = window_index * kWindowSampleCount;
    const size_t end = std::min(start + kWindowSampleCount, samples.size());
    std::vector<float> window_samples(kWindowSampleCount, 0.0f);
    std::copy(samples.begin() + start, samples.begin() + end,
              window_samples.begin());
    const WindowResult window =
        TranscribeWindow(window_samples, prompt_ids, max_new_tokens,
                         options.temperature, cancel);
    const std::string text = swift_port::TrimWhitespace(window.text, true);
    if (!text.empty()) {
      result.text += (result.text.empty() ? "" : " ") + text;
    } else {
    }
    result.generated_token_count += window.generated_token_count;
    if (window.reached_token_limit) {
      result.finish_reason = qwen3_asr::FinishReason::kLength;
    } else {
    }
  }
  return result;
}

} // namespace whisper
