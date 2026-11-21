// SPDX-License-Identifier: Apache-2.0
#include "whisper/whisper_transcriber.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iterator>
#include <stdexcept>

#include "common/generation.h"
#include "common/layers.h"
#include "common/swift_port.h"
#include "common/text.h"
#include "common/transcription.h"
#include "nlohmann/json.hpp"

namespace whisper {

namespace mx = mlx::core;

namespace {

constexpr float kSuppressedLogit = -1e9f;
// Windows decoded together; their cross-attention keys and values take about
// 250 MB each on large-v3.
constexpr size_t kBatchWindows = 8;

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

// "<|en|>" or "EN" to "en".
std::string LanguageKey(const std::string &text) {
  std::string key = omni_common::AsciiLowercase(text);
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
    const std::filesystem::path &model_directory,
    const std::filesystem::path &draft_directory, int draft_steps)
    : model_(model_directory), draft_steps_(draft_steps),
      tokenizer_(model_directory) {
  if (!draft_directory.empty()) {
    draft_.emplace(draft_directory, model_);
  } else {
  }
  const nlohmann::json tokenizer_config =
      omni_common::ReadJson(model_directory / "tokenizer_config.json");
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
      omni_common::ReadJson(model_directory / "generation_config.json");
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
  if (is_multilingual_ && language_ids_.empty()) {
    throw std::invalid_argument(
        "multilingual Whisper needs language token IDs");
  } else {
  }
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

mx::array WhisperTranscriber::NextToken(mx::array logits, bool first_step,
                                        float temperature) const {
  // Note (Jiaxin Deng): the float16 decoder's logits are suppressed and
  // ranked in float32, as the reference Whisper decoder does.
  logits = mx::astype(logits, mx::float32);
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
    return mx::random::categorical(
        mx::divide(logits, mx::array(temperature, logits.dtype())));
  }
}

std::vector<WhisperTranscriber::WindowResult>
WhisperTranscriber::TranscribeWindows(const mx::array &encoder_states,
                                      const std::vector<int> &prompt_ids,
                                      int max_new_tokens, float temperature,
                                      const std::atomic<bool> &cancel) const {
  const int window_count = encoder_states.shape(0);
  std::vector<DecoderLayerCache> caches(model_.config().text_layer_count);
  const int prompt_length = static_cast<int>(prompt_ids.size());
  const mx::array prompt = mx::broadcast_to(
      mx::array(prompt_ids.data(), {1, prompt_length}, mx::int32),
      {window_count, prompt_length});
  mx::array tokens = NextToken(
      model_.Decode(prompt, 0, encoder_states, caches, 1), true, temperature);
  mx::async_eval({tokens});
  if (draft_.has_value() && temperature <= 0.0f) {
    return {DecodeWithDraft(encoder_states, prompt, tokens, caches,
                            max_new_tokens, cancel)};
  } else {
  }
  std::vector<WindowResult> results(window_count);
  std::vector<std::vector<int>> generated_ids(window_count);
  // The window each batch row decodes; a finished window's row leaves the
  // batch, so later steps read only the unfinished windows' caches.
  std::vector<int> row_windows(window_count);
  for (int window = 0; window < window_count; ++window) {
    row_windows[window] = window;
  }
  for (int step = 0; !row_windows.empty(); ++step) {
    if (cancel.load()) {
      throw omni_common::GenerationCancelled();
    } else if (step == max_new_tokens) {
      for (const int window : row_windows) {
        results[window].reached_token_limit = true;
      }
      break;
    } else {
    }
    const int batch = static_cast<int>(row_windows.size());
    const mx::array current = tokens;
    // Note (khazic): queue the next step before reading this token, so the GPU
    // decodes while the end of text is checked.
    if (step + 1 < max_new_tokens) {
      tokens = NextToken(model_.Decode(mx::reshape(current, {batch, 1}),
                                       prompt_length + step, encoder_states,
                                       caches, 1),
                         false, temperature);
      mx::async_eval({tokens});
    } else {
    }
    mx::eval(current);
    const uint32_t *token_ids = current.data<uint32_t>();
    std::vector<int32_t> kept_rows;
    std::vector<int> kept_windows;
    for (int row = 0; row < batch; ++row) {
      const int window = row_windows[row];
      if (static_cast<int>(token_ids[row]) == end_of_text_id_) {
        continue;
      } else {
      }
      generated_ids[window].push_back(static_cast<int>(token_ids[row]));
      kept_rows.push_back(row);
      kept_windows.push_back(window);
    }
    if (kept_windows.size() < row_windows.size() && !kept_windows.empty()) {
      const mx::array rows(kept_rows.data(),
                           {static_cast<int>(kept_rows.size())}, mx::int32);
      tokens = mx::take(tokens, rows, 0);
      KeepCacheRows(caches, rows);
    } else {
    }
    row_windows = std::move(kept_windows);
  }
  for (int window = 0; window < window_count; ++window) {
    results[window].text = DecodeText(generated_ids[window]);
    results[window].generated_token_count =
        static_cast<int>(generated_ids[window].size());
  }
  return results;
}

WhisperTranscriber::WindowResult WhisperTranscriber::DecodeWithDraft(
    const mx::array &encoder_states, const mx::array &prompt,
    const mx::array &first_token, std::vector<DecoderLayerCache> &caches,
    int max_new_tokens, const std::atomic<bool> &cancel) const {
  std::vector<DecoderLayerCache> draft_caches(
      draft_->config().text_layer_count);
  // Note (XXXXRT666): only the draft's caches of the prompt are used; they are
  // evaluated with the first round's proposals.
  draft_->Decode(prompt, 0, encoder_states, draft_caches, 1);
  WindowResult result;
  std::vector<int> generated_ids;
  // Positions the target's caches hold: the prompt, then the output before
  // the anchor.
  int position = prompt.shape(1);
  // Output before the anchor that the draft has not decoded yet.
  std::vector<int> undrafted_ids;
  int anchor = static_cast<int>(first_token.item<uint32_t>());
  while (true) {
    if (cancel.load()) {
      throw omni_common::GenerationCancelled();
    } else if (anchor == end_of_text_id_) {
      break;
    } else {
    }
    generated_ids.push_back(anchor);
    const int remaining =
        max_new_tokens - static_cast<int>(generated_ids.size());
    if (remaining == 0) {
      result.reached_token_limit = true;
      break;
    } else {
    }
    // Note (XXXXRT666): the target's token after the last accepted proposal
    // takes the last place of the budget, so the draft proposes one fewer.
    const int proposal_count = std::min(draft_steps_, remaining - 1);
    // Note (XXXXRT666): each proposal feeds the draft's next step on the GPU,
    // so a round reads back only the target's verdict.
    std::vector<mx::array> block = {mx::array({static_cast<uint32_t>(anchor)})};
    if (proposal_count > 0) {
      int draft_position = position - static_cast<int>(undrafted_ids.size());
      for (const int token_id : undrafted_ids) {
        draft_->Decode(mx::array(&token_id, {1, 1}, mx::int32), draft_position,
                       encoder_states, draft_caches, 1);
        ++draft_position;
      }
      undrafted_ids.clear();
      for (int i = 0; i < proposal_count; ++i) {
        block.push_back(NextToken(
            draft_->Decode(mx::reshape(block.back(), {1, 1}), position + i,
                           encoder_states, draft_caches, 1),
            false, 0.0f));
      }
    } else {
      undrafted_ids.push_back(anchor);
    }
    const int block_length = proposal_count + 1;
    const mx::array block_ids = mx::concatenate(block);
    const mx::array predictions =
        NextToken(model_.Decode(mx::reshape(block_ids, {1, block_length}),
                                position, encoder_states, caches, block_length),
                  false, 0.0f);
    mx::eval({block_ids, predictions});
    const uint32_t *block_values = block_ids.data<uint32_t>();
    const uint32_t *predicted_ids = predictions.data<uint32_t>();
    int accepted = 0;
    bool stopped = false;
    while (accepted < proposal_count &&
           block_values[accepted + 1] == predicted_ids[accepted]) {
      ++accepted;
      if (static_cast<int>(block_values[accepted]) == end_of_text_id_) {
        stopped = true;
        break;
      } else {
        generated_ids.push_back(static_cast<int>(block_values[accepted]));
      }
    }
    ++result.verify_round_count;
    result.proposed_draft_token_count += proposal_count;
    result.accepted_draft_token_count += accepted;
    if (stopped) {
      break;
    } else if (accepted < proposal_count) {
      TruncateCaches(caches, position + accepted + 1);
      TruncateCaches(draft_caches, position + accepted + 1);
    } else if (proposal_count > 0) {
      // Note (XXXXRT666): the draft proposed the last token without decoding
      // it.
      undrafted_ids.push_back(static_cast<int>(block_values[proposal_count]));
    } else {
    }
    anchor = static_cast<int>(predicted_ids[accepted]);
    position += accepted + 1;
  }
  result.text = DecodeText(generated_ids);
  result.generated_token_count = static_cast<int>(generated_ids.size());
  return result;
}

omni_common::TranscriptionResult
WhisperTranscriber::Transcribe(const std::vector<float> &samples,
                               const WhisperOptions &options,
                               const std::atomic<bool> &cancel) const {
  omni_common::TranscriptionResult result;
  result.finish_reason = omni_common::FinishReason::kStop;
  // Note (ruiling): an unspecified language is inserted after detection on
  // the first window, before the transcription decoder consumes this prefix.
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

  // Note (khazic): audio up to 30 s is one window; longer audio is cut every 30
  // s. Each window is padded with silence, or trimmed, to exactly 30 s and
  // decoded on its own.
  const size_t window_count = std::max<size_t>(
      1, (samples.size() + kWindowSampleCount - 1) / kWindowSampleCount);
  // Note (XXXXRT666): up to kBatchWindows windows decode together, so a step
  // costs about what one window's does; draft verification has per-window
  // cache positions, so the draft decodes one window at a time.
  const size_t max_batch_size =
      draft_.has_value() && options.temperature <= 0.0f ? 1 : kBatchWindows;
  for (size_t first_window = 0; first_window < window_count;) {
    if (cancel.load()) {
      throw omni_common::GenerationCancelled();
    } else {
    }
    const size_t batch_size =
        std::min(max_batch_size, window_count - first_window);
    const size_t start = first_window * kWindowSampleCount;
    const size_t batch_sample_count = batch_size * kWindowSampleCount;
    const size_t end = std::min(start + batch_sample_count, samples.size());
    std::vector<float> window_samples(batch_sample_count, 0.0f);
    std::copy(samples.begin() + start, samples.begin() + end,
              window_samples.begin());
    // Each window is encoded on its own, which bounds the encoder's peak
    // memory; only the decoder steps run as a batch.
    std::vector<mx::array> window_states;
    for (size_t window = 0; window < batch_size; ++window) {
      window_states.push_back(model_.Encode(
          std::span<const float>(window_samples)
              .subspan(window * kWindowSampleCount, kWindowSampleCount),
          cancel));
    }
    const mx::array encoder_states = batch_size == 1
                                         ? window_states.front()
                                         : mx::concatenate(window_states, 0);
    if (cancel.load()) {
      throw omni_common::GenerationCancelled();
    } else {
    }
    if (is_multilingual_ && !result.language.has_value()) {
      std::vector<DecoderLayerCache> detection_caches(
          model_.config().text_layer_count);
      const mx::array start_token(&start_of_transcript_id_, {1, 1}, mx::int32);
      const mx::array logits = model_.Decode(
          start_token, 0, window_states.front(), detection_caches, 1);
      std::vector<int> language_token_ids;
      for (const auto &[code, id] : language_ids_) {
        language_token_ids.push_back(id);
      }
      const mx::array candidates(language_token_ids.data(),
                                 {static_cast<int>(language_token_ids.size())},
                                 mx::int32);
      const mx::array selected = mx::argmax(mx::take(logits, candidates, -1));
      mx::eval(selected);
      const auto language =
          std::next(language_ids_.begin(), selected.item<uint32_t>());
      result.language = language->first;
      prompt_ids.insert(prompt_ids.begin() + 1, language->second);
    } else {
    }
    const int max_new_tokens = std::max(
        1, std::min(requested_tokens,
                    context_length - static_cast<int>(prompt_ids.size()) - 1));
    const std::vector<WindowResult> windows =
        TranscribeWindows(encoder_states, prompt_ids, max_new_tokens,
                          options.temperature, cancel);
    for (const WindowResult &window : windows) {
      const std::string text = swift_port::TrimWhitespace(window.text, true);
      if (!text.empty()) {
        result.text += (result.text.empty() ? "" : " ") + text;
      } else {
      }
      if (window_count > 1) {
        result.window_texts.push_back(text);
      } else {
      }
      result.generated_token_count += window.generated_token_count;
      result.verify_round_count += window.verify_round_count;
      result.proposed_draft_token_count += window.proposed_draft_token_count;
      result.accepted_draft_token_count += window.accepted_draft_token_count;
      if (window.reached_token_limit) {
        result.finish_reason = omni_common::FinishReason::kLength;
      } else {
      }
    }
    first_window += batch_size;
  }
  return result;
}

} // namespace whisper
