// SPDX-License-Identifier: Apache-2.0
#include "qwen3_asr/qwen3_asr_transcriber.h"
#include "common/text.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <map>

namespace qwen3_asr {

using omni_common::IsTokenLoop;
using omni_common::IsUnicodeSpace;

namespace mx = mlx::core;

namespace {

constexpr const char *kAsrTextMarker = "<asr_text>";
constexpr const char *kAudioPad = "<|audio_pad|>";
constexpr int kOutputTokensPerAudioSecond = 10;
constexpr int kMinDefaultOutputTokens = 128;

const std::map<std::string, std::string> &LanguageCodeToName() {
  static const std::map<std::string, std::string> table = {
      {"ar", "Arabic"},     {"yue", "Cantonese"}, {"zh", "Chinese"},
      {"cs", "Czech"},      {"da", "Danish"},     {"nl", "Dutch"},
      {"en", "English"},    {"fil", "Filipino"},  {"fi", "Finnish"},
      {"fr", "French"},     {"de", "German"},     {"el", "Greek"},
      {"hi", "Hindi"},      {"hu", "Hungarian"},  {"id", "Indonesian"},
      {"it", "Italian"},    {"ja", "Japanese"},   {"ko", "Korean"},
      {"mk", "Macedonian"}, {"ms", "Malay"},      {"fa", "Persian"},
      {"pl", "Polish"},     {"pt", "Portuguese"}, {"ro", "Romanian"},
      {"ru", "Russian"},    {"es", "Spanish"},    {"sv", "Swedish"},
      {"th", "Thai"},       {"tr", "Turkish"},    {"vi", "Vietnamese"},
  };
  return table;
}

std::string Strip(const std::string &text) {
  const auto is_space = [](unsigned char c) { return std::isspace(c) != 0; };
  size_t begin = 0;
  size_t end = text.size();
  while (begin < end && is_space(text[begin]))
    ++begin;
  while (end > begin && is_space(text[end - 1]))
    --end;
  return text.substr(begin, end - begin);
}

std::string Lowercase(std::string text) {
  std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return text;
}

std::optional<size_t> FindSubsequence(const std::vector<int> &values,
                                      const std::vector<int> &pattern) {
  if (pattern.size() > values.size()) {
    return std::nullopt;
  } else {
  }
  for (size_t start = 0; start + pattern.size() <= values.size(); ++start) {
    if (std::equal(pattern.begin(), pattern.end(), values.begin() + start)) {
      return start;
    } else {
    }
  }
  return std::nullopt;
}

} // namespace

const char *FinishReasonName(FinishReason reason) {
  return reason == FinishReason::kStop ? "stop" : "length";
}

nlohmann::ordered_json
SpeakerSegmentsJson(const std::vector<SpeakerSegment> &segments) {
  nlohmann::ordered_json array = nlohmann::ordered_json::array();
  for (const SpeakerSegment &segment : segments) {
    nlohmann::ordered_json json = {{"start", segment.start_seconds},
                                   {"end", segment.end_seconds},
                                   {"speaker", segment.speaker},
                                   {"text", segment.text}};
    for (const auto &[name, value] : {std::pair{"language", &segment.language},
                                      std::pair{"emotion", &segment.emotion},
                                      std::pair{"event", &segment.event}}) {
      if (value->has_value()) {
        json[name] = **value;
      } else {
      }
    }
    array.push_back(std::move(json));
  }
  return array;
}

std::optional<std::string> NormalizeLanguage(const std::string &language) {
  const std::string stripped = Strip(language);
  const std::string normalized = Lowercase(stripped);
  if (stripped.empty()) {
    return std::nullopt;
  } else if (normalized == "cn" || normalized.rfind("zh-", 0) == 0 ||
             normalized.rfind("zh_", 0) == 0) {
    return std::string("Chinese");
  } else {
  }
  const auto code = LanguageCodeToName().find(normalized);
  if (code != LanguageCodeToName().end()) {
    return code->second;
  } else {
  }
  for (const auto &[unused_code, name] : LanguageCodeToName()) {
    if (Lowercase(name) == normalized) {
      return name;
    } else {
    }
  }
  return stripped;
}

Qwen3ASRTranscriber::Qwen3ASRTranscriber(
    const std::filesystem::path &model_directory,
    const std::filesystem::path &draft_directory, int draft_quantization_bits)
    : model_(model_directory), tokenizer_(model_directory),
      audio_pad_id_(tokenizer_.AddedTokenId(kAudioPad)),
      end_of_text_id_(tokenizer_.AddedTokenId("<|endoftext|>")),
      im_end_id_(tokenizer_.AddedTokenId("<|im_end|>")),
      asr_text_ids_(tokenizer_.Encode(kAsrTextMarker)) {
  if (!draft_directory.empty()) {
    draft_.emplace(draft_directory, draft_quantization_bits,
                   model_.decoder().config());
  } else {
  }
}

std::vector<int>
Qwen3ASRTranscriber::PromptIds(int audio_token_count,
                               const TranscriptionOptions &options) const {
  std::string prompt = "<|im_start|>system\n" +
                       Strip(options.context.value_or("")) +
                       "<|im_end|>\n<|im_start|>user\n<|audio_start|>";
  for (int i = 0; i < audio_token_count; ++i)
    prompt += kAudioPad;
  prompt += "<|audio_end|><|im_end|>\n<|im_start|>assistant\n";
  const std::optional<std::string> language =
      NormalizeLanguage(options.language.value_or(""));
  if (language.has_value()) {
    prompt += "language " + *language + kAsrTextMarker;
  } else {
  }
  std::vector<int> ids = tokenizer_.Encode(prompt);
  ids.insert(ids.end(), options.prefix_token_ids.begin(),
             options.prefix_token_ids.end());
  return ids;
}

std::pair<std::vector<int>, std::string>
Qwen3ASRTranscriber::RetainedPrefix(const std::string &text,
                                    int rollback_token_count) const {
  const std::vector<int> token_ids = tokenizer_.Encode(text);
  const size_t keep =
      token_ids.size() > static_cast<size_t>(rollback_token_count)
          ? token_ids.size() - rollback_token_count
          : 0;
  std::vector<int> retained(token_ids.begin(), token_ids.begin() + keep);
  static const std::string replacement = "\xEF\xBF\xBD";
  while (!retained.empty()) {
    const std::string decoded = tokenizer_.Decode(retained, false);
    if (decoded.size() >= replacement.size() &&
        decoded.compare(decoded.size() - replacement.size(), replacement.size(),
                        replacement) == 0) {
      retained.pop_back();
    } else {
      return {retained, decoded};
    }
  }
  return {{}, ""};
}

TranscriptionResult
Qwen3ASRTranscriber::Transcribe(const std::vector<float> &samples,
                                const TranscriptionOptions &options,
                                const std::atomic<bool> &cancel) const {
  if (cancel.load()) {
    throw TranscriptionCancelled();
  } else {
  }
  const mx::array mel = LogMel(samples, options.layout);
  const int audio_token_count = TokenCount(mel.shape(-1), options.layout);
  const std::vector<int> prompt_ids = PromptIds(audio_token_count, options);
  const mx::array audio_features = model_.EncodeAudio(mel, options.layout);
  const mx::array input_ids(
      prompt_ids.data(), {1, static_cast<int>(prompt_ids.size())}, mx::int32);
  mx::array embeddings = model_.EmbedTokens(input_ids);
  const int audio_start = static_cast<int>(
      std::find(prompt_ids.begin(), prompt_ids.end(), audio_pad_id_) -
      prompt_ids.begin());
  // Note (Jiaxin Deng): the Swift layout can reserve more placeholders than
  // encoder rows; those keep the audio_pad embedding, surplus rows are dropped.
  const int filled = std::min(audio_token_count, audio_features.shape(0));
  const int hidden = embeddings.shape(2);
  embeddings = mx::slice_update(
      embeddings,
      mx::expand_dims(mx::astype(mx::slice(audio_features, {0, 0},
                                           {filled, audio_features.shape(1)}),
                                 embeddings.dtype()),
                      0),
      {0, audio_start, 0}, {1, audio_start + filled, hidden});

  // Note (Jiaxin Deng): an unset or zero budget means the default.
  const int max_new_tokens =
      options.max_new_tokens.value_or(0) != 0
          ? *options.max_new_tokens
          : std::max(
                kMinDefaultOutputTokens,
                static_cast<int>(std::ceil(static_cast<double>(samples.size()) /
                                           omni_common::kSampleRate *
                                           kOutputTokensPerAudioSecond)));
  const auto is_stop = [&](int token_id) {
    return token_id == im_end_id_ ||
           (options.stop_at_end_of_text && token_id == end_of_text_id_);
  };
  std::vector<int> output_ids;
  FinishReason finish_reason = FinishReason::kLength;
  // Adds a token to the output; true once the output is complete.
  const auto commit = [&](int token_id) {
    output_ids.push_back(token_id);
    if (is_stop(token_id)) {
      finish_reason = FinishReason::kStop;
      return true;
    } else if (options.stop_on_token_loop && IsTokenLoop(output_ids)) {
      finish_reason = FinishReason::kStop;
      return true;
    } else {
      return static_cast<int>(output_ids.size()) >= max_new_tokens;
    }
  };
  int verify_round_count = 0;
  if (draft_.has_value()) {
    verify_round_count = DecodeWithDraft(embeddings, commit, cancel);
  } else {
    std::vector<KVCache> caches = model_.NewCaches();
    mx::array next_token = mx::argmax(model_.Decode(embeddings, caches));
    mx::async_eval({next_token});
    bool finished = false;
    while (!finished) {
      if (cancel.load()) {
        throw TranscriptionCancelled();
      } else {
      }
      const mx::array token = next_token;
      // Note (Jiaxin Deng): queue the next step before reading this token, so
      // the GPU decodes while the stop rules are checked.
      next_token = mx::argmax(model_.Decode(
          model_.EmbedTokens(mx::reshape(token, {1, 1})), caches));
      mx::async_eval({next_token});
      finished = commit(static_cast<int>(token.item<uint32_t>()));
    }
  }
  auto [text, language] = SplitOutput(output_ids, options);
  const bool ended_on_stop = !output_ids.empty() && is_stop(output_ids.back());
  return {options.prefix_text + text,
          language,
          static_cast<int>(output_ids.size()) - (ended_on_stop ? 1 : 0),
          finish_reason,
          {},
          verify_round_count};
}

int Qwen3ASRTranscriber::DecodeWithDraft(
    const mx::array &embeddings, const std::function<bool(int)> &commit,
    const std::atomic<bool> &cancel) const {
  const Qwen3Decoder &decoder = model_.decoder();
  const int block_size = draft_->block_size();
  std::vector<KVCache> caches = model_.NewCaches();
  std::vector<KVCache> draft_caches = draft_->NewCaches();
  std::vector<mx::array> aux_hidden_states;
  const mx::array prompt_hidden = decoder.Forward(
      embeddings, caches, draft_->aux_layers(), &aux_hidden_states);
  draft_->AppendContext(aux_hidden_states, embeddings.shape(1), draft_caches);
  int anchor_id = static_cast<int>(
      mx::argmax(decoder.LastLogits(prompt_hidden)).item<uint32_t>());
  bool finished = commit(anchor_id);
  int round_count = 0;
  while (!finished) {
    if (cancel.load()) {
      throw TranscriptionCancelled();
    } else {
    }
    std::vector<int> block_ids(block_size, draft_->mask_token_id());
    block_ids[0] = anchor_id;
    const mx::array proposed_ids =
        draft_->ProposeBlock(model_.EmbedTokens(mx::array(
                                 block_ids.data(), {1, block_size}, mx::int32)),
                             draft_caches);
    const mx::array verify_ids = mx::concatenate(
        {mx::array(&anchor_id, {1}, mx::int32), proposed_ids}, 0);
    aux_hidden_states.clear();
    const mx::array hidden = decoder.Forward(
        model_.EmbedTokens(mx::reshape(verify_ids, {1, block_size})), caches,
        draft_->aux_layers(), &aux_hidden_states);
    const mx::array predictions = mx::argmax(decoder.Logits(hidden), -1);
    mx::eval({proposed_ids, predictions});
    ++round_count;
    // note (wilsonzheng0327): prediction i follows block token i, so proposal
    // i is accepted while every earlier one was and it matches prediction i;
    // the prediction after the last accepted token is committed as well.
    const int32_t *proposed = proposed_ids.data<int32_t>();
    const uint32_t *predicted = predictions.data<uint32_t>();
    int accepted_count = 0;
    while (!finished && accepted_count < block_size - 1 &&
           proposed[accepted_count] ==
               static_cast<int>(predicted[accepted_count])) {
      finished = commit(proposed[accepted_count]);
      ++accepted_count;
    }
    if (!finished) {
      anchor_id = static_cast<int>(predicted[accepted_count]);
      finished = commit(anchor_id);
    } else {
    }
    // note (wilsonzheng0327): the caches keep the anchor and the accepted
    // proposals; the committed positions' hidden states become the draft's
    // context.
    for (KVCache &cache : caches) {
      cache.Trim(block_size - 1 - accepted_count);
    }
    draft_->AppendContext(aux_hidden_states, accepted_count + 1, draft_caches);
  }
  return round_count;
}

std::pair<std::string, std::optional<std::string>>
Qwen3ASRTranscriber::SplitOutput(const std::vector<int> &output_ids,
                                 const TranscriptionOptions &options) const {
  const std::optional<size_t> marker =
      FindSubsequence(output_ids, asr_text_ids_);
  std::optional<std::string> detected;
  if (!options.language.has_value() && marker.has_value()) {
    const std::string prefix = Strip(tokenizer_.Decode(
        std::vector<int>(output_ids.begin(), output_ids.begin() + *marker),
        true));
    const size_t separator = prefix.find(' ');
    if (separator != std::string::npos &&
        Lowercase(prefix.substr(0, separator)) == "language") {
      const std::string value = Strip(prefix.substr(separator + 1));
      if (!value.empty() && Lowercase(value) != "none") {
        detected = value;
      } else {
      }
    } else if (!prefix.empty()) {
      detected = prefix;
    } else {
    }
  } else {
  }
  const std::vector<int> transcript_ids =
      marker.has_value() ? std::vector<int>(output_ids.begin() + *marker +
                                                asr_text_ids_.size(),
                                            output_ids.end())
                         : output_ids;
  const std::string text = tokenizer_.Decode(transcript_ids, true);
  std::optional<std::string> language =
      NormalizeLanguage(options.language.value_or(""));
  if (!language.has_value() && detected.has_value()) {
    language = NormalizeLanguage(*detected);
  } else {
  }
  return {text, language};
}

} // namespace qwen3_asr
