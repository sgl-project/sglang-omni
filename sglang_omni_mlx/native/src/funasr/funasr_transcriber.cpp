// SPDX-License-Identifier: Apache-2.0
#include "funasr/funasr_transcriber.h"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <string_view>

#include "common/audio.h"
#include "common/generation.h"
#include "common/kv_cache.h"
#include "common/text.h"

namespace funasr {

namespace mx = mlx::core;
using omni_common::FinishReason;
using omni_common::GenerationCancelled;
using omni_common::KVCache;
using omni_common::TranscriptionResult;

namespace {

constexpr const char *kPromptPrefix =
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\n";
constexpr const char *kPromptSuffix = "<|im_end|>\n<|im_start|>assistant\n";
constexpr const char *kEndOfTurn = "<|im_end|>";
// The reference's default budget: 200 tokens for 30 s, in proportion to the
// audio, and at least 16.
constexpr size_t kDefaultBudgetSampleCount = 30 * omni_common::kSampleRate;
constexpr double kDefaultTokensPerWindow = 200.0;
constexpr int kMinDefaultTokens = 16;
// Note (wilsonzheng0327): the checkpoint writes this marker where the audio
// pauses; it is not speech, so the text leaves it out.
constexpr std::string_view kSilenceMarker = "/sil";

// The language the prompt names; none for Chinese and auto, which leave the
// model to choose.
std::optional<std::string>
PromptLanguage(const std::optional<std::string> &language) {
  if (!language.has_value()) {
    return std::nullopt;
  } else {
  }
  const std::string stripped = omni_common::StripUnicodeWhitespace(*language);
  const std::string lowered = omni_common::AsciiLowercase(stripped);
  if (lowered.empty() || lowered == "auto" || lowered == "null" ||
      lowered == "none" || lowered == "zh" || lowered == "cn" ||
      lowered == "chinese" || stripped == "中文") {
    return std::nullopt;
  } else if (lowered == "en" || lowered == "english" || stripped == "英文") {
    return std::string("英文");
  } else {
    return stripped;
  }
}

// The user turn's text before the audio, as the reference renders it.
std::string PromptText(const FunAsrOptions &options) {
  std::string hotwords;
  std::istringstream prompt(options.prompt.value_or(""));
  for (std::string term; std::getline(prompt, term, ',');) {
    term = omni_common::StripUnicodeWhitespace(term);
    if (!term.empty()) {
      hotwords += (hotwords.empty() ? "" : ", ") + term;
    } else {
    }
  }
  std::string text;
  if (!hotwords.empty()) {
    text += "请结合上下文信息，更加准确地完成语音转写任务。"
            "如果没有相关信息，我们会留空。\n\n\n**上下文信息：**\n\n\n"
            "热词列表：[" +
            hotwords + "]\n";
  } else {
  }
  const std::optional<std::string> language = PromptLanguage(options.language);
  return text + (language.has_value() ? "语音转写成" + *language : "语音转写") +
         "：";
}

int DefaultMaxNewTokens(size_t sample_count) {
  return std::max(kMinDefaultTokens,
                  static_cast<int>(std::ceil(static_cast<double>(sample_count) /
                                             kDefaultBudgetSampleCount *
                                             kDefaultTokensPerWindow)));
}

} // namespace

FunAsrTranscriber::FunAsrTranscriber(
    const std::filesystem::path &model_directory)
    : model_(model_directory), tokenizer_(model_directory),
      suffix_embeddings_(PromptEmbeddings(kPromptSuffix)),
      im_end_id_(tokenizer_.AddedTokenId(kEndOfTurn)) {
  if (im_end_id_ < 0) {
    throw std::runtime_error(std::string("tokenizer has no ") + kEndOfTurn);
  } else {
  }
}

mx::array FunAsrTranscriber::PromptEmbeddings(const std::string &text) const {
  const std::vector<int> ids = tokenizer_.Encode(text);
  return model_.decoder().EmbedTokens(
      mx::array(ids.data(), {1, static_cast<int>(ids.size())}, mx::int32));
}

TranscriptionResult FunAsrTranscriber::TranscribeWindow(
    const std::vector<float> &samples, const mx::array &prompt_embeddings,
    int max_new_tokens, float repetition_penalty,
    const std::atomic<bool> &cancel) const {
  if (cancel.load()) {
    throw GenerationCancelled();
  } else {
  }
  TranscriptionResult result;
  result.finish_reason = FinishReason::kStop;
  const mx::array features = model_.Features(samples);
  // Note (wilsonzheng0327): audio shorter than one 25 ms window has no frames
  // to encode, so it transcribes as nothing.
  if (features.shape(0) == 0) {
    return result;
  } else {
  }
  const mx::array embeddings = mx::concatenate(
      {prompt_embeddings, mx::expand_dims(model_.EncodeAudio(features), 0),
       suffix_embeddings_},
      1);
  mx::eval(embeddings);
  if (cancel.load()) {
    throw GenerationCancelled();
  } else {
  }
  std::vector<int> output_ids;
  result.finish_reason = FinishReason::kLength;
  const auto commit = [&](int token_id) {
    if (token_id == im_end_id_) {
      result.finish_reason = FinishReason::kStop;
      return true;
    } else {
    }
    output_ids.push_back(token_id);
    return static_cast<int>(output_ids.size()) >= max_new_tokens;
  };
  std::vector<KVCache> caches = model_.decoder().NewCaches();
  mx::array next_token = mx::argmax(model_.decoder().LastLogits(
      model_.decoder().Forward(embeddings, caches)));
  mx::async_eval({next_token});
  model_.decoder().DecodePlain(next_token, caches, repetition_penalty, commit,
                               cancel);
  const std::string decoded = tokenizer_.Decode(output_ids, true);
  std::vector<std::string> spoken_parts;
  for (size_t begin = 0;;) {
    const size_t marker = decoded.find(kSilenceMarker, begin);
    spoken_parts.push_back(decoded.substr(begin, marker - begin));
    if (marker == std::string::npos) {
      break;
    } else {
      begin = marker + kSilenceMarker.size();
    }
  }
  result.text = omni_common::JoinTranscriptParts(spoken_parts);
  result.generated_token_count = static_cast<int>(output_ids.size());
  return result;
}

TranscriptionResult
FunAsrTranscriber::Transcribe(const std::vector<float> &samples,
                              const FunAsrOptions &options,
                              const std::atomic<bool> &cancel) const {
  if (samples.empty()) {
    throw std::invalid_argument("audio must contain at least one sample");
  } else {
  }
  const mx::array prompt_embeddings =
      PromptEmbeddings(kPromptPrefix + PromptText(options));
  // Note (wilsonzheng0327): an explicit budget is one for the whole request,
  // as on the other ASR servers; the default is the reference's per window.
  const int request_budget = options.max_new_tokens.value_or(0);
  // Note (wilsonzheng0327): the reference accepts at most 30 s, its VAD's
  // segment limit, so longer audio is decoded one window at a time.
  TranscriptionResult combined = omni_common::TranscribeEachWindow(
      samples, omni_common::ThirtySecondWindows(samples),
      [&](const std::pair<size_t, size_t> &window, int generated_token_count) {
        return request_budget != 0
                   ? request_budget - generated_token_count
                   : DefaultMaxNewTokens(window.second - window.first);
      },
      [&](const std::vector<float> &window_samples, int max_new_tokens) {
        return TranscribeWindow(window_samples, prompt_embeddings,
                                max_new_tokens, options.repetition_penalty,
                                cancel);
      });
  combined.language = options.language;
  return combined;
}

} // namespace funasr
