// SPDX-License-Identifier: Apache-2.0
#include "arkasr/arkasr_transcriber.h"

#include <algorithm>
#include <iterator>
#include <stdexcept>

#include "common/audio.h"

namespace arkasr {

namespace mx = mlx::core;
using qwen3_asr::FinishReason;
using qwen3_asr::kSampleRate;
using qwen3_asr::KVCache;
using qwen3_asr::StripUnicodeWhitespace;
using qwen3_asr::TranscriptionCancelled;
using qwen3_asr::TranscriptionResult;

namespace {

constexpr const char *kPromptPrefix = "<|user|><|begin_of_audio|>";
constexpr const char *kPromptSuffix =
    "<|end_of_audio|>Please transcribe this audio.<|assistant|>";
constexpr const char *kEndOfTurn = "<|im_end|>";
// Note (AzizbekMuminoff): the processor truncates at Whisper's 30 s window,
// so longer audio is decoded one window at a time.
constexpr size_t kWindowSampleCount = 30 * kSampleRate;
constexpr size_t kCutSearchSampleCount = 5 * kSampleRate;
constexpr size_t kCutEnergySampleCount = kSampleRate / 10;
constexpr int kDefaultMaxNewTokens = 256;

// [start, end) sample ranges of at most one window each, every cut but the
// last at the quietest 100 ms in the window's final seconds.
std::vector<std::pair<size_t, size_t>>
SplitAtQuietPoints(const std::vector<float> &samples) {
  std::vector<std::pair<size_t, size_t>> windows;
  size_t start = 0;
  while (samples.size() - start > kWindowSampleCount) {
    const size_t end = start + kWindowSampleCount;
    const size_t search_start = end - kCutSearchSampleCount;
    double energy = 0.0;
    for (size_t i = search_start; i < search_start + kCutEnergySampleCount;
         ++i) {
      energy += samples[i] * samples[i];
    }
    double min_energy = energy;
    size_t min_start = search_start;
    for (size_t window_start = search_start + 1;
         window_start + kCutEnergySampleCount <= end; ++window_start) {
      const float leaving = samples[window_start - 1];
      const float entering = samples[window_start + kCutEnergySampleCount - 1];
      energy += entering * entering - leaving * leaving;
      if (energy < min_energy) {
        min_energy = energy;
        min_start = window_start;
      } else {
      }
    }
    const size_t cut = min_start + kCutEnergySampleCount / 2;
    windows.emplace_back(start, cut);
    start = cut;
  }
  windows.emplace_back(start, samples.size());
  return windows;
}

} // namespace

ArkTranscriber::ArkTranscriber(const std::filesystem::path &model_directory)
    : model_(model_directory), tokenizer_(model_directory),
      prompt_prefix_ids_(tokenizer_.Encode(kPromptPrefix)),
      prompt_suffix_ids_(tokenizer_.Encode(kPromptSuffix)),
      end_of_turn_id_(tokenizer_.AddedTokenId(kEndOfTurn)) {
  if (end_of_turn_id_ < 0) {
    throw std::runtime_error(std::string("tokenizer has no ") + kEndOfTurn);
  } else {
  }
  const nlohmann::json config =
      qwen3_asr::ReadJson(model_directory / "tokenizer_config.json");
  for (const auto &[id, token] : config.at("added_tokens_decoder").items()) {
    const std::string content = token.at("content").get<std::string>();
    if (!content.empty() && content.front() == '<' && content.back() == '>' &&
        content != kEndOfTurn) {
      marker_token_ids_.insert(std::stoi(id));
    } else {
    }
  }
}

std::vector<int> ArkTranscriber::PromptIds(int audio_token_count) const {
  std::vector<int> ids = prompt_prefix_ids_;
  ids.insert(ids.end(), audio_token_count, model_.audio_token_id());
  ids.insert(ids.end(), prompt_suffix_ids_.begin(), prompt_suffix_ids_.end());
  return ids;
}

TranscriptionResult
ArkTranscriber::TranscribeWindow(const std::vector<float> &samples,
                                 int max_new_tokens,
                                 const std::atomic<bool> &cancel) const {
  if (cancel.load()) {
    throw TranscriptionCancelled();
  } else {
  }
  const mx::array mel =
      qwen3_asr::LogMel(samples, qwen3_asr::AudioLayout::kReference);
  const int audio_token_count = model_.AudioTokenCount(mel.shape(1));
  const std::vector<int> prompt_ids = PromptIds(audio_token_count);
  const int prompt_length = static_cast<int>(prompt_ids.size());
  const mx::array audio_features = model_.EncodeAudio(mel);
  if (audio_features.shape(0) != audio_token_count) {
    throw std::runtime_error("audio features and audio tokens do not match");
  } else {
  }
  const mx::array token_embeddings = model_.decoder().EmbedTokens(
      mx::array(prompt_ids.data(), {1, prompt_length}, mx::int32));
  const int hidden = token_embeddings.shape(2);
  const int audio_start = static_cast<int>(prompt_prefix_ids_.size());
  const int audio_end = audio_start + audio_token_count;
  const mx::array embeddings = mx::concatenate(
      {mx::slice(token_embeddings, {0, 0, 0}, {1, audio_start, hidden}),
       mx::expand_dims(audio_features, 0),
       mx::slice(token_embeddings, {0, audio_end, 0},
                 {1, prompt_length, hidden})},
      1);
  // Note (AzizbekMuminoff): encoding a window takes about a second, so a
  // request cancelled meanwhile stops before its prefill too.
  mx::eval(embeddings);
  if (cancel.load()) {
    throw TranscriptionCancelled();
  } else {
  }

  std::vector<KVCache> caches = model_.decoder().NewCaches();
  mx::array next_token = mx::argmax(model_.decoder().LastLogits(
      model_.decoder().Forward(embeddings, caches)));
  mx::async_eval({next_token});
  std::vector<int> output_ids;
  FinishReason finish_reason = FinishReason::kLength;
  while (static_cast<int>(output_ids.size()) < max_new_tokens) {
    if (cancel.load()) {
      throw TranscriptionCancelled();
    } else {
    }
    const mx::array token = next_token;
    // Note (AzizbekMuminoff): the next step is queued before this token is
    // read, so the GPU decodes while the stop rule is checked.
    next_token =
        mx::argmax(model_.decoder().LastLogits(model_.decoder().Forward(
            model_.decoder().EmbedTokens(mx::reshape(token, {1, 1})), caches)));
    mx::async_eval({next_token});
    const int token_id = static_cast<int>(token.item<uint32_t>());
    if (token_id == end_of_turn_id_) {
      finish_reason = FinishReason::kStop;
      break;
    } else {
    }
    output_ids.push_back(token_id);
  }
  std::vector<int> text_ids;
  std::copy_if(output_ids.begin(), output_ids.end(),
               std::back_inserter(text_ids),
               [&](int id) { return marker_token_ids_.count(id) == 0; });
  TranscriptionResult result;
  result.text = StripUnicodeWhitespace(tokenizer_.Decode(text_ids, true));
  result.generated_token_count = static_cast<int>(output_ids.size());
  result.finish_reason = finish_reason;
  return result;
}

TranscriptionResult
ArkTranscriber::Transcribe(const std::vector<float> &samples,
                           const ArkOptions &options,
                           const std::atomic<bool> &cancel) const {
  if (samples.empty()) {
    throw std::invalid_argument("audio must contain at least one sample");
  } else {
  }
  const int max_new_tokens =
      options.max_new_tokens.value_or(kDefaultMaxNewTokens);
  TranscriptionResult combined;
  combined.language = options.language;
  combined.finish_reason = FinishReason::kStop;
  for (const auto &[start, end] : SplitAtQuietPoints(samples)) {
    const TranscriptionResult window =
        TranscribeWindow({samples.begin() + start, samples.begin() + end},
                         max_new_tokens, cancel);
    // Note (AzizbekMuminoff): Chinese and other scripts written without spaces
    // join windows directly; ASCII words on both sides get a space.
    if (window.text.empty()) {
    } else if (combined.text.empty() ||
               static_cast<unsigned char>(combined.text.back()) >= 0x80 ||
               static_cast<unsigned char>(window.text.front()) >= 0x80) {
      combined.text += window.text;
    } else {
      combined.text += " " + window.text;
    }
    combined.generated_token_count += window.generated_token_count;
    if (window.finish_reason == FinishReason::kLength) {
      combined.finish_reason = FinishReason::kLength;
    } else {
    }
    mx::clear_cache();
  }
  return combined;
}

} // namespace arkasr
