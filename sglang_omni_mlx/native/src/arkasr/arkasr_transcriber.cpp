// SPDX-License-Identifier: Apache-2.0
#include "arkasr/arkasr_transcriber.h"

#include <algorithm>
#include <iterator>
#include <stdexcept>

#include "common/audio.h"
#include "common/audio_chunks.h"
#include "common/generation.h"
#include "common/kv_cache.h"
#include "common/text.h"

namespace arkasr {

namespace mx = mlx::core;
using omni_common::FinishReason;
using omni_common::GenerationCancelled;
using omni_common::KVCache;
using omni_common::TranscriptionResult;

namespace {

constexpr const char *kPromptPrefix = "<|user|><|begin_of_audio|>";
constexpr const char *kPromptSuffix =
    "<|end_of_audio|>Please transcribe this audio.<|assistant|>";
constexpr const char *kEndOfTurn = "<|im_end|>";
// Note (AzizbekMuminoff): the processor truncates at Whisper's 30 s window,
// so longer audio is decoded one window at a time.
constexpr size_t kWindowSampleCount = omni_common::kWhisperWindowSampleCount;
constexpr size_t kCutSearchSampleCount = 5 * omni_common::kSampleRate;
constexpr size_t kCutEnergySampleCount = omni_common::kSampleRate / 10;
// Note (XXXXRT666): the search stops a second before the audio ends, so the
// last window never becomes a sliver of padding the model transcribes as
// rubbish.
constexpr size_t kMinTailSampleCount = omni_common::kSampleRate;
constexpr int kDefaultMaxNewTokens = 256;

} // namespace

ArkTranscriber::ArkTranscriber(const std::filesystem::path &model_directory)
    : model_(model_directory), tokenizer_(model_directory),
      prefix_embeddings_(PromptEmbeddings(kPromptPrefix)),
      suffix_embeddings_(PromptEmbeddings(kPromptSuffix)),
      end_of_turn_id_(tokenizer_.AddedTokenId(kEndOfTurn)) {
  if (end_of_turn_id_ < 0) {
    throw std::runtime_error(std::string("tokenizer has no ") + kEndOfTurn);
  } else {
  }
  const nlohmann::json config =
      omni_common::ReadJson(model_directory / "tokenizer_config.json");
  for (const auto &[id, token] : config.at("added_tokens_decoder").items()) {
    const std::string content = token.at("content").get<std::string>();
    if (!content.empty() && content.front() == '<' && content.back() == '>' &&
        content != kEndOfTurn) {
      marker_token_ids_.insert(std::stoi(id));
    } else {
    }
  }
}

mx::array ArkTranscriber::PromptEmbeddings(const char *text) const {
  const std::vector<int> ids = tokenizer_.Encode(text);
  return model_.decoder().EmbedTokens(
      mx::array(ids.data(), {1, static_cast<int>(ids.size())}, mx::int32));
}

TranscriptionResult
ArkTranscriber::TranscribeWindow(const std::vector<float> &samples,
                                 int max_new_tokens,
                                 const std::atomic<bool> &cancel) const {
  if (cancel.load()) {
    throw GenerationCancelled();
  } else {
  }
  const mx::array mel =
      omni_common::LogMel(samples, omni_common::FinalStftFrame::kDropped);
  const mx::array audio_features = model_.EncodeAudio(mel);
  const mx::array embeddings =
      mx::concatenate({prefix_embeddings_, mx::expand_dims(audio_features, 0),
                       suffix_embeddings_},
                      1);
  // Note (AzizbekMuminoff): encoding a window takes about a second, so a
  // request cancelled meanwhile stops before its prefill too.
  mx::eval(embeddings);
  if (cancel.load()) {
    throw GenerationCancelled();
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
      throw GenerationCancelled();
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
  result.text =
      omni_common::StripUnicodeWhitespace(tokenizer_.Decode(text_ids, true));
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
  const std::vector<std::pair<size_t, size_t>> windows =
      omni_common::BoundedEnergyCutChunks(
          samples, kWindowSampleCount, kCutSearchSampleCount,
          kCutEnergySampleCount, kMinTailSampleCount);
  // Note (AzizbekMuminoff): one budget for the whole request, as on the
  // other ASR servers; the default is the per-window one scaled to the
  // request, as the checkpoint's processor sees at most one window.
  const int request_budget = options.max_new_tokens.value_or(0);
  int remaining_tokens =
      request_budget != 0
          ? request_budget
          : kDefaultMaxNewTokens * static_cast<int>(windows.size());
  std::vector<std::string> parts;
  int generated_token_count = 0;
  FinishReason finish_reason = FinishReason::kStop;
  for (const auto &[start, end] : windows) {
    if (remaining_tokens <= 0) {
      finish_reason = FinishReason::kLength;
      break;
    } else {
    }
    const TranscriptionResult window =
        TranscribeWindow({samples.begin() + start, samples.begin() + end},
                         remaining_tokens, cancel);
    if (window.text.empty()) {
    } else {
      parts.push_back(window.text);
    }
    generated_token_count += window.generated_token_count;
    remaining_tokens -= window.generated_token_count;
    if (window.finish_reason == FinishReason::kLength) {
      finish_reason = FinishReason::kLength;
    } else {
    }
    mx::clear_cache();
  }
  TranscriptionResult combined;
  combined.text = omni_common::JoinTranscriptParts(parts);
  combined.language = options.language;
  combined.generated_token_count = generated_token_count;
  combined.finish_reason = finish_reason;
  if (windows.size() > 1) {
    combined.window_texts = parts;
  } else {
  }
  return combined;
}

} // namespace arkasr
