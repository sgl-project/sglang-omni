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
using omni_common::DraftStatistics;
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

bool WindowIsAllZero(const std::vector<float> &samples,
                     const std::pair<size_t, size_t> &window) {
  return std::all_of(samples.begin() + window.first,
                     samples.begin() + window.second,
                     [](float sample) { return sample == 0.0f; });
}

} // namespace

ArkTranscriber::ArkTranscriber(const std::filesystem::path &model_directory,
                               const std::filesystem::path &draft_directory,
                               int draft_quantization_bits)
    : model_(model_directory), tokenizer_(model_directory),
      prefix_embeddings_(PromptEmbeddings(kPromptPrefix)),
      suffix_embeddings_(PromptEmbeddings(kPromptSuffix)),
      end_of_turn_id_(tokenizer_.AddedTokenId(kEndOfTurn)) {
  if (!draft_directory.empty()) {
    draft_.emplace(draft_directory, draft_quantization_bits,
                   model_.decoder().config());
  } else {
  }
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

  std::vector<int> output_ids;
  FinishReason finish_reason = FinishReason::kLength;
  // Adds a token to the output; true once the window is finished.
  const auto commit = [&](int token_id) {
    if (token_id == end_of_turn_id_) {
      finish_reason = FinishReason::kStop;
      return true;
    } else {
      output_ids.push_back(token_id);
      return static_cast<int>(output_ids.size()) >= max_new_tokens;
    }
  };
  DraftStatistics draft_statistics;
  if (draft_.has_value()) {
    draft_statistics =
        draft_->Decode(model_.decoder(), embeddings, max_new_tokens,
                       /*prefill_step_tokens=*/0, commit, cancel);
  } else {
    std::vector<KVCache> caches = model_.decoder().NewCaches();
    mx::array next_token = mx::argmax(model_.decoder().LastLogits(
        model_.decoder().Forward(embeddings, caches)));
    mx::async_eval({next_token});
    model_.decoder().DecodePlain(next_token, caches, 1.0f, commit, cancel);
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
  result.verify_round_count = draft_statistics.verify_round_count;
  result.proposed_draft_token_count = draft_statistics.proposed_token_count;
  result.accepted_draft_token_count = draft_statistics.accepted_token_count;
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
  TranscriptionResult combined;
  combined.finish_reason = FinishReason::kStop;
  std::vector<std::string> parts;
  for (const auto &[start, end] : windows) {
    if (remaining_tokens <= 0) {
      combined.finish_reason = FinishReason::kLength;
      break;
    } else {
    }
    // Note (AzizbekMuminoff): a split can leave a window that is only the
    // request's own zero padding; the checkpoint reads it as silence and
    // hallucinates ("Okay." on the frozen silence controls), and that text
    // must not join the real speech. Only exact zeros are skipped, so quiet
    // speech still decodes; a single-window request always decodes, as the
    // frozen corpus pins its silence output.
    if (windows.size() > 1 && WindowIsAllZero(samples, {start, end})) {
      continue;
    } else {
    }
    const TranscriptionResult window =
        TranscribeWindow({samples.begin() + start, samples.begin() + end},
                         remaining_tokens, cancel);
    if (window.text.empty()) {
    } else {
      parts.push_back(window.text);
    }
    combined.generated_token_count += window.generated_token_count;
    combined.verify_round_count += window.verify_round_count;
    combined.proposed_draft_token_count += window.proposed_draft_token_count;
    combined.accepted_draft_token_count += window.accepted_draft_token_count;
    remaining_tokens -= window.generated_token_count;
    if (window.finish_reason == FinishReason::kLength) {
      combined.finish_reason = FinishReason::kLength;
    } else {
    }
    mx::clear_cache();
  }
  combined.text = omni_common::JoinTranscriptParts(parts);
  combined.language = options.language;
  if (windows.size() > 1) {
    combined.window_texts = parts;
  } else {
  }
  return combined;
}

} // namespace arkasr
