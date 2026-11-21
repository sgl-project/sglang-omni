// SPDX-License-Identifier: Apache-2.0
#include "qwen3_asr/qwen3_asr_transcriber.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <map>
#include <utility>

#include "common/audio.h"
#include "common/audio_chunks.h"
#include "common/generation.h"
#include "common/kv_cache.h"
#include "common/text.h"
#include "common/transcription.h"

namespace qwen3_asr {

using omni_common::DraftPolicy;
using omni_common::DraftStatistics;

using omni_common::KVCache;

using omni_common::CollapseTokenLoop;
using omni_common::FinishReason;
using omni_common::GenerationCancelled;
using omni_common::IsTokenLoop;
using omni_common::IsUnicodeSpace;
using omni_common::TranscriptionResult;

namespace mx = mlx::core;

namespace {

constexpr const char *kAsrTextMarker = "<asr_text>";
constexpr const char *kAudioPad = "<|audio_pad|>";
constexpr int kOutputTokensPerAudioSecond = 10;
constexpr int kMinDefaultOutputTokens = 128;
// The last mel frames, whose STFT windows take the reflected tail, change as
// realtime audio arrives.
constexpr int kReflectedTailFrames = 2;
constexpr double kChunkSearchSeconds = 5.0;
constexpr double kChunkEnergyWindowSeconds = 0.1;
constexpr double kMinimumTailSeconds = 5.0;
// Past about 10 minutes a single prompt can loop on its
// first sentence; 120 s keeps every shorter request in one piece.
constexpr double kMaxChunkSeconds = 120.0;

size_t MaxChunkSamples() {
  return static_cast<size_t>(kMaxChunkSeconds * omni_common::kSampleRate);
}

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

// The log-mel features the prompt's audio is encoded from.
mx::array PromptMel(const std::vector<float> &samples, AudioLayout layout) {
  return omni_common::LogMel(samples,
                             layout == AudioLayout::kReference
                                 ? omni_common::FinalStftFrame::kDropped
                                 : omni_common::FinalStftFrame::kKept);
}

} // namespace

std::optional<std::string> NormalizeLanguage(const std::string &language) {
  const std::string stripped = Strip(language);
  const std::string normalized = omni_common::AsciiLowercase(stripped);
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
    if (omni_common::AsciiLowercase(name) == normalized) {
      return name;
    } else {
    }
  }
  return stripped;
}

Qwen3ASRTranscriber::Qwen3ASRTranscriber(
    const std::filesystem::path &model_directory,
    const std::filesystem::path &draft_directory, int draft_quantization_bits,
    DraftPolicy draft_policy)
    : model_(model_directory), tokenizer_(model_directory),
      audio_pad_id_(tokenizer_.AddedTokenId(kAudioPad)),
      end_of_text_id_(tokenizer_.AddedTokenId("<|endoftext|>")),
      im_end_id_(tokenizer_.AddedTokenId("<|im_end|>")),
      asr_text_ids_(tokenizer_.Encode(kAsrTextMarker)),
      draft_policy_(draft_policy) {
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

RetainedText
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
      return {retained, decoded,
              std::vector<int>(token_ids.begin() + retained.size(),
                               token_ids.end())};
    }
  }
  return {{}, "", token_ids};
}

int Qwen3ASRTranscriber::MaxNewTokens(size_t sample_count,
                                      const TranscriptionOptions &options) {
  // Note (Jiaxin Deng): an unset or zero budget means the default.
  return options.max_new_tokens.value_or(0) != 0
             ? *options.max_new_tokens
             : std::max(kMinDefaultOutputTokens,
                        static_cast<int>(
                            std::ceil(static_cast<double>(sample_count) /
                                      omni_common::kSampleRate *
                                      kOutputTokensPerAudioSecond)));
}

Qwen3ASRTranscriber::Prompt
Qwen3ASRTranscriber::BuildPrompt(const std::vector<float> &samples,
                                 const TranscriptionOptions &options) const {
  const mx::array mel = PromptMel(samples, options.layout);
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
  return {embeddings, MaxNewTokens(samples.size(), options)};
}

Qwen3ASRTranscriber::RefreshPrompt Qwen3ASRTranscriber::BuildRefreshPrompt(
    const std::vector<float> &samples, const TranscriptionOptions &options,
    bool uses_draft, RefreshReuse &held) const {
  const mx::array mel = PromptMel(samples, options.layout);
  const int mel_bins = mel.shape(0);
  const int frame_count = mel.shape(-1);
  const int audio_token_count = TokenCount(frame_count, options.layout);
  const std::vector<int> prompt_ids = PromptIds(audio_token_count, options);
  const int audio_start = static_cast<int>(
      std::find(prompt_ids.begin(), prompt_ids.end(), audio_pad_id_) -
      prompt_ids.begin());
  const std::vector<int> head_ids(prompt_ids.begin(),
                                  prompt_ids.begin() + audio_start);
  int held_frames = held.mel.has_value() ? held.mel->shape(1) : 0;
  int held_rows = held.audio_rows.has_value() ? held.audio_rows->shape(0) : 0;
  bool is_reusable = held_frames > 0 && held_frames <= frame_count &&
                     held.head_ids == head_ids && !held.caches.empty() &&
                     held.caches[0].offset() == audio_start + held_rows &&
                     (!uses_draft || (!held.draft_caches.empty() &&
                                      held.draft_caches[0].offset() ==
                                          audio_start + held_rows));
  if (is_reusable) {
    is_reusable =
        mx::array_equal(mx::slice(mel, {0, 0}, {mel_bins, held_frames}),
                        *held.mel)
            .item<bool>();
  } else {
  }
  if (!is_reusable) {
    held = RefreshReuse();
    held.caches = model_.NewCaches();
    if (uses_draft) {
      held.draft_caches = draft_->NewCaches();
    } else {
    }
    held_frames = 0;
    held_rows = 0;
  } else {
  }
  const mx::array new_rows = model_.EncodeAudio(
      mx::slice(mel, {0, held_frames}, {mel_bins, frame_count}),
      options.layout);
  const int reused_length = held_rows > 0 ? audio_start + held_rows : 0;
  const std::vector<int> rest_ids(prompt_ids.begin() + reused_length,
                                  prompt_ids.end());
  mx::array embeddings = model_.EmbedTokens(mx::array(
      rest_ids.data(), {1, static_cast<int>(rest_ids.size())}, mx::int32));
  // Note (Jiaxin Deng): the Swift layout can reserve more placeholders than
  // encoder rows; those keep the audio_pad embedding, surplus rows are dropped.
  const int filled =
      std::min(audio_token_count, held_rows + new_rows.shape(0)) - held_rows;
  const int rows_start = audio_start + held_rows - reused_length;
  const int hidden = embeddings.shape(2);
  embeddings = mx::slice_update(
      embeddings,
      mx::expand_dims(
          mx::astype(mx::slice(new_rows, {0, 0}, {filled, new_rows.shape(1)}),
                     embeddings.dtype()),
          0),
      {0, rows_start, 0}, {1, rows_start + filled, hidden});

  const int window_frame_count = model_.WindowFrameCount();
  const int complete_frames = std::max(0, frame_count - kReflectedTailFrames) /
                              window_frame_count * window_frame_count;
  const int complete_rows =
      complete_frames > 0 ? TokenCount(complete_frames, options.layout) : 0;
  if (complete_frames > held_frames && complete_rows <= held_rows + filled) {
    const mx::array completed_rows = mx::slice(
        new_rows, {0, 0}, {complete_rows - held_rows, new_rows.shape(1)});
    held.mel = mx::slice(mel, {0, 0}, {mel_bins, complete_frames});
    held.audio_rows =
        held_rows > 0 ? mx::concatenate({*held.audio_rows, completed_rows}, 0)
                      : completed_rows;
  } else {
  }
  held.head_ids = head_ids;
  return {embeddings, held.audio_rows.has_value()
                          ? audio_start + held.audio_rows->shape(0)
                          : 0};
}

bool Qwen3ASRTranscriber::ChunksAudio(size_t sample_count,
                                      const TranscriptionOptions &options) {
  // A piece would repeat the text already shown and the guesses.
  return sample_count > MaxChunkSamples() && options.prefix_token_ids.empty() &&
         options.prefix_text.empty() && options.guessed_token_ids.empty();
}

TranscriptionResult Qwen3ASRTranscriber::Transcribe(
    const std::vector<float> &samples, const TranscriptionOptions &options,
    const std::atomic<bool> &cancel, RefreshReuse *reuse) const {
  if (reuse != nullptr || !ChunksAudio(samples.size(), options)) {
    return TranscribeWindow(samples, options, cancel, reuse);
  } else {
  }
  const size_t max_chunk_samples = MaxChunkSamples();
  const auto chunks = omni_common::BoundedEnergyCutChunks(
      samples, max_chunk_samples,
      static_cast<size_t>(kChunkSearchSeconds * omni_common::kSampleRate),
      static_cast<size_t>(kChunkEnergyWindowSeconds * omni_common::kSampleRate),
      static_cast<size_t>(kMinimumTailSeconds * omni_common::kSampleRate));
  TranscriptionResult merged;
  merged.finish_reason = FinishReason::kStop;
  int remaining_tokens = options.max_new_tokens.value_or(0);
  const bool has_token_budget = remaining_tokens != 0;
  for (const auto &[start, end] : chunks) {
    TranscriptionOptions chunk_options = options;
    if (has_token_budget) {
      chunk_options.max_new_tokens = remaining_tokens;
    } else {
    }
    const TranscriptionResult chunk = TranscribeWindow(
        std::vector<float>(samples.begin() + start, samples.begin() + end),
        chunk_options, cancel, nullptr);
    // Scripts written with spaces need one between pieces; CJK text does not.
    if (!merged.text.empty() && !chunk.text.empty() &&
        static_cast<unsigned char>(merged.text.back()) < 0x80 &&
        static_cast<unsigned char>(chunk.text.front()) < 0x80 &&
        !std::isspace(static_cast<unsigned char>(merged.text.back()))) {
      merged.text += ' ';
    } else {
    }
    merged.text += chunk.text;
    if (!merged.language.has_value()) {
      merged.language = chunk.language;
    } else {
    }
    merged.generated_token_count += chunk.generated_token_count;
    merged.verify_round_count += chunk.verify_round_count;
    merged.proposed_draft_token_count += chunk.proposed_draft_token_count;
    merged.accepted_draft_token_count += chunk.accepted_draft_token_count;
    if (chunk.finish_reason == FinishReason::kLength) {
      merged.finish_reason = FinishReason::kLength;
    } else {
    }
    if (has_token_budget) {
      remaining_tokens -= chunk.generated_token_count;
      if (remaining_tokens <= 0) {
        break;
      } else {
      }
    } else {
    }
  }
  return merged;
}

TranscriptionResult Qwen3ASRTranscriber::TranscribeWindow(
    const std::vector<float> &samples, const TranscriptionOptions &options,
    const std::atomic<bool> &cancel, RefreshReuse *reuse) const {
  if (cancel.load()) {
    throw GenerationCancelled();
  } else {
  }
  const int max_new_tokens = MaxNewTokens(samples.size(), options);
  const auto is_stop = [&](int token_id) {
    return IsStopToken(token_id, options);
  };
  std::vector<int> output_ids;
  FinishReason finish_reason = FinishReason::kLength;
  bool stopped_on_token_loop = false;
  // Adds a token to the output; true once the output is complete.
  const auto commit = [&](int token_id) {
    output_ids.push_back(token_id);
    if (is_stop(token_id)) {
      finish_reason = FinishReason::kStop;
      return true;
    } else if (options.stop_on_token_loop && IsTokenLoop(output_ids)) {
      finish_reason = FinishReason::kStop;
      stopped_on_token_loop = true;
      return true;
    } else {
      return static_cast<int>(output_ids.size()) >= max_new_tokens;
    }
  };
  DraftStatistics draft_statistics;
  // note (wilsonzheng0327): a penalty would have to be applied per block
  // position with the block's own earlier tokens; a request that asks for one
  // decodes without the draft.
  const bool uses_draft =
      draft_.has_value() && options.repetition_penalty == 1.0f;
  if (reuse == nullptr && uses_draft) {
    draft_statistics = draft_->Decode(
        model_.decoder(), BuildPrompt(samples, options).embeddings,
        max_new_tokens, 0, commit, cancel, draft_policy_);
  } else {
    // A failed refresh leaves reuse empty, so the next one starts over.
    RefreshReuse held;
    std::optional<RefreshPrompt> refresh;
    if (reuse != nullptr) {
      held = std::exchange(*reuse, RefreshReuse());
      refresh = BuildRefreshPrompt(samples, options, uses_draft, held);
    } else {
      held.caches = model_.NewCaches();
    }
    std::vector<KVCache> &caches = held.caches;
    const mx::array embeddings = refresh.has_value()
                                     ? refresh->embeddings
                                     : BuildPrompt(samples, options).embeddings;
    if (uses_draft) {
      draft_statistics =
          draft_->Decode(model_.decoder(), embeddings,
                         options.guessed_token_ids, max_new_tokens, 0, commit,
                         cancel, draft_policy_, caches, held.draft_caches);
    } else {
      const std::vector<int> &guessed_ids = options.guessed_token_ids;
      // A penalty would have to apply to the guesses' logits too.
      const int guessed_count = options.repetition_penalty == 1.0f
                                    ? static_cast<int>(guessed_ids.size())
                                    : 0;
      std::optional<mx::array> next_token;
      bool finished = false;
      if (guessed_count == 0) {
        next_token = mx::argmax(model_.Decode(embeddings, caches));
      } else {
        // Note (XXXXRT666): the guesses ride in the prompt's pass; the logits
        // of the prompt's last position and of each guess check them.
        const mx::array hidden_states = model_.decoder().Forward(
            mx::concatenate({embeddings, model_.EmbedTokens(mx::array(
                                             guessed_ids.data(),
                                             {1, guessed_count}, mx::int32))},
                            1),
            caches);
        const int length = hidden_states.shape(1);
        const mx::array predicted =
            mx::argmax(model_.decoder().Logits(mx::slice(
                           hidden_states, {0, length - guessed_count - 1, 0},
                           {1, length, hidden_states.shape(2)})),
                       -1);
        mx::eval(predicted);
        const uint32_t *predicted_ids = predicted.data<uint32_t>();
        int accepted_count = 0;
        while (!finished && accepted_count < guessed_count &&
               static_cast<int>(predicted_ids[accepted_count]) ==
                   guessed_ids[accepted_count]) {
          finished = commit(guessed_ids[accepted_count]);
          ++accepted_count;
        }
        for (KVCache &cache : caches) {
          cache.Trim(guessed_count - accepted_count);
        }
        next_token = mx::reshape(
            mx::slice(predicted, {accepted_count}, {accepted_count + 1}), {});
      }
      mx::async_eval({*next_token});
      if (!finished) {
        model_.decoder().DecodePlain(
            *next_token, caches, options.repetition_penalty, commit, cancel);
      } else {
      }
    }
    if (refresh.has_value()) {
      for (KVCache &cache : caches) {
        cache.Trim(cache.offset() - refresh->kept_length);
      }
      for (KVCache &cache : held.draft_caches) {
        cache.Trim(cache.offset() - refresh->kept_length);
      }
      *reuse = std::move(held);
    } else {
    }
  }
  return FinishedResult(output_ids, options, finish_reason,
                        stopped_on_token_loop, draft_statistics);
}

bool Qwen3ASRTranscriber::IsStopToken(
    int token_id, const TranscriptionOptions &options) const {
  return token_id == im_end_id_ ||
         (options.stop_at_end_of_text && token_id == end_of_text_id_);
}

TranscriptionResult Qwen3ASRTranscriber::FinishedResult(
    const std::vector<int> &output_ids, const TranscriptionOptions &options,
    FinishReason finish_reason, bool stopped_on_token_loop,
    const DraftStatistics &draft_statistics) const {
  const bool ended_on_stop =
      !output_ids.empty() && IsStopToken(output_ids.back(), options);
  // Note (Minyi Tang): the repeats are the model feeding on its own output,
  // not speech, so the transcript keeps one copy of the looping unit.
  auto [text, language] = SplitOutput(
      stopped_on_token_loop ? CollapseTokenLoop(output_ids) : output_ids,
      options);
  return {options.prefix_text + text,
          language,
          static_cast<int>(output_ids.size()) - (ended_on_stop ? 1 : 0),
          finish_reason,
          {},
          draft_statistics.verify_round_count,
          draft_statistics.proposed_token_count,
          draft_statistics.accepted_token_count};
}

struct Qwen3ASRTranscriber::BatchRow {
  size_t request_index;
  int max_new_tokens;
  omni_common::PrefilledSequence sequence;
  std::vector<int> output_ids;
};

void Qwen3ASRTranscriber::TranscribeBatch(
    const std::vector<BatchRequest> &requests,
    const BatchFinished &on_finished) const {
  std::vector<BatchRow> rows = PrefillRows(requests, on_finished);
  if (!rows.empty()) {
    DecodeRows(requests, rows, on_finished);
  } else {
  }
}

// Each request prefills alone: prompts differ in length, and the encoder and
// prefill run once per request either way. A request that fails here ends with
// its error and joins no batch.
std::vector<Qwen3ASRTranscriber::BatchRow>
Qwen3ASRTranscriber::PrefillRows(const std::vector<BatchRequest> &requests,
                                 const BatchFinished &on_finished) const {
  std::vector<BatchRow> rows;
  for (size_t i = 0; i < requests.size(); ++i) {
    try {
      if (requests[i].cancel->load()) {
        throw GenerationCancelled();
      } else {
      }
      const Prompt prompt =
          BuildPrompt(*requests[i].samples, *requests[i].options);
      std::vector<KVCache> caches = model_.NewCaches();
      const mx::array first_token =
          mx::argmax(model_.Decode(prompt.embeddings, caches));
      mx::async_eval({first_token});
      rows.push_back(
          {i,
           prompt.max_new_tokens,
           {std::move(caches), prompt.embeddings.shape(1), first_token},
           {}});
    } catch (...) {
      on_finished(i, {std::nullopt, std::current_exception()});
    }
  }
  return rows;
}

bool Qwen3ASRTranscriber::AcceptToken(BatchRow &row, int token_id,
                                      const TranscriptionOptions &options,
                                      BatchOutcome &outcome) const {
  row.output_ids.push_back(token_id);
  if (IsStopToken(token_id, options)) {
    outcome.result =
        FinishedResult(row.output_ids, options, FinishReason::kStop, false);
    return true;
  } else if (options.stop_on_token_loop && IsTokenLoop(row.output_ids)) {
    outcome.result =
        FinishedResult(row.output_ids, options, FinishReason::kStop, true);
    return true;
  } else if (static_cast<int>(row.output_ids.size()) >= row.max_new_tokens) {
    outcome.result =
        FinishedResult(row.output_ids, options, FinishReason::kLength, false);
    return true;
  } else {
    return false;
  }
}

void Qwen3ASRTranscriber::DecodeRows(const std::vector<BatchRequest> &requests,
                                     std::vector<BatchRow> &rows,
                                     const BatchFinished &on_finished) const {
  std::vector<omni_common::PrefilledSequence> sequences;
  for (BatchRow &row : rows) {
    sequences.push_back(std::move(row.sequence));
  }
  model_.decoder().DecodeGreedyBatch(
      sequences, [&](size_t index, int token_id) {
        BatchRow &row = rows[index];
        const BatchRequest &request = requests[row.request_index];
        BatchOutcome outcome;
        bool is_done = true;
        if (request.cancel->load()) {
          outcome.error = std::make_exception_ptr(GenerationCancelled());
        } else {
          is_done = AcceptToken(row, token_id, *request.options, outcome);
        }
        if (is_done) {
          on_finished(row.request_index, std::move(outcome));
        } else {
        }
        return is_done;
      });
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
        omni_common::AsciiLowercase(prefix.substr(0, separator)) ==
            "language") {
      const std::string value = Strip(prefix.substr(separator + 1));
      if (!value.empty() && omni_common::AsciiLowercase(value) != "none") {
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

omni_server::BatchRunner BatchableTranscriptionRunner() {
  return {[](const omni_server::Transcription &transcription) {
            return transcription.target<BatchableTranscription>() != nullptr;
          },
          [](const std::vector<omni_server::BatchItem> &items,
             const omni_server::BatchCompletion &finish) {
            std::vector<BatchRequest> requests;
            const Qwen3ASRTranscriber *transcriber = nullptr;
            for (const omni_server::BatchItem &item : items) {
              const auto *batchable =
                  item.transcription->target<BatchableTranscription>();
              transcriber = batchable->transcriber;
              requests.push_back(
                  {&batchable->samples, &batchable->options, item.cancel});
            }
            transcriber->TranscribeBatch(
                requests, [&](size_t i, BatchOutcome outcome) {
                  finish(i, std::move(outcome.result), outcome.error);
                });
          }};
}

} // namespace qwen3_asr
