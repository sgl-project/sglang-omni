// SPDX-License-Identifier: Apache-2.0
#include "qwen3_asr/qwen3_asr_transcriber.h"
#include "common/text.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <map>

namespace qwen3_asr {

using omni_common::CollapseTokenLoop;
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

Qwen3ASRTranscriber::Prompt
Qwen3ASRTranscriber::BuildPrompt(const std::vector<float> &samples,
                                 const TranscriptionOptions &options) const {
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
  return {embeddings, max_new_tokens};
}

TranscriptionResult
Qwen3ASRTranscriber::Transcribe(const std::vector<float> &samples,
                                const TranscriptionOptions &options,
                                const std::atomic<bool> &cancel) const {
  if (cancel.load()) {
    throw TranscriptionCancelled();
  } else {
  }
  const Prompt prompt = BuildPrompt(samples, options);
  const mx::array &embeddings = prompt.embeddings;
  const int max_new_tokens = prompt.max_new_tokens;
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
  if (draft_.has_value() && options.repetition_penalty == 1.0f) {
    draft_statistics = draft_->Decode(model_.decoder(), embeddings,
                                      max_new_tokens, 0, commit, cancel);
  } else {
    std::vector<KVCache> caches = model_.NewCaches();
    const mx::array prefill_logits = model_.Decode(embeddings, caches);
    mx::array next_token = mx::argmax(prefill_logits);
    mx::async_eval({next_token});
    // Note (Minyi Tang): only generated tokens are penalized, as on CUDA; the
    // scales stay lazy so penalizing adds no host sync to the queued step.
    mx::array penalty_by_token_id =
        mx::ones(prefill_logits.shape(), mx::float32);
    bool finished = false;
    while (!finished) {
      if (cancel.load()) {
        throw TranscriptionCancelled();
      } else {
      }
      const mx::array token = next_token;
      // Note (Jiaxin Deng): queue the next step before reading this token, so
      // the GPU decodes while the stop rules are checked.
      mx::array logits =
          model_.Decode(model_.EmbedTokens(mx::reshape(token, {1, 1})), caches);
      if (options.repetition_penalty != 1.0f) {
        penalty_by_token_id =
            mx::put_along_axis(penalty_by_token_id, mx::reshape(token, {1}),
                               mx::array(options.repetition_penalty), 0);
        logits = mx::astype(logits, mx::float32);
        logits = mx::where(mx::less(logits, mx::array(0.0f)),
                           mx::multiply(logits, penalty_by_token_id),
                           mx::divide(logits, penalty_by_token_id));
      } else {
      }
      next_token = mx::argmax(logits);
      mx::async_eval({next_token});
      finished = commit(static_cast<int>(token.item<uint32_t>()));
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
  int prompt_length;
  int max_new_tokens;
  std::vector<KVCache> prefill_caches;
  mx::array first_token;
  std::vector<int> output_ids;
  bool is_done = false;
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
        throw TranscriptionCancelled();
      } else {
      }
      const Prompt prompt =
          BuildPrompt(*requests[i].samples, *requests[i].options);
      std::vector<KVCache> caches = model_.NewCaches();
      const mx::array first_token =
          mx::argmax(model_.Decode(prompt.embeddings, caches));
      mx::async_eval({first_token});
      rows.push_back({i,
                      prompt.embeddings.shape(1),
                      prompt.max_new_tokens,
                      std::move(caches),
                      first_token,
                      {}});
    } catch (...) {
      on_finished(i, {std::nullopt, std::current_exception()});
    }
  }
  return rows;
}

std::vector<BatchedKVCache>
Qwen3ASRTranscriber::StackCaches(const std::vector<BatchRow> &rows) {
  const int layer_count = static_cast<int>(rows[0].prefill_caches.size());
  std::vector<BatchedKVCache> caches;
  caches.reserve(layer_count);
  for (int layer = 0; layer < layer_count; ++layer) {
    std::vector<std::pair<mx::array, mx::array>> filled;
    for (const BatchRow &row : rows) {
      filled.push_back(row.prefill_caches[layer].Filled());
    }
    caches.emplace_back(filled);
  }
  return caches;
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
  const int batch = static_cast<int>(rows.size());
  int longest_prompt = 0;
  std::vector<mx::array> first_tokens;
  // Rows are left-padded to the longest prompt; the pad slots are masked out.
  std::vector<int32_t> pads;
  // Each row's true position, which differs from the shared cache index.
  std::vector<int32_t> positions;
  for (const BatchRow &row : rows) {
    longest_prompt = std::max(longest_prompt, row.prompt_length);
    first_tokens.push_back(mx::reshape(row.first_token, {1}));
    positions.push_back(row.prompt_length);
  }
  for (const BatchRow &row : rows) {
    pads.push_back(longest_prompt - row.prompt_length);
  }
  std::vector<BatchedKVCache> caches = StackCaches(rows);
  // The stacked caches hold the prefill keys and values; drop the per-row
  // copies so the decode does not keep both alive.
  for (BatchRow &row : rows) {
    row.prefill_caches.clear();
  }
  const mx::array pad_counts(pads.data(), {batch, 1}, mx::int32);
  mx::array next_tokens = mx::concatenate(first_tokens, 0);
  mx::async_eval({next_tokens});
  int active_count = batch;
  while (active_count > 0) {
    const mx::array tokens = next_tokens;
    // Queue the next step for the whole batch before reading this one, as the
    // single-request loop does.
    next_tokens = mx::argmax(
        model_.DecodeBatch(
            model_.EmbedTokens(mx::reshape(tokens, {batch, 1})), caches,
            mx::array(positions.data(), {batch}, mx::int32), pad_counts),
        -1);
    mx::async_eval({next_tokens});
    mx::eval(tokens);
    const uint32_t *token_ids = tokens.data<uint32_t>();
    for (int i = 0; i < batch; ++i) {
      BatchRow &row = rows[i];
      ++positions[i];
      if (row.is_done) {
        continue;
      } else {
      }
      const BatchRequest &request = requests[row.request_index];
      BatchOutcome outcome;
      if (request.cancel->load()) {
        outcome.error = std::make_exception_ptr(TranscriptionCancelled());
        row.is_done = true;
      } else if (AcceptToken(row, static_cast<int>(token_ids[i]),
                             *request.options, outcome)) {
        row.is_done = true;
      } else {
      }
      if (row.is_done) {
        --active_count;
        on_finished(row.request_index, std::move(outcome));
      } else {
      }
    }
  }
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
