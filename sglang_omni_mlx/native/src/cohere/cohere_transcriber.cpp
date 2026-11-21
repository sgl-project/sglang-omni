// SPDX-License-Identifier: Apache-2.0
#include "cohere/cohere_transcriber.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>

#include "common/audio.h"
#include "common/audio_chunks.h"
#include "common/generation.h"
#include "common/layers.h"
#include "common/swift_port.h"
#include "common/text.h"
#include "common/transcription.h"
#include "nlohmann/json.hpp"

namespace cohere_transcribe {

namespace mx = mlx::core;

namespace {

// The energy-cut search around a chunk end and its smallest window.
constexpr float kChunkSearchSeconds = 5.0f;
// Note (XXXXRT666): chunks decoded together; each row keeps its cross- and
// self-attention keys, about 20 MB a row at 24 s and a few hundred tokens.
constexpr size_t kBatchChunks = 8;
constexpr float kChunkEnergyWindowMilliseconds = 100.0f;

// The code of a language code or English name; English when unknown.
std::string LanguageCode(const std::string &language) {
  static const std::map<std::string, std::string> table = {
      {"english", "en"},    {"en", "en"}, {"french", "fr"},     {"fr", "fr"},
      {"german", "de"},     {"de", "de"}, {"spanish", "es"},    {"es", "es"},
      {"italian", "it"},    {"it", "it"}, {"portuguese", "pt"}, {"pt", "pt"},
      {"dutch", "nl"},      {"nl", "nl"}, {"polish", "pl"},     {"pl", "pl"},
      {"greek", "el"},      {"el", "el"}, {"arabic", "ar"},     {"ar", "ar"},
      {"japanese", "ja"},   {"ja", "ja"}, {"chinese", "zh"},    {"zh", "zh"},
      {"vietnamese", "vi"}, {"vi", "vi"}, {"korean", "ko"},     {"ko", "ko"},
  };
  std::string key = language;
  key = omni_common::AsciiLowercase(key);
  const auto found = table.find(key);
  return found == table.end() ? "en" : found->second;
}

// [start, end) sample ranges: audio up to chunk_seconds is one chunk; longer
// audio is cut at the quietest 100 ms within 5 s of each chunk end.
std::vector<std::pair<size_t, size_t>>
EnergyCutChunks(const std::vector<float> &samples, float chunk_seconds) {
  const size_t total = samples.size();
  const float sample_rate = static_cast<float>(omni_common::kSampleRate);
  if (static_cast<float>(total) / sample_rate <= chunk_seconds) {
    return {{0, total}};
  } else {
  }
  const size_t max_chunk = static_cast<size_t>(chunk_seconds * sample_rate);
  const size_t search = static_cast<size_t>(kChunkSearchSeconds * sample_rate);
  const size_t energy_window = static_cast<size_t>(
      kChunkEnergyWindowMilliseconds * sample_rate / 1000.0f);
  std::vector<std::pair<size_t, size_t>> chunks;
  size_t start = 0;
  while (start < total) {
    const size_t end = std::min(start + max_chunk, total);
    if (end >= total) {
      chunks.emplace_back(start, total);
      break;
    } else {
    }
    const size_t search_start =
        std::max(start, end > search ? end - search : 0);
    const size_t search_end = std::min(total, end + search);
    size_t cut = end;
    if (search_end - search_start > energy_window) {
      cut = omni_common::QuietestAudioPosition(samples, search_start,
                                               search_end, energy_window);
    } else {
    }
    cut = std::max(cut, start + static_cast<size_t>(omni_common::kSampleRate));
    chunks.emplace_back(start, std::min(cut, total));
    start = cut;
  }
  return chunks;
}

} // namespace

CohereTranscriber::CohereTranscriber(
    const std::filesystem::path &model_directory)
    : model_(model_directory),
      vocabulary_(model_directory / "tokenizer.model") {
  const nlohmann::json tokenizer_config = nlohmann::json::parse(
      omni_common::ReadFile(model_directory / "tokenizer_config.json"));
  for (const auto &[id_text, token] :
       tokenizer_config.at("added_tokens_decoder").items()) {
    const int id = std::stoi(id_text);
    special_token_ids_[token.at("content").get<std::string>()] = id;
    special_ids_.insert(id);
  }
  end_of_text_id_ = special_token_ids_.at("<|endoftext|>");
}

std::string
CohereTranscriber::DecodeText(const std::vector<int> &token_ids) const {
  return vocabulary_.Decode(token_ids, special_ids_);
}

std::vector<CohereTranscriber::ChunkResult> CohereTranscriber::TranscribeChunks(
    const std::vector<std::vector<float>> &chunks,
    const std::vector<int> &prompt_ids, int max_new_tokens, float temperature,
    const std::atomic<bool> &cancel, bool preview) const {
  const int rows = static_cast<int>(chunks.size());
  std::vector<ChunkResult> results(rows);
  const int prompt_length = static_cast<int>(prompt_ids.size());
  const int token_budget =
      std::min(max_new_tokens, std::max(0, model_.config().max_sequence_length -
                                               prompt_length));
  if (token_budget <= 0) {
    return results;
  } else {
  }
  // Note (XXXXRT666): one chunk's encoder is queued while the one before it
  // finishes, so a cancel waits for at most one more chunk.
  std::vector<mx::array> states;
  std::vector<int> frame_counts;
  for (const std::vector<float> &samples : chunks) {
    states.push_back(model_.Encode(samples, preview ? &cancel : nullptr));
    mx::async_eval(states.back());
    frame_counts.push_back(states.back().shape(1));
    if (states.size() > 1) {
      mx::eval(states[states.size() - 2]);
    } else {
    }
    if (cancel.load()) {
      throw omni_common::GenerationCancelled();
    } else {
    }
  }
  // Note (XXXXRT666): the chunks decode as rows of one batch: encoder states
  // padded to the longest, the padding hidden from cross-attention.
  const int frames =
      *std::max_element(frame_counts.begin(), frame_counts.end());
  std::optional<mx::array> encoder_mask;
  if (*std::min_element(frame_counts.begin(), frame_counts.end()) < frames) {
    for (int row = 0; row < rows; ++row) {
      states[row] = mx::pad(states[row],
                            {{0, 0}, {0, frames - frame_counts[row]}, {0, 0}});
    }
    const mx::array lengths = mx::reshape(
        mx::array(frame_counts.data(), {rows}, mx::int32), {rows, 1, 1, 1});
    encoder_mask = mx::astype(
        mx::where(mx::less(mx::reshape(mx::arange(frames, mx::int32),
                                       {1, 1, 1, frames}),
                           lengths),
                  mx::array(0.0f),
                  mx::array(-std::numeric_limits<float>::infinity())),
        states.front().dtype());
  } else {
  }
  mx::array encoder_states =
      rows == 1 ? states.front() : mx::concatenate(states, 0);
  std::vector<DecoderLayerCache> caches(model_.config().decoder_layer_count);
  const auto next_token = [&](const mx::array &logits) {
    if (temperature == 0.0f) {
      return mx::argmax(logits, -1);
    } else {
      return mx::random::categorical(
          mx::divide(logits, mx::array(temperature, logits.dtype())));
    }
  };
  std::vector<int32_t> prompt_rows;
  for (int row = 0; row < rows; ++row) {
    prompt_rows.insert(prompt_rows.end(), prompt_ids.begin(), prompt_ids.end());
  }
  const mx::array prompt(prompt_rows.data(), {rows, prompt_length}, mx::int32);
  mx::array token = next_token(
      model_.Decode(prompt, 0, encoder_states, encoder_mask, caches));
  mx::async_eval({token});
  std::vector<std::vector<int>> generated_ids(rows);
  // Note (XXXXRT666): the batch rows still decoding, by chunk; a finished row
  // leaves the batch, so the others stop paying for it.
  std::vector<int> active(rows);
  for (int row = 0; row < rows; ++row) {
    active[row] = row;
  }
  for (int step = 0; !active.empty(); ++step) {
    if (cancel.load()) {
      throw omni_common::GenerationCancelled();
    } else if (step == token_budget) {
      for (const int row : active) {
        results[row].reached_token_limit = true;
      }
      break;
    } else {
    }
    const int batch = static_cast<int>(active.size());
    const mx::array current = token;
    // Note (khazic): queue the next step before reading this token, so the GPU
    // decodes while the end of text is checked.
    if (step + 1 < token_budget) {
      token = next_token(model_.Decode(mx::reshape(current, {batch, 1}),
                                       prompt_length + step, encoder_states,
                                       encoder_mask, caches));
      mx::async_eval({token});
    } else {
    }
    mx::eval(current);
    const uint32_t *token_ids = current.data<uint32_t>();
    std::vector<int32_t> kept;
    for (int position = 0; position < batch; ++position) {
      if (static_cast<int>(token_ids[position]) == end_of_text_id_) {
        continue;
      } else {
      }
      generated_ids[active[position]].push_back(
          static_cast<int>(token_ids[position]));
      kept.push_back(position);
    }
    if (static_cast<int>(kept.size()) < batch && !kept.empty()) {
      const mx::array keep(kept.data(), {static_cast<int>(kept.size())},
                           mx::int32);
      token = mx::take(token, keep, 0);
      encoder_states = mx::take(encoder_states, keep, 0);
      if (encoder_mask.has_value()) {
        encoder_mask = mx::take(*encoder_mask, keep, 0);
      } else {
      }
      for (DecoderLayerCache &cache : caches) {
        cache.self_attention.KeepRows(keep);
        cache.cross_keys = mx::take(*cache.cross_keys, keep, 0);
        cache.cross_values = mx::take(*cache.cross_values, keep, 0);
      }
      // Note (XXXXRT666): the smaller batch's buffers fit none of the larger
      // one's, which would otherwise stay cached.
      mx::clear_cache();
    } else {
    }
    std::vector<int> still_active;
    for (const int position : kept) {
      still_active.push_back(active[position]);
    }
    active = std::move(still_active);
  }
  for (int row = 0; row < rows; ++row) {
    results[row].token_ids = std::move(generated_ids[row]);
  }
  return results;
}

omni_common::TranscriptionResult
CohereTranscriber::Transcribe(const std::vector<float> &samples,
                              const CohereOptions &options,
                              const std::atomic<bool> &cancel) const {
  // Note (khazic): context, transcript start, emotion, the language twice,
  // punctuation, no inverse text normalization, no timestamps and no
  // diarization.
  const std::string language = LanguageCode(options.language);
  const std::string language_token = "<|" + language + "|>";
  std::vector<int> prompt_ids;
  for (const std::string &token :
       {std::string("<|startofcontext|>"), std::string("<|startoftranscript|>"),
        std::string("<|emo:undefined|>"), language_token, language_token,
        std::string(options.use_punctuation ? "<|pnc|>" : "<|nopnc|>"),
        std::string("<|noitn|>"), std::string("<|notimestamp|>"),
        std::string("<|nodiarize|>")}) {
    const auto found = special_token_ids_.find(token);
    if (found != special_token_ids_.end()) {
      prompt_ids.push_back(found->second);
    } else {
    }
  }
  int remaining_tokens = options.max_new_tokens != 0
                             ? options.max_new_tokens
                             : model_.config().max_sequence_length;
  const size_t min_chunk_samples =
      static_cast<size_t>(options.min_chunk_duration_seconds *
                          static_cast<float>(omni_common::kSampleRate));
  omni_common::TranscriptionResult result;
  // Note (khazic): the language actually decoded: an unknown one decodes as
  // English.
  result.language = language;
  result.finish_reason = omni_common::FinishReason::kStop;
  const bool cuts_at_speech = options.voice_activity_detector != nullptr;
  const std::vector<std::pair<size_t, size_t>> chunks =
      cuts_at_speech
          ? silero_vad::SegmentSpeech(*options.voice_activity_detector, samples,
                                      options.speech_segments)
          : EnergyCutChunks(samples, options.chunk_duration_seconds);
  // Note (XXXXRT666): chunks decode a batch at a time, each row with the
  // budget left when its batch starts; greedy decoding stops a row's tokens
  // where a smaller budget would, so the budget is then spent in chunk order.
  for (size_t first = 0; first < chunks.size() && remaining_tokens > 0;
       first += kBatchChunks) {
    std::vector<std::vector<float>> batch;
    for (size_t index = first;
         index < std::min(chunks.size(), first + kBatchChunks); ++index) {
      // Note (khazic): of several energy-cut chunks, one shorter than the
      // minimum is padded with silence; speech segments and a single chunk
      // are decoded as cut.
      std::vector<float> chunk(samples.begin() + chunks[index].first,
                               samples.begin() + chunks[index].second);
      if (!cuts_at_speech && chunks.size() > 1 &&
          chunk.size() < min_chunk_samples) {
        chunk.resize(min_chunk_samples, 0.0f);
      } else {
      }
      batch.push_back(std::move(chunk));
    }
    for (ChunkResult &chunk_result :
         TranscribeChunks(batch, prompt_ids, remaining_tokens,
                          options.temperature, cancel, options.preview)) {
      if (remaining_tokens <= 0) {
        break;
      } else {
      }
      std::vector<int> &token_ids = chunk_result.token_ids;
      if (static_cast<int>(token_ids.size()) >= remaining_tokens) {
        token_ids.resize(remaining_tokens);
        chunk_result.reached_token_limit = true;
      } else {
      }
      const int generated_token_count = static_cast<int>(token_ids.size());
      remaining_tokens = std::max(0, remaining_tokens - generated_token_count);
      result.generated_token_count += generated_token_count;
      if (chunk_result.reached_token_limit) {
        result.finish_reason = omni_common::FinishReason::kLength;
      } else {
      }
      const std::string text =
          swift_port::TrimWhitespace(DecodeText(token_ids), true);
      if (!text.empty()) {
        result.text += (result.text.empty() ? "" : "\n") + text;
      } else {
      }
    }
  }
  return result;
}

} // namespace cohere_transcribe
