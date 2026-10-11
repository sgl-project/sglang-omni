// SPDX-License-Identifier: Apache-2.0
#include "cohere_transcriber.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "audio.h"
#include "nlohmann/json.hpp"
#include "swift_port.h"

namespace cohere_transcribe {

namespace mx = mlx::core;

namespace {

// The energy-cut search around a chunk end and its smallest window.
constexpr float kChunkSearchSeconds = 5.0f;
constexpr float kChunkEnergyWindowMilliseconds = 100.0f;

std::string ReadFile(const std::filesystem::path &path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) {
    throw std::runtime_error("cannot read " + path.string());
  } else {
  }
  std::ostringstream contents;
  contents << stream.rdbuf();
  return contents.str();
}

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
  std::transform(key.begin(), key.end(), key.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  const auto found = table.find(key);
  return found == table.end() ? "en" : found->second;
}

// [start, end) sample ranges: audio up to chunk_seconds is one chunk; longer
// audio is cut at the quietest 100 ms within 5 s of each chunk end.
std::vector<std::pair<size_t, size_t>>
EnergyCutChunks(const std::vector<float> &samples, float chunk_seconds) {
  const size_t total = samples.size();
  const float sample_rate = static_cast<float>(qwen3_asr::kSampleRate);
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
      const size_t energy_count = search_end - search_start - energy_window + 1;
      const float inverse_window = 1.0f / static_cast<float>(energy_window);
      float window_sum = 0.0f;
      for (size_t i = 0; i < energy_window; ++i) {
        const float square =
            samples[search_start + i] * samples[search_start + i];
        window_sum += square;
      }
      float minimum_energy = window_sum * inverse_window;
      size_t minimum_index = 0;
      for (size_t i = 1; i < energy_count; ++i) {
        const float leaving = samples[search_start + i - 1];
        const float entering = samples[search_start + i + energy_window - 1];
        const float entering_square = entering * entering;
        const float leaving_square = leaving * leaving;
        window_sum += entering_square - leaving_square;
        const float energy = window_sum * inverse_window;
        if (energy < minimum_energy) {
          minimum_energy = energy;
          minimum_index = i;
        } else {
        }
      }
      cut = search_start + minimum_index + energy_window / 2;
    } else {
    }
    cut = std::max(cut, start + static_cast<size_t>(qwen3_asr::kSampleRate));
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
      ReadFile(model_directory / "tokenizer_config.json"));
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

CohereTranscriber::ChunkResult
CohereTranscriber::TranscribeChunk(const std::vector<float> &samples,
                                   const std::vector<int> &prompt_ids,
                                   int max_new_tokens, float temperature,
                                   const std::atomic<bool> &cancel) const {
  const mx::array encoder_states = model_.Encode(samples);
  std::vector<DecoderLayerCache> caches(model_.config().decoder_layer_count);
  const auto next_token = [&](const mx::array &logits) {
    if (temperature == 0.0f) {
      return mx::argmax(logits, -1);
    } else {
      return mx::reshape(
          mx::random::categorical(mx::expand_dims(
              mx::divide(logits, mx::array(temperature, logits.dtype())), 0)),
          {});
    }
  };
  const int prompt_length = static_cast<int>(prompt_ids.size());
  const int token_budget =
      std::min(max_new_tokens, std::max(0, model_.config().max_sequence_length -
                                               prompt_length));
  ChunkResult result;
  if (token_budget <= 0) {
    return result;
  } else {
  }
  const mx::array prompt(prompt_ids.data(), {1, prompt_length}, mx::int32);
  mx::array token =
      next_token(model_.Decode(prompt, 0, encoder_states, caches));
  mx::async_eval({token});
  std::vector<int> generated_ids;
  for (int step = 0;; ++step) {
    if (cancel.load()) {
      throw qwen3_asr::TranscriptionCancelled();
    } else if (step == token_budget) {
      result.reached_token_limit = true;
      break;
    } else {
    }
    const mx::array current = token;
    // Note (khazic): queue the next step before reading this token, so the GPU
    // decodes while the end of text is checked.
    if (step + 1 < token_budget) {
      token = next_token(model_.Decode(mx::reshape(current, {1, 1}),
                                       prompt_length + step, encoder_states,
                                       caches));
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
  result.text = swift_port::TrimWhitespace(DecodeText(generated_ids), true);
  result.generated_token_count = static_cast<int>(generated_ids.size());
  return result;
}

qwen3_asr::TranscriptionResult
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
                          static_cast<float>(qwen3_asr::kSampleRate));
  qwen3_asr::TranscriptionResult result;
  // Note (khazic): the language actually decoded: an unknown one decodes as
  // English.
  result.language = language;
  result.finish_reason = qwen3_asr::FinishReason::kStop;
  const bool cuts_at_speech = options.voice_activity_detector != nullptr;
  const std::vector<std::pair<size_t, size_t>> chunks =
      cuts_at_speech
          ? silero_vad::SegmentSpeech(*options.voice_activity_detector, samples,
                                      options.speech_segments)
          : EnergyCutChunks(samples, options.chunk_duration_seconds);
  for (const auto &[start, end] : chunks) {
    if (remaining_tokens <= 0) {
      break;
    } else {
    }
    // Note (khazic): of several energy-cut chunks, one shorter than the minimum
    // is padded with silence; speech segments and a single chunk are decoded as
    // cut.
    std::vector<float> chunk(samples.begin() + start, samples.begin() + end);
    if (!cuts_at_speech && chunks.size() > 1 &&
        chunk.size() < min_chunk_samples) {
      chunk.resize(min_chunk_samples, 0.0f);
    } else {
    }
    const ChunkResult chunk_result = TranscribeChunk(
        chunk, prompt_ids, remaining_tokens, options.temperature, cancel);
    remaining_tokens =
        std::max(0, remaining_tokens - chunk_result.generated_token_count);
    result.generated_token_count += chunk_result.generated_token_count;
    if (chunk_result.reached_token_limit) {
      result.finish_reason = qwen3_asr::FinishReason::kLength;
    } else {
    }
    if (!chunk_result.text.empty()) {
      result.text += (result.text.empty() ? "" : "\n") + chunk_result.text;
    } else {
    }
  }
  return result;
}

} // namespace cohere_transcribe
