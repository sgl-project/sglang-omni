// SPDX-License-Identifier: Apache-2.0
#include "moss_transcriber.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <regex>

#include "audio.h"
#include "nlohmann/json.hpp"

namespace moss {

namespace mx = mlx::core;
using qwen3_asr::FinishReason;
using qwen3_asr::IsTokenLoop;
using qwen3_asr::kHopLength;
using qwen3_asr::kSampleRate;
using qwen3_asr::KVCache;
using qwen3_asr::kWhisperWindowSampleCount;
using qwen3_asr::SpeakerSegment;
using qwen3_asr::StripUnicodeWhitespace;
using qwen3_asr::TranscriptionCancelled;
using qwen3_asr::TranscriptionResult;
using qwen3_asr::WhisperWindowFeatures;

namespace {

constexpr const char *kAudioPad = "<|audio_pad|>";
constexpr const char *kAudioStart = "<|audio_start|>";
constexpr const char *kAudioEnd = "<|audio_end|>";
constexpr const char *kDefaultPrompt =
    "Transcribe the audio into text. Start each segment with the start "
    "timestamp and speaker label ([S01], [S02], [S03], ...), write the "
    "corresponding spoken content, and end each segment with the ending "
    "timestamp to clearly mark the segment range.";
// Note (Dayuxiaoshui): with the 10 ms hop and the adaptor's frame merge, this
// stride sets how many samples make one audio token.
constexpr int kWhisperEncoderStride = 2;
constexpr int kPrefillStepTokens = 2048;
// Note (Dayuxiaoshui): Voxt's final-pass budget, 28 tokens per second of
// audio plus headroom, clamped.
constexpr double kDefaultTokensPerAudioSecond = 28.0;
constexpr int kDefaultExtraTokens = 64;
constexpr int kDefaultMinTokens = 256;
constexpr int kDefaultMaxTokens = 8192;
// Note (Dayuxiaoshui): Voxt's final pass cuts at these lengths.
constexpr float kChunkDurationSeconds = 1200.0f;
constexpr float kMinChunkDurationSeconds = 1.0f;
// Note (Dayuxiaoshui): a cut lands on the quietest 100 ms within 5 s of the
// chunk boundary, as MLXAudio's splitter cuts.
constexpr float kCutSearchSeconds = 5.0f;
constexpr float kCutEnergyWindowMilliseconds = 100.0f;
const char *const kTimestampPattern = R"(\[(\d+(?:[\.,]\d+)?)\])";
const char *const kSegmentPattern =
    R"(\[(\d+(?:[\.,]\d+)?)\]\[(S\d+)\]([\s\S]*?)\[(\d+(?:[\.,]\d+)?)\])";

double TimestampValue(std::string text) {
  std::replace(text.begin(), text.end(), ',', '.');
  // Note (Dayuxiaoshui): strtod, not stod: a huge digit run reads as infinity,
  // as Voxt's Swift Double() reads it, instead of throwing.
  return std::strtod(text.c_str(), nullptr);
}

std::string FormatTimestamp(double seconds) {
  char buffer[32];
  std::snprintf(buffer, sizeof(buffer), "[%.2f]", seconds);
  return buffer;
}

struct AudioChunk {
  std::vector<float> samples;
  double offset_seconds = 0.0;
};

std::vector<float> PaddedTo(std::vector<float> samples, size_t min_samples) {
  if (samples.size() < min_samples) {
    samples.resize(min_samples, 0.0f);
  } else {
  }
  return samples;
}

// Note (Dayuxiaoshui): a chunk shorter than the minimum is padded up to it, as
// MLXAudio pads it.
std::vector<AudioChunk> SplitAudio(const std::vector<float> &samples,
                                   float chunk_duration_seconds,
                                   float min_chunk_duration_seconds) {
  const int sample_rate = kSampleRate;
  const int total_samples = static_cast<int>(samples.size());
  const float total_seconds =
      static_cast<float>(total_samples) / static_cast<float>(sample_rate);
  const size_t min_samples =
      static_cast<size_t>(min_chunk_duration_seconds * sample_rate);
  if (total_seconds <= chunk_duration_seconds) {
    return {{PaddedTo(samples, min_samples), 0.0}};
  } else {
  }
  const int max_chunk_samples =
      static_cast<int>(chunk_duration_seconds * sample_rate);
  const int search_samples = static_cast<int>(kCutSearchSeconds * sample_rate);
  const int energy_window_samples =
      static_cast<int>(kCutEnergyWindowMilliseconds * sample_rate / 1000.0f);
  std::vector<AudioChunk> chunks;
  int start_sample = 0;
  while (start_sample < total_samples) {
    const int end_sample =
        std::min(start_sample + max_chunk_samples, total_samples);
    const double offset_seconds =
        static_cast<float>(start_sample) / static_cast<float>(sample_rate);
    if (end_sample >= total_samples) {
      chunks.push_back(
          {PaddedTo({samples.begin() + start_sample, samples.end()},
                    min_samples),
           offset_seconds});
      break;
    } else {
    }
    const int search_start =
        std::max(start_sample, end_sample - search_samples);
    const int search_end = std::min(total_samples, end_sample + search_samples);
    int cut_sample = end_sample;
    if (search_end - search_start > energy_window_samples) {
      const int energy_count =
          search_end - search_start - energy_window_samples + 1;
      const float inverse_window =
          1.0f / static_cast<float>(energy_window_samples);
      float window_sum = 0.0f;
      for (int i = 0; i < energy_window_samples; ++i) {
        const float value = samples[search_start + i];
        window_sum += value * value;
      }
      float min_energy = window_sum * inverse_window;
      int min_index = 0;
      for (int i = 1; i < energy_count; ++i) {
        const float old_value = samples[search_start + i - 1];
        const float new_value =
            samples[search_start + i + energy_window_samples - 1];
        window_sum += new_value * new_value - old_value * old_value;
        const float energy = window_sum * inverse_window;
        if (energy < min_energy) {
          min_energy = energy;
          min_index = i;
        } else {
        }
      }
      cut_sample = search_start + min_index + energy_window_samples / 2;
    } else {
    }
    cut_sample = std::max(cut_sample, start_sample + sample_rate);
    const int actual_end = std::min(cut_sample, total_samples);
    chunks.push_back({PaddedTo({samples.begin() + start_sample,
                                samples.begin() + actual_end},
                               min_samples),
                      offset_seconds});
    start_sample = cut_sample;
  }
  return chunks;
}

} // namespace

std::string OffsetTimestampTags(const std::string &text,
                                double offset_seconds) {
  if (offset_seconds == 0.0) {
    return text;
  } else {
  }
  static const std::regex pattern(kTimestampPattern);
  std::string output;
  auto cursor = text.cbegin();
  for (std::sregex_iterator match(text.begin(), text.end(), pattern), end;
       match != end; ++match) {
    output.append(cursor, text.cbegin() + match->position(0));
    output +=
        FormatTimestamp(TimestampValue((*match)[1].str()) + offset_seconds);
    cursor = text.cbegin() + match->position(0) + match->length(0);
  }
  output.append(cursor, text.cend());
  return output;
}

std::vector<SpeakerSegment> ParseSegments(const std::string &text,
                                          double duration_seconds,
                                          double offset_seconds) {
  static const std::regex pattern(kSegmentPattern);
  std::vector<SpeakerSegment> segments;
  for (std::sregex_iterator match(text.begin(), text.end(), pattern), end;
       match != end; ++match) {
    const double start = TimestampValue((*match)[1].str());
    const double end_seconds = TimestampValue((*match)[4].str());
    const std::string segment_text = StripUnicodeWhitespace((*match)[3].str());
    if (end_seconds < start || segment_text.empty()) {
      continue;
    } else {
    }
    segments.push_back({start + offset_seconds, end_seconds + offset_seconds,
                        (*match)[2].str(), segment_text});
  }
  if (segments.empty()) {
    segments.push_back({offset_seconds,
                        offset_seconds + std::max(duration_seconds, 0.0), "",
                        text});
  } else {
  }
  return segments;
}

void ValidatePrompt(const std::optional<std::string> &prompt) {
  const std::string &instruction = prompt.value_or("");
  const size_t first = instruction.find(kAudioPad);
  if (first != std::string::npos &&
      instruction.find(kAudioPad, first + std::string(kAudioPad).size()) !=
          std::string::npos) {
    throw std::invalid_argument(
        "prompt must carry at most one <|audio_pad|> placeholder");
  } else {
  }
}

MossTranscriber::MossTranscriber(const std::filesystem::path &model_directory)
    : model_(model_directory), tokenizer_(model_directory) {
  for (char digit = '0'; digit <= '9'; ++digit) {
    const std::vector<int> ids = tokenizer_.Encode(std::string(1, digit));
    if (ids.size() != 1) {
      throw std::runtime_error(std::string("digit ") + digit +
                               " is not a single token");
    } else {
    }
    digit_token_ids_.push_back(ids[0]);
  }
  end_of_text_id_ = tokenizer_.AddedTokenId("<|endoftext|>");
  im_end_id_ = tokenizer_.AddedTokenId("<|im_end|>");
  std::ifstream processor_stream(model_directory / "processor_config.json");
  if (processor_stream) {
    const nlohmann::json processor = nlohmann::json::parse(processor_stream);
    audio_tokens_per_second_ =
        processor.value("audio_tokens_per_second", audio_tokens_per_second_);
    time_marker_every_seconds_ = processor.value("time_marker_every_seconds",
                                                 time_marker_every_seconds_);
    enable_time_marker_ =
        processor.value("enable_time_marker", enable_time_marker_);
  } else {
  }
}

std::vector<int> MossTranscriber::AudioSpanIds(int audio_token_count) const {
  const int audio_token_id = model_.audio_token_id();
  const int tokens_per_marker =
      static_cast<int>(audio_tokens_per_second_ *
                       static_cast<float>(time_marker_every_seconds_));
  if (!enable_time_marker_ || audio_token_count <= 0 ||
      time_marker_every_seconds_ <= 0 || tokens_per_marker <= 0) {
    return std::vector<int>(std::max(audio_token_count, 0), audio_token_id);
  } else {
  }
  // Note (Dayuxiaoshui): tokens per marker are truncated as the reference
  // truncates them, so markers drift half a token every 5 s there too.
  const float duration_seconds =
      static_cast<float>(audio_token_count) / audio_tokens_per_second_;
  std::vector<int> ids;
  int consumed = 0;
  for (int seconds = time_marker_every_seconds_;
       seconds <= static_cast<int>(duration_seconds);
       seconds += time_marker_every_seconds_) {
    const int position =
        (seconds / time_marker_every_seconds_) * tokens_per_marker;
    if (position > consumed) {
      ids.insert(ids.end(), position - consumed, audio_token_id);
      consumed = position;
    } else {
    }
    for (const char digit : std::to_string(seconds)) {
      ids.push_back(digit_token_ids_[digit - '0']);
    }
  }
  if (audio_token_count > consumed) {
    ids.insert(ids.end(), audio_token_count - consumed, audio_token_id);
  } else {
  }
  return ids;
}

std::vector<int>
MossTranscriber::PromptIds(int audio_token_count,
                           const std::optional<std::string> &prompt) const {
  ValidatePrompt(prompt);
  const std::string instruction =
      StripUnicodeWhitespace(prompt.value_or("")).empty()
          ? std::string(kDefaultPrompt)
          : *prompt;
  const std::string rendered =
      instruction.find(kAudioPad) != std::string::npos
          ? instruction
          : "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n" +
                std::string(kAudioStart) + kAudioPad + kAudioEnd + "\n" +
                instruction + "<|im_end|>\n<|im_start|>assistant\n";
  const size_t pad = rendered.find(kAudioPad);
  const std::string after =
      rendered.substr(pad + std::string(kAudioPad).size());
  std::vector<int> ids = tokenizer_.Encode(rendered.substr(0, pad));
  const std::vector<int> span = AudioSpanIds(audio_token_count);
  ids.insert(ids.end(), span.begin(), span.end());
  const std::vector<int> suffix = tokenizer_.Encode(after);
  ids.insert(ids.end(), suffix.begin(), suffix.end());
  return ids;
}

TranscriptionResult
MossTranscriber::TranscribeChunk(const std::vector<float> &samples,
                                 const MossOptions &options, int max_new_tokens,
                                 double offset_seconds,
                                 const std::atomic<bool> &cancel) const {
  const int sample_count = static_cast<int>(samples.size());
  const int samples_per_token =
      kHopLength * kWhisperEncoderStride * model_.audio_merge_size();
  std::vector<mx::array> windows;
  std::vector<int> window_token_counts;
  for (int start = 0; start < sample_count;
       start += kWhisperWindowSampleCount) {
    const int length =
        std::min(kWhisperWindowSampleCount, sample_count - start);
    window_token_counts.push_back(
        (std::max(1, length) - 1) / samples_per_token + 1);
    windows.push_back(WhisperWindowFeatures(samples.data() + start, length,
                                            model_.mel_bin_count()));
  }
  int audio_token_count = 0;
  for (const int count : window_token_counts)
    audio_token_count += count;

  const std::vector<int> prompt_ids =
      PromptIds(audio_token_count, options.prompt);
  const int prompt_length = static_cast<int>(prompt_ids.size());
  const mx::array audio_features =
      model_.EncodeAudio(mx::stack(windows), window_token_counts);
  const mx::array token_embeddings = model_.decoder().EmbedTokens(
      mx::array(prompt_ids.data(), {1, prompt_length}, mx::int32));
  const int hidden = token_embeddings.shape(2);
  // Note (Dayuxiaoshui): the digit markers between runs of audio placeholders
  // keep their own token embeddings.
  std::vector<mx::array> pieces;
  int cursor = 0;
  int audio_row = 0;
  const int audio_token_id = model_.audio_token_id();
  for (int position = 0; position < prompt_length;) {
    if (prompt_ids[position] != audio_token_id) {
      ++position;
      continue;
    } else {
    }
    int run_end = position;
    while (run_end < prompt_length && prompt_ids[run_end] == audio_token_id)
      ++run_end;
    if (position > cursor) {
      pieces.push_back(mx::reshape(
          mx::slice(token_embeddings, {0, cursor, 0}, {1, position, hidden}),
          {position - cursor, hidden}));
    } else {
    }
    const int run_length = run_end - position;
    pieces.push_back(mx::slice(audio_features, {audio_row, 0},
                               {audio_row + run_length, hidden}));
    audio_row += run_length;
    cursor = run_end;
    position = run_end;
  }
  if (audio_row != audio_features.shape(0)) {
    throw std::runtime_error("audio features and audio tokens do not match");
  } else {
  }
  if (cursor < prompt_length) {
    pieces.push_back(mx::reshape(
        mx::slice(token_embeddings, {0, cursor, 0}, {1, prompt_length, hidden}),
        {prompt_length - cursor, hidden}));
  } else {
  }
  const mx::array embeddings = mx::expand_dims(mx::concatenate(pieces, 0), 0);
  mx::eval(embeddings);

  // Note (Dayuxiaoshui): the last prompt token stays out of the chunked
  // prefill, so its forward pass also gives the first logits.
  std::vector<KVCache> caches = model_.decoder().NewCaches();
  int processed = 0;
  while (prompt_length - processed > 1) {
    if (cancel.load()) {
      throw TranscriptionCancelled();
    } else {
    }
    const int step =
        std::min(kPrefillStepTokens, prompt_length - processed - 1);
    mx::eval(model_.decoder().Forward(
        mx::slice(embeddings, {0, processed, 0}, {1, processed + step, hidden}),
        caches));
    processed += step;
  }
  mx::array next_token =
      mx::argmax(model_.decoder().LastLogits(model_.decoder().Forward(
          mx::slice(embeddings, {0, processed, 0}, {1, prompt_length, hidden}),
          caches)));
  mx::async_eval({next_token});

  const auto is_end = [&](int token_id) {
    return token_id == im_end_id_ ||
           (options.stop_at_end_of_text && token_id == end_of_text_id_);
  };
  std::vector<int> output_ids;
  FinishReason finish_reason = FinishReason::kLength;
  while (static_cast<int>(output_ids.size()) < max_new_tokens) {
    if (cancel.load()) {
      throw TranscriptionCancelled();
    } else {
    }
    const mx::array token = next_token;
    // Note (Dayuxiaoshui): queue the next step before reading this token, so
    // the GPU decodes while the stop rules are checked.
    next_token =
        mx::argmax(model_.decoder().LastLogits(model_.decoder().Forward(
            model_.decoder().EmbedTokens(mx::reshape(token, {1, 1})), caches)));
    mx::async_eval({next_token});
    const int token_id = static_cast<int>(token.item<uint32_t>());
    if (is_end(token_id)) {
      finish_reason = FinishReason::kStop;
      break;
    } else {
    }
    output_ids.push_back(token_id);
    if (options.stop_on_token_loop && IsTokenLoop(output_ids)) {
      finish_reason = FinishReason::kStop;
      break;
    } else {
    }
  }
  const std::string raw_text =
      StripUnicodeWhitespace(tokenizer_.Decode(output_ids, true));
  TranscriptionResult result;
  result.text = OffsetTimestampTags(raw_text, offset_seconds);
  result.segments =
      ParseSegments(raw_text, static_cast<double>(sample_count) / kSampleRate,
                    offset_seconds);
  result.generated_token_count = static_cast<int>(output_ids.size());
  result.finish_reason = finish_reason;
  return result;
}

TranscriptionResult
MossTranscriber::Transcribe(const std::vector<float> &samples,
                            const MossOptions &options,
                            const std::atomic<bool> &cancel) const {
  if (samples.empty()) {
    throw std::invalid_argument("audio must contain at least one sample");
  } else if (options.window_offset_seconds.has_value()) {
    if (options.max_new_tokens.value_or(0) <= 0) {
      throw std::invalid_argument("a realtime window needs max_new_tokens");
    } else {
    }
    return TranscribeChunk(samples, options, *options.max_new_tokens,
                           *options.window_offset_seconds, cancel);
  } else {
  }
  const double duration_seconds =
      static_cast<double>(samples.size()) / kSampleRate;
  const int max_new_tokens =
      options.max_new_tokens.value_or(0) != 0
          ? *options.max_new_tokens
          : std::min(
                kDefaultMaxTokens,
                std::max(kDefaultMinTokens,
                         static_cast<int>(std::ceil(
                             duration_seconds * kDefaultTokensPerAudioSecond)) +
                             kDefaultExtraTokens));
  TranscriptionResult combined;
  combined.finish_reason = FinishReason::kStop;
  for (const AudioChunk &chunk :
       SplitAudio(samples, kChunkDurationSeconds, kMinChunkDurationSeconds)) {
    TranscriptionResult result = TranscribeChunk(
        chunk.samples, options, max_new_tokens, chunk.offset_seconds, cancel);
    const std::string text = StripUnicodeWhitespace(result.text);
    if (!text.empty()) {
      combined.text += (combined.text.empty() ? "" : "\n") + text;
    } else {
    }
    combined.segments.insert(combined.segments.end(), result.segments.begin(),
                             result.segments.end());
    combined.generated_token_count += result.generated_token_count;
    // Note (Dayuxiaoshui): one truncated chunk cuts the whole transcript short.
    if (result.finish_reason == FinishReason::kLength) {
      combined.finish_reason = FinishReason::kLength;
    } else {
    }
    mx::clear_cache();
  }
  return combined;
}

} // namespace moss
