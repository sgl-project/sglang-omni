// SPDX-License-Identifier: Apache-2.0
#include "moss/moss_transcriber.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <regex>

#include "common/audio.h"
#include "common/audio_chunks.h"
#include "common/generation.h"
#include "common/kv_cache.h"
#include "common/text.h"
#include "common/transcription.h"
#include "nlohmann/json.hpp"

namespace moss {

namespace mx = mlx::core;
using omni_common::DraftStatistics;
using omni_common::FinishReason;
using omni_common::GenerationCancelled;
using omni_common::IsTokenLoop;
using omni_common::kHopLength;
using omni_common::kSampleRate;
using omni_common::KVCache;
using omni_common::kWhisperWindowSampleCount;
using omni_common::PrefilledSequence;
using omni_common::SpeakerSegment;
using omni_common::StripUnicodeWhitespace;
using omni_common::TranscriptionResult;
using omni_common::WhisperWindowFeatures;

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

// The token budget per chunk of a final pass.
int FinalMaxNewTokens(size_t sample_count, const MossOptions &options) {
  const double duration_seconds =
      static_cast<double>(sample_count) / kSampleRate;
  return options.max_new_tokens.value_or(0) != 0
             ? *options.max_new_tokens
             : std::min(kDefaultMaxTokens,
                        std::max(kDefaultMinTokens,
                                 static_cast<int>(
                                     std::ceil(duration_seconds *
                                               kDefaultTokensPerAudioSecond)) +
                                     kDefaultExtraTokens));
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
      cut_sample = static_cast<int>(omni_common::QuietestAudioPosition(
          samples, search_start, search_end, energy_window_samples));
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
    SpeakerSegment segment;
    segment.start_seconds = start + offset_seconds;
    segment.end_seconds = end_seconds + offset_seconds;
    segment.speaker = (*match)[2].str();
    segment.text = segment_text;
    segments.push_back(std::move(segment));
  }
  if (segments.empty()) {
    SpeakerSegment segment;
    segment.start_seconds = offset_seconds;
    segment.end_seconds = offset_seconds + std::max(duration_seconds, 0.0);
    segment.text = text;
    segments.push_back(std::move(segment));
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

MossTranscriber::MossTranscriber(const std::filesystem::path &model_directory,
                                 const std::filesystem::path &draft_directory,
                                 int draft_quantization_bits)
    : model_(model_directory), tokenizer_(model_directory) {
  if (!draft_directory.empty()) {
    draft_.emplace(draft_directory, draft_quantization_bits,
                   model_.decoder().config());
  } else {
  }
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

mx::array MossTranscriber::PromptEmbeddings(
    const std::vector<float> &samples,
    const std::optional<std::string> &prompt) const {
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

  const std::vector<int> prompt_ids = PromptIds(audio_token_count, prompt);
  const int prompt_length = static_cast<int>(prompt_ids.size());
  // Short windows keep their silent padding so the encoder's bidirectional
  // attention sees the same context as the original runtime.
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
  mx::async_eval(embeddings);
  return embeddings;
}

PrefilledSequence
MossTranscriber::Prefill(const mx::array &embeddings,
                         const std::atomic<bool> &cancel) const {
  // Note (Dayuxiaoshui): the last prompt token stays out of the chunked
  // prefill, so its forward pass also gives the first logits.
  const int prompt_length = embeddings.shape(1);
  const int hidden = embeddings.shape(2);
  std::vector<KVCache> caches = model_.decoder().NewCaches();
  int processed = 0;
  while (prompt_length - processed > 1) {
    if (cancel.load()) {
      throw GenerationCancelled();
    } else {
    }
    const int step =
        std::min(kPrefillStepTokens, prompt_length - processed - 1);
    mx::async_eval(model_.decoder().Forward(
        mx::slice(embeddings, {0, processed, 0}, {1, processed + step, hidden}),
        caches));
    processed += step;
  }
  const mx::array first_token =
      mx::argmax(model_.decoder().LastLogits(model_.decoder().Forward(
          mx::slice(embeddings, {0, processed, 0}, {1, prompt_length, hidden}),
          caches)));
  mx::async_eval({first_token});
  return {std::move(caches), prompt_length, first_token};
}

bool MossTranscriber::Commit(ChunkOutput &output, int token_id,
                             const MossOptions &options,
                             int max_new_tokens) const {
  if (token_id == im_end_id_ ||
      (options.stop_at_end_of_text && token_id == end_of_text_id_)) {
    output.finish_reason = FinishReason::kStop;
    return true;
  } else {
  }
  output.output_ids.push_back(token_id);
  if (options.stop_on_token_loop && IsTokenLoop(output.output_ids)) {
    output.finish_reason = FinishReason::kStop;
    return true;
  } else {
    return static_cast<int>(output.output_ids.size()) >= max_new_tokens;
  }
}

TranscriptionResult
MossTranscriber::ChunkResult(const ChunkOutput &output, int sample_count,
                             double offset_seconds,
                             const DraftStatistics &draft_statistics) const {
  const std::string raw_text =
      StripUnicodeWhitespace(tokenizer_.Decode(output.output_ids, true));
  TranscriptionResult result;
  result.text = OffsetTimestampTags(raw_text, offset_seconds);
  result.segments =
      ParseSegments(raw_text, static_cast<double>(sample_count) / kSampleRate,
                    offset_seconds);
  result.generated_token_count = static_cast<int>(output.output_ids.size());
  result.finish_reason = output.finish_reason;
  result.verify_round_count = draft_statistics.verify_round_count;
  result.proposed_draft_token_count = draft_statistics.proposed_token_count;
  result.accepted_draft_token_count = draft_statistics.accepted_token_count;
  return result;
}

TranscriptionResult
MossTranscriber::TranscribeChunk(const std::vector<float> &samples,
                                 const MossOptions &options, int max_new_tokens,
                                 double offset_seconds,
                                 const std::atomic<bool> &cancel) const {
  const mx::array embeddings = PromptEmbeddings(samples, options.prompt);
  ChunkOutput output;
  const auto commit = [&](int token_id) {
    return Commit(output, token_id, options, max_new_tokens);
  };
  DraftStatistics draft_statistics;
  if (draft_.has_value()) {
    draft_statistics =
        draft_->Decode(model_.decoder(), embeddings, max_new_tokens,
                       kPrefillStepTokens, commit, cancel);
  } else {
    PrefilledSequence sequence = Prefill(embeddings, cancel);
    mx::array next_token = sequence.first_token;
    bool finished = false;
    while (!finished) {
      if (cancel.load()) {
        throw GenerationCancelled();
      } else {
      }
      const mx::array token = next_token;
      next_token =
          mx::argmax(model_.decoder().LastLogits(model_.decoder().Forward(
              model_.decoder().EmbedTokens(mx::reshape(token, {1, 1})),
              sequence.caches)));
      mx::async_eval({next_token});
      finished = commit(static_cast<int>(token.item<uint32_t>()));
    }
  }
  return ChunkResult(output, static_cast<int>(samples.size()), offset_seconds,
                     draft_statistics);
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
  const int max_new_tokens = FinalMaxNewTokens(samples.size(), options);
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
    combined.verify_round_count += result.verify_round_count;
    combined.proposed_draft_token_count += result.proposed_draft_token_count;
    combined.accepted_draft_token_count += result.accepted_draft_token_count;
    // Note (Dayuxiaoshui): one truncated chunk cuts the whole transcript short.
    if (result.finish_reason == FinishReason::kLength) {
      combined.finish_reason = FinishReason::kLength;
    } else {
    }
    mx::clear_cache();
  }
  return combined;
}

bool MossTranscriber::CanBatch(const std::vector<float> &samples,
                               const MossOptions &options) const {
  if (draft_.has_value() || samples.empty()) {
    return false;
  } else if (options.window_offset_seconds.has_value()) {
    return options.max_new_tokens.value_or(0) > 0;
  } else {
    return samples.size() <=
           static_cast<size_t>(kChunkDurationSeconds * kSampleRate);
  }
}

void MossTranscriber::TranscribeBatch(const std::vector<BatchRequest> &requests,
                                      const BatchFinished &on_finished) const {
  // A request decoded as one chunk: a realtime window as it is, else the
  // final pass's single chunk, padded to the minimum length.
  struct Row {
    size_t request_index;
    std::vector<float> samples;
    double offset_seconds;
    int max_new_tokens;
    ChunkOutput output;
  };
  std::vector<Row> rows;
  std::vector<PrefilledSequence> sequences;
  // Note (XXXXRT666): each request is encoded and prefilled alone, since the
  // prompts differ in length; the decode steps then run for all together.
  for (size_t i = 0; i < requests.size(); ++i) {
    try {
      const BatchRequest &request = requests[i];
      if (request.cancel->load()) {
        throw GenerationCancelled();
      } else {
      }
      const MossOptions &options = *request.options;
      Row row{i, {}, 0.0, 0, {}};
      if (options.window_offset_seconds.has_value()) {
        row.samples = *request.samples;
        row.offset_seconds = *options.window_offset_seconds;
        row.max_new_tokens = *options.max_new_tokens;
      } else {
        row.samples = PaddedTo(
            *request.samples,
            static_cast<size_t>(kMinChunkDurationSeconds * kSampleRate));
        row.max_new_tokens =
            FinalMaxNewTokens(request.samples->size(), options);
      }
      sequences.push_back(Prefill(PromptEmbeddings(row.samples, options.prompt),
                                  *request.cancel));
      rows.push_back(std::move(row));
    } catch (...) {
      on_finished(i, {std::nullopt, std::current_exception()});
    }
  }
  if (rows.empty()) {
    return;
  } else {
  }
  model_.decoder().DecodeGreedyBatch(
      sequences, [&](size_t index, int token_id) {
        Row &row = rows[index];
        const BatchRequest &request = requests[row.request_index];
        if (request.cancel->load()) {
          on_finished(
              row.request_index,
              {std::nullopt, std::make_exception_ptr(GenerationCancelled())});
          return true;
        } else if (!Commit(row.output, token_id, *request.options,
                           row.max_new_tokens)) {
          return false;
        } else {
        }
        TranscriptionResult result =
            ChunkResult(row.output, static_cast<int>(row.samples.size()),
                        row.offset_seconds, {});
        if (!request.options->window_offset_seconds.has_value()) {
          result.text = StripUnicodeWhitespace(result.text);
        } else {
        }
        on_finished(row.request_index, {std::move(result), nullptr});
        return true;
      });
}

omni_server::BatchRunner BatchableTranscriptionRunner() {
  return {[](const omni_server::Transcription &transcription) {
            return transcription.target<BatchableTranscription>() != nullptr;
          },
          [](const std::vector<omni_server::BatchItem> &items,
             const omni_server::BatchCompletion &finish) {
            std::vector<BatchRequest> requests;
            const MossTranscriber *transcriber = nullptr;
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

} // namespace moss
