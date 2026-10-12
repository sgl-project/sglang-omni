// SPDX-License-Identifier: Apache-2.0
#include "nemotron_transcriber.h"

#include <algorithm>
#include <cmath>

#include <CoreFoundation/CoreFoundation.h>

#include "swift_port.h"

namespace nemotron {

namespace mx = mlx::core;

namespace {

// Note (Dayuxiaoshui): the 8x causal subsampling looks back fewer than 16 mel
// frames, so carrying the last 16 into the next window reproduces the
// subsampled frames of one pass over all the audio.
constexpr int kMelCacheFrames = 16;
constexpr const char *kWordMarker = "\xE2\x96\x81"; // U+2581
// Note (Dayuxiaoshui): the front end needs two samples for preemphasis; one
// sample alone (which Swift decodes as one mel frame) gives no text here.
constexpr int kMinFrontEndSamples = 2;

bool Contains(const std::string &text, const char *needle) {
  return text.find(needle) != std::string::npos;
}

// Note (Dayuxiaoshui): Swift's String.contains(" ") compares grapheme
// clusters, so a space that a combining mark attaches to is not a space; a
// word marker before a Thai vowel sign is one such piece.
bool ContainsStandaloneSpace(const std::string &text) {
  const std::vector<uint32_t> code_points = qwen3_asr::CodePoints(text);
  const CFCharacterSetRef combining =
      CFCharacterSetGetPredefined(kCFCharacterSetNonBase);
  for (size_t index = 0; index < code_points.size(); ++index) {
    if (code_points[index] == ' ' && (index + 1 == code_points.size() ||
                                      !CFCharacterSetIsLongCharacterMember(
                                          combining, code_points[index + 1]))) {
      return true;
    } else {
    }
  }
  return false;
}

sortformer::ProcessorConfig FrontEnd(const std::filesystem::path &directory) {
  const nlohmann::json config =
      qwen3_asr::ReadJson(directory / "config.json").at("preprocessor");
  const int sample_rate = config.at("sample_rate").get<int>();
  sortformer::ProcessorConfig front_end;
  front_end.feature_size = config.at("features").get<int>();
  front_end.sampling_rate = sample_rate;
  front_end.fft_size = config.at("n_fft").get<int>();
  front_end.window_length = static_cast<int>(
      std::lround(config.at("window_size").get<double>() * sample_rate));
  front_end.hop_length = static_cast<int>(
      std::lround(config.at("window_stride").get<double>() * sample_rate));
  front_end.preemphasis = config.at("preemph").get<float>();
  return front_end;
}

} // namespace

NemotronTranscriber::NemotronTranscriber(
    const std::filesystem::path &model_directory)
    : model_(model_directory), front_end_(FrontEnd(model_directory)),
      features_(front_end_) {
  for (const std::string &piece : model_.config().vocabulary) {
    std::string text = piece;
    for (size_t marker = text.find(kWordMarker); marker != std::string::npos;
         marker = text.find(kWordMarker, marker + 1)) {
      text.replace(marker, std::string(kWordMarker).size(), " ");
    }
    standalone_spaces_.push_back(ContainsStandaloneSpace(text));
    token_texts_.push_back(text);
    const bool language_tag = piece.size() > 1 && piece.front() == '<' &&
                              piece.back() == '>' && Contains(piece, "-");
    const bool control = piece.size() > 3 && piece.rfind("<|", 0) == 0 &&
                         piece.compare(piece.size() - 2, 2, "|>") == 0;
    special_tokens_.push_back(language_tag || control || piece == "<unk>" ||
                              piece == "<pad>");
  }
}

double NemotronTranscriber::frame_seconds() const {
  return static_cast<double>(model_.config().subsampling_factor *
                             front_end_.hop_length) /
         front_end_.sampling_rate;
}

bool NemotronTranscriber::IsSpecialToken(int token_id) const {
  return token_id >= 0 && token_id < static_cast<int>(special_tokens_.size()) &&
         special_tokens_[token_id];
}

const std::string &NemotronTranscriber::TokenText(int token_id) const {
  return token_texts_.at(token_id);
}

bool NemotronTranscriber::HasStandaloneSpace(int token_id) const {
  return standalone_spaces_.at(token_id);
}

qwen3_asr::TranscriptionResult
NemotronTranscriber::Transcribe(const std::vector<float> &samples,
                                const NemotronOptions &options,
                                const std::atomic<bool> &cancel) const {
  NemotronStream stream(*this, options);
  stream.Finish(samples, cancel);
  return stream.Result();
}

NemotronStream::NemotronStream(const NemotronTranscriber &transcriber,
                               const NemotronOptions &options)
    : transcriber_(transcriber), language_(options.language),
      prompt_index_(transcriber.model().PromptIndex(options.language)),
      chunk_frames_(options.chunk_frames.value_or(
          transcriber.model().config().right_context_frames + 1)),
      layer_caches_(transcriber.model().config().layer_count),
      last_token_(transcriber.model().blank_id()) {
  if (chunk_frames_ < 1) {
    throw std::invalid_argument("chunk_frames must be positive");
  } else {
  }
}

void NemotronStream::Append(const std::vector<float> &samples,
                            const std::atomic<bool> &cancel) {
  const int half_window = transcriber_.front_end().fft_size / 2;
  const int hop = transcriber_.front_end().hop_length;
  // Note (Jiaxin Deng): one chunk of audio at a time, so however much one call
  // brings, the mel and its FFT frames cover about a chunk, not all of it.
  const size_t chunk_samples =
      static_cast<size_t>(chunk_frames_) *
      transcriber_.model().config().subsampling_factor * hop;
  for (size_t start = 0; start < samples.size(); start += chunk_samples) {
    const auto first = samples.begin() + static_cast<std::ptrdiff_t>(start);
    samples_.insert(samples_.end(), first,
                    first + static_cast<std::ptrdiff_t>(std::min(
                                chunk_samples, samples.size() - start)));
    const int sample_count = sample_offset_ + static_cast<int>(samples_.size());
    // Note (Dayuxiaoshui): mel frame m reads samples up to m * hop + n_fft / 2,
    // so frames below this limit no longer change as audio arrives.
    const int frozen_frames =
        sample_count < half_window ? 0 : (sample_count - half_window) / hop + 1;
    Decode(frozen_frames, false, cancel);
  }
}

void NemotronStream::Finish(const std::vector<float> &samples,
                            const std::atomic<bool> &cancel) {
  Append(samples, cancel);
  Decode(-1, true, cancel);
}

void NemotronStream::Decode(int mel_frame_limit, bool flush,
                            const std::atomic<bool> &cancel) {
  const NemotronModel &model = transcriber_.model();
  const NemotronConfig &config = model.config();
  const int factor = config.subsampling_factor;
  const int chunk_mel_frames = chunk_frames_ * factor;
  // Note (Dayuxiaoshui): centered framing gives 1 + samples / hop mel frames,
  // so a call that cannot fill a whole chunk returns before the mel.
  const int hop = transcriber_.front_end().hop_length;
  const int sample_count = sample_offset_ + static_cast<int>(samples_.size());
  const int mel_frames = 1 + sample_count / hop;
  const int limit =
      mel_frame_limit < 0 ? mel_frames : std::min(mel_frames, mel_frame_limit);
  if (sample_count < kMinFrontEndSamples ||
      (!flush && limit - consumed_mel_frames_ < chunk_mel_frames)) {
    return;
  } else {
  }
  // Note (Dayuxiaoshui): only the kept audio's mel is computed. It starts a
  // whole number of hops in, so its frame i is frame first_frame + i of the
  // full recording; its first frames, which read the padding or a sample
  // whose preemphasis lost its predecessor, are already decoded.
  const int first_frame = sample_offset_ / hop;
  const mx::array mel =
      mx::astype(mx::transpose(transcriber_.features()(samples_), {0, 2, 1}),
                 mx::bfloat16);
  while (consumed_mel_frames_ < limit) {
    if (cancel.load()) {
      throw qwen3_asr::TranscriptionCancelled();
    } else {
    }
    const int end = std::min(consumed_mel_frames_ + chunk_mel_frames, limit);
    if (!flush && end - consumed_mel_frames_ < chunk_mel_frames) {
      break;
    } else {
    }
    const mx::array chunk =
        mx::slice(mel, {0, consumed_mel_frames_ - first_frame, 0},
                  {1, end - first_frame, config.feature_count});
    const int cached_frames = mel_cache_.has_value() ? mel_cache_->shape(1) : 0;
    const mx::array window = mel_cache_.has_value()
                                 ? mx::concatenate({*mel_cache_, chunk}, 1)
                                 : chunk;
    const mx::array subsampled = model.Subsample(window);
    const int window_frames = window.shape(1);
    const bool last_chunk = flush && end >= limit;
    const int base = (consumed_mel_frames_ - cached_frames) / factor;
    const int low = emitted_frames_ - base;
    const int high = last_chunk ? subsampled.shape(1) : end / factor - base;
    consumed_mel_frames_ = end;
    mel_cache_ =
        mx::slice(window, {0, std::max(0, window_frames - kMelCacheFrames), 0},
                  {1, window_frames, config.feature_count});
    if (high <= low) {
      emitted_frames_ = base + std::max(low, high);
      continue;
    } else {
    }
    emitted_frames_ = base + high;
    mx::array hidden =
        mx::slice(subsampled, {0, low, 0}, {1, high, config.model_width});
    const int attention_frames =
        layer_caches_[0].attention_inputs.has_value()
            ? layer_caches_[0].attention_inputs->shape(1)
            : 0;
    const mx::array positions =
        model.RelativePositions(attention_frames + high - low, hidden.dtype());
    for (int layer = 0; layer < config.layer_count; ++layer) {
      hidden =
          model.StreamLayer(hidden, layer, positions, layer_caches_[layer]);
    }
    DecodeChunk(model.ApplyPrompt(hidden, prompt_index_));
  }
  std::vector<mx::array> carried;
  if (mel_cache_.has_value()) {
    carried.push_back(*mel_cache_);
  } else {
  }
  for (const LayerCache &cache : layer_caches_) {
    if (cache.attention_inputs.has_value()) {
      carried.push_back(*cache.attention_inputs);
      carried.push_back(*cache.convolution_inputs);
    } else {
    }
  }
  mx::eval(carried);
  // Note (Dayuxiaoshui): mel frame m reads samples from m * hop - n_fft / 2,
  // and preemphasis one sample before that; keep whole hops back to there.
  const int lookback_hops =
      (transcriber_.front_end().fft_size / 2 + 1 + hop - 1) / hop;
  const int keep_from = std::max(0, consumed_mel_frames_ - lookback_hops) * hop;
  if (keep_from > sample_offset_) {
    samples_.erase(samples_.begin(),
                   samples_.begin() + (keep_from - sample_offset_));
    sample_offset_ = keep_from;
  } else {
  }
  // Note (Dayuxiaoshui): as Swift's session does after each step, return this
  // call's buffers.
  mx::clear_cache();
}

void NemotronStream::DecodeChunk(const mx::array &prompted) {
  const NemotronModel &model = transcriber_.model();
  const int blank = model.blank_id();
  const int width = prompted.shape(2);
  const int frame_count = prompted.shape(1);
  const double frame_seconds = transcriber_.frame_seconds();
  // Note (Dayuxiaoshui): the prediction only changes when a token is emitted,
  // so it is reused across blanks; Swift reruns it to the same values.
  std::optional<std::pair<mx::array, LstmState>> prediction;
  int frame = 0;
  int symbols_this_frame = 0;
  while (frame < frame_count) {
    if (!prediction.has_value()) {
      auto [output, state] = model.Predict(
          last_token_ == blank ? std::nullopt : std::optional<int>(last_token_),
          prediction_state_);
      const mx::Dtype dtype = prompted.dtype();
      for (size_t layer = 0; layer < state.hidden.size(); ++layer) {
        state.hidden[layer] = mx::astype(state.hidden[layer], dtype);
        state.cell[layer] = mx::astype(state.cell[layer], dtype);
      }
      prediction.emplace(mx::astype(output, dtype), std::move(state));
    } else {
    }
    const mx::array encoder_frame =
        mx::slice(prompted, {0, frame, 0}, {1, frame + 1, width});
    const int token = static_cast<int>(
        mx::argmax(model.JointLogits(encoder_frame, prediction->first), -1)
            .item<uint32_t>());
    if (token == blank) {
      ++frame;
      symbols_this_frame = 0;
      continue;
    } else {
    }
    last_token_ = token;
    prediction_state_ = std::move(prediction->second);
    prediction.reset();
    if (!transcriber_.IsSpecialToken(token)) {
      tokens_.push_back({transcriber_.TokenText(token),
                         transcriber_.HasStandaloneSpace(token),
                         (decoded_frames_ + frame) * frame_seconds});
    } else {
    }
    ++symbols_this_frame;
    if (model.config().max_symbols_per_frame.has_value() &&
        symbols_this_frame >= *model.config().max_symbols_per_frame) {
      ++frame;
      symbols_this_frame = 0;
    } else {
    }
  }
  decoded_frames_ += frame_count;
}

std::string NemotronStream::Text() const {
  std::string text;
  for (const Token &token : tokens_) {
    text += token.text;
  }
  return swift_port::TrimWhitespace(text, true);
}

qwen3_asr::TranscriptionResult NemotronStream::Result() const {
  const double frame_seconds = transcriber_.frame_seconds();
  qwen3_asr::TranscriptionResult result;
  result.text = Text();
  result.language = language_;
  result.generated_token_count = static_cast<int>(tokens_.size());
  result.finish_reason = qwen3_asr::FinishReason::kStop;
  // Note (Dayuxiaoshui): a sentence closes on ! ? and their CJK forms, or on a
  // period that ends the transcript or precedes a word boundary.
  std::string sentence;
  double sentence_start = 0.0;
  for (size_t index = 0; index < tokens_.size(); ++index) {
    const Token &token = tokens_[index];
    if (sentence.empty()) {
      sentence_start = token.start_seconds;
    } else {
    }
    sentence += token.text;
    const bool closes = Contains(token.text, "!") ||
                        Contains(token.text, "?") ||
                        Contains(token.text, "\xE3\x80\x82") || // 。
                        Contains(token.text, "\xEF\xBC\x9F") || // ？
                        Contains(token.text, "\xEF\xBC\x81") || // ！
                        (Contains(token.text, ".") &&
                         (index + 1 == tokens_.size() ||
                          tokens_[index + 1].has_standalone_space));
    if (closes || index + 1 == tokens_.size()) {
      result.segments.push_back(
          {sentence_start, token.start_seconds + frame_seconds, "", sentence});
      sentence.clear();
    } else {
    }
  }
  return result;
}

} // namespace nemotron
