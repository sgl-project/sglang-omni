// SPDX-License-Identifier: Apache-2.0
#include "parakeet/parakeet_transcriber.h"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <stdexcept>

#include <CoreFoundation/CoreFoundation.h>

#include "common/swift_port.h"
#include "common/text.h"

namespace parakeet {

namespace mx = mlx::core;

namespace {

constexpr const char *kWordMarker = "\xE2\x96\x81"; // U+2581
constexpr double kWholeRecordingChunkSeconds = 1199.0;
// Note (Mengwei Liu): quiet cuts and a useful final tail reduce blank
// predictions on short crops without unbounded encoder attention.
constexpr double kFinalChunkSeconds = 30.0;
// Note (Mengwei Liu): these bounded settings recover the frozen short-tail
// and seed-2888 blank-chunk regressions; they are not universal optima.
constexpr double kFinalMinimumTailSeconds = 10.0;
constexpr double kRecoveryPaddingSeconds = 0.25;
constexpr double kRecoveryGapSeconds = 3.0;
constexpr int kRecoveryAttemptCount = 2;
// Note (Mengwei Liu): the -50 dBFS RMS gate excludes silence, not noise.
constexpr double kRecoveryMinimumMeanSquaredAmplitude = 1e-5;
constexpr double kChunkSearchSeconds = 5.0;
constexpr double kChunkEnergyWindowSeconds = 0.1;
constexpr size_t kBufferCacheLimitBytes = 256ULL << 20;
constexpr double kMinChunkSeconds = 0.5;
constexpr double kChunkOverlapSeconds = 1.0;

bool Contains(const std::string &text, const char *needle) {
  return text.find(needle) != std::string::npos;
}

// Note (Dayuxiaoshui): Swift's String.contains(" ") compares grapheme
// clusters, so a space that a combining mark attaches to is not a space.
bool ContainsStandaloneSpace(const std::string &text) {
  const std::vector<uint32_t> code_points = omni_common::CodePoints(text);
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

struct Sentence {
  std::string text;
  std::vector<AlignedToken> tokens;
};

// Note (Dayuxiaoshui): NemoAlignment.tokensToSentences: a sentence ends at
// "!", "?" or their CJK forms, and at "." before a token starting a word.
std::vector<Sentence>
TokensToSentences(const std::vector<AlignedToken> &tokens) {
  std::vector<Sentence> sentences;
  Sentence current;
  const auto close = [&]() {
    for (const AlignedToken &token : current.tokens) {
      current.text += token.text;
    }
    std::stable_sort(current.tokens.begin(), current.tokens.end(),
                     [](const AlignedToken &left, const AlignedToken &right) {
                       return left.start_seconds < right.start_seconds;
                     });
    sentences.push_back(std::move(current));
    current = Sentence();
  };
  for (size_t index = 0; index < tokens.size(); ++index) {
    const AlignedToken &token = tokens[index];
    current.tokens.push_back(token);
    const std::string &text = token.text;
    const bool closes =
        Contains(text, "!") || Contains(text, "?") ||
        Contains(text, "\xE3\x80\x82") || Contains(text, "\xEF\xBC\x9F") ||
        Contains(text, "\xEF\xBC\x81") ||
        (Contains(text, ".") && (index + 1 == tokens.size() ||
                                 tokens[index + 1].has_standalone_space));
    if (closes) {
      close();
    } else {
    }
  }
  if (!current.tokens.empty()) {
    close();
  } else {
  }
  return sentences;
}

bool Matches(const AlignedToken &left, const AlignedToken &right,
             double tolerance_seconds) {
  return left.id == right.id &&
         std::abs(left.start_seconds - right.start_seconds) < tolerance_seconds;
}

// Swift's a.filter { $0.end <= cutoff } + b.filter { $0.start >= cutoff }.
std::vector<AlignedToken> CutAtMidpoint(const std::vector<AlignedToken> &a,
                                        const std::vector<AlignedToken> &b) {
  const double cutoff = (a.back().end_seconds() + b.front().start_seconds) / 2;
  std::vector<AlignedToken> merged;
  std::copy_if(
      a.begin(), a.end(), std::back_inserter(merged),
      [&](const AlignedToken &token) { return token.end_seconds() <= cutoff; });
  std::copy_if(
      b.begin(), b.end(), std::back_inserter(merged),
      [&](const AlignedToken &token) { return token.start_seconds >= cutoff; });
  return merged;
}

// The tokens of a up to its first matched pair, each matched token of a with
// the longer of the two gaps after it, then b past its last matched token;
// pairs index the overlapping tokens of a (from a_start) and of b.
std::vector<AlignedToken>
JoinOnPairs(const std::vector<AlignedToken> &a,
            const std::vector<AlignedToken> &b, size_t a_start,
            const std::vector<std::pair<size_t, size_t>> &pairs) {
  std::vector<AlignedToken> merged(a.begin(),
                                   a.begin() + (a_start + pairs[0].first));
  for (size_t index = 0; index < pairs.size(); ++index) {
    const size_t ia = a_start + pairs[index].first;
    const size_t ib = pairs[index].second;
    merged.push_back(a[ia]);
    if (index + 1 < pairs.size()) {
      const size_t next_a = a_start + pairs[index + 1].first;
      const size_t next_b = pairs[index + 1].second;
      if (next_b - (ib + 1) > next_a - (ia + 1)) {
        merged.insert(merged.end(), b.begin() + (ib + 1), b.begin() + next_b);
      } else {
        merged.insert(merged.end(), a.begin() + (ia + 1), a.begin() + next_a);
      }
    } else {
    }
  }
  merged.insert(merged.end(), b.begin() + (pairs.back().second + 1), b.end());
  return merged;
}

// Note (Dayuxiaoshui): the Swift port's mergeTokenSequences: the longest
// contiguous run of matching overlap tokens when it covers half the
// overlap, else their longest common subsequence.
std::vector<AlignedToken> MergeTokens(const std::vector<AlignedToken> &a,
                                      const std::vector<AlignedToken> &b,
                                      double overlap_seconds) {
  if (a.empty()) {
    return b;
  } else if (b.empty()) {
    return a;
  } else {
  }
  const double a_end = a.back().end_seconds();
  const double b_start = b.front().start_seconds;
  if (a_end <= b_start) {
    std::vector<AlignedToken> merged = a;
    merged.insert(merged.end(), b.begin(), b.end());
    return merged;
  } else {
  }
  std::vector<AlignedToken> overlap_a;
  std::copy_if(a.begin(), a.end(), std::back_inserter(overlap_a),
               [&](const AlignedToken &token) {
                 return token.end_seconds() > b_start - overlap_seconds;
               });
  std::vector<AlignedToken> overlap_b;
  std::copy_if(b.begin(), b.end(), std::back_inserter(overlap_b),
               [&](const AlignedToken &token) {
                 return token.start_seconds < a_end + overlap_seconds;
               });
  if (overlap_a.size() < 2 || overlap_b.size() < 2) {
    return CutAtMidpoint(a, b);
  } else {
  }
  const double tolerance = overlap_seconds / 2;
  const size_t a_start = a.size() - overlap_a.size();

  std::vector<std::pair<size_t, size_t>> best;
  for (size_t i = 0; i < overlap_a.size(); ++i) {
    for (size_t j = 0; j < overlap_b.size(); ++j) {
      std::vector<std::pair<size_t, size_t>> chain;
      for (size_t k = i, l = j; k < overlap_a.size() && l < overlap_b.size() &&
                                Matches(overlap_a[k], overlap_b[l], tolerance);
           ++k, ++l) {
        chain.emplace_back(k, l);
      }
      if (chain.size() > best.size()) {
        best = std::move(chain);
      } else {
      }
    }
  }
  if (best.size() >= overlap_a.size() / 2) {
    return JoinOnPairs(a, b, a_start, best);
  } else {
  }

  const size_t rows = overlap_a.size() + 1;
  const size_t columns = overlap_b.size() + 1;
  std::vector<std::vector<int>> lengths(rows, std::vector<int>(columns, 0));
  for (size_t i = 1; i < rows; ++i) {
    for (size_t j = 1; j < columns; ++j) {
      lengths[i][j] = Matches(overlap_a[i - 1], overlap_b[j - 1], tolerance)
                          ? lengths[i - 1][j - 1] + 1
                          : std::max(lengths[i - 1][j], lengths[i][j - 1]);
    }
  }
  std::vector<std::pair<size_t, size_t>> pairs;
  for (size_t i = overlap_a.size(), j = overlap_b.size(); i > 0 && j > 0;) {
    if (Matches(overlap_a[i - 1], overlap_b[j - 1], tolerance)) {
      pairs.emplace_back(i - 1, j - 1);
      --i;
      --j;
    } else if (lengths[i - 1][j] > lengths[i][j - 1]) {
      --i;
    } else {
      --j;
    }
  }
  if (pairs.empty()) {
    return CutAtMidpoint(a, b);
  } else {
  }
  std::reverse(pairs.begin(), pairs.end());
  return JoinOnPairs(a, b, a_start, pairs);
}

} // namespace

ParakeetTranscriber::ParakeetTranscriber(
    const std::filesystem::path &model_directory)
    : model_(model_directory) {
  // Note (Mengwei Liu): each process serves one model; bound freed buffers
  // for both the server and multi-file CLI, not only when a server is idle.
  mx::set_cache_limit(kBufferCacheLimitBytes);
  for (const std::string &piece : model_.config().vocabulary) {
    std::string text = piece;
    for (size_t marker = text.find(kWordMarker); marker != std::string::npos;
         marker = text.find(kWordMarker, marker + 1)) {
      text.replace(marker, std::string(kWordMarker).size(), " ");
    }
    standalone_spaces_.push_back(ContainsStandaloneSpace(text));
    token_texts_.push_back(text);
    special_tokens_.push_back((piece.size() >= 4 && piece.rfind("<|", 0) == 0 &&
                               piece.compare(piece.size() - 2, 2, "|>") == 0) ||
                              piece == "<unk>" || piece == "<pad>");
  }
}

std::vector<AlignedToken>
ParakeetTranscriber::DecodeChunk(const std::vector<float> &samples,
                                 const std::atomic<bool> &cancel) const {
  if (cancel.load()) {
    throw qwen3_asr::TranscriptionCancelled();
  } else {
  }
  const ParakeetConfig &config = model_.config();
  const mx::array encoded = model_.Encode(model_.Features(samples));
  mx::eval(encoded);
  const int frame_count = encoded.shape(1);
  const int blank = model_.blank_id();
  auto [hidden, cell] = model_.InitialState();
  int last_token = blank;
  int new_symbols = 0;
  std::vector<AlignedToken> tokens;
  for (int frame = 0; frame < frame_count;) {
    if (cancel.load()) {
      throw qwen3_asr::TranscriptionCancelled();
    } else {
    }
    const DecoderStep step = model_.Decode(
        mx::slice(encoded, {0, frame, 0}, {1, frame + 1, config.model_width}),
        last_token, hidden, cell);
    // Note (Dayuxiaoshui): NemoDecodingLogic.tdtStep; an unknown duration
    // class advances one frame.
    const int jump =
        step.duration_index >= 0 &&
                step.duration_index < static_cast<int>(config.durations.size())
            ? config.durations[step.duration_index]
            : 1;
    const int symbol_frame = frame;
    frame += jump;
    if (jump != 0) {
      new_symbols = 0;
    } else if (++new_symbols >= config.max_symbols_per_frame) {
      frame += 1;
      new_symbols = 0;
    } else {
    }
    if (step.token != blank) {
      last_token = step.token;
      hidden = step.hidden;
      cell = step.cell;
      if (step.token >= 0 && step.token < blank &&
          !special_tokens_[step.token]) {
        tokens.push_back({step.token, token_texts_[step.token],
                          model_.FrameSeconds(symbol_frame),
                          model_.FrameSeconds(jump),
                          standalone_spaces_[step.token]});
      } else {
      }
    } else {
    }
  }
  std::vector<AlignedToken> ordered;
  for (Sentence &sentence : TokensToSentences(tokens)) {
    ordered.insert(ordered.end(), sentence.tokens.begin(),
                   sentence.tokens.end());
  }
  return ordered;
}

std::vector<AlignedToken>
ParakeetTranscriber::DecodeFinalChunk(const std::vector<float> &samples,
                                      const std::atomic<bool> &cancel) const {
  std::vector<AlignedToken> tokens = DecodeChunk(samples, cancel);
  const size_t sample_rate_hertz = omni_common::kSampleRate;
  const size_t padding_samples =
      static_cast<size_t>(kRecoveryPaddingSeconds * sample_rate_hertz);
  const size_t max_chunk_samples =
      static_cast<size_t>(kFinalChunkSeconds * sample_rate_hertz);
  const size_t recovery_context_samples =
      static_cast<size_t>(kChunkOverlapSeconds * sample_rate_hertz);
  for (int attempt = 0; attempt < kRecoveryAttemptCount && !cancel.load();
       ++attempt) {
    const double decoded_end_seconds =
        tokens.empty() ? 0.0 : tokens.back().end_seconds();
    const size_t decoded_end_sample =
        std::min(samples.size(),
                 static_cast<size_t>(decoded_end_seconds * sample_rate_hertz));
    if (samples.size() - decoded_end_sample <
        static_cast<size_t>(kRecoveryGapSeconds * sample_rate_hertz)) {
      break;
    } else {
    }
    double tail_squared_amplitude_sum = 0.0;
    for (size_t i = decoded_end_sample; i < samples.size(); ++i) {
      tail_squared_amplitude_sum +=
          static_cast<double>(samples[i]) * samples[i];
    }
    if (tail_squared_amplitude_sum / (samples.size() - decoded_end_sample) <
        kRecoveryMinimumMeanSquaredAmplitude) {
      break;
    } else {
    }
    const size_t start_sample =
        decoded_end_sample > recovery_context_samples
            ? decoded_end_sample - recovery_context_samples
            : 0;
    const size_t requested_padding_samples =
        tokens.empty() ? (attempt + 1) * padding_samples
                       : (attempt > 0 ? padding_samples : 0);
    // Note (Mengwei Liu): padding uses spare encoder capacity, never space
    // obtained by discarding the source tail.
    const size_t leading_padding_samples =
        std::min(requested_padding_samples,
                 max_chunk_samples - (samples.size() - start_sample));
    const size_t end_sample = samples.size();
    std::vector<float> retry_samples(leading_padding_samples, 0.0f);
    retry_samples.insert(retry_samples.end(), samples.begin() + start_sample,
                         samples.begin() + end_sample);
    std::vector<AlignedToken> retry_tokens = DecodeChunk(retry_samples, cancel);
    const double shift_seconds =
        (static_cast<double>(start_sample) - leading_padding_samples) /
        sample_rate_hertz;
    for (AlignedToken &token : retry_tokens) {
      const double end_seconds =
          std::clamp(token.end_seconds() + shift_seconds, 0.0,
                     static_cast<double>(samples.size()) / sample_rate_hertz);
      token.start_seconds =
          std::clamp(token.start_seconds + shift_seconds, 0.0, end_seconds);
      token.duration_seconds = end_seconds - token.start_seconds;
    }
    // Note (Mengwei Liu): empty or nonadvancing retries keep the original.
    // A successful retry may revise tokens in the one-second overlap.
    if (!retry_tokens.empty() &&
        retry_tokens.back().end_seconds() > decoded_end_seconds) {
      tokens = MergeTokens(tokens, retry_tokens, kChunkOverlapSeconds);
    } else {
    }
  }
  return tokens;
}

qwen3_asr::TranscriptionResult
ParakeetTranscriber::Transcribe(const std::vector<float> &samples,
                                const ParakeetOptions &options,
                                const std::atomic<bool> &cancel) const {
  const double sample_rate = omni_common::kSampleRate;
  const bool whole_recording =
      options.chunk_duration_seconds >= kWholeRecordingChunkSeconds;
  const double chunk_seconds =
      whole_recording
          ? kFinalChunkSeconds
          : std::max(kMinChunkSeconds, options.chunk_duration_seconds);
  const size_t chunk_samples =
      std::max<size_t>(1, static_cast<size_t>(chunk_seconds * sample_rate));
  const size_t overlap_samples =
      std::min(chunk_samples - 1,
               static_cast<size_t>(kChunkOverlapSeconds * sample_rate));
  std::vector<std::pair<size_t, size_t>> chunks;
  if (whole_recording) {
    chunks = omni_common::EnergyCutChunks(
        samples, chunk_samples,
        static_cast<size_t>(kChunkSearchSeconds * sample_rate),
        static_cast<size_t>(kChunkEnergyWindowSeconds * sample_rate),
        static_cast<size_t>(kFinalMinimumTailSeconds * sample_rate));
  } else {
    const size_t step_samples = chunk_samples - overlap_samples;
    for (size_t start = 0; start < samples.size(); start += step_samples) {
      const size_t end = std::min(start + chunk_samples, samples.size());
      chunks.emplace_back(start, end);
      if (end >= samples.size()) {
        break;
      } else {
      }
    }
  }
  std::vector<AlignedToken> tokens;
  for (const auto &[start, end] : chunks) {
    // Note (Dayuxiaoshui): the front end's preemphasis needs two samples;
    // a shorter chunk has no speech to decode.
    const std::vector<float> chunk(samples.begin() + start,
                                   samples.begin() + end);
    std::vector<AlignedToken> chunk_tokens;
    if (chunk.size() >= 2) {
      chunk_tokens = whole_recording ? DecodeFinalChunk(chunk, cancel)
                                     : DecodeChunk(chunk, cancel);
    } else {
    }
    for (AlignedToken &token : chunk_tokens) {
      token.start_seconds += static_cast<double>(start) / sample_rate;
    }
    if (whole_recording) {
      tokens.insert(tokens.end(), chunk_tokens.begin(), chunk_tokens.end());
    } else {
      tokens = MergeTokens(tokens, chunk_tokens, kChunkOverlapSeconds);
    }
  }

  qwen3_asr::TranscriptionResult result;
  result.finish_reason = qwen3_asr::FinishReason::kStop;
  result.generated_token_count = static_cast<int>(tokens.size());
  std::string text;
  for (const Sentence &sentence : TokensToSentences(tokens)) {
    text += sentence.text;
    qwen3_asr::SpeakerSegment segment;
    segment.start_seconds = sentence.tokens.front().start_seconds;
    segment.end_seconds = sentence.tokens.back().end_seconds();
    segment.text = sentence.text;
    result.segments.push_back(std::move(segment));
  }
  result.text = swift_port::TrimWhitespace(text, true);
  return result;
}

} // namespace parakeet
