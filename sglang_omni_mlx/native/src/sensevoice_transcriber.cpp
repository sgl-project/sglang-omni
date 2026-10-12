// SPDX-License-Identifier: Apache-2.0
#include "sensevoice_transcriber.h"

#include <algorithm>
#include <cctype>
#include <map>
#include <stdexcept>

#include <CoreFoundation/CoreFoundation.h>

#include "audio.h"

namespace sensevoice {

namespace mx = mlx::core;

namespace {

constexpr int kBlankId = 0;
// The language, emotion and event query frames before the speech.
constexpr int kQueryFrameCount = 4;

// Note (Dayuxiaoshui): Voxt's hint normalization and the Swift port's query
// rows; anything else detects the language.
int LanguageQueryId(const std::optional<std::string> &language) {
  static const std::map<std::string, int> ids = {{"zh", 3},  {"en", 4},
                                                 {"yue", 7}, {"ja", 11},
                                                 {"ko", 12}, {"nospeech", 13}};
  if (!language.has_value()) {
    return 0;
  } else {
  }
  std::string key = swift_port::TrimWhitespace(*language, true);
  std::transform(key.begin(), key.end(), key.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  const auto found = ids.find(key);
  return found == ids.end() ? 0 : found->second;
}

std::string Label(const std::map<int, std::string> &labels, int id,
                  const std::string &fallback) {
  const auto found = labels.find(id);
  return found == labels.end() ? fallback : found->second;
}

std::filesystem::path
SentencePieceFile(const std::filesystem::path &model_directory) {
  std::vector<std::filesystem::path> files;
  for (const auto &entry :
       std::filesystem::directory_iterator(model_directory)) {
    if (entry.path().extension() == ".model") {
      files.push_back(entry.path());
    } else {
    }
  }
  if (files.empty()) {
    throw std::runtime_error("no SentencePiece .model file in " +
                             model_directory.string());
  } else {
  }
  // Note (Dayuxiaoshui): the Swift port takes the first .model by name.
  std::sort(files.begin(), files.end(),
            [](const auto &left, const auto &right) {
              return left.filename().string() < right.filename().string();
            });
  return files.front();
}

std::string CFStringToUtf8(CFStringRef string) {
  const CFIndex length = CFStringGetLength(string);
  const CFIndex capacity =
      CFStringGetMaximumSizeForEncoding(length, kCFStringEncodingUTF8) + 1;
  std::string buffer(static_cast<size_t>(capacity), '\0');
  if (!CFStringGetCString(string, buffer.data(), capacity,
                          kCFStringEncodingUTF8)) {
    return "";
  } else {
  }
  buffer.resize(std::char_traits<char>::length(buffer.c_str()));
  return buffer;
}

// One Swift Character: its text, and its NFC form, under which Swift
// compares Characters.
struct Character {
  std::string text;
  std::string canonical;
};

// Note (Dayuxiaoshui): text split into composed character sequences, the
// public CoreFoundation form of the grapheme clusters Array(String) splits
// it into in Swift; the two agree on combining marks, Hangul and surrogate
// pairs.
std::vector<Character> Characters(const std::string &text) {
  std::vector<Character> characters;
  CFStringRef string = CFStringCreateWithBytes(
      kCFAllocatorDefault, reinterpret_cast<const UInt8 *>(text.data()),
      static_cast<CFIndex>(text.size()), kCFStringEncodingUTF8, false);
  if (string == nullptr) {
    return characters;
  } else {
  }
  const CFIndex length = CFStringGetLength(string);
  for (CFIndex index = 0; index < length;) {
    const CFRange range =
        CFStringGetRangeOfComposedCharactersAtIndex(string, index);
    CFStringRef cluster =
        CFStringCreateWithSubstring(kCFAllocatorDefault, string, range);
    CFMutableStringRef canonical =
        CFStringCreateMutableCopy(kCFAllocatorDefault, 0, cluster);
    CFStringNormalize(canonical, kCFStringNormalizationFormC);
    characters.push_back({CFStringToUtf8(cluster), CFStringToUtf8(canonical)});
    CFRelease(canonical);
    CFRelease(cluster);
    index = range.location + range.length;
  }
  CFRelease(string);
  return characters;
}

bool SameCharacters(const std::vector<Character> &left, size_t left_start,
                    const std::vector<Character> &right, size_t right_start,
                    size_t count) {
  for (size_t offset = 0; offset < count; ++offset) {
    if (left[left_start + offset].canonical !=
        right[right_start + offset].canonical) {
      return false;
    } else {
    }
  }
  return true;
}

bool IsAlphanumeric(uint32_t code_point) {
  return CFCharacterSetIsLongCharacterMember(
      CFCharacterSetGetPredefined(kCFCharacterSetAlphaNumeric), code_point);
}

// Note (Dayuxiaoshui): Voxt's
// MLXTranscriptionPlanning.mergeSequentialTranscript on two chunk texts: a text
// the other already holds at its edge is kept once, an overlap of at least 3
// characters between alphanumerics (2 otherwise) is joined, and other texts
// join with a space only between alphanumerics.
std::string MergeSequentialTranscript(const std::string &base,
                                      const std::string &next) {
  const std::string left = swift_port::TrimWhitespace(base, true);
  const std::string right = swift_port::TrimWhitespace(next, true);
  if (left.empty()) {
    return right;
  } else if (right.empty()) {
    return left;
  } else {
  }
  const std::vector<Character> left_characters = Characters(left);
  const std::vector<Character> right_characters = Characters(right);
  const size_t left_count = left_characters.size();
  const size_t right_count = right_characters.size();
  if (right_count <= left_count &&
      SameCharacters(left_characters, left_count - right_count,
                     right_characters, 0, right_count)) {
    return left;
  } else if (left_count <= right_count &&
             SameCharacters(right_characters, 0, left_characters, 0,
                            left_count)) {
    return right;
  } else {
  }
  const bool alphanumeric_edges =
      IsAlphanumeric(qwen3_asr::CodePoints(left).back()) &&
      IsAlphanumeric(qwen3_asr::CodePoints(right).front());
  const size_t minimum_overlap = alphanumeric_edges ? 3 : 2;
  size_t overlap = 0;
  for (size_t candidate = std::min(left_count, right_count); candidate >= 1;
       --candidate) {
    if (SameCharacters(left_characters, left_count - candidate,
                       right_characters, 0, candidate)) {
      overlap = candidate;
      break;
    } else {
    }
  }
  if (overlap >= minimum_overlap) {
    std::string merged = left;
    for (size_t index = overlap; index < right_count; ++index) {
      merged += right_characters[index].text;
    }
    return merged;
  } else {
  }
  return alphanumeric_edges ? left + " " + right : left + right;
}

// [start, end) cut into chunks of at most max_samples, each after the first
// starting overlap_samples before the previous one ends.
std::vector<std::pair<size_t, size_t>> SplitRange(size_t start, size_t end,
                                                  size_t max_samples,
                                                  size_t overlap_samples) {
  if (max_samples == 0 || end - start <= max_samples) {
    return start < end ? std::vector<std::pair<size_t, size_t>>{{start, end}}
                       : std::vector<std::pair<size_t, size_t>>{};
  } else {
  }
  std::vector<std::pair<size_t, size_t>> ranges;
  size_t cursor = start;
  while (cursor < end) {
    const size_t upper = std::min(cursor + max_samples, end);
    ranges.emplace_back(cursor, upper);
    if (upper >= end) {
      break;
    } else {
    }
    cursor = std::max(start, upper > overlap_samples ? upper - overlap_samples
                                                     : size_t{0});
  }
  return ranges;
}

} // namespace

std::pair<size_t, size_t> ChunkSampleCounts(double max_chunk_seconds,
                                            double chunk_overlap_seconds) {
  const double sample_rate = qwen3_asr::kSampleRate;
  if (!(max_chunk_seconds * sample_rate >= 1.0) ||
      !(chunk_overlap_seconds >= 0.0)) {
    throw std::invalid_argument(
        "a chunk must hold a sample and its overlap must be nonnegative");
  } else {
  }
  const size_t max_samples =
      static_cast<size_t>(max_chunk_seconds * sample_rate);
  const size_t overlap_samples =
      static_cast<size_t>(chunk_overlap_seconds * sample_rate);
  if (overlap_samples >= max_samples) {
    throw std::invalid_argument("the overlap must be shorter than a chunk");
  } else {
  }
  return {max_samples, overlap_samples};
}

SenseVoiceTranscriber::SenseVoiceTranscriber(
    const std::filesystem::path &model_directory)
    : model_(model_directory), vocabulary_(SentencePieceFile(model_directory)) {
}

SenseVoiceTranscriber::Pass
SenseVoiceTranscriber::TranscribePass(const std::vector<float> &samples,
                                      int language_id, bool use_itn) const {
  static const std::map<int, std::string> kLanguages = {
      {24884, "zh"}, {24885, "en"}, {24888, "yue"},
      {24892, "ja"}, {24896, "ko"}, {24992, "nospeech"}};
  static const std::map<int, std::string> kEmotions = {
      {25001, "happy"},     {25002, "sad"},     {25003, "angry"},
      {25004, "neutral"},   {25005, "fearful"}, {25006, "disgusted"},
      {25007, "surprised"}, {25008, "other"},   {25009, "unk"}};
  static const std::map<int, std::string> kEvents = {{24993, "Speech"},
                                                     {24995, "BGM"},
                                                     {24997, "Laughter"},
                                                     {24999, "Applause"}};
  const mx::array log_probabilities =
      model_.LogProbabilities(model_.Features(samples), language_id, use_itn);
  const mx::array predictions =
      mx::astype(mx::argmax(log_probabilities, -1), mx::int32);
  mx::eval(predictions);
  const int32_t *ids = predictions.data<int32_t>();
  const int frame_count = static_cast<int>(predictions.size());
  // Note (Dayuxiaoshui): the Swift port reads the language from the first
  // query frame, the emotion from the second and the event from the third.
  Pass pass;
  pass.language = Label(kLanguages, ids[0], "unknown");
  pass.emotion = Label(kEmotions, ids[1], "token_" + std::to_string(ids[1]));
  pass.event = Label(kEvents, ids[2], "token_" + std::to_string(ids[2]));
  std::vector<int> tokens;
  std::optional<int> previous;
  for (int frame = kQueryFrameCount; frame < frame_count; ++frame) {
    const int id = ids[frame];
    if (previous != id) {
      if (id != kBlankId) {
        tokens.push_back(id);
      } else {
      }
      previous = id;
    } else {
    }
  }
  pass.text = swift_port::TrimWhitespace(vocabulary_.Decode(tokens), false);
  pass.token_count = static_cast<int>(tokens.size());
  return pass;
}

qwen3_asr::TranscriptionResult
SenseVoiceTranscriber::Transcribe(const std::vector<float> &samples,
                                  const SenseVoiceOptions &options,
                                  const std::atomic<bool> &cancel) const {
  const int language_id = LanguageQueryId(options.language);
  const double sample_rate = model_.config().sample_rate;
  qwen3_asr::TranscriptionResult result;
  result.finish_reason = qwen3_asr::FinishReason::kStop;
  const auto add_segment = [&](const Pass &pass, size_t start, size_t end) {
    qwen3_asr::SpeakerSegment segment;
    segment.start_seconds = static_cast<double>(start) / sample_rate;
    segment.end_seconds = static_cast<double>(end) / sample_rate;
    segment.text = pass.text;
    segment.language = pass.language;
    segment.emotion = pass.emotion;
    segment.event = pass.event;
    result.segments.push_back(std::move(segment));
    result.generated_token_count += pass.token_count;
  };
  if (options.voice_activity_detector == nullptr) {
    if (cancel.load()) {
      throw qwen3_asr::TranscriptionCancelled();
    } else {
    }
    const Pass pass = TranscribePass(samples, language_id, options.use_itn);
    result.text = pass.text;
    result.language = pass.language;
    add_segment(pass, 0, samples.size());
    return result;
  } else {
  }

  const std::vector<silero_vad::Timestamp> speech =
      silero_vad::ProbabilitiesToTimestamps(
          options.voice_activity_detector->PredictProbabilities(samples),
          static_cast<long>(samples.size()), options.speech);
  const auto [max_samples, overlap_samples] = ChunkSampleCounts(
      options.max_chunk_seconds, options.chunk_overlap_seconds);
  std::map<std::string, int> language_counts;
  std::vector<std::string> language_order;
  for (const silero_vad::Timestamp &stamp : speech) {
    const size_t start = static_cast<size_t>(
        std::clamp<long>(stamp.start, 0, static_cast<long>(samples.size())));
    const size_t end = static_cast<size_t>(
        std::clamp<long>(stamp.end, 0, static_cast<long>(samples.size())));
    for (const auto &[chunk_start, chunk_end] :
         SplitRange(start, end, max_samples, overlap_samples)) {
      if (cancel.load()) {
        throw qwen3_asr::TranscriptionCancelled();
      } else {
      }
      const Pass pass =
          TranscribePass(std::vector<float>(samples.begin() + chunk_start,
                                            samples.begin() + chunk_end),
                         language_id, options.use_itn);
      result.text = MergeSequentialTranscript(result.text, pass.text);
      if (!swift_port::TrimWhitespace(pass.text, true).empty() &&
          language_counts[pass.language]++ == 0) {
        language_order.push_back(pass.language);
      } else {
      }
      add_segment(pass, chunk_start, chunk_end);
    }
  }
  // Note (Dayuxiaoshui): the most frequent language, the earliest seen on a
  // tie, as Voxt aggregates its SenseVoice metadata.
  for (const std::string &language : language_order) {
    if (!result.language.has_value() ||
        language_counts[language] > language_counts[*result.language]) {
      result.language = language;
    } else {
    }
  }
  return result;
}

} // namespace sensevoice
