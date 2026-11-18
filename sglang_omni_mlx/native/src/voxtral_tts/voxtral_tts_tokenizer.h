// SPDX-License-Identifier: Apache-2.0
// Voxtral TTS's Tekken tokenizer (byte-pair ranks split by tekken.json's
// pattern) and the speech prompt around a preset voice.
#pragma once

#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace voxtral_tts {

class SpeechTokenizer {
public:
  // Reads tekken.json from the checkpoint.
  explicit SpeechTokenizer(const std::filesystem::path &model_directory);
  ~SpeechTokenizer();
  SpeechTokenizer(const SpeechTokenizer &) = delete;
  SpeechTokenizer &operator=(const SpeechTokenizer &) = delete;

  // Text to token ids; special-token text encodes as ordinary text. Throws
  // std::invalid_argument for text that is not UTF-8.
  std::vector<int> Encode(const std::string &text) const;
  // [BOS][BEGIN_AUDIO], one [AUDIO] per voice frame, [NEXT_AUDIO_TEXT], the
  // text, [REPEAT_AUDIO_TEXT][BEGIN_AUDIO]. Throws std::invalid_argument for
  // a voice the checkpoint does not list.
  std::vector<int> EncodeSpeechRequest(const std::string &text,
                                       const std::string &voice) const;
  // Each preset voice's frame count.
  const std::map<std::string, int> &voice_frame_counts() const {
    return voice_frame_counts_;
  }
  int audio_token_id() const { return audio_token_id_; }

private:
  struct Pcre2Regex;

  void EncodePiece(const std::string &piece, std::vector<int> &ids) const;

  std::unordered_map<std::string, int> ranks_;
  int special_token_count_ = 0;
  int begin_of_sequence_token_id_ = 0;
  int begin_audio_token_id_ = 0;
  int audio_token_id_ = 0;
  int text_to_audio_token_id_ = 0;
  int audio_to_text_token_id_ = 0;
  std::map<std::string, int> voice_frame_counts_;
  std::unique_ptr<Pcre2Regex> split_pattern_;
};

} // namespace voxtral_tts
