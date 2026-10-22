// SPDX-License-Identifier: Apache-2.0
// Tekken speech prompt encoding for Voxtral TTS.
#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "nlohmann/json.hpp"

namespace voxtral_tts {

class SpeechTokenizer {
public:
  explicit SpeechTokenizer(const std::filesystem::path &checkpoint_directory);
  ~SpeechTokenizer();

  std::vector<int> EncodeSpeechRequest(const std::string &text,
                                       const std::string &voice) const;

private:
  struct Regex;

  std::vector<int> EncodeText(const std::string &text) const;
  std::vector<int> EncodePiece(const std::string &piece) const;

  int special_token_count_ = 0;
  int beginning_of_sequence_id_ = -1;
  int begin_audio_id_ = -1;
  int audio_id_ = -1;
  int text_to_audio_id_ = -1;
  int audio_to_text_id_ = -1;
  std::unordered_map<std::string, int> byte_ranks_;
  std::unordered_map<std::string, int> voice_audio_token_counts_;
  std::unique_ptr<Regex> split_pattern_;
};

} // namespace voxtral_tts
