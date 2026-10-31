// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "qwen3_omni/model.h"

namespace qwen3_omni {
enum class ContentKind { Text, Image, Audio, Video };
struct ContentPart {
  ContentKind kind;
  std::string content;
};
struct ChatMessage {
  std::string role;
  std::vector<ContentPart> content;
};
struct Sampling {
  float temperature;
  float top_p;
  int top_k;
  float repetition_penalty;
};
struct ChatRequest {
  std::string model;
  bool stream;
  bool include_usage;
  bool speech;
  std::string voice;
  Sampling sampling;
  Sampling talker_sampling;
  int max_tokens;
  int max_speech_frames;
  std::optional<uint64_t> seed;
  std::vector<ChatMessage> messages;
  std::vector<ContentPart> extra_media;
  double video_fps;
  int video_max_frames;
  int video_min_pixels;
  int video_max_pixels;
  int video_total_pixels;
  bool use_audio_in_video;
};
ChatRequest
parse_request(const Json &root, const std::string &model_name,
              const std::vector<std::filesystem::path> &allowed_paths);
void validate_media(const std::string &source,
                    const std::vector<std::filesystem::path> &allowed_paths);
std::vector<uint8_t> decode_base64(const std::string &encoded);
std::string encode_base64(const std::vector<uint8_t> &bytes);
} // namespace qwen3_omni
