// SPDX-License-Identifier: Apache-2.0
#include "qwen3_omni/preprocessing.h"
#include <cmath>

namespace qwen3_omni {
namespace {
float next_position(const std::vector<std::array<float, 3>> &positions) {
  const auto &last = positions.back();
  return *std::max_element(last.begin(), last.end()) + 1;
}
} // namespace
std::pair<mx::array, float>
position_ids(const std::vector<int> &tokens,
             const std::vector<MediaEntry> &entries) {
  std::vector<float> positions(3 * tokens.size());
  size_t index = 0, entry_index = 0;
  float position = 0;
  while (index < tokens.size()) {
    if (entry_index < entries.size() &&
        tokens[index] == entries[entry_index].tokens.front()) {
      const auto &entry = entries[entry_index];
      if (index + entry.tokens.size() > tokens.size() ||
          !std::equal(entry.tokens.begin(), entry.tokens.end(),
                      tokens.begin() + index)) {
        throw std::invalid_argument("template/media token ordering mismatch");
      } else {
      }
      for (size_t i = 0; i < entry.positions.size(); ++i) {
        for (int axis = 0; axis < 3; ++axis) {
          positions[axis * tokens.size() + index + i] =
              position + entry.positions[i][axis];
        }
      }
      position += next_position(entry.positions);
      index += entry.tokens.size();
      ++entry_index;
    } else {
      if (tokens[index] == 151652 || tokens[index] == 151669 ||
          tokens[index] == 151655 || tokens[index] == 151656 ||
          tokens[index] == 151675) {
        throw std::invalid_argument("unmatched media placeholder");
      } else {
      }
      for (int axis = 0; axis < 3; ++axis) {
        positions[axis * tokens.size() + index] = position;
      }
      ++position;
      ++index;
    }
  }
  if (entry_index != entries.size() || positions.empty()) {
    throw std::invalid_argument("template removed media or empty prompt");
  } else {
    float delta = *std::max_element(positions.begin(), positions.end()) + 1 -
                  float(tokens.size());
    return {mx::array(positions.data(), {3, 1, int(tokens.size())}), delta};
  }
}
PreparedInput prepare(const ChatRequest &request,
                      const omni_common::Tokenizer &tokenizer,
                      const Cancellation &cancelled) {
  std::string prompt;
  size_t media_token_count = 0;
  std::vector<MediaEntry> entries;
  std::optional<double> video_cadence;
  int last_user = 0;
  for (int i = 0; i < int(request.messages.size()); ++i) {
    if (request.messages[i].role == "user") {
      last_user = i;
    } else {
    }
  }
  for (int message_index = 0; message_index < int(request.messages.size());
       ++message_index) {
    const auto &message = request.messages[message_index];
    auto parts = message.content;
    if (message_index == last_user) {
      parts.insert(parts.begin(), request.extra_media.begin(),
                   request.extra_media.end());
    } else {
    }
    std::string content;
    for (const auto &part : parts) {
      check_cancelled(cancelled);
      if (part.kind == ContentKind::Text) {
        content += part.content;
        continue;
      } else {
      }
      MediaEntry entry;
      entry.user_owned = message.role == "user";
      int vision_start = tokenizer.AddedTokenId("<|vision_start|>");
      int vision_end = tokenizer.AddedTokenId("<|vision_end|>");
      int audio_start = tokenizer.AddedTokenId("<|audio_start|>");
      int audio_end = tokenizer.AddedTokenId("<|audio_end|>");
      if (part.kind == ContentKind::Audio) {
        entry.mel = audio_mel(audio_samples(part.content, cancelled));
        int count = AudioEncoder::output_length(entry.mel->shape(1));
        entry.tokens.push_back(audio_start);
        entry.tokens.insert(entry.tokens.end(), count, 151675);
        entry.tokens.push_back(audio_end);
        for (int i = 0; i < int(entry.tokens.size()); ++i) {
          entry.positions.push_back({float(i), float(i), float(i)});
        }
      } else {
        auto visual = part.kind == ContentKind::Video
                          ? video_pixels(part.content, request, cancelled)
                          : image_pixels(part.content, cancelled);
        entry.pixels = visual.patches;
        entry.grid = visual.grid;
        entry.visual_token = part.kind == ContentKind::Video ? 151656 : 151655;
        double seconds_per_grid =
            part.kind == ContentKind::Video ? 2 / visual.sampled_fps : 1;
        if (part.kind == ContentKind::Video) {
          if (video_cadence &&
              std::abs(*video_cadence - visual.sampled_fps) > 1e-6) {
            throw std::invalid_argument(
                "videos must share effective sampled FPS");
          } else {
            video_cadence = visual.sampled_fps;
          }
          if (request.use_audio_in_video) {
            entry.mel = audio_mel(audio_samples(part.content, cancelled));
          } else {
          }
        } else {
        }
        std::vector<std::array<float, 3>> visual_positions;
        float start = entry.mel ? 2 : 1;
        for (int frame = 0; frame < visual.grid.frames; ++frame) {
          for (int row = 0; row < visual.grid.height / 2; ++row) {
            for (int column = 0; column < visual.grid.width / 2; ++column) {
              visual_positions.push_back(
                  {float(frame * seconds_per_grid * 13) + start, row + start,
                   column + start});
            }
          }
        }
        entry.tokens.push_back(vision_start);
        entry.positions.push_back({0, 0, 0});
        if (entry.mel) {
          entry.tokens.push_back(audio_start);
          entry.positions.push_back({1, 1, 1});
        } else {
        }
        int audio_count =
            entry.mel ? AudioEncoder::output_length(entry.mel->shape(1)) : 0;
        int visual_index = 0, audio_index = 0;
        while (visual_index < int(visual_positions.size()) ||
               audio_index < audio_count) {
          if (visual_index < int(visual_positions.size()) &&
              (audio_index >= audio_count ||
               visual_positions[visual_index][0] <= audio_index + start)) {
            entry.tokens.push_back(entry.visual_token);
            entry.positions.push_back(visual_positions[visual_index++]);
          } else {
            entry.tokens.push_back(151675);
            entry.positions.push_back({audio_index + start, audio_index + start,
                                       audio_index + start});
            ++audio_index;
          }
        }
        float maximum = next_position(entry.positions);
        if (entry.mel) {
          entry.tokens.push_back(audio_end);
          entry.positions.push_back({maximum, maximum, maximum});
          ++maximum;
        } else {
        }
        entry.tokens.push_back(vision_end);
        entry.positions.push_back({maximum, maximum, maximum});
      }
      content += tokenizer.Decode(entry.tokens, false);
      media_token_count += entry.tokens.size();
      entries.push_back(std::move(entry));
      if (media_token_count > 8192) {
        throw std::invalid_argument("expanded prompt exceeds 8192 tokens");
      } else {
      }
    }
    if (message.role == "assistant") {
      std::string reasoning;
      size_t end_thinking = content.find("</think>");
      if (end_thinking != std::string::npos) {
        reasoning = content.substr(0, end_thinking);
        auto start_thinking = reasoning.rfind("<think>");
        reasoning = reasoning.substr(
            start_thinking == std::string::npos ? 0 : start_thinking + 7);
        reasoning.erase(0, reasoning.find_first_not_of('\n'));
        reasoning.erase(reasoning.find_last_not_of('\n') + 1);
        content = content.substr(content.rfind("</think>") + 8);
        content.erase(0, content.find_first_not_of('\n'));
      } else {
      }
      if (message_index > last_user &&
          (message_index + 1 == int(request.messages.size()) ||
           !reasoning.empty())) {
        content.erase(0, content.find_first_not_of('\n'));
        content = "<think>\n" + reasoning + "\n</think>\n\n" + content;
      } else {
      }
    } else {
    }
    prompt += "<|im_start|>" + message.role + "\n" + content + "<|im_end|>\n";
  }
  prompt += "<|im_start|>assistant\n";
  auto tokens = tokenizer.Encode(prompt);
  if (tokens.size() > 8192) {
    throw std::invalid_argument("expanded prompt exceeds 8192 tokens");
  } else {
  }
  auto [positions, delta] = position_ids(tokens, entries);
  return {std::move(tokens), positions, delta, std::move(entries)};
}
} // namespace qwen3_omni
