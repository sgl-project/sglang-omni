// SPDX-License-Identifier: Apache-2.0
#include "fishaudio_drama3/fishaudio_drama3_prompt.h"

#include <stdexcept>

#include "fishaudio_drama3/fishaudio_drama3_text.h"

namespace fishaudio_drama3 {

namespace mx = mlx::core;

std::vector<std::vector<int32_t>>
ReferenceCodes(const nlohmann::ordered_json &vq_codes, int codebook_count,
               int codebook_size, size_t max_frames,
               const std::string &location) {
  const std::string shape_error = location + ".vq_codes must be a [" +
                                  std::to_string(codebook_count) +
                                  ", frames] grid of integers";
  if (!vq_codes.is_array() ||
      vq_codes.size() != static_cast<size_t>(codebook_count) ||
      !vq_codes.at(0).is_array() || vq_codes.at(0).empty()) {
    throw std::invalid_argument(shape_error);
  } else if (vq_codes.at(0).size() > max_frames) {
    throw std::invalid_argument(location + ".vq_codes must have at most " +
                                std::to_string(max_frames) + " frames");
  } else {
  }
  const size_t frame_count = vq_codes.at(0).size();
  std::vector<std::vector<int32_t>> codes;
  for (const nlohmann::ordered_json &row : vq_codes) {
    if (!row.is_array() || row.size() != frame_count) {
      throw std::invalid_argument(shape_error);
    } else {
    }
    std::vector<int32_t> &codebook = codes.emplace_back();
    for (const nlohmann::ordered_json &code : row) {
      if (!code.is_number_integer()) {
        throw std::invalid_argument(shape_error);
      } else if (code.get<int64_t>() < 0 ||
                 code.get<int64_t>() >= codebook_size) {
        throw std::invalid_argument(location + ".vq_codes must be in [0, " +
                                    std::to_string(codebook_size) + ")");
      } else {
      }
      codebook.push_back(code.get<int32_t>());
    }
  }
  return codes;
}

PromptFrames BuildFrames(const omni_common::Tokenizer &tokenizer,
                         const std::vector<SpeechTurn> &references,
                         const std::vector<SpeechTurn> &history,
                         const std::string &target, int codebook_count,
                         int semantic_begin) {
  // A segment is text, or the index of a turn's codes.
  struct Segment {
    std::string text;
    const mx::array *codes = nullptr;
  };
  std::vector<Segment> segments = {{"<|im_start|>system\n"}};
  if (!references.empty()) {
    segments.push_back(
        {"convert the provided text to speech reference to the following:"});
    for (size_t index = 0; index < references.size(); ++index) {
      segments.push_back(
          {"\n\n[Reference " + std::to_string(index + 1) + "]\nText: " +
           ReferenceTranscript(references[index].text) + "\nSpeech: "});
      segments.push_back({"", &references[index].codes});
    }
  } else {
    segments.push_back({"convert the provided text to speech"});
  }
  segments.push_back({"<|im_end|>\n"});
  const size_t system_segments = segments.size();
  for (size_t index = 0; index < history.size(); ++index) {
    segments.push_back({"<|im_start|>user\n"});
    segments.push_back({WithSpeakerTag(history[index].text)});
    segments.push_back({"<|im_end|>\n"});
    segments.push_back({"<|im_start|>assistant\n<|voice|>"});
    segments.push_back({"", &history[index].codes});
    segments.push_back({"<|im_end|>\n"});
  }
  segments.push_back({"<|im_start|>user\n"});
  segments.push_back({WithSpeakerTag(target)});
  segments.push_back({"<|im_end|>\n"});
  segments.push_back({"<|im_start|>assistant\n<|voice|>"});

  const int frame_width = 1 + codebook_count;
  std::vector<mx::array> blocks;
  int system_rows = 0;
  for (size_t index = 0; index < segments.size(); ++index) {
    mx::array block = mx::zeros({0, frame_width}, mx::int32);
    if (segments[index].codes == nullptr) {
      const std::vector<int> ids = tokenizer.Encode(segments[index].text);
      std::vector<int32_t> rows(ids.size() * frame_width, 0);
      for (size_t row = 0; row < ids.size(); ++row) {
        rows[row * frame_width] = ids[row];
      }
      block = mx::array(rows.data(),
                        {static_cast<int>(ids.size()), frame_width}, mx::int32);
    } else {
      const mx::array codes = mx::astype(*segments[index].codes, mx::int32);
      const mx::array main_tokens =
          mx::add(mx::slice(codes, {0, 0}, {1, codes.shape(1)}),
                  mx::array(semantic_begin, mx::int32));
      block = mx::transpose(mx::concatenate({main_tokens, codes}, 0));
    }
    if (index < system_segments) {
      system_rows += block.shape(0);
    } else {
    }
    blocks.push_back(block);
  }
  mx::array frames = mx::concatenate(blocks, 0);
  mx::eval(frames);
  return {frames, system_rows};
}

} // namespace fishaudio_drama3
