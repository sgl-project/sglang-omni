// SPDX-License-Identifier: Apache-2.0
#include "fishaudio_drama3/fishaudio_drama3_prompt.h"

#include <stdexcept>

#include "fishaudio_drama3/fishaudio_drama3_text.h"

namespace fishaudio_drama3 {

namespace mx = mlx::core;

void CheckCodes(const mx::array &codes, int codebook_count, int codebook_size,
                const std::string &location) {
  if (codes.ndim() != 2 || codes.shape(0) != codebook_count) {
    throw std::invalid_argument(location + ": codes must be [" +
                                std::to_string(codebook_count) + ", frames]");
  } else if (codes.shape(1) == 0) {
    throw std::invalid_argument(location + ": codes are empty");
  } else if (!mx::issubdtype(codes.dtype(), mx::integer)) {
    throw std::invalid_argument(location + ": codes must be integers");
  } else if (mx::min(mx::astype(codes, mx::int64)).item<int64_t>() < 0 ||
             mx::max(mx::astype(codes, mx::int64)).item<int64_t>() >=
                 codebook_size) {
    throw std::invalid_argument(location + ": codes must be in [0, " +
                                std::to_string(codebook_size) + ")");
  } else {
  }
}

PromptFrames BuildFrames(const omni_common::Tokenizer &tokenizer,
                         const std::vector<SpeechTurn> &references,
                         const std::vector<SpeechTurn> &history,
                         const std::string &target, int codebook_count,
                         int codebook_size, int semantic_begin) {
  ValidateText(target, "text");
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
      const std::string location = "reference " + std::to_string(index + 1);
      ValidateText(references[index].text, location);
      CheckCodes(references[index].codes, codebook_count, codebook_size,
                 location);
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
    const std::string location = "history " + std::to_string(index + 1);
    ValidateText(history[index].text, location);
    CheckCodes(history[index].codes, codebook_count, codebook_size, location);
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
